"""
main.py
-------
Main file.  Allocates all GPU arrays, applies initial conditions, then starts the SSP-RK3 time loop until t = T_END.

Per-stage  (repeated 3× per time step):
    cons → prim  →  periodic BCs  →  cons
    velocity derivatives  →  Ducros sensor
    zero residual
    WCNS-X  →  HLLC-X  →  residual
    WCNS-Y  →  HLLC-Y  →  residual
    WCNS-Z  →  HLLC-Z  →  residual
    viscous derivatives  →  temperature  →  temperature gradients
    viscous-X  →  viscous-Y  →  viscous-Z  →  residual
    SSP-RK3 update
"""

import time
import numpy as np
import warp as wp

from constants import (
    NX, NY, NZ, GX, GY, GZ, G, GHOSTP,
    NTMAX, FILE_SAVE, CFL_PY, T_END,
    XMIN_PY, XMAX_PY, YMIN_PY, YMAX_PY, ZMIN_PY, ZMAX_PY,
    DEVICE, USE_SUTH,
)
from initial_bc   import (init_cond_kernel, bc_periodic_x_kernel,
                          bc_periodic_y_kernel, bc_periodic_z_kernel,
                          prim_to_cons_kernel, cons_to_prim_kernel)
from sensors      import velocity_deriv_kernel, ducros_kernel
from wcns         import wcns_x_kernel, wcns_y_kernel, wcns_z_kernel
from fluxes       import (flux_x_residual_kernel, flux_y_residual_kernel,
                          flux_z_residual_kernel)
from viscous      import (viscous_deriv_2nd_kernel, temperature_kernel,
                          temperature_deriv_kernel,
                          viscous_flux_x_kernel, viscous_flux_y_kernel,
                          viscous_flux_z_kernel)
from timestepping import rk3_step_kernel, zero_residual_kernel, dt_local_kernel
from diagnostics  import save_npz, compute_ke, compute_enstrophy, save_png_slice


def main():
    print(f"3D TGV  N={NX}  device={DEVICE}  layout=SoA cons[v,ix,iy,iz]")

    dx_py = (XMAX_PY-XMIN_PY)/NX
    dy_py = (YMAX_PY-YMIN_PY)/NY
    dz_py = (ZMAX_PY-ZMIN_PY)/NZ

    x_np = np.array([XMIN_PY+(i-G-0.5)*dx_py for i in range(GX)], dtype=np.float64)
    y_np = np.array([YMIN_PY+(j-G-0.5)*dy_py for j in range(GY)], dtype=np.float64)
    z_np = np.array([ZMIN_PY+(k-G-0.5)*dz_py for k in range(GZ)], dtype=np.float64)

    x_d = wp.array(x_np, dtype=wp.float64, device=DEVICE)
    y_d = wp.array(y_np, dtype=wp.float64, device=DEVICE)
    z_d = wp.array(z_np, dtype=wp.float64, device=DEVICE)

    # Primitives: (GX,GY,GZ) AoS scalar fields
    rho_d  = wp.zeros((GX,GY,GZ), dtype=wp.float64, device=DEVICE)
    u_d    = wp.zeros((GX,GY,GZ), dtype=wp.float64, device=DEVICE)
    v_d    = wp.zeros((GX,GY,GZ), dtype=wp.float64, device=DEVICE)
    w_d    = wp.zeros((GX,GY,GZ), dtype=wp.float64, device=DEVICE)
    p_d    = wp.zeros((GX,GY,GZ), dtype=wp.float64, device=DEVICE)
    snd_d  = wp.zeros((GX,GY,GZ), dtype=wp.float64, device=DEVICE)

    # Conservatives: SoA (5,GX,GY,GZ) — coalesced along z
    cons0  = wp.zeros((5,GX,GY,GZ), dtype=wp.float64, device=DEVICE)
    cons1  = wp.zeros((5,GX,GY,GZ), dtype=wp.float64, device=DEVICE)
    cons2  = wp.zeros((5,GX,GY,GZ), dtype=wp.float64, device=DEVICE)
    consl  = wp.zeros((5,GX,GY,GZ), dtype=wp.float64, device=DEVICE)
    consr  = wp.zeros((5,GX,GY,GZ), dtype=wp.float64, device=DEVICE)
    resid  = wp.zeros((5,GX,GY,GZ), dtype=wp.float64, device=DEVICE)

    # Velocity gradient arrays (shared by Ducros sensor and viscous fluxes)
    der_ux = wp.zeros((GX,GY,GZ), dtype=wp.float64, device=DEVICE)
    der_uy = wp.zeros((GX,GY,GZ), dtype=wp.float64, device=DEVICE)
    der_uz = wp.zeros((GX,GY,GZ), dtype=wp.float64, device=DEVICE)
    der_vx = wp.zeros((GX,GY,GZ), dtype=wp.float64, device=DEVICE)
    der_vy = wp.zeros((GX,GY,GZ), dtype=wp.float64, device=DEVICE)
    der_vz = wp.zeros((GX,GY,GZ), dtype=wp.float64, device=DEVICE)
    der_wx = wp.zeros((GX,GY,GZ), dtype=wp.float64, device=DEVICE)
    der_wy = wp.zeros((GX,GY,GZ), dtype=wp.float64, device=DEVICE)
    der_wz = wp.zeros((GX,GY,GZ), dtype=wp.float64, device=DEVICE)
    ducros = wp.zeros((GX,GY,GZ), dtype=wp.float64, device=DEVICE)

    # Viscous flux arrays
    temp_d = wp.zeros((GX,GY,GZ), dtype=wp.float64, device=DEVICE)
    dTx_d  = wp.zeros((GX,GY,GZ), dtype=wp.float64, device=DEVICE)
    dTy_d  = wp.zeros((GX,GY,GZ), dtype=wp.float64, device=DEVICE)
    dTz_d  = wp.zeros((GX,GY,GZ), dtype=wp.float64, device=DEVICE)

    # GPU dt: single element, atomic min — only 8 bytes per step to CPU
    dt_min = wp.zeros(1, dtype=wp.float64, device=DEVICE)

    # ── Initial conditions ────────────────────────────────────────────────────
    wp.launch(init_cond_kernel, dim=(GX,GY,GZ),
              inputs=[x_d,y_d,z_d,rho_d,u_d,v_d,w_d,p_d,snd_d], device=DEVICE)
    wp.launch(prim_to_cons_kernel, dim=(GX,GY,GZ),
              inputs=[rho_d,u_d,v_d,w_d,p_d,cons0], device=DEVICE)

    rho_np=rho_d.numpy(); u_np=u_d.numpy()
    v_np=v_d.numpy();     w_np=w_d.numpy(); p_np=p_d.numpy()
    ke0  = compute_ke(rho_np, u_np, v_np, w_np)
    ent0 = compute_enstrophy(u_np, v_np, w_np, dx_py, dy_py, dz_py)
    print(f"t=0  KE={ke0:.6f}  Enstrophy={ent0:.6f}")
    save_npz(rho_np, u_np, v_np, w_np, p_np, x_np, y_np, z_np, step=0, time_sim=0.0)

    fke = open("ke_tgv3d.txt", "w")
    fke.write("# time  KE/KE0  Enstrophy/Ent0\n")
    fke.write(f"0.000000  1.000000  1.000000\n"); fke.flush()

    time_sim = 0.0; N = 1; t0 = time.perf_counter()

    while time_sim < T_END and N <= NTMAX:

        # ── GPU CFL — atomic min, reset then launch ───────────────────────────
        dt_min.fill_(wp.float64(1.0e30))
        wp.launch(dt_local_kernel, dim=(NX+2,NY+2,NZ+2),
                  inputs=[u_d,v_d,w_d,p_d,rho_d,snd_d,dt_min,
                          wp.float64(dx_py),wp.float64(dy_py),wp.float64(dz_py),
                          NX,NY,NZ,G],
                  device=DEVICE)
        dt = float(CFL_PY * dt_min.numpy()[0])
        if time_sim+dt > T_END: dt = T_END-time_sim
        time_sim += dt

        if N % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  step {N:5d}  t={time_sim:.4e}  dt={dt:.3e}  wall={elapsed:.1f}s")

        # ── SSP-RK3 ───────────────────────────────────────────────────────────
        rk_stages = [
            (0.0,  1.0,  cons0, cons1),   # u1 = u0 + dt*L(u0)
            (0.75, 0.25, cons1, cons2),   # u2 = 3/4*u0 + 1/4*(u1+dt*L(u1))
            (1/3,  2/3,  cons2, cons0),   # u0 = 1/3*u0 + 2/3*(u2+dt*L(u2))
        ]
        for alpha, beta, c_in, c_out in rk_stages:
            wp.launch(cons_to_prim_kernel, dim=(NX+2,NY+2,NZ+2),
                      inputs=[c_in,rho_d,u_d,v_d,w_d,p_d,NX,NY,NZ,G], device=DEVICE)
            wp.launch(bc_periodic_x_kernel, dim=(G,GY,GZ),
                      inputs=[rho_d,u_d,v_d,w_d,p_d,NX,NY,NZ,G], device=DEVICE)
            wp.launch(bc_periodic_y_kernel, dim=(GX,G,GZ),
                      inputs=[rho_d,u_d,v_d,w_d,p_d,NX,NY,NZ,G], device=DEVICE)
            wp.launch(bc_periodic_z_kernel, dim=(GX,GY,G),
                      inputs=[rho_d,u_d,v_d,w_d,p_d,NX,NY,NZ,G], device=DEVICE)
            wp.launch(prim_to_cons_kernel, dim=(GX,GY,GZ),
                      inputs=[rho_d,u_d,v_d,w_d,p_d,c_in], device=DEVICE)

            wp.launch(velocity_deriv_kernel, dim=(GX-2,GY-2,GZ-2),
                      inputs=[u_d,v_d,w_d,
                               der_ux,der_uy,der_uz,der_vx,der_vy,der_vz,
                               der_wx,der_wy,der_wz,GX,GY,GZ], device=DEVICE)
            wp.launch(ducros_kernel, dim=(GX-2,GY-2,GZ-2),
                      inputs=[der_ux,der_uy,der_uz,der_vx,der_vy,der_vz,
                               der_wx,der_wy,der_wz,ducros,GX,GY,GZ], device=DEVICE)

            wp.launch(zero_residual_kernel, dim=(NX+2,NY+2,NZ+2),
                      inputs=[resid,NX,NY,NZ,G], device=DEVICE)

            # Convective fluxes (X → Y → Z)
            wp.launch(wcns_x_kernel, dim=(GX,GY,GZ),
                      inputs=[c_in,p_d,ducros,consl,consr,NX,NY,NZ,G], device=DEVICE)
            wp.launch(flux_x_residual_kernel, dim=(NX+2,NY+2,NZ+2),
                      inputs=[c_in,consl,consr,resid,wp.float64(dx_py),NX,NY,NZ,G],
                      device=DEVICE)

            wp.launch(wcns_y_kernel, dim=(GX,GY,GZ),
                      inputs=[c_in,p_d,ducros,consl,consr,NX,NY,NZ,G], device=DEVICE)
            wp.launch(flux_y_residual_kernel, dim=(NX+2,NY+2,NZ+2),
                      inputs=[c_in,consl,consr,resid,wp.float64(dy_py),NX,NY,NZ,G],
                      device=DEVICE)

            wp.launch(wcns_z_kernel, dim=(GX,GY,GZ),
                      inputs=[c_in,p_d,ducros,consl,consr,NX,NY,NZ,G], device=DEVICE)
            wp.launch(flux_z_residual_kernel, dim=(NX+2,NY+2,NZ+2),
                      inputs=[c_in,consl,consr,resid,wp.float64(dz_py),NX,NY,NZ,G],
                      device=DEVICE)

            # Viscous fluxes
            wp.launch(viscous_deriv_2nd_kernel, dim=(GX-2,GY-2,GZ-2),
                      inputs=[u_d,v_d,w_d,
                               der_ux,der_uy,der_uz,der_vx,der_vy,der_vz,
                               der_wx,der_wy,der_wz,GX,GY,GZ], device=DEVICE)
            wp.launch(temperature_kernel, dim=(GX,GY,GZ),
                      inputs=[rho_d,p_d,temp_d,GX,GY,GZ], device=DEVICE)
            wp.launch(temperature_deriv_kernel, dim=(GX-2,GY-2,GZ-2),
                      inputs=[temp_d,dTx_d,dTy_d,dTz_d,GX,GY,GZ], device=DEVICE)
            wp.launch(viscous_flux_x_kernel, dim=(NX+2,NY+2,NZ+2),
                      inputs=[u_d,v_d,w_d,temp_d,
                               der_ux,der_uy,der_uz,der_vx,der_vy,der_vz,
                               der_wx,der_wy,der_wz,dTx_d,resid,
                               wp.float64(dx_py),NX,NY,NZ,G,USE_SUTH], device=DEVICE)
            wp.launch(viscous_flux_y_kernel, dim=(NX+2,NY+2,NZ+2),
                      inputs=[u_d,v_d,w_d,temp_d,
                               der_ux,der_uy,der_uz,der_vx,der_vy,der_vz,
                               der_wx,der_wy,der_wz,dTy_d,resid,
                               wp.float64(dy_py),NX,NY,NZ,G,USE_SUTH], device=DEVICE)
            wp.launch(viscous_flux_z_kernel, dim=(NX+2,NY+2,NZ+2),
                      inputs=[u_d,v_d,w_d,temp_d,
                               der_ux,der_uy,der_uz,der_vx,der_vy,der_vz,
                               der_wx,der_wy,der_wz,dTz_d,resid,
                               wp.float64(dz_py),NX,NY,NZ,G,USE_SUTH], device=DEVICE)

            wp.launch(rk3_step_kernel, dim=(NX+2,NY+2,NZ+2),
                      inputs=[cons0,c_in,resid,c_out,
                               wp.float64(alpha),wp.float64(beta),wp.float64(dt),
                               NX,NY,NZ,G], device=DEVICE)

        wp.launch(cons_to_prim_kernel, dim=(NX+2,NY+2,NZ+2),
                  inputs=[cons0,rho_d,u_d,v_d,w_d,p_d,NX,NY,NZ,G], device=DEVICE)

        if N % 100 == 0:
            rho_np=rho_d.numpy(); u_np=u_d.numpy()
            v_np=v_d.numpy();     w_np=w_d.numpy()
            ke  = compute_ke(rho_np, u_np, v_np, w_np)
            ent = compute_enstrophy(u_np, v_np, w_np, dx_py, dy_py, dz_py)
            fke.write(f"{time_sim:.6f}  {ke/ke0:.10f}  {ent/ent0:.10f}\n"); fke.flush()

        if N % FILE_SAVE == 0:
            rho_np=rho_d.numpy(); u_np=u_d.numpy()
            v_np=v_d.numpy();     w_np=w_d.numpy(); p_np=p_d.numpy()
            save_npz(rho_np,u_np,v_np,w_np,p_np,x_np,y_np,z_np,step=N,time_sim=time_sim)
            save_png_slice(rho_np, step=N, time_sim=time_sim)

        if abs(time_sim-T_END) < 1.0e-12: break
        N += 1

    wp.synchronize()
    elapsed = time.perf_counter()-t0
    print(f"\nDone. steps={N}  wall={elapsed:.2f}s  ({elapsed/N*1000:.1f} ms/step)")

    rho_np=rho_d.numpy(); u_np=u_d.numpy()
    v_np=v_d.numpy();     w_np=w_d.numpy(); p_np=p_d.numpy()
    save_npz(rho_np,u_np,v_np,w_np,p_np,x_np,y_np,z_np,step=N,time_sim=time_sim)
    save_png_slice(rho_np, step=N, time_sim=time_sim)
    ke  = compute_ke(rho_np, u_np, v_np, w_np)
    ent = compute_enstrophy(u_np, v_np, w_np, dx_py, dy_py, dz_py)
    fke.write(f"{time_sim:.6f}  {ke/ke0:.10f}  {ent/ent0:.10f}\n")
    fke.close()


if __name__ == "__main__":
    main()
