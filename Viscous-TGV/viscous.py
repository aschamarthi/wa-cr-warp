"""
viscous.py
----------
Viscous flux kernels for the compressible Navier-Stokes equations.

Uses the Nishikawa alpha=4 face-interpolation scheme (4th-order):
    u_face = 0.5*(u_L + u_R),   where  u_L = u_i + 0.5*du_i  (cell-centred grad)
    du/dn_face = 0.5*(du_L + du_R) + (4/3)*(u_R - u_L) * inv_ds

The 8th-order central stencil for velocity derivatives is retained as
commented-out code in viscous_deriv_2nd_kernel for reference.

Viscous contributions are added to the residual with the opposite sign to
the convective fluxes (resid += viscous divergence).

viscous_stress is a helper function defining the full 3D stress tensor;
the directional flux kernels compute only the stress components they need.
"""

import warp as wp
from constants import (
    GAMMA, HALF, F1, F2, F1_3, F4_3,
    RE, PR, GM1, SUTH_C,
    C8_1, C8_2, C8_3, C8_4,
)


# =============================================================================
#  Sutherland's law  (non-dimensional)
# =============================================================================

@wp.func
def mu_suth(T: wp.float64) -> wp.float64:
    # μ(T) = T^(3/2) * (1 + C) / (T + C),  normalised so that μ(1) = 1.
    return T*wp.sqrt(T)*(F1+SUTH_C)/(T+SUTH_C)


# =============================================================================
#  S_V1  Velocity derivatives — 2nd-order central  (no /ds)
#
#  The 8th-order stencil (commented out) requires GHOSTP≥4 and is retained
#  for reference.  The active 2nd-order stencil is consistent with the
#  viscous flux reconstruction used below.
# =============================================================================

@wp.kernel
def viscous_deriv_2nd_kernel(
    u_vel:wp.array(dtype=wp.float64, ndim=3),
    v_vel:wp.array(dtype=wp.float64, ndim=3),
    w_vel:wp.array(dtype=wp.float64, ndim=3),
    der_ux:wp.array(dtype=wp.float64, ndim=3), der_uy:wp.array(dtype=wp.float64, ndim=3),
    der_uz:wp.array(dtype=wp.float64, ndim=3), der_vx:wp.array(dtype=wp.float64, ndim=3),
    der_vy:wp.array(dtype=wp.float64, ndim=3), der_vz:wp.array(dtype=wp.float64, ndim=3),
    der_wx:wp.array(dtype=wp.float64, ndim=3), der_wy:wp.array(dtype=wp.float64, ndim=3),
    der_wz:wp.array(dtype=wp.float64, ndim=3),
    gx:int, gy:int, gz:int
):
    ti, tj, tk = wp.tid()
    ix = ti+1; iy = tj+1; iz = tk+1
    if ix>=gx-1 or iy>=gy-1 or iz>=gz-1: return

    # 8th-order stencil
    # # Coeff: +C8_1*f(i-4) -C8_2*f(i-3) +C8_3*f(i-2) -C8_4*f(i-1) +C8_4*f(i+1) -C8_3*f(i+2) +C8_2*f(i+3) -C8_1*f(i+4)
    # der_ux[ix,iy,iz]=(C8_1*u_vel[ix-4,iy,iz]-C8_2*u_vel[ix-3,iy,iz]+C8_3*u_vel[ix-2,iy,iz]-C8_4*u_vel[ix-1,iy,iz]+C8_4*u_vel[ix+1,iy,iz]-C8_3*u_vel[ix+2,iy,iz]+C8_2*u_vel[ix+3,iy,iz]-C8_1*u_vel[ix+4,iy,iz])
    # der_uy[ix,iy,iz]=(C8_1*u_vel[ix,iy-4,iz]-C8_2*u_vel[ix,iy-3,iz]+C8_3*u_vel[ix,iy-2,iz]-C8_4*u_vel[ix,iy-1,iz]+C8_4*u_vel[ix,iy+1,iz]-C8_3*u_vel[ix,iy+2,iz]+C8_2*u_vel[ix,iy+3,iz]-C8_1*u_vel[ix,iy+4,iz])
    # der_uz[ix,iy,iz]=(C8_1*u_vel[ix,iy,iz-4]-C8_2*u_vel[ix,iy,iz-3]+C8_3*u_vel[ix,iy,iz-2]-C8_4*u_vel[ix,iy,iz-1]+C8_4*u_vel[ix,iy,iz+1]-C8_3*u_vel[ix,iy,iz+2]+C8_2*u_vel[ix,iy,iz+3]-C8_1*u_vel[ix,iy,iz+4])
    # der_vx[ix,iy,iz]=(C8_1*v_vel[ix-4,iy,iz]-C8_2*v_vel[ix-3,iy,iz]+C8_3*v_vel[ix-2,iy,iz]-C8_4*v_vel[ix-1,iy,iz]+C8_4*v_vel[ix+1,iy,iz]-C8_3*v_vel[ix+2,iy,iz]+C8_2*v_vel[ix+3,iy,iz]-C8_1*v_vel[ix+4,iy,iz])
    # der_vy[ix,iy,iz]=(C8_1*v_vel[ix,iy-4,iz]-C8_2*v_vel[ix,iy-3,iz]+C8_3*v_vel[ix,iy-2,iz]-C8_4*v_vel[ix,iy-1,iz]+C8_4*v_vel[ix,iy+1,iz]-C8_3*v_vel[ix,iy+2,iz]+C8_2*v_vel[ix,iy+3,iz]-C8_1*v_vel[ix,iy+4,iz])
    # der_vz[ix,iy,iz]=(C8_1*v_vel[ix,iy,iz-4]-C8_2*v_vel[ix,iy,iz-3]+C8_3*v_vel[ix,iy,iz-2]-C8_4*v_vel[ix,iy,iz-1]+C8_4*v_vel[ix,iy,iz+1]-C8_3*v_vel[ix,iy,iz+2]+C8_2*v_vel[ix,iy,iz+3]-C8_1*v_vel[ix,iy,iz+4])
    # der_wx[ix,iy,iz]=(C8_1*w_vel[ix-4,iy,iz]-C8_2*w_vel[ix-3,iy,iz]+C8_3*w_vel[ix-2,iy,iz]-C8_4*w_vel[ix-1,iy,iz]+C8_4*w_vel[ix+1,iy,iz]-C8_3*w_vel[ix+2,iy,iz]+C8_2*w_vel[ix+3,iy,iz]-C8_1*w_vel[ix+4,iy,iz])
    # der_wy[ix,iy,iz]=(C8_1*w_vel[ix,iy-4,iz]-C8_2*w_vel[ix,iy-3,iz]+C8_3*w_vel[ix,iy-2,iz]-C8_4*w_vel[ix,iy-1,iz]+C8_4*w_vel[ix,iy+1,iz]-C8_3*w_vel[ix,iy+2,iz]+C8_2*w_vel[ix,iy+3,iz]-C8_1*w_vel[ix,iy+4,iz])
    # der_wz[ix,iy,iz]=(C8_1*w_vel[ix,iy,iz-4]-C8_2*w_vel[ix,iy,iz-3]+C8_3*w_vel[ix,iy,iz-2]-C8_4*w_vel[ix,iy,iz-1]+C8_4*w_vel[ix,iy,iz+1]-C8_3*w_vel[ix,iy,iz+2]+C8_2*w_vel[ix,iy,iz+3]-C8_1*w_vel[ix,iy,iz+4])

    # 2nd-order central differences (no /ds)
    # du/dx, du/dy, du/dz
    der_ux[ix,iy,iz] = HALF*(-u_vel[ix-1,iy,iz]+u_vel[ix+1,iy,iz])
    der_uy[ix,iy,iz] = HALF*(-u_vel[ix,iy-1,iz]+u_vel[ix,iy+1,iz])
    der_uz[ix,iy,iz] = HALF*(-u_vel[ix,iy,iz-1]+u_vel[ix,iy,iz+1])
    # dv/dx, dv/dy, dv/dz
    der_vx[ix,iy,iz] = HALF*(-v_vel[ix-1,iy,iz]+v_vel[ix+1,iy,iz])
    der_vy[ix,iy,iz] = HALF*(-v_vel[ix,iy-1,iz]+v_vel[ix,iy+1,iz])
    der_vz[ix,iy,iz] = HALF*(-v_vel[ix,iy,iz-1]+v_vel[ix,iy,iz+1])
    # dw/dx, dw/dy, dw/dz
    der_wx[ix,iy,iz] = HALF*(-w_vel[ix-1,iy,iz]+w_vel[ix+1,iy,iz])
    der_wy[ix,iy,iz] = HALF*(-w_vel[ix,iy-1,iz]+w_vel[ix,iy+1,iz])
    der_wz[ix,iy,iz] = HALF*(-w_vel[ix,iy,iz-1]+w_vel[ix,iy,iz+1])


# =============================================================================
#  S_V2  Temperature field + 2nd-order gradients  (no /ds)
# =============================================================================

@wp.kernel
def temperature_kernel(
    rho:wp.array(dtype=wp.float64, ndim=3),
    pres:wp.array(dtype=wp.float64, ndim=3),
    temp:wp.array(dtype=wp.float64, ndim=3),
    gx:int, gy:int, gz:int
):
    # Non-dimensional temperature T = p/ρ  (ideal gas, Rgas = 1).
    # Evaluated over the full ghost-padded domain.
    ix, iy, iz = wp.tid()
    if ix>=gx or iy>=gy or iz>=gz: return
    temp[ix,iy,iz] = pres[ix,iy,iz]/rho[ix,iy,iz]   # Rgas=1

@wp.kernel
def temperature_deriv_kernel(
    temp:wp.array(dtype=wp.float64, ndim=3),
    dTx:wp.array(dtype=wp.float64, ndim=3),
    dTy:wp.array(dtype=wp.float64, ndim=3),
    dTz:wp.array(dtype=wp.float64, ndim=3),
    gx:int, gy:int, gz:int
):
    # 2nd-order central temperature gradients (no /ds).
    ti, tj, tk = wp.tid()
    ix = ti+1; iy = tj+1; iz = tk+1
    if ix>=gx-1 or iy>=gy-1 or iz>=gz-1: return
    dTx[ix,iy,iz] = HALF*(-temp[ix-1,iy,iz]+temp[ix+1,iy,iz])
    dTy[ix,iy,iz] = HALF*(-temp[ix,iy-1,iz]+temp[ix,iy+1,iz])
    dTz[ix,iy,iz] = HALF*(-temp[ix,iy,iz-1]+temp[ix,iy,iz+1])


# =============================================================================
#  S_V3  Full stress tensor helper  (reference; not called by flux kernels)
#
#  The directional flux kernels compute only their required stress components
#  inline for efficiency.  This helper is retained for clarity and testing.
# =============================================================================

@wp.func
def viscous_stress(
    du_dx:wp.float64, du_dy:wp.float64, du_dz:wp.float64,
    dv_dx:wp.float64, dv_dy:wp.float64, dv_dz:wp.float64,
    dw_dx:wp.float64, dw_dy:wp.float64, dw_dz:wp.float64,
    uf:wp.float64, vf:wp.float64, wf:wp.float64,
    dT_n:wp.float64, mu_r:wp.float64
) -> wp.types.vector(5, wp.float64):
    # Returns the X-face viscous flux vector [0, Txx, Txy, Txz, u·T+κ·dT].
    kap  = GAMMA*mu_r/(PR*GM1)
    divV = du_dx+dv_dy+dw_dz
    Txx  = F2*mu_r*(du_dx-divV*F1_3)
    Txy  = mu_r*(du_dy+dv_dx)
    Txz  = mu_r*(du_dz+dw_dx)
    Tyy  = F2*mu_r*(dv_dy-divV*F1_3)
    Tyz  = mu_r*(dv_dz+dw_dy)
    Tzz  = F2*mu_r*(dw_dz-divV*F1_3)
    fv4  = uf*Txx+vf*Txy+wf*Txz+kap*dT_n
    return wp.vector(wp.float64(0.0), Txx, Txy, Txz, fv4, length=5)


# =============================================================================
#  S_V4  Viscous flux X  (+= to residual)
#
#  Nishikawa alpha=4 face interpolation:
#    u_face = (u_L + u_R)/2
#    du/dn  = 0.5*(du_L + du_R) + (4/3)*(u_R - u_L)*inv_dx
# =============================================================================

@wp.kernel
def viscous_flux_x_kernel(
    u_vel:wp.array(dtype=wp.float64, ndim=3),
    v_vel:wp.array(dtype=wp.float64, ndim=3),
    w_vel:wp.array(dtype=wp.float64, ndim=3),
    temp:wp.array(dtype=wp.float64, ndim=3),
    der_ux:wp.array(dtype=wp.float64, ndim=3), der_uy:wp.array(dtype=wp.float64, ndim=3),
    der_uz:wp.array(dtype=wp.float64, ndim=3), der_vx:wp.array(dtype=wp.float64, ndim=3),
    der_vy:wp.array(dtype=wp.float64, ndim=3), der_vz:wp.array(dtype=wp.float64, ndim=3),
    der_wx:wp.array(dtype=wp.float64, ndim=3), der_wy:wp.array(dtype=wp.float64, ndim=3),
    der_wz:wp.array(dtype=wp.float64, ndim=3),
    dTx:wp.array(dtype=wp.float64, ndim=3),
    resid:wp.array(dtype=wp.float64, ndim=4),
    dx:wp.float64, nx:int, ny:int, nz:int, gp:int, use_suth:int
):
    i, j, k = wp.tid()
    if i<1 or i>nx or j<1 or j>ny or k<1 or k>nz: return
    ix=i+gp; iy=j+gp; iz=k+gp
    inv_dx = F1/dx

    # ── right face (ix, iy, iz) ───────────────────────────────────────────────
    fx = ix
    uL=u_vel[fx,iy,iz]+HALF*der_ux[fx,iy,iz];   uR=u_vel[fx+1,iy,iz]-HALF*der_ux[fx+1,iy,iz]
    vL=v_vel[fx,iy,iz]+HALF*der_vx[fx,iy,iz];   vR=v_vel[fx+1,iy,iz]-HALF*der_vx[fx+1,iy,iz]
    wL=w_vel[fx,iy,iz]+HALF*der_wx[fx,iy,iz];   wR=w_vel[fx+1,iy,iz]-HALF*der_wx[fx+1,iy,iz]
    uf=HALF*(uL+uR); vf=HALF*(vL+vR); wf=HALF*(wL+wR)
    ju=uR-uL; jv=vR-vL; jw=wR-wL
    du_dx=(HALF*(der_ux[fx,iy,iz]+der_ux[fx+1,iy,iz])+F4_3*ju)*inv_dx
    du_dy=(HALF*(der_uy[fx,iy,iz]+der_uy[fx+1,iy,iz])+F4_3*ju)*inv_dx
    du_dz=(HALF*(der_uz[fx,iy,iz]+der_uz[fx+1,iy,iz])+F4_3*ju)*inv_dx
    dv_dx=(HALF*(der_vx[fx,iy,iz]+der_vx[fx+1,iy,iz])+F4_3*jv)*inv_dx
    dv_dy=(HALF*(der_vy[fx,iy,iz]+der_vy[fx+1,iy,iz])+F4_3*jv)*inv_dx
    dv_dz=(HALF*(der_vz[fx,iy,iz]+der_vz[fx+1,iy,iz])+F4_3*jv)*inv_dx
    dw_dx=(HALF*(der_wx[fx,iy,iz]+der_wx[fx+1,iy,iz])+F4_3*jw)*inv_dx
    dw_dy=(HALF*(der_wy[fx,iy,iz]+der_wy[fx+1,iy,iz])+F4_3*jw)*inv_dx
    dw_dz=(HALF*(der_wz[fx,iy,iz]+der_wz[fx+1,iy,iz])+F4_3*jw)*inv_dx
    TL_f=temp[fx,iy,iz]+HALF*dTx[fx,iy,iz]; TR_f=temp[fx+1,iy,iz]-HALF*dTx[fx+1,iy,iz]
    Tf=HALF*(TL_f+TR_f)
    dT_n=(HALF*(dTx[fx,iy,iz]+dTx[fx+1,iy,iz])+F4_3*(TR_f-TL_f))*inv_dx
    if use_suth==1: mu_r=mu_suth(Tf)/RE
    else:           mu_r=F1/RE
    kap=GAMMA*mu_r/(PR*GM1); divV=du_dx+dv_dy+dw_dz
    Txx=F2*mu_r*(du_dx-divV*F1_3); Txy=mu_r*(du_dy+dv_dx); Txz=mu_r*(du_dz+dw_dx)
    # X-face: sf=(1,0,0) → fv=(0, Txx, Txy, Txz, u*Txx+v*Txy+w*Txz+κ*dT/dx)
    fr1=Txx; fr2=Txy; fr3=Txz; fr4=uf*Txx+vf*Txy+wf*Txz+kap*dT_n

    # ── left face (ix-1) ─────────────────────────────────────────────────────
    fx = ix-1
    uL=u_vel[fx,iy,iz]+HALF*der_ux[fx,iy,iz];   uR=u_vel[fx+1,iy,iz]-HALF*der_ux[fx+1,iy,iz]
    vL=v_vel[fx,iy,iz]+HALF*der_vx[fx,iy,iz];   vR=v_vel[fx+1,iy,iz]-HALF*der_vx[fx+1,iy,iz]
    wL=w_vel[fx,iy,iz]+HALF*der_wx[fx,iy,iz];   wR=w_vel[fx+1,iy,iz]-HALF*der_wx[fx+1,iy,iz]
    uf=HALF*(uL+uR); vf=HALF*(vL+vR); wf=HALF*(wL+wR)
    ju=uR-uL; jv=vR-vL; jw=wR-wL
    du_dx=(HALF*(der_ux[fx,iy,iz]+der_ux[fx+1,iy,iz])+F4_3*ju)*inv_dx
    du_dy=(HALF*(der_uy[fx,iy,iz]+der_uy[fx+1,iy,iz])+F4_3*ju)*inv_dx
    du_dz=(HALF*(der_uz[fx,iy,iz]+der_uz[fx+1,iy,iz])+F4_3*ju)*inv_dx
    dv_dx=(HALF*(der_vx[fx,iy,iz]+der_vx[fx+1,iy,iz])+F4_3*jv)*inv_dx
    dv_dy=(HALF*(der_vy[fx,iy,iz]+der_vy[fx+1,iy,iz])+F4_3*jv)*inv_dx
    dv_dz=(HALF*(der_vz[fx,iy,iz]+der_vz[fx+1,iy,iz])+F4_3*jv)*inv_dx
    dw_dx=(HALF*(der_wx[fx,iy,iz]+der_wx[fx+1,iy,iz])+F4_3*jw)*inv_dx
    dw_dy=(HALF*(der_wy[fx,iy,iz]+der_wy[fx+1,iy,iz])+F4_3*jw)*inv_dx
    dw_dz=(HALF*(der_wz[fx,iy,iz]+der_wz[fx+1,iy,iz])+F4_3*jw)*inv_dx
    TL_f=temp[fx,iy,iz]+HALF*dTx[fx,iy,iz]; TR_f=temp[fx+1,iy,iz]-HALF*dTx[fx+1,iy,iz]
    Tf=HALF*(TL_f+TR_f)
    dT_n=(HALF*(dTx[fx,iy,iz]+dTx[fx+1,iy,iz])+F4_3*(TR_f-TL_f))*inv_dx
    if use_suth==1: mu_r=mu_suth(Tf)/RE
    else:           mu_r=F1/RE
    kap=GAMMA*mu_r/(PR*GM1); divV=du_dx+dv_dy+dw_dz
    Txx=F2*mu_r*(du_dx-divV*F1_3); Txy=mu_r*(du_dy+dv_dx); Txz=mu_r*(du_dz+dw_dx)
    fl1=Txx; fl2=Txy; fl3=Txz; fl4=uf*Txx+vf*Txy+wf*Txz+kap*dT_n

    resid[1,ix,iy,iz]=resid[1,ix,iy,iz]+(fr1-fl1)*inv_dx
    resid[2,ix,iy,iz]=resid[2,ix,iy,iz]+(fr2-fl2)*inv_dx
    resid[3,ix,iy,iz]=resid[3,ix,iy,iz]+(fr3-fl3)*inv_dx
    resid[4,ix,iy,iz]=resid[4,ix,iy,iz]+(fr4-fl4)*inv_dx


# =============================================================================
#  S_V5  Viscous flux Y
# =============================================================================

@wp.kernel
def viscous_flux_y_kernel(
    u_vel:wp.array(dtype=wp.float64, ndim=3),
    v_vel:wp.array(dtype=wp.float64, ndim=3),
    w_vel:wp.array(dtype=wp.float64, ndim=3),
    temp:wp.array(dtype=wp.float64, ndim=3),
    der_ux:wp.array(dtype=wp.float64, ndim=3), der_uy:wp.array(dtype=wp.float64, ndim=3),
    der_uz:wp.array(dtype=wp.float64, ndim=3), der_vx:wp.array(dtype=wp.float64, ndim=3),
    der_vy:wp.array(dtype=wp.float64, ndim=3), der_vz:wp.array(dtype=wp.float64, ndim=3),
    der_wx:wp.array(dtype=wp.float64, ndim=3), der_wy:wp.array(dtype=wp.float64, ndim=3),
    der_wz:wp.array(dtype=wp.float64, ndim=3),
    dTy:wp.array(dtype=wp.float64, ndim=3),
    resid:wp.array(dtype=wp.float64, ndim=4),
    dy:wp.float64, nx:int, ny:int, nz:int, gp:int, use_suth:int
):
    i, j, k = wp.tid()
    if i<1 or i>nx or j<1 or j>ny or k<1 or k>nz: return
    ix=i+gp; iy=j+gp; iz=k+gp
    inv_dy = F1/dy

    # ── right face ────────────────────────────────────────────────────────────
    fy = iy
    uL=u_vel[ix,fy,iz]+HALF*der_uy[ix,fy,iz];   uR=u_vel[ix,fy+1,iz]-HALF*der_uy[ix,fy+1,iz]
    vL=v_vel[ix,fy,iz]+HALF*der_vy[ix,fy,iz];   vR=v_vel[ix,fy+1,iz]-HALF*der_vy[ix,fy+1,iz]
    wL=w_vel[ix,fy,iz]+HALF*der_wy[ix,fy,iz];   wR=w_vel[ix,fy+1,iz]-HALF*der_wy[ix,fy+1,iz]
    uf=HALF*(uL+uR); vf=HALF*(vL+vR); wf=HALF*(wL+wR)
    ju=uR-uL; jv=vR-vL; jw=wR-wL
    du_dx=(HALF*(der_ux[ix,fy,iz]+der_ux[ix,fy+1,iz])+F4_3*ju)*inv_dy
    du_dy=(HALF*(der_uy[ix,fy,iz]+der_uy[ix,fy+1,iz])+F4_3*ju)*inv_dy
    du_dz=(HALF*(der_uz[ix,fy,iz]+der_uz[ix,fy+1,iz])+F4_3*ju)*inv_dy
    dv_dx=(HALF*(der_vx[ix,fy,iz]+der_vx[ix,fy+1,iz])+F4_3*jv)*inv_dy
    dv_dy=(HALF*(der_vy[ix,fy,iz]+der_vy[ix,fy+1,iz])+F4_3*jv)*inv_dy
    dv_dz=(HALF*(der_vz[ix,fy,iz]+der_vz[ix,fy+1,iz])+F4_3*jv)*inv_dy
    dw_dx=(HALF*(der_wx[ix,fy,iz]+der_wx[ix,fy+1,iz])+F4_3*jw)*inv_dy
    dw_dy=(HALF*(der_wy[ix,fy,iz]+der_wy[ix,fy+1,iz])+F4_3*jw)*inv_dy
    dw_dz=(HALF*(der_wz[ix,fy,iz]+der_wz[ix,fy+1,iz])+F4_3*jw)*inv_dy
    TL_f=temp[ix,fy,iz]+HALF*dTy[ix,fy,iz]; TR_f=temp[ix,fy+1,iz]-HALF*dTy[ix,fy+1,iz]
    Tf=HALF*(TL_f+TR_f)
    dT_n=(HALF*(dTy[ix,fy,iz]+dTy[ix,fy+1,iz])+F4_3*(TR_f-TL_f))*inv_dy
    if use_suth==1: mu_r=mu_suth(Tf)/RE
    else:           mu_r=F1/RE
    kap=GAMMA*mu_r/(PR*GM1); divV=du_dx+dv_dy+dw_dz
    Txy=mu_r*(du_dy+dv_dx); Tyy=F2*mu_r*(dv_dy-divV*F1_3); Tyz=mu_r*(dv_dz+dw_dy)
    # Y-face: sf=(0,1,0) → fv=(0, Txy, Tyy, Tyz, u*Txy+v*Tyy+w*Tyz+κ*dT/dy)
    fr1=Txy; fr2=Tyy; fr3=Tyz; fr4=uf*Txy+vf*Tyy+wf*Tyz+kap*dT_n

    # ── left face ─────────────────────────────────────────────────────────────
    fy = iy-1
    uL=u_vel[ix,fy,iz]+HALF*der_uy[ix,fy,iz];   uR=u_vel[ix,fy+1,iz]-HALF*der_uy[ix,fy+1,iz]
    vL=v_vel[ix,fy,iz]+HALF*der_vy[ix,fy,iz];   vR=v_vel[ix,fy+1,iz]-HALF*der_vy[ix,fy+1,iz]
    wL=w_vel[ix,fy,iz]+HALF*der_wy[ix,fy,iz];   wR=w_vel[ix,fy+1,iz]-HALF*der_wy[ix,fy+1,iz]
    uf=HALF*(uL+uR); vf=HALF*(vL+vR); wf=HALF*(wL+wR)
    ju=uR-uL; jv=vR-vL; jw=wR-wL
    du_dx=(HALF*(der_ux[ix,fy,iz]+der_ux[ix,fy+1,iz])+F4_3*ju)*inv_dy
    du_dy=(HALF*(der_uy[ix,fy,iz]+der_uy[ix,fy+1,iz])+F4_3*ju)*inv_dy
    du_dz=(HALF*(der_uz[ix,fy,iz]+der_uz[ix,fy+1,iz])+F4_3*ju)*inv_dy
    dv_dx=(HALF*(der_vx[ix,fy,iz]+der_vx[ix,fy+1,iz])+F4_3*jv)*inv_dy
    dv_dy=(HALF*(der_vy[ix,fy,iz]+der_vy[ix,fy+1,iz])+F4_3*jv)*inv_dy
    dv_dz=(HALF*(der_vz[ix,fy,iz]+der_vz[ix,fy+1,iz])+F4_3*jv)*inv_dy
    dw_dx=(HALF*(der_wx[ix,fy,iz]+der_wx[ix,fy+1,iz])+F4_3*jw)*inv_dy
    dw_dy=(HALF*(der_wy[ix,fy,iz]+der_wy[ix,fy+1,iz])+F4_3*jw)*inv_dy
    dw_dz=(HALF*(der_wz[ix,fy,iz]+der_wz[ix,fy+1,iz])+F4_3*jw)*inv_dy
    TL_f=temp[ix,fy,iz]+HALF*dTy[ix,fy,iz]; TR_f=temp[ix,fy+1,iz]-HALF*dTy[ix,fy+1,iz]
    Tf=HALF*(TL_f+TR_f)
    dT_n=(HALF*(dTy[ix,fy,iz]+dTy[ix,fy+1,iz])+F4_3*(TR_f-TL_f))*inv_dy
    if use_suth==1: mu_r=mu_suth(Tf)/RE
    else:           mu_r=F1/RE
    kap=GAMMA*mu_r/(PR*GM1); divV=du_dx+dv_dy+dw_dz
    Txy=mu_r*(du_dy+dv_dx); Tyy=F2*mu_r*(dv_dy-divV*F1_3); Tyz=mu_r*(dv_dz+dw_dy)
    fl1=Txy; fl2=Tyy; fl3=Tyz; fl4=uf*Txy+vf*Tyy+wf*Tyz+kap*dT_n

    resid[1,ix,iy,iz]=resid[1,ix,iy,iz]+(fr1-fl1)*inv_dy
    resid[2,ix,iy,iz]=resid[2,ix,iy,iz]+(fr2-fl2)*inv_dy
    resid[3,ix,iy,iz]=resid[3,ix,iy,iz]+(fr3-fl3)*inv_dy
    resid[4,ix,iy,iz]=resid[4,ix,iy,iz]+(fr4-fl4)*inv_dy


# =============================================================================
#  S_V6  Viscous flux Z
# =============================================================================

@wp.kernel
def viscous_flux_z_kernel(
    u_vel:wp.array(dtype=wp.float64, ndim=3),
    v_vel:wp.array(dtype=wp.float64, ndim=3),
    w_vel:wp.array(dtype=wp.float64, ndim=3),
    temp:wp.array(dtype=wp.float64, ndim=3),
    der_ux:wp.array(dtype=wp.float64, ndim=3), der_uy:wp.array(dtype=wp.float64, ndim=3),
    der_uz:wp.array(dtype=wp.float64, ndim=3), der_vx:wp.array(dtype=wp.float64, ndim=3),
    der_vy:wp.array(dtype=wp.float64, ndim=3), der_vz:wp.array(dtype=wp.float64, ndim=3),
    der_wx:wp.array(dtype=wp.float64, ndim=3), der_wy:wp.array(dtype=wp.float64, ndim=3),
    der_wz:wp.array(dtype=wp.float64, ndim=3),
    dTz:wp.array(dtype=wp.float64, ndim=3),
    resid:wp.array(dtype=wp.float64, ndim=4),
    dz:wp.float64, nx:int, ny:int, nz:int, gp:int, use_suth:int
):
    i, j, k = wp.tid()
    if i<1 or i>nx or j<1 or j>ny or k<1 or k>nz: return
    ix=i+gp; iy=j+gp; iz=k+gp
    inv_dz = F1/dz

    # ── right face ────────────────────────────────────────────────────────────
    fz = iz
    uL=u_vel[ix,iy,fz]+HALF*der_uz[ix,iy,fz];   uR=u_vel[ix,iy,fz+1]-HALF*der_uz[ix,iy,fz+1]
    vL=v_vel[ix,iy,fz]+HALF*der_vz[ix,iy,fz];   vR=v_vel[ix,iy,fz+1]-HALF*der_vz[ix,iy,fz+1]
    wL=w_vel[ix,iy,fz]+HALF*der_wz[ix,iy,fz];   wR=w_vel[ix,iy,fz+1]-HALF*der_wz[ix,iy,fz+1]
    uf=HALF*(uL+uR); vf=HALF*(vL+vR); wf=HALF*(wL+wR)
    ju=uR-uL; jv=vR-vL; jw=wR-wL
    du_dx=(HALF*(der_ux[ix,iy,fz]+der_ux[ix,iy,fz+1])+F4_3*ju)*inv_dz
    du_dy=(HALF*(der_uy[ix,iy,fz]+der_uy[ix,iy,fz+1])+F4_3*ju)*inv_dz
    du_dz=(HALF*(der_uz[ix,iy,fz]+der_uz[ix,iy,fz+1])+F4_3*ju)*inv_dz
    dv_dx=(HALF*(der_vx[ix,iy,fz]+der_vx[ix,iy,fz+1])+F4_3*jv)*inv_dz
    dv_dy=(HALF*(der_vy[ix,iy,fz]+der_vy[ix,iy,fz+1])+F4_3*jv)*inv_dz
    dv_dz=(HALF*(der_vz[ix,iy,fz]+der_vz[ix,iy,fz+1])+F4_3*jv)*inv_dz
    dw_dx=(HALF*(der_wx[ix,iy,fz]+der_wx[ix,iy,fz+1])+F4_3*jw)*inv_dz
    dw_dy=(HALF*(der_wy[ix,iy,fz]+der_wy[ix,iy,fz+1])+F4_3*jw)*inv_dz
    dw_dz=(HALF*(der_wz[ix,iy,fz]+der_wz[ix,iy,fz+1])+F4_3*jw)*inv_dz
    TL_f=temp[ix,iy,fz]+HALF*dTz[ix,iy,fz]; TR_f=temp[ix,iy,fz+1]-HALF*dTz[ix,iy,fz+1]
    Tf=HALF*(TL_f+TR_f)
    dT_n=(HALF*(dTz[ix,iy,fz]+dTz[ix,iy,fz+1])+F4_3*(TR_f-TL_f))*inv_dz
    if use_suth==1: mu_r=mu_suth(Tf)/RE
    else:           mu_r=F1/RE
    kap=GAMMA*mu_r/(PR*GM1); divV=du_dx+dv_dy+dw_dz
    Txz=mu_r*(du_dz+dw_dx); Tyz=mu_r*(dv_dz+dw_dy); Tzz=F2*mu_r*(dw_dz-divV*F1_3)
    # Z-face: sf=(0,0,1) → fv=(0, Txz, Tyz, Tzz, u*Txz+v*Tyz+w*Tzz+κ*dT/dz)
    fr1=Txz; fr2=Tyz; fr3=Tzz; fr4=uf*Txz+vf*Tyz+wf*Tzz+kap*dT_n

    # ── left face ─────────────────────────────────────────────────────────────
    fz = iz-1
    uL=u_vel[ix,iy,fz]+HALF*der_uz[ix,iy,fz];   uR=u_vel[ix,iy,fz+1]-HALF*der_uz[ix,iy,fz+1]
    vL=v_vel[ix,iy,fz]+HALF*der_vz[ix,iy,fz];   vR=v_vel[ix,iy,fz+1]-HALF*der_vz[ix,iy,fz+1]
    wL=w_vel[ix,iy,fz]+HALF*der_wz[ix,iy,fz];   wR=w_vel[ix,iy,fz+1]-HALF*der_wz[ix,iy,fz+1]
    uf=HALF*(uL+uR); vf=HALF*(vL+vR); wf=HALF*(wL+wR)
    ju=uR-uL; jv=vR-vL; jw=wR-wL
    du_dx=(HALF*(der_ux[ix,iy,fz]+der_ux[ix,iy,fz+1])+F4_3*ju)*inv_dz
    du_dy=(HALF*(der_uy[ix,iy,fz]+der_uy[ix,iy,fz+1])+F4_3*ju)*inv_dz
    du_dz=(HALF*(der_uz[ix,iy,fz]+der_uz[ix,iy,fz+1])+F4_3*ju)*inv_dz
    dv_dx=(HALF*(der_vx[ix,iy,fz]+der_vx[ix,iy,fz+1])+F4_3*jv)*inv_dz
    dv_dy=(HALF*(der_vy[ix,iy,fz]+der_vy[ix,iy,fz+1])+F4_3*jv)*inv_dz
    dv_dz=(HALF*(der_vz[ix,iy,fz]+der_vz[ix,iy,fz+1])+F4_3*jv)*inv_dz
    dw_dx=(HALF*(der_wx[ix,iy,fz]+der_wx[ix,iy,fz+1])+F4_3*jw)*inv_dz
    dw_dy=(HALF*(der_wy[ix,iy,fz]+der_wy[ix,iy,fz+1])+F4_3*jw)*inv_dz
    dw_dz=(HALF*(der_wz[ix,iy,fz]+der_wz[ix,iy,fz+1])+F4_3*jw)*inv_dz
    TL_f=temp[ix,iy,fz]+HALF*dTz[ix,iy,fz]; TR_f=temp[ix,iy,fz+1]-HALF*dTz[ix,iy,fz+1]
    Tf=HALF*(TL_f+TR_f)
    dT_n=(HALF*(dTz[ix,iy,fz]+dTz[ix,iy,fz+1])+F4_3*(TR_f-TL_f))*inv_dz
    if use_suth==1: mu_r=mu_suth(Tf)/RE
    else:           mu_r=F1/RE
    kap=GAMMA*mu_r/(PR*GM1); divV=du_dx+dv_dy+dw_dz
    Txz=mu_r*(du_dz+dw_dx); Tyz=mu_r*(dv_dz+dw_dy); Tzz=F2*mu_r*(dw_dz-divV*F1_3)
    fl1=Txz; fl2=Tyz; fl3=Tzz; fl4=uf*Txz+vf*Tyz+wf*Tzz+kap*dT_n

    resid[1,ix,iy,iz]=resid[1,ix,iy,iz]+(fr1-fl1)*inv_dz
    resid[2,ix,iy,iz]=resid[2,ix,iy,iz]+(fr2-fl2)*inv_dz
    resid[3,ix,iy,iz]=resid[3,ix,iy,iz]+(fr3-fl3)*inv_dz
    resid[4,ix,iy,iz]=resid[4,ix,iy,iz]+(fr4-fl4)*inv_dz
