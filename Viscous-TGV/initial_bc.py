"""
initial_bc.py
-------------
Initial condition, periodic boundary conditions, and primitive <-> conservative
conversion kernels.

cons layout: SoA  cons[var, ix, iy, iz]  with var ∈ {ρ, ρu, ρv, ρw, ρE}  --- This is the change that helped me speed up :)
prim layout: AoS  separate scalar arrays (rho, u_vel, v_vel, w_vel, pres)
"""

import warp as wp
from constants import GAMMA, GM1, HALF, F0, F1, F2


@wp.kernel
def init_cond_kernel(
    x:wp.array(dtype=wp.float64, ndim=1),
    y:wp.array(dtype=wp.float64, ndim=1),
    z:wp.array(dtype=wp.float64, ndim=1),
    rho:wp.array(dtype=wp.float64, ndim=3),
    u_vel:wp.array(dtype=wp.float64, ndim=3),
    v_vel:wp.array(dtype=wp.float64, ndim=3),
    w_vel:wp.array(dtype=wp.float64, ndim=3),
    pres:wp.array(dtype=wp.float64, ndim=3),
    snd:wp.array(dtype=wp.float64, ndim=3),
):
    # Taylor-Green Vortex initial conditions (subsonic, p~100 for low Ma).
    # ρ=1, w=0, u=sin(x)cos(y)cos(z), v=-cos(x)sin(y)cos(z)
    # p = 100 + (cos(2x)+cos(2y))*(cos(2z)+2)/16 - 1/8
    ix, iy, iz = wp.tid()
    xi = x[ix]; yi = y[iy]; zi = z[iz]
    r = F1
    u =  wp.sin(xi)*wp.cos(yi)*wp.cos(zi)
    v = -wp.cos(xi)*wp.sin(yi)*wp.cos(zi)
    w = F0
    t1 = wp.cos(F2*xi)+wp.cos(F2*yi)
    t2 = wp.cos(F2*zi)+F2
    p  = wp.float64(100.0)+(t1*t2-F2)/wp.float64(16.0)
    rho[ix,iy,iz]   = r
    u_vel[ix,iy,iz] = u
    v_vel[ix,iy,iz] = v
    w_vel[ix,iy,iz] = w
    pres[ix,iy,iz]  = p
    snd[ix,iy,iz]   = wp.sqrt(GAMMA*p/r)


@wp.kernel
def bc_periodic_x_kernel(
    rho:wp.array(dtype=wp.float64, ndim=3),
    u_vel:wp.array(dtype=wp.float64, ndim=3),
    v_vel:wp.array(dtype=wp.float64, ndim=3),
    w_vel:wp.array(dtype=wp.float64, ndim=3),
    pres:wp.array(dtype=wp.float64, ndim=3),
    nx:int, ny:int, nz:int, gp:int
):
    layer, iy, iz = wp.tid()
    if layer>=gp: return
    i  = layer+1
    gl = gp+1-i;  sr = gp+nx+1-i;  gr = gp+nx+i;  sl = gp+i
    rho[gl,iy,iz]=rho[sr,iy,iz];     rho[gr,iy,iz]=rho[sl,iy,iz]
    u_vel[gl,iy,iz]=u_vel[sr,iy,iz]; u_vel[gr,iy,iz]=u_vel[sl,iy,iz]
    v_vel[gl,iy,iz]=v_vel[sr,iy,iz]; v_vel[gr,iy,iz]=v_vel[sl,iy,iz]
    w_vel[gl,iy,iz]=w_vel[sr,iy,iz]; w_vel[gr,iy,iz]=w_vel[sl,iy,iz]
    pres[gl,iy,iz]=pres[sr,iy,iz];   pres[gr,iy,iz]=pres[sl,iy,iz]


@wp.kernel
def bc_periodic_y_kernel(
    rho:wp.array(dtype=wp.float64, ndim=3),
    u_vel:wp.array(dtype=wp.float64, ndim=3),
    v_vel:wp.array(dtype=wp.float64, ndim=3),
    w_vel:wp.array(dtype=wp.float64, ndim=3),
    pres:wp.array(dtype=wp.float64, ndim=3),
    nx:int, ny:int, nz:int, gp:int
):
    ix, layer, iz = wp.tid()
    if layer>=gp: return
    j  = layer+1
    gb = gp+1-j;  st = gp+ny+1-j;  gt = gp+ny+j;  sb = gp+j
    rho[ix,gb,iz]=rho[ix,st,iz];     rho[ix,gt,iz]=rho[ix,sb,iz]
    u_vel[ix,gb,iz]=u_vel[ix,st,iz]; u_vel[ix,gt,iz]=u_vel[ix,sb,iz]
    v_vel[ix,gb,iz]=v_vel[ix,st,iz]; v_vel[ix,gt,iz]=v_vel[ix,sb,iz]
    w_vel[ix,gb,iz]=w_vel[ix,st,iz]; w_vel[ix,gt,iz]=w_vel[ix,sb,iz]
    pres[ix,gb,iz]=pres[ix,st,iz];   pres[ix,gt,iz]=pres[ix,sb,iz]


@wp.kernel
def bc_periodic_z_kernel(
    rho:wp.array(dtype=wp.float64, ndim=3),
    u_vel:wp.array(dtype=wp.float64, ndim=3),
    v_vel:wp.array(dtype=wp.float64, ndim=3),
    w_vel:wp.array(dtype=wp.float64, ndim=3),
    pres:wp.array(dtype=wp.float64, ndim=3),
    nx:int, ny:int, nz:int, gp:int
):
    ix, iy, layer = wp.tid()
    if layer>=gp: return
    k  = layer+1
    gf = gp+1-k;  sb = gp+nz+1-k;  gk = gp+nz+k;  sf = gp+k
    rho[ix,iy,gf]=rho[ix,iy,sb];     rho[ix,iy,gk]=rho[ix,iy,sf]
    u_vel[ix,iy,gf]=u_vel[ix,iy,sb]; u_vel[ix,iy,gk]=u_vel[ix,iy,sf]
    v_vel[ix,iy,gf]=v_vel[ix,iy,sb]; v_vel[ix,iy,gk]=v_vel[ix,iy,sf]
    w_vel[ix,iy,gf]=w_vel[ix,iy,sb]; w_vel[ix,iy,gk]=w_vel[ix,iy,sf]
    pres[ix,iy,gf]=pres[ix,iy,sb];   pres[ix,iy,gk]=pres[ix,iy,sf]


@wp.kernel
def prim_to_cons_kernel(
    rho:wp.array(dtype=wp.float64, ndim=3),
    u_vel:wp.array(dtype=wp.float64, ndim=3),
    v_vel:wp.array(dtype=wp.float64, ndim=3),
    w_vel:wp.array(dtype=wp.float64, ndim=3),
    pres:wp.array(dtype=wp.float64, ndim=3),
    cons:wp.array(dtype=wp.float64, ndim=4),
):
    ix, iy, iz = wp.tid()
    r=rho[ix,iy,iz]; u=u_vel[ix,iy,iz]; v=v_vel[ix,iy,iz]
    w=w_vel[ix,iy,iz]; p=pres[ix,iy,iz]
    cons[0,ix,iy,iz]=r
    cons[1,ix,iy,iz]=r*u
    cons[2,ix,iy,iz]=r*v
    cons[3,ix,iy,iz]=r*w
    cons[4,ix,iy,iz]=p/GM1+HALF*r*(u*u+v*v+w*w)


@wp.kernel
def cons_to_prim_kernel(
    cons:wp.array(dtype=wp.float64, ndim=4),
    rho:wp.array(dtype=wp.float64, ndim=3),
    u_vel:wp.array(dtype=wp.float64, ndim=3),
    v_vel:wp.array(dtype=wp.float64, ndim=3),
    w_vel:wp.array(dtype=wp.float64, ndim=3),
    pres:wp.array(dtype=wp.float64, ndim=3),
    nx:int, ny:int, nz:int, gp:int
):
    i, j, k = wp.tid()
    if i<1 or i>nx or j<1 or j>ny or k<1 or k>nz: return
    ix=i+gp; iy=j+gp; iz=k+gp
    r=cons[0,ix,iy,iz]; ru=cons[1,ix,iy,iz]; rv=cons[2,ix,iy,iz]
    rw=cons[3,ix,iy,iz]; E=cons[4,ix,iy,iz]
    u=ru/r; v=rv/r; w=rw/r
    p=GM1*(E-HALF*r*(u*u+v*v+w*w))
    rho[ix,iy,iz]=r; u_vel[ix,iy,iz]=u; v_vel[ix,iy,iz]=v
    w_vel[ix,iy,iz]=w; pres[ix,iy,iz]=p
