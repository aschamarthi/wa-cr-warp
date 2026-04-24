"""
timestepping.py
---------------
SSP-RK3 update kernel, residual zeroing kernel, and GPU CFL time-step kernel.
"""

import warp as wp
from constants import GAMMA, HALF, F0, F1, TINY


@wp.kernel
def rk3_step_kernel(
    cons0:wp.array(dtype=wp.float64, ndim=4),
    cons_in:wp.array(dtype=wp.float64, ndim=4),
    resid:wp.array(dtype=wp.float64, ndim=4),
    cons_out:wp.array(dtype=wp.float64, ndim=4),
    alpha:wp.float64, beta:wp.float64, dt:wp.float64,
    nx:int, ny:int, nz:int, gp:int
):
    # Shu-Osher SSP-RK3:  cons_out = alpha*cons0 + beta*(cons_in + dt*resid)
    # Stage 1: alpha=0,   beta=1    → cons1 = cons0 + dt*L(cons0)
    # Stage 2: alpha=3/4, beta=1/4  → cons2 = 3/4*cons0 + 1/4*(cons1+dt*L(cons1))
    # Stage 3: alpha=1/3, beta=2/3  → cons0 = 1/3*cons0 + 2/3*(cons2+dt*L(cons2))
    i, j, k = wp.tid()
    if i<1 or i>nx or j<1 or j>ny or k<1 or k>nz: return
    ix=i+gp; iy=j+gp; iz=k+gp
    cons_out[0,ix,iy,iz]=alpha*cons0[0,ix,iy,iz]+beta*(cons_in[0,ix,iy,iz]+dt*resid[0,ix,iy,iz])
    cons_out[1,ix,iy,iz]=alpha*cons0[1,ix,iy,iz]+beta*(cons_in[1,ix,iy,iz]+dt*resid[1,ix,iy,iz])
    cons_out[2,ix,iy,iz]=alpha*cons0[2,ix,iy,iz]+beta*(cons_in[2,ix,iy,iz]+dt*resid[2,ix,iy,iz])
    cons_out[3,ix,iy,iz]=alpha*cons0[3,ix,iy,iz]+beta*(cons_in[3,ix,iy,iz]+dt*resid[3,ix,iy,iz])
    cons_out[4,ix,iy,iz]=alpha*cons0[4,ix,iy,iz]+beta*(cons_in[4,ix,iy,iz]+dt*resid[4,ix,iy,iz])


@wp.kernel
def zero_residual_kernel(
    resid:wp.array(dtype=wp.float64, ndim=4),
    nx:int, ny:int, nz:int, gp:int
):
    # Zeroes the residual over interior cells before each RK stage.
    i, j, k = wp.tid()
    if i<1 or i>nx or j<1 or j>ny or k<1 or k>nz: return
    ix=i+gp; iy=j+gp; iz=k+gp
    resid[0,ix,iy,iz]=F0; resid[1,ix,iy,iz]=F0; resid[2,ix,iy,iz]=F0
    resid[3,ix,iy,iz]=F0; resid[4,ix,iy,iz]=F0


@wp.kernel
def dt_local_kernel(
    u_vel:wp.array(dtype=wp.float64, ndim=3),
    v_vel:wp.array(dtype=wp.float64, ndim=3),
    w_vel:wp.array(dtype=wp.float64, ndim=3),
    pres:wp.array(dtype=wp.float64, ndim=3),
    rho:wp.array(dtype=wp.float64, ndim=3),
    snd:wp.array(dtype=wp.float64, ndim=3),
    dt_min:wp.array(dtype=wp.float64, ndim=1),
    dx:wp.float64, dy:wp.float64, dz:wp.float64,
    nx:int, ny:int, nz:int, gp:int
):
    # CFL time step: dt_cell = min(dx/(|u|+c), dy/(|v|+c), dz/(|w|+c)).
    # Global minimum via GPU atomic_min — only 8 bytes transferred per step.
    i, j, k = wp.tid()
    if i<1 or i>nx or j<1 or j>ny or k<1 or k>nz: return
    ix=i+gp; iy=j+gp; iz=k+gp
    c = wp.sqrt(GAMMA*pres[ix,iy,iz]/rho[ix,iy,iz])
    snd[ix,iy,iz] = c
    sx = wp.abs(u_vel[ix,iy,iz])+c
    sy = wp.abs(v_vel[ix,iy,iz])+c
    sz = wp.abs(w_vel[ix,iy,iz])+c
    dt_cell = wp.min(dx/sx, wp.min(dy/sy, dz/sz))
    wp.atomic_min(dt_min, 0, dt_cell)
