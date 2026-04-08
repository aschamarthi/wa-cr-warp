"""
2D Compressible Euler Solver — Explosion test case
Using Nvidia Warp 

## References

1. Chamarthi, Hoffmann, Frankel — *A wave appropriate discontinuity sensor approach for compressible flows*, **Phys. Fluids** 35, 066107 (2023)
2. Hoffmann, Chamarthi, Frankel — *Centralized gradient-based reconstruction for wall modeledlarge eddy simulations of hypersonic boundary layer transition*, **J. Comput. Phys.** (2024)
3. Chamarthi — *Wave-appropriate multidimensional upwinding approach for compressible multiphase flows*, **J. Comput. Phys.** 538, 114157 (2025)
4. Chamarthi — *Physics appropriate interface capturing reconstruction approach for viscous compressible multicomponent flows*, **Comput. Fluids** 303, 106858 (2025)
5. Chamarthi — *Wave-appropriate reconstruction of compressible flows: physics-constrained acoustic dissipation and rank-1 entropy wave correction*, preprint (2026)


Algorithm is from ref. 5. WA-3
## Author

**Amareshwara Sainadh Chamarthi** sainath@caltech.edu
Problem
-------
  Domain  : [-1, 1] × [-1, 1]
  IC      : r < R0 → ρ=1.0, p=1.0;  r >= R0 → ρ=0.125, p=0.1;  u=v=0
  BC      : reflecting walls on all four sides
  t_end   : 0.25  (blast wave well inside domain)

Usage
-----
  python explosion_warp.py            # GPU (CUDA) if available
  python explosion_warp.py --cpu      # force CPU
"""

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
# plt.rc('font', family='arial') !!! If you want to use Latex for visualization
plt.rcParams.update({'font.size': 12})

import numpy as np
import warp as wp

# ─────────────────────────────────── CLI ─────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true")
args, _ = parser.parse_known_args()
DEVICE = "cpu" if args.cpu else "cpu"

wp.init()

# ─────────────────────────── Problem parameters ───────────────────────────────
NX, NY  = 512, 512
GHOSTP  = 5
GX      = NX + 2 * GHOSTP + 1
GY      = NY + 2 * GHOSTP + 1
G       = GHOSTP

NTMAX     = 200_000
FILE_SAVE = 10000000

CFL_PY   = 0.4
T_END    = 1.1

XMIN_PY, XMAX_PY = 0.0, 1.2
YMIN_PY, YMAX_PY = 0.0, 1.2

X_DIV = 1.0
Y_DIV = 1.0

import math as _math
_U_SHEAR = 4.0 / _math.sqrt(11.0)
"""
Quadrant ICs (cell-centre x,y):
x > 1, y > 1  : ρ=1.5,      u=0,        v=0,        p=1.5
x ≤ 1, y > 1  : ρ=33/62,    u=4/√11,    v=0,        p=0.3
x ≤ 1, y ≤ 1  : ρ=77/558,   u=4/√11,    v=4/√11,    p=9/310
x > 1, y ≤ 1  : ρ=33/62,    u=0,        v=4/√11,    p=0.3
"""
# Quadrant states: (rho, u, v, p)
# Q1: x>1, y>1
RHO_Q1, U_Q1, V_Q1, P_Q1 = 1.5,           0.0,       0.0,       1.5
# Q2: x<=1, y>1
RHO_Q2, U_Q2, V_Q2, P_Q2 = 33.0/62.0,     _U_SHEAR,  0.0,       0.3
# Q3: x<=1, y<=1
RHO_Q3, U_Q3, V_Q3, P_Q3 = 77.0/558.0,    _U_SHEAR,  _U_SHEAR,  9.0/310.0
# Q4: x>1, y<=1
RHO_Q4, U_Q4, V_Q4, P_Q4 = 33.0/62.0,     0.0,       _U_SHEAR,  0.3

# ─── float64 Warp constants ───────────────────────────────────────────────────
GAMMA  = wp.constant(wp.float64(1.4))
GM1    = wp.constant(wp.float64(0.4))
TINY   = wp.constant(wp.float64(1.0e-30))
KAPPA  = wp.constant(wp.float64(-1.0 / 3.0))

F0     = wp.constant(wp.float64(0.0))
F1     = wp.constant(wp.float64(1.0))
F2     = wp.constant(wp.float64(2.0))
HALF   = wp.constant(wp.float64(0.5))
QUART  = wp.constant(wp.float64(0.25))
F5_6   = wp.constant(wp.float64(5.0 / 6.0))
F1_6   = wp.constant(wp.float64(1.0 / 6.0))
F1_3   = wp.constant(wp.float64(1.0 / 3.0))
F7_12  = wp.constant(wp.float64(7.0 / 12.0))
F1_12  = wp.constant(wp.float64(1.0 / 12.0))
F0p54  = wp.constant(wp.float64(0.54))

NF1    = wp.constant(wp.float64(-1.0))
NF1_6  = wp.constant(wp.float64(-1.0 / 6.0))
NF1_12 = wp.constant(wp.float64(-1.0 / 12.0))

# ══════════════════════════════════════════════════════════════════════════════
#  §1  Helper functions  (identical to mach_warp.py)
# ══════════════════════════════════════════════════════════════════════════════

@wp.func
def minmod2(x: wp.float64, y: wp.float64) -> wp.float64:
    return HALF * (wp.sign(x) + wp.sign(y)) * wp.min(wp.abs(x), wp.abs(y))


# ─── Left-eigenvector rows ────────────────────────────────────────────────────

@wp.func
def Lq0(q1: wp.float64, q2: wp.float64, q3: wp.float64, q4: wp.float64,
        u: wp.float64, v: wp.float64, c: wp.float64, c2: wp.float64,
        mx: wp.float64, my: wp.float64, qn: wp.float64, q2vel: wp.float64) -> wp.float64:
    gm1 = GAMMA - F1
    return (HALF*(gm1*HALF*q2vel/c2 + qn/c)*q1
            - HALF*(gm1*u/c2 + mx/c)*q2
            - HALF*(gm1*v/c2 + my/c)*q3
            + gm1/(F2*c2)*q4)

@wp.func
def Lq1(q1: wp.float64, q2: wp.float64, q3: wp.float64, q4: wp.float64,
        lx: wp.float64, ly: wp.float64, ql: wp.float64) -> wp.float64:
    return -ql*q1 + lx*q2 + ly*q3

@wp.func
def Lq2(q1: wp.float64, q2: wp.float64, q3: wp.float64, q4: wp.float64,
        u: wp.float64, v: wp.float64, c2: wp.float64, q2vel: wp.float64) -> wp.float64:
    gm1 = GAMMA - F1
    return ((F1 - gm1*q2vel/(F2*c2))*q1
            + gm1*u/c2*q2
            + gm1*v/c2*q3
            - gm1/c2*q4)

@wp.func
def Lq3(q1: wp.float64, q2: wp.float64, q3: wp.float64, q4: wp.float64,
        u: wp.float64, v: wp.float64, c: wp.float64, c2: wp.float64,
        mx: wp.float64, my: wp.float64, qn: wp.float64, q2vel: wp.float64) -> wp.float64:
    gm1 = GAMMA - F1
    return (HALF*(gm1*HALF*q2vel/c2 - qn/c)*q1
            - HALF*(gm1*u/c2 - mx/c)*q2
            - HALF*(gm1*v/c2 - my/c)*q3
            + gm1/(F2*c2)*q4)


# ─── Right-eigenvector rows ───────────────────────────────────────────────────

@wp.func
def Rv0(v0: wp.float64, v1: wp.float64, v2: wp.float64, v3: wp.float64) -> wp.float64:
    return v0 + v2 + v3

@wp.func
def Rv1(v0: wp.float64, v1: wp.float64, v2: wp.float64, v3: wp.float64,
        u: wp.float64, c: wp.float64, mx: wp.float64, lx: wp.float64) -> wp.float64:
    return v0*(u - c*mx) + v1*lx + v2*u + v3*(u + c*mx)

@wp.func
def Rv2(v0: wp.float64, v1: wp.float64, v2: wp.float64, v3: wp.float64,
        vv: wp.float64, c: wp.float64, my: wp.float64, ly: wp.float64) -> wp.float64:
    return v0*(vv - c*my) + v1*ly + v2*vv + v3*(vv + c*my)

@wp.func
def Rv3(v0: wp.float64, v1: wp.float64, v2: wp.float64, v3: wp.float64,
        enth: wp.float64, qn: wp.float64, c: wp.float64,
        ql: wp.float64, q2vel: wp.float64) -> wp.float64:
    return v0*(enth - qn*c) + v1*ql + v2*(HALF*q2vel) + v3*(enth + qn*c)


# ─── WCNS reconstruction ─────────────────────────────────────────────────────

@wp.func
def wcns_ul_acoustic(vm1: wp.float64, v0: wp.float64,
                     v1: wp.float64,  v2: wp.float64, eta: wp.float64) -> wp.float64:
    dm = v0 - vm1;  do_ = v1 - v0
    muscl = v0 + QUART*((F1-KAPPA)*minmod2(dm, F2*do_) + (F1+KAPPA)*minmod2(do_, F2*dm))
    vor   = eta*(NF1_6*vm1 + F5_6*v0 + F1_3*v1) + (F1-eta)*(F1_3*v0 + F5_6*v1 + NF1_6*v2)
    if eta < F1:
        return vor
    return muscl

@wp.func
def wcns_ur_acoustic(vm1: wp.float64, v0: wp.float64,
                     v1: wp.float64,  v2: wp.float64, eta: wp.float64) -> wp.float64:
    do_ = v1 - v0;  dp = v2 - v1
    muscl = v1 - QUART*((F1-KAPPA)*minmod2(dp, F2*do_) + (F1+KAPPA)*minmod2(do_, F2*dp))
    vor   = (F1-eta)*(NF1_6*vm1 + F5_6*v0 + F1_3*v1) + eta*(F1_3*v0 + F5_6*v1 + NF1_6*v2)
    if eta < F1:
        return vor
    return muscl

@wp.func
def wcns_ul_shear(vm1: wp.float64, v0: wp.float64,
                  v1: wp.float64,  v2: wp.float64, eta: wp.float64) -> wp.float64:
    dm = v0 - vm1;  do_ = v1 - v0
    muscl = v0 + QUART*((F1-KAPPA)*minmod2(dm, F2*do_) + (F1+KAPPA)*minmod2(do_, F2*dm))
    iph   = NF1_12*vm1 + F7_12*v0 + F7_12*v1 + NF1_12*v2
    if eta < F1:
        return iph
    return muscl

@wp.func
def wcns_ur_shear(vm1: wp.float64, v0: wp.float64,
                  v1: wp.float64,  v2: wp.float64, eta: wp.float64) -> wp.float64:
    do_ = v1 - v0;  dp = v2 - v1
    muscl = v1 - QUART*((F1-KAPPA)*minmod2(dp, F2*do_) + (F1+KAPPA)*minmod2(do_, F2*dp))
    iph   = NF1_12*vm1 + F7_12*v0 + F7_12*v1 + NF1_12*v2
    if eta < F1:
        return iph
    return muscl

@wp.func
def wcns_ul_entropy(vm1: wp.float64, v0: wp.float64,
                    v1: wp.float64,  v2: wp.float64) -> wp.float64:
    dm = v0 - vm1;  do_ = v1 - v0
    return v0 + QUART*((F1-KAPPA)*minmod2(dm, F2*do_) + (F1+KAPPA)*minmod2(do_, F2*dm))

@wp.func
def wcns_ur_entropy(vm1: wp.float64, v0: wp.float64,
                    v1: wp.float64,  v2: wp.float64) -> wp.float64:
    do_ = v1 - v0;  dp = v2 - v1
    return v1 - QUART*((F1-KAPPA)*minmod2(dp, F2*do_) + (F1+KAPPA)*minmod2(do_, F2*dp))


# ─── Positivity check ─────────────────────────────────────────────────────────

@wp.func
def is_physical(c0: wp.float64, c1: wp.float64,
                c2: wp.float64, c3: wp.float64) -> int:
    if c0 <= F0:
        return 0
    u_ = c1 / c0;  v_ = c2 / c0
    e_ = c3 / c0 - HALF * (u_*u_ + v_*v_)
    if e_ <= F0:
        return 0
    return 1


# ─── Pressure shock sensor ────────────────────────────────────────────────────

@wp.func
def pressure_sensor_x(pres  : wp.array(dtype=wp.float64, ndim=2),
                       ducros: wp.array(dtype=wp.float64, ndim=2),
                       kx: int, iy: int) -> wp.float64:
    F16 = wp.float64(16.0);  F30 = wp.float64(30.0)
    aa = wp.abs(-pres[kx-2,iy] + F16*pres[kx-1,iy] - F30*pres[kx,iy]
                + F16*pres[kx+1,iy] - pres[kx+2,iy])
    bb = wp.abs( pres[kx-2,iy] + F16*pres[kx-1,iy] + F30*pres[kx,iy]
                + F16*pres[kx+1,iy] + pres[kx+2,iy])
    return aa / (bb + TINY) * ducros[kx, iy]

@wp.func
def pressure_sensor_y(pres  : wp.array(dtype=wp.float64, ndim=2),
                       ducros: wp.array(dtype=wp.float64, ndim=2),
                       ix: int, ky: int) -> wp.float64:
    F16 = wp.float64(16.0);  F30 = wp.float64(30.0)
    aa = wp.abs(-pres[ix,ky-2] + F16*pres[ix,ky-1] - F30*pres[ix,ky]
                + F16*pres[ix,ky+1] - pres[ix,ky+2])
    bb = wp.abs( pres[ix,ky-2] + F16*pres[ix,ky-1] + F30*pres[ix,ky]
                + F16*pres[ix,ky+1] + pres[ix,ky+2])
    return aa / (bb + TINY) * ducros[ix, ky]


# ─── Characteristic transform ─────────────────────────────────────────────────

@wp.func
def char_transform(cons: wp.array(dtype=wp.float64, ndim=3),
                   kx: int, ky: int, row: int,
                   u_r : wp.float64, v_r : wp.float64,
                   c_r : wp.float64, c2  : wp.float64,
                   mx  : wp.float64, my  : wp.float64,
                   qn  : wp.float64, q2v : wp.float64,
                   lx  : wp.float64, ly  : wp.float64,
                   ql  : wp.float64) -> wp.float64:
    q1=cons[kx,ky,0]; q2=cons[kx,ky,1]; q3=cons[kx,ky,2]; q4=cons[kx,ky,3]
    if row == 0:
        return Lq0(q1,q2,q3,q4, u_r,v_r,c_r,c2, mx,my, qn,q2v)
    elif row == 1:
        return Lq1(q1,q2,q3,q4, lx,ly,ql)
    elif row == 2:
        return Lq2(q1,q2,q3,q4, u_r,v_r,c2,q2v)
    else:
        return Lq3(q1,q2,q3,q4, u_r,v_r,c_r,c2, mx,my, qn,q2v)


# ══════════════════════════════════════════════════════════════════════════════
#  §2  Initial conditions — Configuration-3 four-quadrant Riemann
# ══════════════════════════════════════════════════════════════════════════════

@wp.kernel
def init_cond_kernel(
    x    : wp.array(dtype=wp.float64, ndim=1),
    y    : wp.array(dtype=wp.float64, ndim=1),
    rho  : wp.array(dtype=wp.float64, ndim=2),
    u_vel: wp.array(dtype=wp.float64, ndim=2),
    v_vel: wp.array(dtype=wp.float64, ndim=2),
    pres : wp.array(dtype=wp.float64, ndim=2),
    snd  : wp.array(dtype=wp.float64, ndim=2),
    nx: int, ny: int, gp: int,
    x_div: wp.float64, y_div: wp.float64,
    # Q1 (x>xd, y>yd)
    rho1: wp.float64, u1: wp.float64, v1: wp.float64, p1: wp.float64,
    # Q2 (x<=xd, y>yd)
    rho2: wp.float64, u2: wp.float64, v2: wp.float64, p2: wp.float64,
    # Q3 (x<=xd, y<=yd)
    rho3: wp.float64, u3: wp.float64, v3: wp.float64, p3: wp.float64,
    # Q4 (x>xd, y<=yd)
    rho4: wp.float64, u4: wp.float64, v4: wp.float64, p4: wp.float64,
):
    i, j = wp.tid()
    if i < 1 or i > nx or j < 1 or j > ny:
        return
    ix = i + gp;  iy = j + gp
    xi = x[ix];   yi = y[iy]

    rho_c = rho1;  uc = u1;  vc = v1;  p_c = p1
    if xi > x_div and yi > y_div:
        rho_c = rho1;  uc = u1;  vc = v1;  p_c = p1
    elif xi <= x_div and yi > y_div:
        rho_c = rho2;  uc = u2;  vc = v2;  p_c = p2
    elif xi <= x_div and yi <= y_div:
        rho_c = rho3;  uc = u3;  vc = v3;  p_c = p3
    else:
        rho_c = rho4;  uc = u4;  vc = v4;  p_c = p4

    rho[ix, iy]   = rho_c
    u_vel[ix, iy] = uc
    v_vel[ix, iy] = vc
    pres[ix, iy]  = p_c
    snd[ix, iy]   = wp.sqrt(GAMMA * p_c / rho_c)


# ══════════════════════════════════════════════════════════════════════════════
#  §3  Boundary conditions — transmissive on all four walls
#
#  Zero-order extrapolation: all ghost layers on each side copy the value
#  of the nearest interior cell.
# ══════════════════════════════════════════════════════════════════════════════

@wp.kernel
def bc_transmit_x_kernel(
    rho  : wp.array(dtype=wp.float64, ndim=2),
    u_vel: wp.array(dtype=wp.float64, ndim=2),
    v_vel: wp.array(dtype=wp.float64, ndim=2),
    pres : wp.array(dtype=wp.float64, ndim=2),
    nx: int, ny: int, gp: int
):
    """Transmissive (outflow) BC on left and right walls."""
    idx   = wp.tid()
    gy    = ny + 2*gp + 1
    layer = idx // gy
    py_j  = idx % gy
    if layer >= gp:
        return
    i = layer + 1              # ghost layer index 1..gp

    # Left wall: all ghost layers copy from first interior cell gp+1
    gl = gp + 1 - i
    sl = gp + 1
    rho[gl,  py_j] = rho[sl,  py_j]
    u_vel[gl,py_j] = u_vel[sl,py_j]
    v_vel[gl,py_j] = v_vel[sl,py_j]
    pres[gl, py_j] = pres[sl, py_j]

    # Right wall: all ghost layers copy from last interior cell gp+nx
    gr = gp + nx + i
    sr = gp + nx
    rho[gr,  py_j] = rho[sr,  py_j]
    u_vel[gr,py_j] = u_vel[sr,py_j]
    v_vel[gr,py_j] = v_vel[sr,py_j]
    pres[gr, py_j] = pres[sr, py_j]


@wp.kernel
def bc_transmit_y_kernel(
    rho  : wp.array(dtype=wp.float64, ndim=2),
    u_vel: wp.array(dtype=wp.float64, ndim=2),
    v_vel: wp.array(dtype=wp.float64, ndim=2),
    pres : wp.array(dtype=wp.float64, ndim=2),
    nx: int, ny: int, gp: int
):
    """Transmissive (outflow) BC on bottom and top walls."""
    idx   = wp.tid()
    gx    = nx + 2*gp + 1
    layer = idx // gx
    py_i  = idx % gx
    if layer >= gp:
        return
    j = layer + 1              # ghost layer index 1..gp

    # Bottom wall: all ghost layers copy from first interior cell gp+1
    gb = gp + 1 - j
    sb = gp + 1
    rho[py_i,  gb] = rho[py_i,  sb]
    u_vel[py_i,gb] = u_vel[py_i,sb]
    v_vel[py_i,gb] = v_vel[py_i,sb]
    pres[py_i, gb] = pres[py_i, sb]

    # Top wall: all ghost layers copy from last interior cell gp+ny
    gt = gp + ny + j
    st = gp + ny
    rho[py_i,  gt] = rho[py_i,  st]
    u_vel[py_i,gt] = u_vel[py_i,st]
    v_vel[py_i,gt] = v_vel[py_i,st]
    pres[py_i, gt] = pres[py_i, st]


# ══════════════════════════════════════════════════════════════════════════════
#  §4  Primitive ↔ Conservative  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

@wp.kernel
def prim_to_cons_kernel(
    rho  : wp.array(dtype=wp.float64, ndim=2),
    u_vel: wp.array(dtype=wp.float64, ndim=2),
    v_vel: wp.array(dtype=wp.float64, ndim=2),
    pres : wp.array(dtype=wp.float64, ndim=2),
    cons : wp.array(dtype=wp.float64, ndim=3),
):
    ix, iy = wp.tid()
    r = rho[ix,iy]; u = u_vel[ix,iy]; v = v_vel[ix,iy]; p = pres[ix,iy]
    cons[ix,iy,0] = r
    cons[ix,iy,1] = r * u
    cons[ix,iy,2] = r * v
    cons[ix,iy,3] = p / GM1 + HALF * r * (u*u + v*v)


@wp.kernel
def cons_to_prim_kernel(
    cons : wp.array(dtype=wp.float64, ndim=3),
    rho  : wp.array(dtype=wp.float64, ndim=2),
    u_vel: wp.array(dtype=wp.float64, ndim=2),
    v_vel: wp.array(dtype=wp.float64, ndim=2),
    pres : wp.array(dtype=wp.float64, ndim=2),
    nx: int, ny: int, gp: int
):
    i, j = wp.tid()
    if i < 1 or i > nx or j < 1 or j > ny:
        return
    ix = i + gp;  iy = j + gp
    r  = cons[ix,iy,0]; ru = cons[ix,iy,1]
    rv = cons[ix,iy,2]; E  = cons[ix,iy,3]
    u  = ru / r;  v = rv / r
    p  = GM1 * (E - HALF * r * (u*u + v*v))
    rho[ix,iy]   = r
    u_vel[ix,iy] = u
    v_vel[ix,iy] = v
    pres[ix,iy]  = p


# ══════════════════════════════════════════════════════════════════════════════
#  §5  Velocity derivatives + Ducros sensor  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

@wp.kernel
def velocity_deriv_kernel(
    u_vel : wp.array(dtype=wp.float64, ndim=2),
    v_vel : wp.array(dtype=wp.float64, ndim=2),
    der_ux: wp.array(dtype=wp.float64, ndim=2),
    der_vx: wp.array(dtype=wp.float64, ndim=2),
    der_uy: wp.array(dtype=wp.float64, ndim=2),
    der_vy: wp.array(dtype=wp.float64, ndim=2),
    gx: int, gy: int
):
    ti, tj = wp.tid()
    ix = ti + 1;  iy = tj + 1
    if ix >= gx - 1 or iy >= gy - 1:
        return
    der_ux[ix,iy] = HALF*(-u_vel[ix-1,iy] + u_vel[ix+1,iy])
    der_vx[ix,iy] = HALF*(-v_vel[ix-1,iy] + v_vel[ix+1,iy])
    der_uy[ix,iy] = HALF*(-u_vel[ix,iy-1] + u_vel[ix,iy+1])
    der_vy[ix,iy] = HALF*(-v_vel[ix,iy-1] + v_vel[ix,iy+1])


@wp.kernel
def ducros_kernel(
    der_ux: wp.array(dtype=wp.float64, ndim=2),
    der_vx: wp.array(dtype=wp.float64, ndim=2),
    der_uy: wp.array(dtype=wp.float64, ndim=2),
    der_vy: wp.array(dtype=wp.float64, ndim=2),
    ducros: wp.array(dtype=wp.float64, ndim=2),
    gx: int, gy: int
):
    ti, tj = wp.tid()
    ix = ti + 1;  iy = tj + 1
    if ix >= gx - 1 or iy >= gy - 1:
        return
    div   = der_ux[ix,iy] + der_vy[ix,iy]
    curl  = der_vx[ix,iy] - der_uy[ix,iy]
    ducros[ix,iy] = div*div / (div*div + curl*curl + TINY)


# ══════════════════════════════════════════════════════════════════════════════
#  §6  WCNS reconstruction — X  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

@wp.kernel
def wcns_x_kernel(
    cons  : wp.array(dtype=wp.float64, ndim=3),
    pres  : wp.array(dtype=wp.float64, ndim=2),
    ducros: wp.array(dtype=wp.float64, ndim=2),
    consl : wp.array(dtype=wp.float64, ndim=3),
    consr : wp.array(dtype=wp.float64, ndim=3),
    nx: int, ny: int, gp: int
):
    face_ix, iy = wp.tid()
    if face_ix < gp - 1 or face_ix > nx + gp:
        return
    if iy < gp + 1 or iy > ny + gp:
        return

    sig_max = wp.max(pressure_sensor_x(pres, ducros, face_ix-1, iy),
                wp.max(pressure_sensor_x(pres, ducros, face_ix,   iy),
                       pressure_sensor_x(pres, ducros, face_ix+1, iy)))
    f1idx   = face_ix - gp
    is_smooth = (sig_max < wp.float64(0.01)) and (f1idx > 3) and (f1idx < nx - 2)
    eta = F0p54
    if not is_smooth:
        eta = F1

    rL  = cons[face_ix,  iy,0]; rR  = cons[face_ix+1,iy,0]
    ruL = cons[face_ix,  iy,1]; ruR = cons[face_ix+1,iy,1]
    rvL = cons[face_ix,  iy,2]; rvR = cons[face_ix+1,iy,2]
    EL  = cons[face_ix,  iy,3]; ER  = cons[face_ix+1,iy,3]

    sqL = wp.sqrt(rL);  sqR = wp.sqrt(rR);  div = F1/(sqL+sqR)
    pL  = GM1*(EL - HALF*(ruL*ruL+rvL*rvL)/rL)
    pR  = GM1*(ER - HALF*(ruR*ruR+rvR*rvR)/rR)
    HL  = (EL+pL)/rL;  HR = (ER+pR)/rR

    u_r   = (sqL*(ruL/rL) + sqR*(ruR/rR))*div
    v_r   = (sqL*(rvL/rL) + sqR*(rvR/rR))*div
    H_r   = (sqL*HL       + sqR*HR      )*div
    q2v   = u_r*u_r + v_r*v_r
    c2    = wp.max((GAMMA-F1)*(H_r - HALF*q2v), TINY)
    c_r   = wp.sqrt(c2)

    mx = F1; my = F0; lx = F0; ly = F1
    qn = u_r; ql = v_r

    c_vm1=char_transform(cons,face_ix-1,iy,0,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v0 =char_transform(cons,face_ix,  iy,0,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v1 =char_transform(cons,face_ix+1,iy,0,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v2 =char_transform(cons,face_ix+2,iy,0,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    ul0=wcns_ul_acoustic(c_vm1,c_v0,c_v1,c_v2,eta)
    ur0=wcns_ur_acoustic(c_vm1,c_v0,c_v1,c_v2,eta)

    c_vm1=char_transform(cons,face_ix-1,iy,1,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v0 =char_transform(cons,face_ix,  iy,1,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v1 =char_transform(cons,face_ix+1,iy,1,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v2 =char_transform(cons,face_ix+2,iy,1,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    ul1=wcns_ul_shear(c_vm1,c_v0,c_v1,c_v2,eta)
    ur1=wcns_ur_shear(c_vm1,c_v0,c_v1,c_v2,eta)

    c_vm1=char_transform(cons,face_ix-1,iy,2,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v0 =char_transform(cons,face_ix,  iy,2,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v1 =char_transform(cons,face_ix+1,iy,2,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v2 =char_transform(cons,face_ix+2,iy,2,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    ul2=wcns_ul_entropy(c_vm1,c_v0,c_v1,c_v2)
    ur2=wcns_ur_entropy(c_vm1,c_v0,c_v1,c_v2)

    c_vm1=char_transform(cons,face_ix-1,iy,3,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v0 =char_transform(cons,face_ix,  iy,3,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v1 =char_transform(cons,face_ix+1,iy,3,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v2 =char_transform(cons,face_ix+2,iy,3,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    ul3=wcns_ul_acoustic(c_vm1,c_v0,c_v1,c_v2,eta)
    ur3=wcns_ur_acoustic(c_vm1,c_v0,c_v1,c_v2,eta)

    consl[face_ix,iy,0] = Rv0(ul0,ul1,ul2,ul3)
    consl[face_ix,iy,1] = Rv1(ul0,ul1,ul2,ul3, u_r,c_r,mx,lx)
    consl[face_ix,iy,2] = Rv2(ul0,ul1,ul2,ul3, v_r,c_r,my,ly)
    consl[face_ix,iy,3] = Rv3(ul0,ul1,ul2,ul3, H_r,qn,c_r,ql,q2v)

    consr[face_ix,iy,0] = Rv0(ur0,ur1,ur2,ur3)
    consr[face_ix,iy,1] = Rv1(ur0,ur1,ur2,ur3, u_r,c_r,mx,lx)
    consr[face_ix,iy,2] = Rv2(ur0,ur1,ur2,ur3, v_r,c_r,my,ly)
    consr[face_ix,iy,3] = Rv3(ur0,ur1,ur2,ur3, H_r,qn,c_r,ql,q2v)


# ══════════════════════════════════════════════════════════════════════════════
#  §7  WCNS reconstruction — Y  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

@wp.kernel
def wcns_y_kernel(
    cons  : wp.array(dtype=wp.float64, ndim=3),
    pres  : wp.array(dtype=wp.float64, ndim=2),
    ducros: wp.array(dtype=wp.float64, ndim=2),
    consl : wp.array(dtype=wp.float64, ndim=3),
    consr : wp.array(dtype=wp.float64, ndim=3),
    nx: int, ny: int, gp: int
):
    ix, face_iy = wp.tid()
    if ix < gp + 1 or ix > nx + gp:
        return
    if face_iy < gp - 1 or face_iy > ny + gp:
        return

    sig_max = wp.max(pressure_sensor_y(pres, ducros, ix, face_iy-1),
                wp.max(pressure_sensor_y(pres, ducros, ix, face_iy  ),
                       pressure_sensor_y(pres, ducros, ix, face_iy+1)))
    f1idx   = face_iy - gp
    is_smooth = (sig_max < wp.float64(0.01)) and (f1idx > 3) and (f1idx < ny - 2)
    eta = F0p54
    if not is_smooth:
        eta = F1

    rL  = cons[ix,face_iy,  0]; rR  = cons[ix,face_iy+1,0]
    ruL = cons[ix,face_iy,  1]; ruR = cons[ix,face_iy+1,1]
    rvL = cons[ix,face_iy,  2]; rvR = cons[ix,face_iy+1,2]
    EL  = cons[ix,face_iy,  3]; ER  = cons[ix,face_iy+1,3]

    sqL = wp.sqrt(rL);  sqR = wp.sqrt(rR);  div = F1/(sqL+sqR)
    pL  = GM1*(EL - HALF*(ruL*ruL+rvL*rvL)/rL)
    pR  = GM1*(ER - HALF*(ruR*ruR+rvR*rvR)/rR)
    HL  = (EL+pL)/rL;  HR = (ER+pR)/rR

    u_r  = (sqL*(ruL/rL) + sqR*(ruR/rR))*div
    v_r  = (sqL*(rvL/rL) + sqR*(rvR/rR))*div
    H_r  = (sqL*HL       + sqR*HR      )*div
    q2v  = u_r*u_r + v_r*v_r
    c2   = wp.max((GAMMA-F1)*(H_r - HALF*q2v), TINY)
    c_r  = wp.sqrt(c2)

    mx = F0; my = F1; lx = NF1; ly = F0
    qn = v_r; ql = -u_r

    c_vm1=char_transform(cons,ix,face_iy-1,0,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v0 =char_transform(cons,ix,face_iy,  0,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v1 =char_transform(cons,ix,face_iy+1,0,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v2 =char_transform(cons,ix,face_iy+2,0,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    ul0=wcns_ul_acoustic(c_vm1,c_v0,c_v1,c_v2,eta); ur0=wcns_ur_acoustic(c_vm1,c_v0,c_v1,c_v2,eta)

    c_vm1=char_transform(cons,ix,face_iy-1,1,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v0 =char_transform(cons,ix,face_iy,  1,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v1 =char_transform(cons,ix,face_iy+1,1,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v2 =char_transform(cons,ix,face_iy+2,1,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    ul1=wcns_ul_shear(c_vm1,c_v0,c_v1,c_v2,eta); ur1=wcns_ur_shear(c_vm1,c_v0,c_v1,c_v2,eta)

    c_vm1=char_transform(cons,ix,face_iy-1,2,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v0 =char_transform(cons,ix,face_iy,  2,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v1 =char_transform(cons,ix,face_iy+1,2,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v2 =char_transform(cons,ix,face_iy+2,2,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    ul2=wcns_ul_entropy(c_vm1,c_v0,c_v1,c_v2); ur2=wcns_ur_entropy(c_vm1,c_v0,c_v1,c_v2)

    c_vm1=char_transform(cons,ix,face_iy-1,3,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v0 =char_transform(cons,ix,face_iy,  3,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v1 =char_transform(cons,ix,face_iy+1,3,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    c_v2 =char_transform(cons,ix,face_iy+2,3,u_r,v_r,c_r,c2,mx,my,qn,q2v,lx,ly,ql)
    ul3=wcns_ul_acoustic(c_vm1,c_v0,c_v1,c_v2,eta); ur3=wcns_ur_acoustic(c_vm1,c_v0,c_v1,c_v2,eta)

    consl[ix,face_iy,0] = Rv0(ul0,ul1,ul2,ul3)
    consl[ix,face_iy,1] = Rv1(ul0,ul1,ul2,ul3, u_r,c_r,mx,lx)
    consl[ix,face_iy,2] = Rv2(ul0,ul1,ul2,ul3, v_r,c_r,my,ly)
    consl[ix,face_iy,3] = Rv3(ul0,ul1,ul2,ul3, H_r,qn,c_r,ql,q2v)

    consr[ix,face_iy,0] = Rv0(ur0,ur1,ur2,ur3)
    consr[ix,face_iy,1] = Rv1(ur0,ur1,ur2,ur3, u_r,c_r,mx,lx)
    consr[ix,face_iy,2] = Rv2(ur0,ur1,ur2,ur3, v_r,c_r,my,ly)
    consr[ix,face_iy,3] = Rv3(ur0,ur1,ur2,ur3, H_r,qn,c_r,ql,q2v)


# ══════════════════════════════════════════════════════════════════════════════
#  §8  HLLC flux + residual — X  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

@wp.func
def hllc_x(cL0: wp.float64, cL1: wp.float64, cL2: wp.float64, cL3: wp.float64,
            cR0: wp.float64, cR1: wp.float64, cR2: wp.float64, cR3: wp.float64) -> wp.vec4d:
    rL=cL0; rR=cR0
    uL=cL1/rL; uR=cR1/rR; vL=cL2/rL; vR=cR2/rR
    pL=GM1*(cL3 - HALF*rL*(uL*uL+vL*vL)); pR=GM1*(cR3 - HALF*rR*(uR*uR+vR*vR))
    HL=(cL3+pL)/rL; HR=(cR3+pR)/rR
    sqL=wp.sqrt(rL); sqR=wp.sqrt(rR); div=F1/(sqL+sqR)
    uav=(sqL*uL+sqR*uR)*div; vav=(sqL*vL+sqR*vR)*div; Hav=(sqL*HL+sqR*HR)*div
    q2av=uav*uav+vav*vav
    cav=wp.sqrt(wp.max((GAMMA-F1)*(Hav-HALF*q2av),TINY))
    cL_s=wp.sqrt(wp.max(GAMMA*pL/rL,TINY)); cR_s=wp.sqrt(wp.max(GAMMA*pR/rR,TINY))
    SL=wp.min(uL-cL_s, uav-cav); SR=wp.max(uR+cR_s, uav+cav)
    denom_x = rL*(SL-uL) - rR*(SR-uR)
    if wp.abs(denom_x) < TINY:
        denom_x = TINY
    SP=(pR-pL + rL*uL*(SL-uL) - rR*uR*(SR-uR)) / denom_x

    fL0=cL1; fL1=cL1*uL+pL; fL2=cL1*vL; fL3=uL*(cL3+pL)
    fR0=cR1; fR1=cR1*uR+pR; fR2=cR1*vR; fR3=uR*(cR3+pR)

    f0=F0; f1=F0; f2=F0; f3=F0
    if SL > F0:
        f0=fL0; f1=fL1; f2=fL2; f3=fL3
    elif SL <= F0 and F0 < SP:
        fac=rL*(SL-uL)/(SL-SP)
        EL_=cL3/rL+(SP-uL)*(SP+pL/(rL*(SL-uL)))
        f0=fL0+SL*(fac-cL0); f1=fL1+SL*(fac*SP-cL1)
        f2=fL2+SL*(fac*vL-cL2); f3=fL3+SL*(fac*EL_-cL3)
    elif SP <= F0 and F0 <= SR:
        fac=rR*(SR-uR)/(SR-SP)
        ER_=cR3/rR+(SP-uR)*(SP+pR/(rR*(SR-uR)))
        f0=fR0+SR*(fac-cR0); f1=fR1+SR*(fac*SP-cR1)
        f2=fR2+SR*(fac*vR-cR2); f3=fR3+SR*(fac*ER_-cR3)
    else:
        f0=fR0; f1=fR1; f2=fR2; f3=fR3
    return wp.vec4d(f0,f1,f2,f3)


@wp.kernel
def flux_x_residual_kernel(
    cons  : wp.array(dtype=wp.float64, ndim=3),
    consl : wp.array(dtype=wp.float64, ndim=3),
    consr : wp.array(dtype=wp.float64, ndim=3),
    resid : wp.array(dtype=wp.float64, ndim=3),
    dx    : wp.float64,
    nx: int, ny: int, gp: int
):
    i, j = wp.tid()
    if i < 1 or i > nx or j < 1 or j > ny:
        return
    ix=i+gp; iy=j+gp

    rL0=consl[ix,  iy,0]; rL1=consl[ix,  iy,1]; rL2=consl[ix,  iy,2]; rL3=consl[ix,  iy,3]
    rR0=consr[ix,  iy,0]; rR1=consr[ix,  iy,1]; rR2=consr[ix,  iy,2]; rR3=consr[ix,  iy,3]
    if is_physical(rL0,rL1,rL2,rL3) == 0 or is_physical(rR0,rR1,rR2,rR3) == 0:
        rL0=cons[ix,  iy,0]; rL1=cons[ix,  iy,1]; rL2=cons[ix,  iy,2]; rL3=cons[ix,  iy,3]
        rR0=cons[ix+1,iy,0]; rR1=cons[ix+1,iy,1]; rR2=cons[ix+1,iy,2]; rR3=cons[ix+1,iy,3]
    fr=hllc_x(rL0,rL1,rL2,rL3, rR0,rR1,rR2,rR3)

    lL0=consl[ix-1,iy,0]; lL1=consl[ix-1,iy,1]; lL2=consl[ix-1,iy,2]; lL3=consl[ix-1,iy,3]
    lR0=consr[ix-1,iy,0]; lR1=consr[ix-1,iy,1]; lR2=consr[ix-1,iy,2]; lR3=consr[ix-1,iy,3]
    if is_physical(lL0,lL1,lL2,lL3) == 0 or is_physical(lR0,lR1,lR2,lR3) == 0:
        lL0=cons[ix-1,iy,0]; lL1=cons[ix-1,iy,1]; lL2=cons[ix-1,iy,2]; lL3=cons[ix-1,iy,3]
        lR0=cons[ix,  iy,0]; lR1=cons[ix,  iy,1]; lR2=cons[ix,  iy,2]; lR3=cons[ix,  iy,3]
    fl=hllc_x(lL0,lL1,lL2,lL3, lR0,lR1,lR2,lR3)

    resid[ix,iy,0] = resid[ix,iy,0] - (fr[0]-fl[0])/dx
    resid[ix,iy,1] = resid[ix,iy,1] - (fr[1]-fl[1])/dx
    resid[ix,iy,2] = resid[ix,iy,2] - (fr[2]-fl[2])/dx
    resid[ix,iy,3] = resid[ix,iy,3] - (fr[3]-fl[3])/dx


# ══════════════════════════════════════════════════════════════════════════════
#  §9  HLLC flux + residual — Y  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

@wp.func
def hllc_y(cL0: wp.float64, cL1: wp.float64, cL2: wp.float64, cL3: wp.float64,
            cR0: wp.float64, cR1: wp.float64, cR2: wp.float64, cR3: wp.float64) -> wp.vec4d:
    rL=cL0; rR=cR0
    uL=cL1/rL; uR=cR1/rR; vL=cL2/rL; vR=cR2/rR
    pL=GM1*(cL3 - HALF*rL*(uL*uL+vL*vL)); pR=GM1*(cR3 - HALF*rR*(uR*uR+vR*vR))
    HL=(cL3+pL)/rL; HR=(cR3+pR)/rR
    sqL=wp.sqrt(rL); sqR=wp.sqrt(rR); div=F1/(sqL+sqR)
    uav=(sqL*uL+sqR*uR)*div; vav=(sqL*vL+sqR*vR)*div; Hav=(sqL*HL+sqR*HR)*div
    q2av=uav*uav+vav*vav
    cav=wp.sqrt(wp.max((GAMMA-F1)*(Hav-HALF*q2av),TINY))
    cL_s=wp.sqrt(wp.max(GAMMA*pL/rL,TINY)); cR_s=wp.sqrt(wp.max(GAMMA*pR/rR,TINY))
    SL=wp.min(vL-cL_s, vav-cav); SR=wp.max(vR+cR_s, vav+cav)
    denom_y = rL*(SL-vL) - rR*(SR-vR)
    if wp.abs(denom_y) < TINY:
        denom_y = TINY
    SP=(pR-pL + rL*vL*(SL-vL) - rR*vR*(SR-vR)) / denom_y

    fL0=cL2; fL1=cL2*uL; fL2=cL2*vL+pL; fL3=cL2*HL
    fR0=cR2; fR1=cR2*uR; fR2=cR2*vR+pR; fR3=cR2*HR

    f0=F0; f1=F0; f2=F0; f3=F0
    if SL > F0:
        f0=fL0; f1=fL1; f2=fL2; f3=fL3
    elif SL <= F0 and F0 < SP:
        fac=rL*(SL-vL)/(SL-SP)
        EL_=cL3/rL+(SP-vL)*(SP+pL/(rL*(SL-vL)))
        f0=fL0+SL*(fac-cL0); f1=fL1+SL*(fac*uL-cL1)
        f2=fL2+SL*(fac*SP-cL2); f3=fL3+SL*(fac*EL_-cL3)
    elif SP <= F0 and F0 <= SR:
        fac=rR*(SR-vR)/(SR-SP)
        ER_=cR3/rR+(SP-vR)*(SP+pR/(rR*(SR-vR)))
        f0=fR0+SR*(fac-cR0); f1=fR1+SR*(fac*uR-cR1)
        f2=fR2+SR*(fac*SP-cR2); f3=fR3+SR*(fac*ER_-cR3)
    else:
        f0=fR0; f1=fR1; f2=fR2; f3=fR3
    return wp.vec4d(f0,f1,f2,f3)


@wp.kernel
def flux_y_residual_kernel(
    cons  : wp.array(dtype=wp.float64, ndim=3),
    consl : wp.array(dtype=wp.float64, ndim=3),
    consr : wp.array(dtype=wp.float64, ndim=3),
    resid : wp.array(dtype=wp.float64, ndim=3),
    dy    : wp.float64,
    nx: int, ny: int, gp: int
):
    i, j = wp.tid()
    if i < 1 or i > nx or j < 1 or j > ny:
        return
    ix=i+gp; iy=j+gp

    rL0=consl[ix,iy,  0]; rL1=consl[ix,iy,  1]; rL2=consl[ix,iy,  2]; rL3=consl[ix,iy,  3]
    rR0=consr[ix,iy,  0]; rR1=consr[ix,iy,  1]; rR2=consr[ix,iy,  2]; rR3=consr[ix,iy,  3]
    if is_physical(rL0,rL1,rL2,rL3) == 0 or is_physical(rR0,rR1,rR2,rR3) == 0:
        rL0=cons[ix,iy,  0]; rL1=cons[ix,iy,  1]; rL2=cons[ix,iy,  2]; rL3=cons[ix,iy,  3]
        rR0=cons[ix,iy+1,0]; rR1=cons[ix,iy+1,1]; rR2=cons[ix,iy+1,2]; rR3=cons[ix,iy+1,3]
    fr=hllc_y(rL0,rL1,rL2,rL3, rR0,rR1,rR2,rR3)

    lL0=consl[ix,iy-1,0]; lL1=consl[ix,iy-1,1]; lL2=consl[ix,iy-1,2]; lL3=consl[ix,iy-1,3]
    lR0=consr[ix,iy-1,0]; lR1=consr[ix,iy-1,1]; lR2=consr[ix,iy-1,2]; lR3=consr[ix,iy-1,3]
    if is_physical(lL0,lL1,lL2,lL3) == 0 or is_physical(lR0,lR1,lR2,lR3) == 0:
        lL0=cons[ix,iy-1,0]; lL1=cons[ix,iy-1,1]; lL2=cons[ix,iy-1,2]; lL3=cons[ix,iy-1,3]
        lR0=cons[ix,iy,  0]; lR1=cons[ix,iy,  1]; lR2=cons[ix,iy,  2]; lR3=cons[ix,iy,  3]
    fl=hllc_y(lL0,lL1,lL2,lL3, lR0,lR1,lR2,lR3)

    resid[ix,iy,0] = resid[ix,iy,0] - (fr[0]-fl[0])/dy
    resid[ix,iy,1] = resid[ix,iy,1] - (fr[1]-fl[1])/dy
    resid[ix,iy,2] = resid[ix,iy,2] - (fr[2]-fl[2])/dy
    resid[ix,iy,3] = resid[ix,iy,3] - (fr[3]-fl[3])/dy


# ══════════════════════════════════════════════════════════════════════════════
#  §10  SSP-RK3  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

@wp.kernel
def rk3_step_kernel(
    cons0   : wp.array(dtype=wp.float64, ndim=3),
    cons_in : wp.array(dtype=wp.float64, ndim=3),
    resid   : wp.array(dtype=wp.float64, ndim=3),
    cons_out: wp.array(dtype=wp.float64, ndim=3),
    alpha: wp.float64, beta: wp.float64, dt: wp.float64,
    nx: int, ny: int, gp: int
):
    i, j = wp.tid()
    if i < 1 or i > nx or j < 1 or j > ny:
        return
    ix=i+gp; iy=j+gp
    for k in range(4):
        cons_out[ix,iy,k] = alpha*cons0[ix,iy,k] + beta*(cons_in[ix,iy,k] + dt*resid[ix,iy,k])


@wp.kernel
def zero_residual_kernel(resid: wp.array(dtype=wp.float64, ndim=3),
                         nx: int, ny: int, gp: int):
    i, j = wp.tid()
    if i < 1 or i > nx or j < 1 or j > ny:
        return
    ix=i+gp; iy=j+gp
    resid[ix,iy,0]=F0; resid[ix,iy,1]=F0; resid[ix,iy,2]=F0; resid[ix,iy,3]=F0


# ══════════════════════════════════════════════════════════════════════════════
#  §11  Time step  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

@wp.kernel
def dt_local_kernel(
    u_vel : wp.array(dtype=wp.float64, ndim=2),
    v_vel : wp.array(dtype=wp.float64, ndim=2),
    pres  : wp.array(dtype=wp.float64, ndim=2),
    rho   : wp.array(dtype=wp.float64, ndim=2),
    snd   : wp.array(dtype=wp.float64, ndim=2),
    dt_arr: wp.array(dtype=wp.float64, ndim=1),
    dx: wp.float64, dy: wp.float64,
    nx: int, ny: int, gp: int
):
    i, j = wp.tid()
    if i < 1 or i > nx or j < 1 or j > ny:
        return
    ix=i+gp; iy=j+gp
    c  = wp.sqrt(GAMMA * pres[ix,iy] / rho[ix,iy])
    snd[ix,iy] = c
    sx = wp.abs(u_vel[ix,iy]) + c
    sy = wp.abs(v_vel[ix,iy]) + c
    dt_cell = wp.min(dx/sx, dy/sy)
    dt_arr[(i-1)*ny + (j-1)] = dt_cell


# ══════════════════════════════════════════════════════════════════════════════
#  §12  Output  (full 2-D grid)
# ══════════════════════════════════════════════════════════════════════════════


def save_npz(rho_np, u_np, v_np, p_np, x_np, y_np, step, time_sim):
    fname = f"riem_{step:06d}.npz"
    np.savez_compressed(fname,
                        time=np.float64(time_sim),
                        x=x_np[G:G+NX],
                        y=y_np[G:G+NY],
                        rho=rho_np[G:G+NX, G:G+NY],
                        p=p_np[G:G+NX, G:G+NY],
                        u=u_np[G:G+NX, G:G+NY],
                        v=v_np[G:G+NX, G:G+NY])
    print(f"  → {fname}")


def save_png(rho_np, x_np, y_np, step, time_sim):
    # Build 2-D coordinate grids (j-outer, i-inner — same layout as R.py reshape)
    xg = np.array([[x_np[i + G] for i in range(NX)] for j in range(NY)])
    yg = np.array([[y_np[j + G] for i in range(NX)] for j in range(NY)])
    d  = rho_np[G:G+NX, G:G+NY].T   # transpose so first index = j

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contour(xg, yg, d, 30, linewidths=0.3, colors='k')
    # ax.set_xlabel(r'\textbf{x}')
    # ax.set_ylabel(r'\textbf{y}')
    ax.set_title(f't = {time_sim:.4f}')
    ax.set_aspect('equal')
    fname = f"riem_{step:06d}.png"
    fig.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"  → {fname}")

# ══════════════════════════════════════════════════════════════════════════════
#  §13  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"2D Euler/Warp  Explosion  NX=NY={NX}  device={DEVICE}")

    dx_py = (XMAX_PY - XMIN_PY) / NX
    dy_py = (YMAX_PY - YMIN_PY) / NY

    # Cell-centre coordinates (ghost cells included)
    x_np = np.array([XMIN_PY + (i - G - 0.5)*dx_py for i in range(GX)], dtype=np.float64)
    y_np = np.array([YMIN_PY + (j - G - 0.5)*dy_py for j in range(GY)], dtype=np.float64)

    x_d = wp.array(x_np, dtype=wp.float64, device=DEVICE)
    y_d = wp.array(y_np, dtype=wp.float64, device=DEVICE)

    rho_d = wp.zeros((GX,GY), dtype=wp.float64, device=DEVICE)
    u_d   = wp.zeros((GX,GY), dtype=wp.float64, device=DEVICE)
    v_d   = wp.zeros((GX,GY), dtype=wp.float64, device=DEVICE)
    p_d   = wp.zeros((GX,GY), dtype=wp.float64, device=DEVICE)
    snd_d = wp.zeros((GX,GY), dtype=wp.float64, device=DEVICE)

    cons0 = wp.zeros((GX,GY,4), dtype=wp.float64, device=DEVICE)
    cons1 = wp.zeros((GX,GY,4), dtype=wp.float64, device=DEVICE)
    cons2 = wp.zeros((GX,GY,4), dtype=wp.float64, device=DEVICE)

    consl = wp.zeros((GX,GY,4), dtype=wp.float64, device=DEVICE)
    consr = wp.zeros((GX,GY,4), dtype=wp.float64, device=DEVICE)
    resid = wp.zeros((GX,GY,4), dtype=wp.float64, device=DEVICE)

    der_ux = wp.zeros((GX,GY), dtype=wp.float64, device=DEVICE)
    der_vx = wp.zeros((GX,GY), dtype=wp.float64, device=DEVICE)
    der_uy = wp.zeros((GX,GY), dtype=wp.float64, device=DEVICE)
    der_vy = wp.zeros((GX,GY), dtype=wp.float64, device=DEVICE)
    ducros = wp.zeros((GX,GY), dtype=wp.float64, device=DEVICE)
    dt_arr = wp.zeros(NX*NY,   dtype=wp.float64, device=DEVICE)

    # ── Initial conditions ────────────────────────────────────────────────────
    wp.launch(init_cond_kernel, dim=(NX+2,NY+2),
              inputs=[x_d,y_d, rho_d,u_d,v_d,p_d,snd_d, NX,NY,G,
                      wp.float64(X_DIV), wp.float64(Y_DIV),
                      wp.float64(RHO_Q1), wp.float64(U_Q1), wp.float64(V_Q1), wp.float64(P_Q1),
                      wp.float64(RHO_Q2), wp.float64(U_Q2), wp.float64(V_Q2), wp.float64(P_Q2),
                      wp.float64(RHO_Q3), wp.float64(U_Q3), wp.float64(V_Q3), wp.float64(P_Q3),
                      wp.float64(RHO_Q4), wp.float64(U_Q4), wp.float64(V_Q4), wp.float64(P_Q4)],
              device=DEVICE)
    wp.launch(bc_transmit_x_kernel, dim=(G*GY,),
              inputs=[rho_d,u_d,v_d,p_d, NX,NY,G], device=DEVICE)
    wp.launch(bc_transmit_y_kernel, dim=(G*GX,),
              inputs=[rho_d,u_d,v_d,p_d, NX,NY,G], device=DEVICE)
    wp.launch(prim_to_cons_kernel, dim=(GX,GY),
              inputs=[rho_d,u_d,v_d,p_d,cons0], device=DEVICE)

    rho_np0 = rho_d.numpy(); u_np0 = u_d.numpy()
    v_np0   = v_d.numpy();   p_np0 = p_d.numpy()
    save_npz(rho_np0, u_np0, v_np0, p_np0, x_np, y_np, step=0, time_sim=0.0)

    time_sim = 0.0;  N = 1;  t0 = time.perf_counter()
    print(f"{'step':>8}  {'time':>12}")

    while time_sim < T_END and N <= NTMAX:

        # ── CFL dt ───────────────────────────────────────────────────────────
        wp.launch(dt_local_kernel, dim=(NX+2,NY+2),
                  inputs=[u_d,v_d,p_d,rho_d,snd_d,dt_arr,
                          wp.float64(dx_py), wp.float64(dy_py), NX,NY,G],
                  device=DEVICE)
        dt = float(CFL_PY * dt_arr.numpy().min())
        if time_sim + dt > T_END:
            dt = T_END - time_sim
        time_sim += dt
        print(f"{N:>8}  {time_sim:>12.6f}")

        # ── SSP-RK3 ──────────────────────────────────────────────────────────
        rk_stages = [
            (0.0, 1.0,   cons0, cons1),   # U1 = U0 + dt*L(U0)
            (0.75, 0.25, cons1, cons2),   # U2 = 3/4 U0 + 1/4(U1 + dt*L(U1))
            (1/3,  2/3,  cons2, cons0),   # Un+1 = 1/3 U0 + 2/3(U2 + dt*L(U2))
        ]

        for alpha, beta, c_in, c_out in rk_stages:
            wp.launch(cons_to_prim_kernel, dim=(NX+2,NY+2),
                      inputs=[c_in,rho_d,u_d,v_d,p_d, NX,NY,G], device=DEVICE)
            wp.launch(bc_transmit_x_kernel, dim=(G*GY,),
                      inputs=[rho_d,u_d,v_d,p_d, NX,NY,G], device=DEVICE)
            wp.launch(bc_transmit_y_kernel, dim=(G*GX,),
                      inputs=[rho_d,u_d,v_d,p_d, NX,NY,G], device=DEVICE)
            wp.launch(prim_to_cons_kernel, dim=(GX,GY),
                      inputs=[rho_d,u_d,v_d,p_d,c_in], device=DEVICE)

            wp.launch(velocity_deriv_kernel, dim=(GX-2,GY-2),
                      inputs=[u_d,v_d, der_ux,der_vx,der_uy,der_vy, GX,GY],
                      device=DEVICE)
            wp.launch(ducros_kernel, dim=(GX-2,GY-2),
                      inputs=[der_ux,der_vx,der_uy,der_vy,ducros, GX,GY],
                      device=DEVICE)

            wp.launch(zero_residual_kernel, dim=(NX+2,NY+2),
                      inputs=[resid,NX,NY,G], device=DEVICE)

            wp.launch(wcns_x_kernel, dim=(GX,GY),
                      inputs=[c_in,p_d,ducros,consl,consr, NX,NY,G], device=DEVICE)
            wp.launch(flux_x_residual_kernel, dim=(NX+2,NY+2),
                      inputs=[c_in,consl,consr,resid, wp.float64(dx_py), NX,NY,G],
                      device=DEVICE)

            wp.launch(wcns_y_kernel, dim=(GX,GY),
                      inputs=[c_in,p_d,ducros,consl,consr, NX,NY,G], device=DEVICE)
            wp.launch(flux_y_residual_kernel, dim=(NX+2,NY+2),
                      inputs=[c_in,consl,consr,resid, wp.float64(dy_py), NX,NY,G],
                      device=DEVICE)

            wp.launch(rk3_step_kernel, dim=(NX+2,NY+2),
                      inputs=[cons0,c_in,resid,c_out,
                               wp.float64(alpha), wp.float64(beta), wp.float64(dt),
                               NX,NY,G],
                      device=DEVICE)

        wp.launch(cons_to_prim_kernel, dim=(NX+2,NY+2),
                  inputs=[cons0,rho_d,u_d,v_d,p_d, NX,NY,G], device=DEVICE)

        if N % FILE_SAVE == 0:
            rho_np = rho_d.numpy(); u_np = u_d.numpy()
            v_np   = v_d.numpy();   p_np = p_d.numpy()
            save_npz(rho_np, u_np, v_np, p_np, x_np, y_np, step=N, time_sim=time_sim)
            save_png(rho_np, x_np, y_np, step=N, time_sim=time_sim)

        if abs(time_sim - T_END) < 1.0e-12:
            break
        N += 1

    wp.synchronize()
    print(f"\nDone. steps={N}  wall={time.perf_counter()-t0:.2f}s")
    rho_np = rho_d.numpy(); u_np = u_d.numpy()
    v_np   = v_d.numpy();   p_np = p_d.numpy()
    save_npz(rho_np, u_np, v_np, p_np, x_np, y_np, step=N, time_sim=time_sim)
    save_png(rho_np, x_np, y_np, step=N, time_sim=time_sim)


if __name__ == "__main__":
    main()
