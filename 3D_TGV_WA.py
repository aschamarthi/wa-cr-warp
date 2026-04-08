"""
3D Compressible Euler — Taylor-Green Vortex
Warp GPU solver  |  Wave appropriate reconstruction

## References

1. Chamarthi, Hoffmann, Frankel — *A wave appropriate discontinuity sensor approach for compressible flows*, **Phys. Fluids** 35, 066107 (2023)
2. Hoffmann, Chamarthi, Frankel — *Centralized gradient-based reconstruction for wall modeled large eddy simulations of hypersonic boundary layer transition*, **J. Comput. Phys.** (2024)
3. Chamarthi — *Wave-appropriate multidimensional upwinding approach for compressible multiphase flows*, **J. Comput. Phys.** 538, 114157 (2025)
4. Chamarthi — *Physics appropriate interface capturing reconstruction approach for viscous compressible multicomponent flows*, **Comput. Fluids** 303, 106858 (2025)
5. Chamarthi — *Wave-appropriate reconstruction of compressible flows: physics-constrained acoustic dissipation and rank-1 entropy wave correction*, preprint (2026)


Algorithm is from ref. 5. I made few modifications here and there out of necessity and laziness  compared to the paper.
## Author

**Amareshwara Sainadh Chamarthi** sainath@caltech.edu

GPU optimization vs version 1.0:
  - SoA memory layout: cons[var, ix, iy, iz]  (coalesced reads)
  - Atomic-min dt reduction: only 8 bytes CPU<->GPU per step
  - Inlined HLLC: no custom vector return type (Vec4D array in 2D was unhelpful )

Domain  : [0,2pi]^3   t_end=10   \gamma=5/3   periodic BCs
IC      : Taylor-Green Vortex

Usage
-----
  python tgv3d_warp.py            # GPU, 64^3
  python tgv3d_warp.py --n 128   # 128^3
  python tgv3d_warp.py --cpu
"""

import argparse, time, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
# plt.rc('font', family='arial')
plt.rcParams.update({'font.size': 12})
import numpy as np
import warp as wp

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--n",   type=int, default=64)
args, _ = parser.parse_known_args()
DEVICE = "cpu" if args.cpu else "cuda"
wp.init()

# ── Problem parameters ────────────────────────────────────────────────────────
NX = NY = NZ = args.n
GHOSTP = 5
GX = NX + 2*GHOSTP + 1
GY = NY + 2*GHOSTP + 1
GZ = NZ + 2*GHOSTP + 1
G  = GHOSTP

NTMAX     = 200_000
FILE_SAVE = 200  ## Would be too much on 512^3, use appropriately
CFL_PY    = 0.4
T_END     = 10.0

PI_PY = math.pi
XMIN_PY, XMAX_PY = 0.0, 2.0*PI_PY
YMIN_PY, YMAX_PY = 0.0, 2.0*PI_PY
ZMIN_PY, ZMAX_PY = 0.0, 2.0*PI_PY

# ── Warp float64 constants ────────────────────────────────────────────────────
GAMMA  = wp.constant(wp.float64(5.0/3.0))
GM1    = wp.constant(wp.float64(2.0/3.0))
TINY   = wp.constant(wp.float64(1.0e-30))
F0     = wp.constant(wp.float64(0.0))
F1     = wp.constant(wp.float64(1.0))
F2     = wp.constant(wp.float64(2.0))
HALF   = wp.constant(wp.float64(0.5))
QUART  = wp.constant(wp.float64(0.25))
F5_6   = wp.constant(wp.float64(5.0/6.0))
F1_3   = wp.constant(wp.float64(1.0/3.0))
F0p6   = wp.constant(wp.float64(0.6))
NF1    = wp.constant(wp.float64(-1.0))
NF1_6  = wp.constant(wp.float64(-1.0/6.0))
NF3    = wp.constant(wp.float64(-3.0))
NF13   = wp.constant(wp.float64(-13.0))
F3     = wp.constant(wp.float64(3.0))
F4     = wp.constant(wp.float64(4.0))
F7_6   = wp.constant(wp.float64(7.0/6.0))
NF7_6  = wp.constant(wp.float64(-7.0/6.0))
F11_6  = wp.constant(wp.float64(11.0/6.0))
F13_12 = wp.constant(wp.float64(13.0/12.0))
F3_10  = wp.constant(wp.float64(3.0/10.0))
F6_10  = wp.constant(wp.float64(6.0/10.0))
F1_10  = wp.constant(wp.float64(1.0/10.0))
F1_60  = wp.constant(wp.float64(1.0/60.0))
F27    = wp.constant(wp.float64(27.0))
F47    = wp.constant(wp.float64(47.0))
WENO_EPS = wp.constant(wp.float64(1.0e-40))
DTBIG  = wp.constant(wp.float64(1.0e30))

# =============================================================================
#  S1  Reconstruction helpers
# =============================================================================

@wp.func
def weno5_ul(vm2:wp.float64,vm1:wp.float64,v0:wp.float64,
             v1:wp.float64,v2:wp.float64)->wp.float64:
    p0=F1_3*v0+F5_6*v1+NF1_6*v2
    p1=NF1_6*vm1+F5_6*v0+F1_3*v1
    p2=F1_3*vm2+NF7_6*vm1+F11_6*v0
    b0=F13_12*(v0-F2*v1+v2)**wp.float64(2.0)+QUART*(F3*v0-F4*v1+v2)**wp.float64(2.0)
    b1=F13_12*(vm1-F2*v0+v1)**wp.float64(2.0)+QUART*(vm1-v1)**wp.float64(2.0)
    b2=F13_12*(vm2-F2*vm1+v0)**wp.float64(2.0)+QUART*(vm2-F4*vm1+F3*v0)**wp.float64(2.0)
    tau=wp.abs(b0-b2)
    a0=F3_10*(F1+tau/(WENO_EPS+b0)); a1=F6_10*(F1+tau/(WENO_EPS+b1)); a2=F1_10*(F1+tau/(WENO_EPS+b2))
    s=a0+a1+a2
    return (a0*p0+a1*p1+a2*p2)/s

@wp.func
def weno5_ur(vm1:wp.float64,v0:wp.float64,v1:wp.float64,
             v2:wp.float64,v3:wp.float64)->wp.float64:
    p0=F11_6*v1+NF7_6*v2+F1_3*v3
    p1=F1_3*v0+F5_6*v1+NF1_6*v2
    p2=NF1_6*vm1+F5_6*v0+F1_3*v1
    b0=F13_12*(v1-F2*v2+v3)**wp.float64(2.0)+QUART*(F3*v1-F4*v2+v3)**wp.float64(2.0)
    b1=F13_12*(v0-F2*v1+v2)**wp.float64(2.0)+QUART*(v0-v2)**wp.float64(2.0)
    b2=F13_12*(vm1-F2*v0+v1)**wp.float64(2.0)+QUART*(vm1-F4*v0+F3*v1)**wp.float64(2.0)
    tau=wp.abs(b0-b2)
    a0=F1_10*(F1+tau/(WENO_EPS+b0)); a1=F6_10*(F1+tau/(WENO_EPS+b1)); a2=F3_10*(F1+tau/(WENO_EPS+b2))
    s=a0+a1+a2
    return (a0*p0+a1*p1+a2*p2)/s

# ── MP5 reconstruction (Suresh & Huynh 1997) ─────────────────────────────────
# Constants matching Fortran: B2=4/3, alpha=4, EPSM=1e-40
MP5_B2   = wp.constant(wp.float64(4.0/3.0))
MP5_ALPH = wp.constant(wp.float64(4.0))
MP5_EPS  = wp.constant(wp.float64(1.0e-40))

@wp.func
def _minmod2(x:wp.float64, y:wp.float64)->wp.float64:
    return HALF*(wp.sign(x)+wp.sign(y))*wp.min(wp.abs(x),wp.abs(y))

@wp.func
def _minmod4(w:wp.float64,x:wp.float64,y:wp.float64,z:wp.float64)->wp.float64:
    sw=wp.sign(w); sx=wp.sign(x); sy=wp.sign(y); sz=wp.sign(z)
    return wp.float64(0.125)*(sw+sx)*wp.abs((sw+sy)*(sw+sz))*wp.min(wp.abs(w),wp.min(wp.abs(x),wp.min(wp.abs(y),wp.abs(z))))

@wp.func
def mp5_ul(vm2:wp.float64,vm1:wp.float64,v0:wp.float64,
           v1:wp.float64,v2:wp.float64)->wp.float64:
    # centL: left-biased stencil [vm2,vm1,v0,v1,v2] -> face i+1/2
    # charstencil(-2:2) = [vm2,vm1,v0,v1,v2]
    VOR = F1_60*(F2*vm2+NF13*vm1+F47*v0+F27*v1+NF3*v2)
    VMP = v0 + _minmod2(v1-v0, MP5_ALPH*(v0-vm1))
    if (VOR-v0)*(VOR-VMP) < MP5_EPS:
        return VOR
    DJM1= vm2 - F2*vm1 + v0
    DJ  = vm1 - F2*v0  + v1
    DJP1= v0  - F2*v1  + v2
    DM4JPH = _minmod4(F4*DJ-DJP1, F4*DJP1-DJ, DJ, DJP1)
    DM4JMH = _minmod4(F4*DJ-DJM1, F4*DJM1-DJ, DJ, DJM1)
    VUL  = v0 + MP5_ALPH*(v0-vm1)
    VAV  = HALF*(v0+v1)
    VMD  = VAV - HALF*DM4JPH
    VLC  = v0 + HALF*(v0-vm1) + MP5_B2*DM4JMH
    VMIN = wp.max(wp.min(wp.min(v0,v1),VMD), wp.min(wp.min(v0,VUL),VLC))
    VMAX = wp.min(wp.max(wp.max(v0,v1),VMD), wp.max(wp.max(v0,VUL),VLC))
    return VOR + _minmod2(VMIN-VOR, VMAX-VOR)

@wp.func
def mp5_ur(vm1:wp.float64,v0:wp.float64,v1:wp.float64,
           v2:wp.float64,v3:wp.float64)->wp.float64:
    # centR: right-biased stencil [vm1,v0,v1,v2,v3] -> face i+1/2
    # charstencil(-2:2) = [vm1,v0,v1,v2,v3]  (Fortran v(-1:3))
    VOR = F1_60*(NF3*vm1+F27*v0+F47*v1+NF13*v2+F2*v3)
    VMP = v1 + _minmod2(v0-v1, MP5_ALPH*(v1-v2))
    if (VOR-v1)*(VOR-VMP) < MP5_EPS:
        return VOR
    DJM1= vm1 - F2*v0  + v1
    DJ  = v0  - F2*v1  + v2
    DJP1= v1  - F2*v2  + v3
    DM4JPH = _minmod4(F4*DJ-DJP1, F4*DJP1-DJ, DJ, DJP1)
    DM4JMH = _minmod4(F4*DJ-DJM1, F4*DJM1-DJ, DJ, DJM1)
    VUL  = v1 + MP5_ALPH*(v1-v2)
    VAV  = HALF*(v1+v0)
    VMD  = VAV - HALF*DM4JMH
    VLC  = v1 + HALF*(v1-v2) + MP5_B2*DM4JPH
    VMIN = wp.max(wp.min(wp.min(v1,v0),VMD), wp.min(wp.min(v1,VUL),VLC))
    VMAX = wp.min(wp.max(wp.max(v1,v0),VMD), wp.max(wp.max(v1,VUL),VLC))
    return VOR + _minmod2(VMIN-VOR, VMAX-VOR)

@wp.func
def central6_ul(vm2:wp.float64,vm1:wp.float64,v0:wp.float64,
                v1:wp.float64,v2:wp.float64,v3:wp.float64,sai:wp.float64)->wp.float64:
    lb=F1_60*(F2*vm2+NF13*vm1+F47*v0+F27*v1+NF3*v2)
    rb=F1_60*(NF3*vm1+F27*v0+F47*v1+NF13*v2+F2*v3)
    return sai*lb+(F1-sai)*rb

@wp.func
def central6_ur(vm2:wp.float64,vm1:wp.float64,v0:wp.float64,
                v1:wp.float64,v2:wp.float64,v3:wp.float64,sai:wp.float64)->wp.float64:
    lb=F1_60*(F2*vm2+NF13*vm1+F47*v0+F27*v1+NF3*v2)
    rb=F1_60*(NF3*vm1+F27*v0+F47*v1+NF13*v2+F2*v3)
    return (F1-sai)*lb+sai*rb

# ── 3D left eigenvectors L (5x5) ──────────────────────────────────────────────

@wp.func
def Lq0(q1:wp.float64,q2:wp.float64,q3:wp.float64,q4:wp.float64,q5:wp.float64,
        u:wp.float64,v:wp.float64,w:wp.float64,c:wp.float64,c2:wp.float64,
        nx:wp.float64,ny:wp.float64,nz:wp.float64,qn:wp.float64,q2v:wp.float64)->wp.float64:
    kap=GAMMA-F1
    return (HALF*(kap*HALF*q2v/c2+qn/c)*q1-HALF*(kap*u/c2+nx/c)*q2
            -HALF*(kap*v/c2+ny/c)*q3-HALF*(kap*w/c2+nz/c)*q4+kap/(F2*c2)*q5)

@wp.func
def Lq1(q1:wp.float64,q2:wp.float64,q3:wp.float64,q4:wp.float64,q5:wp.float64,
        lx:wp.float64,ly:wp.float64,lz:wp.float64,ql:wp.float64)->wp.float64:
    return -ql*q1+lx*q2+ly*q3+lz*q4

@wp.func
def Lq2(q1:wp.float64,q2:wp.float64,q3:wp.float64,q4:wp.float64,q5:wp.float64,
        mx:wp.float64,my:wp.float64,mz:wp.float64,qm:wp.float64)->wp.float64:
    return -qm*q1+mx*q2+my*q3+mz*q4

@wp.func
def Lq3(q1:wp.float64,q2:wp.float64,q3:wp.float64,q4:wp.float64,q5:wp.float64,
        u:wp.float64,v:wp.float64,w:wp.float64,c2:wp.float64,q2v:wp.float64)->wp.float64:
    kap=GAMMA-F1
    return ((F1-kap*q2v/(F2*c2))*q1+kap*u/c2*q2+kap*v/c2*q3+kap*w/c2*q4-kap/c2*q5)

@wp.func
def Lq4(q1:wp.float64,q2:wp.float64,q3:wp.float64,q4:wp.float64,q5:wp.float64,
        u:wp.float64,v:wp.float64,w:wp.float64,c:wp.float64,c2:wp.float64,
        nx:wp.float64,ny:wp.float64,nz:wp.float64,qn:wp.float64,q2v:wp.float64)->wp.float64:
    kap=GAMMA-F1
    return (HALF*(kap*HALF*q2v/c2-qn/c)*q1-HALF*(kap*u/c2-nx/c)*q2
            -HALF*(kap*v/c2-ny/c)*q3-HALF*(kap*w/c2-nz/c)*q4+kap/(F2*c2)*q5)

# ── 3D right eigenvectors R ────────────────────────────────────────────────────

@wp.func
def Rv0(v0:wp.float64,v1:wp.float64,v2:wp.float64,v3:wp.float64,v4:wp.float64)->wp.float64:
    return v0+v3+v4

@wp.func
def Rv1(v0:wp.float64,v1:wp.float64,v2:wp.float64,v3:wp.float64,v4:wp.float64,
        u:wp.float64,c:wp.float64,nx:wp.float64,lx:wp.float64,mx:wp.float64)->wp.float64:
    return v0*(u-c*nx)+v1*lx+v2*mx+v3*u+v4*(u+c*nx)

@wp.func
def Rv2(v0:wp.float64,v1:wp.float64,v2:wp.float64,v3:wp.float64,v4:wp.float64,
        vv:wp.float64,c:wp.float64,ny:wp.float64,ly:wp.float64,my:wp.float64)->wp.float64:
    return v0*(vv-c*ny)+v1*ly+v2*my+v3*vv+v4*(vv+c*ny)

@wp.func
def Rv3(v0:wp.float64,v1:wp.float64,v2:wp.float64,v3:wp.float64,v4:wp.float64,
        w:wp.float64,c:wp.float64,nz:wp.float64,lz:wp.float64,mz:wp.float64)->wp.float64:
    return v0*(w-c*nz)+v1*lz+v2*mz+v3*w+v4*(w+c*nz)

@wp.func
def Rv4(v0:wp.float64,v1:wp.float64,v2:wp.float64,v3:wp.float64,v4:wp.float64,
        enth:wp.float64,qn:wp.float64,c:wp.float64,
        ql:wp.float64,qm:wp.float64,q2v:wp.float64)->wp.float64:
    return v0*(enth-qn*c)+v1*ql+v2*qm+v3*(HALF*q2v)+v4*(enth+qn*c)

# ── Positivity check ──────────────────────────────────────────────────────────

@wp.func
def is_physical(c0:wp.float64,c1:wp.float64,c2:wp.float64,
                c3:wp.float64,c4:wp.float64)->int:
    if c0<=F0: return 0
    u_=c1/c0; v_=c2/c0; w_=c3/c0
    if c4/c0-HALF*(u_*u_+v_*v_+w_*w_)<=F0: return 0
    return 1

# ── Pressure shock sensors ─────────────────────────────────────────────────────
# Note: pres is (GX,GY,GZ), ducros is (GX,GY,GZ) — primitives keep AoS

@wp.func
def pressure_sensor_x(pres:wp.array(dtype=wp.float64,ndim=3),
                       ducros:wp.array(dtype=wp.float64,ndim=3),
                       kx:int,iy:int,iz:int)->wp.float64:
    F16=wp.float64(16.0); F30=wp.float64(30.0)
    aa=wp.abs(-pres[kx-2,iy,iz]+F16*pres[kx-1,iy,iz]-F30*pres[kx,iy,iz]+F16*pres[kx+1,iy,iz]-pres[kx+2,iy,iz])
    bb=wp.abs( pres[kx-2,iy,iz]+F16*pres[kx-1,iy,iz]+F30*pres[kx,iy,iz]+F16*pres[kx+1,iy,iz]+pres[kx+2,iy,iz])
    return aa/(bb+TINY)*ducros[kx,iy,iz]

@wp.func
def pressure_sensor_y(pres:wp.array(dtype=wp.float64,ndim=3),
                       ducros:wp.array(dtype=wp.float64,ndim=3),
                       ix:int,ky:int,iz:int)->wp.float64:
    F16=wp.float64(16.0); F30=wp.float64(30.0)
    aa=wp.abs(-pres[ix,ky-2,iz]+F16*pres[ix,ky-1,iz]-F30*pres[ix,ky,iz]+F16*pres[ix,ky+1,iz]-pres[ix,ky+2,iz])
    bb=wp.abs( pres[ix,ky-2,iz]+F16*pres[ix,ky-1,iz]+F30*pres[ix,ky,iz]+F16*pres[ix,ky+1,iz]+pres[ix,ky+2,iz])
    return aa/(bb+TINY)*ducros[ix,ky,iz]

@wp.func
def pressure_sensor_z(pres:wp.array(dtype=wp.float64,ndim=3),
                       ducros:wp.array(dtype=wp.float64,ndim=3),
                       ix:int,iy:int,kz:int)->wp.float64:
    F16=wp.float64(16.0); F30=wp.float64(30.0)
    aa=wp.abs(-pres[ix,iy,kz-2]+F16*pres[ix,iy,kz-1]-F30*pres[ix,iy,kz]+F16*pres[ix,iy,kz+1]-pres[ix,iy,kz+2])
    bb=wp.abs( pres[ix,iy,kz-2]+F16*pres[ix,iy,kz-1]+F30*pres[ix,iy,kz]+F16*pres[ix,iy,kz+1]+pres[ix,iy,kz+2])
    return aa/(bb+TINY)*ducros[ix,iy,kz]

# ── Characteristic transform — SoA: cons[var, ix, iy, iz] ─────────────────────

@wp.func
def char_transform(cons:wp.array(dtype=wp.float64,ndim=4),
                   kx:int,ky:int,kz:int,row:int,
                   u_r:wp.float64,v_r:wp.float64,w_r:wp.float64,
                   c_r:wp.float64,c2:wp.float64,
                   nx:wp.float64,ny:wp.float64,nz:wp.float64,
                   lx:wp.float64,ly:wp.float64,lz:wp.float64,
                   mx:wp.float64,my:wp.float64,mz:wp.float64,
                   qn:wp.float64,ql:wp.float64,qm:wp.float64,q2v:wp.float64)->wp.float64:
    # SoA: first index is variable
    q1=cons[0,kx,ky,kz]; q2=cons[1,kx,ky,kz]; q3=cons[2,kx,ky,kz]
    q4=cons[3,kx,ky,kz]; q5=cons[4,kx,ky,kz]
    if row==0:   return Lq0(q1,q2,q3,q4,q5,u_r,v_r,w_r,c_r,c2,nx,ny,nz,qn,q2v)
    elif row==1: return Lq1(q1,q2,q3,q4,q5,lx,ly,lz,ql)
    elif row==2: return Lq2(q1,q2,q3,q4,q5,mx,my,mz,qm)
    elif row==3: return Lq3(q1,q2,q3,q4,q5,u_r,v_r,w_r,c2,q2v)
    else:        return Lq4(q1,q2,q3,q4,q5,u_r,v_r,w_r,c_r,c2,nx,ny,nz,qn,q2v)


# =============================================================================
#  S2  Initial conditions  (primitives: AoS (GX,GY,GZ) — fine for init)
# =============================================================================

@wp.kernel
def init_cond_kernel(
    x:wp.array(dtype=wp.float64,ndim=1), y:wp.array(dtype=wp.float64,ndim=1),
    z:wp.array(dtype=wp.float64,ndim=1),
    rho:wp.array(dtype=wp.float64,ndim=3),  u_vel:wp.array(dtype=wp.float64,ndim=3),
    v_vel:wp.array(dtype=wp.float64,ndim=3),w_vel:wp.array(dtype=wp.float64,ndim=3),
    pres:wp.array(dtype=wp.float64,ndim=3), snd:wp.array(dtype=wp.float64,ndim=3),
):
    ix,iy,iz = wp.tid()
    xi=x[ix]; yi=y[iy]; zi=z[iz]
    r=F1
    u= wp.sin(xi)*wp.cos(yi)*wp.cos(zi)
    v=-wp.cos(xi)*wp.sin(yi)*wp.cos(zi)
    w=F0
    t1=wp.cos(F2*xi)+wp.cos(F2*yi)
    t2=wp.cos(F2*zi)+F2
    p=wp.float64(100.0)+(t1*t2-F2)/wp.float64(16.0)
    rho[ix,iy,iz]=r; u_vel[ix,iy,iz]=u; v_vel[ix,iy,iz]=v
    w_vel[ix,iy,iz]=w; pres[ix,iy,iz]=p
    snd[ix,iy,iz]=wp.sqrt(GAMMA*p/r)

# =============================================================================
#  S3  Periodic BCs
# =============================================================================

@wp.kernel
def bc_periodic_x_kernel(
    rho:wp.array(dtype=wp.float64,ndim=3),  u_vel:wp.array(dtype=wp.float64,ndim=3),
    v_vel:wp.array(dtype=wp.float64,ndim=3),w_vel:wp.array(dtype=wp.float64,ndim=3),
    pres:wp.array(dtype=wp.float64,ndim=3), nx:int,ny:int,nz:int,gp:int
):
    layer,iy,iz = wp.tid()
    if layer>=gp: return
    i=layer+1
    gl=gp+1-i; sr=gp+nx+1-i; gr=gp+nx+i; sl=gp+i
    rho[gl,iy,iz]=rho[sr,iy,iz];     rho[gr,iy,iz]=rho[sl,iy,iz]
    u_vel[gl,iy,iz]=u_vel[sr,iy,iz]; u_vel[gr,iy,iz]=u_vel[sl,iy,iz]
    v_vel[gl,iy,iz]=v_vel[sr,iy,iz]; v_vel[gr,iy,iz]=v_vel[sl,iy,iz]
    w_vel[gl,iy,iz]=w_vel[sr,iy,iz]; w_vel[gr,iy,iz]=w_vel[sl,iy,iz]
    pres[gl,iy,iz]=pres[sr,iy,iz];   pres[gr,iy,iz]=pres[sl,iy,iz]

@wp.kernel
def bc_periodic_y_kernel(
    rho:wp.array(dtype=wp.float64,ndim=3),  u_vel:wp.array(dtype=wp.float64,ndim=3),
    v_vel:wp.array(dtype=wp.float64,ndim=3),w_vel:wp.array(dtype=wp.float64,ndim=3),
    pres:wp.array(dtype=wp.float64,ndim=3), nx:int,ny:int,nz:int,gp:int
):
    ix,layer,iz = wp.tid()
    if layer>=gp: return
    j=layer+1
    gb=gp+1-j; st=gp+ny+1-j; gt=gp+ny+j; sb=gp+j
    rho[ix,gb,iz]=rho[ix,st,iz];     rho[ix,gt,iz]=rho[ix,sb,iz]
    u_vel[ix,gb,iz]=u_vel[ix,st,iz]; u_vel[ix,gt,iz]=u_vel[ix,sb,iz]
    v_vel[ix,gb,iz]=v_vel[ix,st,iz]; v_vel[ix,gt,iz]=v_vel[ix,sb,iz]
    w_vel[ix,gb,iz]=w_vel[ix,st,iz]; w_vel[ix,gt,iz]=w_vel[ix,sb,iz]
    pres[ix,gb,iz]=pres[ix,st,iz];   pres[ix,gt,iz]=pres[ix,sb,iz]

@wp.kernel
def bc_periodic_z_kernel(
    rho:wp.array(dtype=wp.float64,ndim=3),  u_vel:wp.array(dtype=wp.float64,ndim=3),
    v_vel:wp.array(dtype=wp.float64,ndim=3),w_vel:wp.array(dtype=wp.float64,ndim=3),
    pres:wp.array(dtype=wp.float64,ndim=3), nx:int,ny:int,nz:int,gp:int
):
    ix,iy,layer = wp.tid()
    if layer>=gp: return
    k=layer+1
    gf=gp+1-k; sb=gp+nz+1-k; gk=gp+nz+k; sf=gp+k
    rho[ix,iy,gf]=rho[ix,iy,sb];     rho[ix,iy,gk]=rho[ix,iy,sf]
    u_vel[ix,iy,gf]=u_vel[ix,iy,sb]; u_vel[ix,iy,gk]=u_vel[ix,iy,sf]
    v_vel[ix,iy,gf]=v_vel[ix,iy,sb]; v_vel[ix,iy,gk]=v_vel[ix,iy,sf]
    w_vel[ix,iy,gf]=w_vel[ix,iy,sb]; w_vel[ix,iy,gk]=w_vel[ix,iy,sf]
    pres[ix,iy,gf]=pres[ix,iy,sb];   pres[ix,iy,gk]=pres[ix,iy,sf]

# =============================================================================
#  S4  Primitive <-> Conservative   SoA: cons[var, ix, iy, iz]
# =============================================================================

@wp.kernel
def prim_to_cons_kernel(
    rho:wp.array(dtype=wp.float64,ndim=3),  u_vel:wp.array(dtype=wp.float64,ndim=3),
    v_vel:wp.array(dtype=wp.float64,ndim=3),w_vel:wp.array(dtype=wp.float64,ndim=3),
    pres:wp.array(dtype=wp.float64,ndim=3), cons:wp.array(dtype=wp.float64,ndim=4),
):
    ix,iy,iz = wp.tid()
    r=rho[ix,iy,iz]; u=u_vel[ix,iy,iz]; v=v_vel[ix,iy,iz]
    w=w_vel[ix,iy,iz]; p=pres[ix,iy,iz]
    cons[0,ix,iy,iz]=r
    cons[1,ix,iy,iz]=r*u
    cons[2,ix,iy,iz]=r*v
    cons[3,ix,iy,iz]=r*w
    cons[4,ix,iy,iz]=p/GM1+HALF*r*(u*u+v*v+w*w)

@wp.kernel
def cons_to_prim_kernel(
    cons:wp.array(dtype=wp.float64,ndim=4),
    rho:wp.array(dtype=wp.float64,ndim=3),  u_vel:wp.array(dtype=wp.float64,ndim=3),
    v_vel:wp.array(dtype=wp.float64,ndim=3),w_vel:wp.array(dtype=wp.float64,ndim=3),
    pres:wp.array(dtype=wp.float64,ndim=3), nx:int,ny:int,nz:int,gp:int
):
    i,j,k = wp.tid()
    if i<1 or i>nx or j<1 or j>ny or k<1 or k>nz: return
    ix=i+gp; iy=j+gp; iz=k+gp
    r=cons[0,ix,iy,iz]; ru=cons[1,ix,iy,iz]; rv=cons[2,ix,iy,iz]
    rw=cons[3,ix,iy,iz]; E=cons[4,ix,iy,iz]
    u=ru/r; v=rv/r; w=rw/r
    p=GM1*(E-HALF*r*(u*u+v*v+w*w))
    rho[ix,iy,iz]=r; u_vel[ix,iy,iz]=u; v_vel[ix,iy,iz]=v
    w_vel[ix,iy,iz]=w; pres[ix,iy,iz]=p

# =============================================================================
#  S5  Velocity derivatives + 3D Ducros sensor
# =============================================================================

@wp.kernel
def velocity_deriv_kernel(
    u_vel:wp.array(dtype=wp.float64,ndim=3), v_vel:wp.array(dtype=wp.float64,ndim=3),
    w_vel:wp.array(dtype=wp.float64,ndim=3),
    der_ux:wp.array(dtype=wp.float64,ndim=3),der_uy:wp.array(dtype=wp.float64,ndim=3),
    der_uz:wp.array(dtype=wp.float64,ndim=3),der_vx:wp.array(dtype=wp.float64,ndim=3),
    der_vy:wp.array(dtype=wp.float64,ndim=3),der_vz:wp.array(dtype=wp.float64,ndim=3),
    der_wx:wp.array(dtype=wp.float64,ndim=3),der_wy:wp.array(dtype=wp.float64,ndim=3),
    der_wz:wp.array(dtype=wp.float64,ndim=3), gx:int,gy:int,gz:int
):
    ti,tj,tk = wp.tid()
    ix=ti+1; iy=tj+1; iz=tk+1
    if ix>=gx-1 or iy>=gy-1 or iz>=gz-1: return
    der_ux[ix,iy,iz]=HALF*(-u_vel[ix-1,iy,iz]+u_vel[ix+1,iy,iz])
    der_uy[ix,iy,iz]=HALF*(-u_vel[ix,iy-1,iz]+u_vel[ix,iy+1,iz])
    der_uz[ix,iy,iz]=HALF*(-u_vel[ix,iy,iz-1]+u_vel[ix,iy,iz+1])
    der_vx[ix,iy,iz]=HALF*(-v_vel[ix-1,iy,iz]+v_vel[ix+1,iy,iz])
    der_vy[ix,iy,iz]=HALF*(-v_vel[ix,iy-1,iz]+v_vel[ix,iy+1,iz])
    der_vz[ix,iy,iz]=HALF*(-v_vel[ix,iy,iz-1]+v_vel[ix,iy,iz+1])
    der_wx[ix,iy,iz]=HALF*(-w_vel[ix-1,iy,iz]+w_vel[ix+1,iy,iz])
    der_wy[ix,iy,iz]=HALF*(-w_vel[ix,iy-1,iz]+w_vel[ix,iy+1,iz])
    der_wz[ix,iy,iz]=HALF*(-w_vel[ix,iy,iz-1]+w_vel[ix,iy,iz+1])

@wp.kernel
def ducros_kernel(
    der_ux:wp.array(dtype=wp.float64,ndim=3),der_uy:wp.array(dtype=wp.float64,ndim=3),
    der_uz:wp.array(dtype=wp.float64,ndim=3),der_vx:wp.array(dtype=wp.float64,ndim=3),
    der_vy:wp.array(dtype=wp.float64,ndim=3),der_vz:wp.array(dtype=wp.float64,ndim=3),
    der_wx:wp.array(dtype=wp.float64,ndim=3),der_wy:wp.array(dtype=wp.float64,ndim=3),
    der_wz:wp.array(dtype=wp.float64,ndim=3),
    ducros:wp.array(dtype=wp.float64,ndim=3), gx:int,gy:int,gz:int
):
    ti,tj,tk = wp.tid()
    ix=ti+1; iy=tj+1; iz=tk+1
    if ix>=gx-1 or iy>=gy-1 or iz>=gz-1: return
    div =der_ux[ix,iy,iz]+der_vy[ix,iy,iz]+der_wz[ix,iy,iz]
    cx  =der_wy[ix,iy,iz]-der_vz[ix,iy,iz]
    cy  =der_uz[ix,iy,iz]-der_wx[ix,iy,iz]
    cz  =der_vx[ix,iy,iz]-der_uy[ix,iy,iz]
    curl2=cx*cx+cy*cy+cz*cz
    ducros[ix,iy,iz]=div*div/(div*div+curl2+TINY)


# =============================================================================
#  S6  WCNS X   n=(1,0,0) l=(0,0,1) m=(0,-1,0)   SoA cons[v,ix,iy,iz]
# =============================================================================

@wp.kernel
def wcns_x_kernel(
    cons:wp.array(dtype=wp.float64,ndim=4),  pres:wp.array(dtype=wp.float64,ndim=3),
    ducros:wp.array(dtype=wp.float64,ndim=3),
    consl:wp.array(dtype=wp.float64,ndim=4), consr:wp.array(dtype=wp.float64,ndim=4),
    nx:int,ny:int,nz:int,gp:int
):
    face_ix,iy,iz = wp.tid()
    if face_ix<gp-1 or face_ix>nx+gp: return
    if iy<gp+1 or iy>ny+gp: return
    if iz<gp+1 or iz>nz+gp: return

    sig_max=wp.max(pressure_sensor_x(pres,ducros,face_ix-1,iy,iz),
             wp.max(pressure_sensor_x(pres,ducros,face_ix,  iy,iz),
                    pressure_sensor_x(pres,ducros,face_ix+1,iy,iz)))
    f1idx=face_ix-gp
    is_smooth=(sig_max<wp.float64(0.01)) and (f1idx>6) and (f1idx<nx-5)

    # Roe averages
    rL=cons[0,face_ix,iy,iz];   rR=cons[0,face_ix+1,iy,iz]
    ruL=cons[1,face_ix,iy,iz];  ruR=cons[1,face_ix+1,iy,iz]
    rvL=cons[2,face_ix,iy,iz];  rvR=cons[2,face_ix+1,iy,iz]
    rwL=cons[3,face_ix,iy,iz];  rwR=cons[3,face_ix+1,iy,iz]
    EL=cons[4,face_ix,iy,iz];   ER=cons[4,face_ix+1,iy,iz]
    sqL=wp.sqrt(rL); sqR=wp.sqrt(rR); dv=F1/(sqL+sqR)
    pL=GM1*(EL-HALF*(ruL*ruL+rvL*rvL+rwL*rwL)/rL)
    pR=GM1*(ER-HALF*(ruR*ruR+rvR*rvR+rwR*rwR)/rR)
    HL=(EL+pL)/rL; HR=(ER+pR)/rR
    u_r=(sqL*(ruL/rL)+sqR*(ruR/rR))*dv
    v_r=(sqL*(rvL/rL)+sqR*(rvR/rR))*dv
    w_r=(sqL*(rwL/rL)+sqR*(rwR/rR))*dv
    H_r=(sqL*HL+sqR*HR)*dv
    q2v=u_r*u_r+v_r*v_r+w_r*w_r
    c2=wp.max((GAMMA-F1)*(H_r-HALF*q2v),TINY); c_r=wp.sqrt(c2)
    # X eigenvector data
    NXV=F1; NYV=F0; NZV=F0; LX=F0; LY=F0; LZ=F1; MX=F0; MY=NF1; MZ=F0
    qn=u_r; ql=w_r; qm=-v_r

    if is_smooth:
        sai=F0p6
        ul0=central6_ul(cons[0,face_ix-2,iy,iz],cons[0,face_ix-1,iy,iz],cons[0,face_ix,iy,iz],cons[0,face_ix+1,iy,iz],cons[0,face_ix+2,iy,iz],cons[0,face_ix+3,iy,iz],HALF)
        ur0=central6_ur(cons[0,face_ix-2,iy,iz],cons[0,face_ix-1,iy,iz],cons[0,face_ix,iy,iz],cons[0,face_ix+1,iy,iz],cons[0,face_ix+2,iy,iz],cons[0,face_ix+3,iy,iz],HALF)
        ul1=central6_ul(cons[1,face_ix-2,iy,iz],cons[1,face_ix-1,iy,iz],cons[1,face_ix,iy,iz],cons[1,face_ix+1,iy,iz],cons[1,face_ix+2,iy,iz],cons[1,face_ix+3,iy,iz],sai)
        ur1=central6_ur(cons[1,face_ix-2,iy,iz],cons[1,face_ix-1,iy,iz],cons[1,face_ix,iy,iz],cons[1,face_ix+1,iy,iz],cons[1,face_ix+2,iy,iz],cons[1,face_ix+3,iy,iz],sai)
        ul2=central6_ul(cons[2,face_ix-2,iy,iz],cons[2,face_ix-1,iy,iz],cons[2,face_ix,iy,iz],cons[2,face_ix+1,iy,iz],cons[2,face_ix+2,iy,iz],cons[2,face_ix+3,iy,iz],HALF)
        ur2=central6_ur(cons[2,face_ix-2,iy,iz],cons[2,face_ix-1,iy,iz],cons[2,face_ix,iy,iz],cons[2,face_ix+1,iy,iz],cons[2,face_ix+2,iy,iz],cons[2,face_ix+3,iy,iz],HALF)
        ul3=central6_ul(cons[3,face_ix-2,iy,iz],cons[3,face_ix-1,iy,iz],cons[3,face_ix,iy,iz],cons[3,face_ix+1,iy,iz],cons[3,face_ix+2,iy,iz],cons[3,face_ix+3,iy,iz],HALF)
        ur3=central6_ur(cons[3,face_ix-2,iy,iz],cons[3,face_ix-1,iy,iz],cons[3,face_ix,iy,iz],cons[3,face_ix+1,iy,iz],cons[3,face_ix+2,iy,iz],cons[3,face_ix+3,iy,iz],HALF)
        ul4=central6_ul(cons[4,face_ix-2,iy,iz],cons[4,face_ix-1,iy,iz],cons[4,face_ix,iy,iz],cons[4,face_ix+1,iy,iz],cons[4,face_ix+2,iy,iz],cons[4,face_ix+3,iy,iz],HALF)
        ur4=central6_ur(cons[4,face_ix-2,iy,iz],cons[4,face_ix-1,iy,iz],cons[4,face_ix,iy,iz],cons[4,face_ix+1,iy,iz],cons[4,face_ix+2,iy,iz],cons[4,face_ix+3,iy,iz],HALF)
        e_m2=char_transform(cons,face_ix-2,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_m1=char_transform(cons,face_ix-1,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_0 =char_transform(cons,face_ix  ,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_1 =char_transform(cons,face_ix+1,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_2 =char_transform(cons,face_ix+2,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_3 =char_transform(cons,face_ix+3,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        centL=mp5_ul(e_m2,e_m1,e_0,e_1,e_2); centR=mp5_ur(e_m1,e_0,e_1,e_2,e_3)
        kap=GAMMA-F1
        le0=F1-kap*HALF*q2v/c2; le1=kap*u_r/c2; le2=kap*v_r/c2; le3=kap*w_r/c2; le4=NF1*kap/c2
        dL=centL-(le0*ul0+le1*ul1+le2*ul2+le3*ul3+le4*ul4)
        dR=centR-(le0*ur0+le1*ur1+le2*ur2+le3*ur3+le4*ur4)
        consl[0,face_ix,iy,iz]=ul0+dL;          consr[0,face_ix,iy,iz]=ur0+dR
        consl[1,face_ix,iy,iz]=ul1+dL*u_r;      consr[1,face_ix,iy,iz]=ur1+dR*u_r
        consl[2,face_ix,iy,iz]=ul2+dL*v_r;      consr[2,face_ix,iy,iz]=ur2+dR*v_r
        consl[3,face_ix,iy,iz]=ul3+dL*w_r;      consr[3,face_ix,iy,iz]=ur3+dR*w_r
        consl[4,face_ix,iy,iz]=ul4+dL*HALF*q2v; consr[4,face_ix,iy,iz]=ur4+dR*HALF*q2v
    else:
        c_m2=char_transform(cons,face_ix-2,iy,iz,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_m1=char_transform(cons,face_ix-1,iy,iz,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_0 =char_transform(cons,face_ix  ,iy,iz,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_1 =char_transform(cons,face_ix+1,iy,iz,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_2 =char_transform(cons,face_ix+2,iy,iz,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_3 =char_transform(cons,face_ix+3,iy,iz,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        ul0=weno5_ul(c_m2,c_m1,c_0,c_1,c_2); ur0=weno5_ur(c_m1,c_0,c_1,c_2,c_3)
        c_m2=char_transform(cons,face_ix-2,iy,iz,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_m1=char_transform(cons,face_ix-1,iy,iz,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_0 =char_transform(cons,face_ix  ,iy,iz,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_1 =char_transform(cons,face_ix+1,iy,iz,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_2 =char_transform(cons,face_ix+2,iy,iz,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_3 =char_transform(cons,face_ix+3,iy,iz,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        ul1=weno5_ul(c_m2,c_m1,c_0,c_1,c_2); ur1=weno5_ur(c_m1,c_0,c_1,c_2,c_3)
        c_m2=char_transform(cons,face_ix-2,iy,iz,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_m1=char_transform(cons,face_ix-1,iy,iz,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_0 =char_transform(cons,face_ix  ,iy,iz,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_1 =char_transform(cons,face_ix+1,iy,iz,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_2 =char_transform(cons,face_ix+2,iy,iz,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_3 =char_transform(cons,face_ix+3,iy,iz,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        ul2=weno5_ul(c_m2,c_m1,c_0,c_1,c_2); ur2=weno5_ur(c_m1,c_0,c_1,c_2,c_3)
        c_m2=char_transform(cons,face_ix-2,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_m1=char_transform(cons,face_ix-1,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_0 =char_transform(cons,face_ix  ,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_1 =char_transform(cons,face_ix+1,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_2 =char_transform(cons,face_ix+2,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_3 =char_transform(cons,face_ix+3,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        ul3=weno5_ul(c_m2,c_m1,c_0,c_1,c_2); ur3=weno5_ur(c_m1,c_0,c_1,c_2,c_3)
        c_m2=char_transform(cons,face_ix-2,iy,iz,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_m1=char_transform(cons,face_ix-1,iy,iz,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_0 =char_transform(cons,face_ix  ,iy,iz,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_1 =char_transform(cons,face_ix+1,iy,iz,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_2 =char_transform(cons,face_ix+2,iy,iz,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_3 =char_transform(cons,face_ix+3,iy,iz,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        ul4=weno5_ul(c_m2,c_m1,c_0,c_1,c_2); ur4=weno5_ur(c_m1,c_0,c_1,c_2,c_3)
        consl[0,face_ix,iy,iz]=Rv0(ul0,ul1,ul2,ul3,ul4)
        consl[1,face_ix,iy,iz]=Rv1(ul0,ul1,ul2,ul3,ul4,u_r,c_r,NXV,LX,MX)
        consl[2,face_ix,iy,iz]=Rv2(ul0,ul1,ul2,ul3,ul4,v_r,c_r,NYV,LY,MY)
        consl[3,face_ix,iy,iz]=Rv3(ul0,ul1,ul2,ul3,ul4,w_r,c_r,NZV,LZ,MZ)
        consl[4,face_ix,iy,iz]=Rv4(ul0,ul1,ul2,ul3,ul4,H_r,qn,c_r,ql,qm,q2v)
        consr[0,face_ix,iy,iz]=Rv0(ur0,ur1,ur2,ur3,ur4)
        consr[1,face_ix,iy,iz]=Rv1(ur0,ur1,ur2,ur3,ur4,u_r,c_r,NXV,LX,MX)
        consr[2,face_ix,iy,iz]=Rv2(ur0,ur1,ur2,ur3,ur4,v_r,c_r,NYV,LY,MY)
        consr[3,face_ix,iy,iz]=Rv3(ur0,ur1,ur2,ur3,ur4,w_r,c_r,NZV,LZ,MZ)
        consr[4,face_ix,iy,iz]=Rv4(ur0,ur1,ur2,ur3,ur4,H_r,qn,c_r,ql,qm,q2v)


# =============================================================================
#  S7  WCNS Y   n=(0,1,0) l=(0,0,1) m=(1,0,0)
# =============================================================================

@wp.kernel
def wcns_y_kernel(
    cons:wp.array(dtype=wp.float64,ndim=4),  pres:wp.array(dtype=wp.float64,ndim=3),
    ducros:wp.array(dtype=wp.float64,ndim=3),
    consl:wp.array(dtype=wp.float64,ndim=4), consr:wp.array(dtype=wp.float64,ndim=4),
    nx:int,ny:int,nz:int,gp:int
):
    ix,face_iy,iz = wp.tid()
    if ix<gp+1 or ix>nx+gp: return
    if face_iy<gp-1 or face_iy>ny+gp: return
    if iz<gp+1 or iz>nz+gp: return

    sig_max=wp.max(pressure_sensor_y(pres,ducros,ix,face_iy-1,iz),
             wp.max(pressure_sensor_y(pres,ducros,ix,face_iy  ,iz),
                    pressure_sensor_y(pres,ducros,ix,face_iy+1,iz)))
    f1idx=face_iy-gp
    is_smooth=(sig_max<wp.float64(0.01)) and (f1idx>6) and (f1idx<ny-5)

    rL=cons[0,ix,face_iy,iz];   rR=cons[0,ix,face_iy+1,iz]
    ruL=cons[1,ix,face_iy,iz];  ruR=cons[1,ix,face_iy+1,iz]
    rvL=cons[2,ix,face_iy,iz];  rvR=cons[2,ix,face_iy+1,iz]
    rwL=cons[3,ix,face_iy,iz];  rwR=cons[3,ix,face_iy+1,iz]
    EL=cons[4,ix,face_iy,iz];   ER=cons[4,ix,face_iy+1,iz]
    sqL=wp.sqrt(rL); sqR=wp.sqrt(rR); dv=F1/(sqL+sqR)
    pL=GM1*(EL-HALF*(ruL*ruL+rvL*rvL+rwL*rwL)/rL)
    pR=GM1*(ER-HALF*(ruR*ruR+rvR*rvR+rwR*rwR)/rR)
    HL=(EL+pL)/rL; HR=(ER+pR)/rR
    u_r=(sqL*(ruL/rL)+sqR*(ruR/rR))*dv
    v_r=(sqL*(rvL/rL)+sqR*(rvR/rR))*dv
    w_r=(sqL*(rwL/rL)+sqR*(rwR/rR))*dv
    H_r=(sqL*HL+sqR*HR)*dv
    q2v=u_r*u_r+v_r*v_r+w_r*w_r
    c2=wp.max((GAMMA-F1)*(H_r-HALF*q2v),TINY); c_r=wp.sqrt(c2)
    NXV=F0; NYV=F1; NZV=F0; LX=F0; LY=F0; LZ=F1; MX=F1; MY=F0; MZ=F0
    qn=v_r; ql=w_r; qm=u_r
      # Its ok to consider "HALF" for this particulr test case (or for that matter anything with periodic Bcs) but should use 0.60 or something near the boundaries.
    if is_smooth:
        sai=F0p6
        ul0=central6_ul(cons[0,ix,face_iy-2,iz],cons[0,ix,face_iy-1,iz],cons[0,ix,face_iy,iz],cons[0,ix,face_iy+1,iz],cons[0,ix,face_iy+2,iz],cons[0,ix,face_iy+3,iz],HALF)
        ur0=central6_ur(cons[0,ix,face_iy-2,iz],cons[0,ix,face_iy-1,iz],cons[0,ix,face_iy,iz],cons[0,ix,face_iy+1,iz],cons[0,ix,face_iy+2,iz],cons[0,ix,face_iy+3,iz],HALF)
        ul1=central6_ul(cons[1,ix,face_iy-2,iz],cons[1,ix,face_iy-1,iz],cons[1,ix,face_iy,iz],cons[1,ix,face_iy+1,iz],cons[1,ix,face_iy+2,iz],cons[1,ix,face_iy+3,iz],HALF)
        ur1=central6_ur(cons[1,ix,face_iy-2,iz],cons[1,ix,face_iy-1,iz],cons[1,ix,face_iy,iz],cons[1,ix,face_iy+1,iz],cons[1,ix,face_iy+2,iz],cons[1,ix,face_iy+3,iz],HALF)
        ul2=central6_ul(cons[2,ix,face_iy-2,iz],cons[2,ix,face_iy-1,iz],cons[2,ix,face_iy,iz],cons[2,ix,face_iy+1,iz],cons[2,ix,face_iy+2,iz],cons[2,ix,face_iy+3,iz],sai)
        ur2=central6_ur(cons[2,ix,face_iy-2,iz],cons[2,ix,face_iy-1,iz],cons[2,ix,face_iy,iz],cons[2,ix,face_iy+1,iz],cons[2,ix,face_iy+2,iz],cons[2,ix,face_iy+3,iz],sai)
        ul3=central6_ul(cons[3,ix,face_iy-2,iz],cons[3,ix,face_iy-1,iz],cons[3,ix,face_iy,iz],cons[3,ix,face_iy+1,iz],cons[3,ix,face_iy+2,iz],cons[3,ix,face_iy+3,iz],HALF)
        ur3=central6_ur(cons[3,ix,face_iy-2,iz],cons[3,ix,face_iy-1,iz],cons[3,ix,face_iy,iz],cons[3,ix,face_iy+1,iz],cons[3,ix,face_iy+2,iz],cons[3,ix,face_iy+3,iz],HALF)
        ul4=central6_ul(cons[4,ix,face_iy-2,iz],cons[4,ix,face_iy-1,iz],cons[4,ix,face_iy,iz],cons[4,ix,face_iy+1,iz],cons[4,ix,face_iy+2,iz],cons[4,ix,face_iy+3,iz],HALF)
        ur4=central6_ur(cons[4,ix,face_iy-2,iz],cons[4,ix,face_iy-1,iz],cons[4,ix,face_iy,iz],cons[4,ix,face_iy+1,iz],cons[4,ix,face_iy+2,iz],cons[4,ix,face_iy+3,iz],HALF)
        e_m2=char_transform(cons,ix,face_iy-2,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_m1=char_transform(cons,ix,face_iy-1,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_0 =char_transform(cons,ix,face_iy  ,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_1 =char_transform(cons,ix,face_iy+1,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_2 =char_transform(cons,ix,face_iy+2,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_3 =char_transform(cons,ix,face_iy+3,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        centL=mp5_ul(e_m2,e_m1,e_0,e_1,e_2); centR=mp5_ur(e_m1,e_0,e_1,e_2,e_3)
        kap=GAMMA-F1
        le0=F1-kap*HALF*q2v/c2; le1=kap*u_r/c2; le2=kap*v_r/c2; le3=kap*w_r/c2; le4=NF1*kap/c2
        dL=centL-(le0*ul0+le1*ul1+le2*ul2+le3*ul3+le4*ul4)
        dR=centR-(le0*ur0+le1*ur1+le2*ur2+le3*ur3+le4*ur4)
        consl[0,ix,face_iy,iz]=ul0+dL;          consr[0,ix,face_iy,iz]=ur0+dR
        consl[1,ix,face_iy,iz]=ul1+dL*u_r;      consr[1,ix,face_iy,iz]=ur1+dR*u_r
        consl[2,ix,face_iy,iz]=ul2+dL*v_r;      consr[2,ix,face_iy,iz]=ur2+dR*v_r
        consl[3,ix,face_iy,iz]=ul3+dL*w_r;      consr[3,ix,face_iy,iz]=ur3+dR*w_r
        consl[4,ix,face_iy,iz]=ul4+dL*HALF*q2v; consr[4,ix,face_iy,iz]=ur4+dR*HALF*q2v
    else:
        c_m2=char_transform(cons,ix,face_iy-2,iz,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_m1=char_transform(cons,ix,face_iy-1,iz,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_0 =char_transform(cons,ix,face_iy  ,iz,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_1 =char_transform(cons,ix,face_iy+1,iz,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_2 =char_transform(cons,ix,face_iy+2,iz,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_3 =char_transform(cons,ix,face_iy+3,iz,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        ul0=weno5_ul(c_m2,c_m1,c_0,c_1,c_2); ur0=weno5_ur(c_m1,c_0,c_1,c_2,c_3)
        c_m2=char_transform(cons,ix,face_iy-2,iz,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_m1=char_transform(cons,ix,face_iy-1,iz,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_0 =char_transform(cons,ix,face_iy  ,iz,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_1 =char_transform(cons,ix,face_iy+1,iz,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_2 =char_transform(cons,ix,face_iy+2,iz,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_3 =char_transform(cons,ix,face_iy+3,iz,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        ul1=weno5_ul(c_m2,c_m1,c_0,c_1,c_2); ur1=weno5_ur(c_m1,c_0,c_1,c_2,c_3)
        c_m2=char_transform(cons,ix,face_iy-2,iz,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_m1=char_transform(cons,ix,face_iy-1,iz,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_0 =char_transform(cons,ix,face_iy  ,iz,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_1 =char_transform(cons,ix,face_iy+1,iz,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_2 =char_transform(cons,ix,face_iy+2,iz,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_3 =char_transform(cons,ix,face_iy+3,iz,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        ul2=weno5_ul(c_m2,c_m1,c_0,c_1,c_2); ur2=weno5_ur(c_m1,c_0,c_1,c_2,c_3)
        c_m2=char_transform(cons,ix,face_iy-2,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_m1=char_transform(cons,ix,face_iy-1,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_0 =char_transform(cons,ix,face_iy  ,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_1 =char_transform(cons,ix,face_iy+1,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_2 =char_transform(cons,ix,face_iy+2,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_3 =char_transform(cons,ix,face_iy+3,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        ul3=weno5_ul(c_m2,c_m1,c_0,c_1,c_2); ur3=weno5_ur(c_m1,c_0,c_1,c_2,c_3)
        c_m2=char_transform(cons,ix,face_iy-2,iz,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_m1=char_transform(cons,ix,face_iy-1,iz,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_0 =char_transform(cons,ix,face_iy  ,iz,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_1 =char_transform(cons,ix,face_iy+1,iz,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_2 =char_transform(cons,ix,face_iy+2,iz,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_3 =char_transform(cons,ix,face_iy+3,iz,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        ul4=weno5_ul(c_m2,c_m1,c_0,c_1,c_2); ur4=weno5_ur(c_m1,c_0,c_1,c_2,c_3)
        consl[0,ix,face_iy,iz]=Rv0(ul0,ul1,ul2,ul3,ul4)
        consl[1,ix,face_iy,iz]=Rv1(ul0,ul1,ul2,ul3,ul4,u_r,c_r,NXV,LX,MX)
        consl[2,ix,face_iy,iz]=Rv2(ul0,ul1,ul2,ul3,ul4,v_r,c_r,NYV,LY,MY)
        consl[3,ix,face_iy,iz]=Rv3(ul0,ul1,ul2,ul3,ul4,w_r,c_r,NZV,LZ,MZ)
        consl[4,ix,face_iy,iz]=Rv4(ul0,ul1,ul2,ul3,ul4,H_r,qn,c_r,ql,qm,q2v)
        consr[0,ix,face_iy,iz]=Rv0(ur0,ur1,ur2,ur3,ur4)
        consr[1,ix,face_iy,iz]=Rv1(ur0,ur1,ur2,ur3,ur4,u_r,c_r,NXV,LX,MX)
        consr[2,ix,face_iy,iz]=Rv2(ur0,ur1,ur2,ur3,ur4,v_r,c_r,NYV,LY,MY)
        consr[3,ix,face_iy,iz]=Rv3(ur0,ur1,ur2,ur3,ur4,w_r,c_r,NZV,LZ,MZ)
        consr[4,ix,face_iy,iz]=Rv4(ur0,ur1,ur2,ur3,ur4,H_r,qn,c_r,ql,qm,q2v)


# =============================================================================
#  S8  WCNS Z   n=(0,0,1) l=(0,-1,0) m=(1,0,0)
# =============================================================================

@wp.kernel
def wcns_z_kernel(
    cons:wp.array(dtype=wp.float64,ndim=4),  pres:wp.array(dtype=wp.float64,ndim=3),
    ducros:wp.array(dtype=wp.float64,ndim=3),
    consl:wp.array(dtype=wp.float64,ndim=4), consr:wp.array(dtype=wp.float64,ndim=4),
    nx:int,ny:int,nz:int,gp:int
):
    ix,iy,face_iz = wp.tid()
    if ix<gp+1 or ix>nx+gp: return
    if iy<gp+1 or iy>ny+gp: return
    if face_iz<gp-1 or face_iz>nz+gp: return

    sig_max=wp.max(pressure_sensor_z(pres,ducros,ix,iy,face_iz-1),
             wp.max(pressure_sensor_z(pres,ducros,ix,iy,face_iz  ),
                    pressure_sensor_z(pres,ducros,ix,iy,face_iz+1)))
    f1idx=face_iz-gp
    is_smooth=(sig_max<wp.float64(0.01)) and (f1idx>6) and (f1idx<nz-5)

    rL=cons[0,ix,iy,face_iz];   rR=cons[0,ix,iy,face_iz+1]
    ruL=cons[1,ix,iy,face_iz];  ruR=cons[1,ix,iy,face_iz+1]
    rvL=cons[2,ix,iy,face_iz];  rvR=cons[2,ix,iy,face_iz+1]
    rwL=cons[3,ix,iy,face_iz];  rwR=cons[3,ix,iy,face_iz+1]
    EL=cons[4,ix,iy,face_iz];   ER=cons[4,ix,iy,face_iz+1]
    sqL=wp.sqrt(rL); sqR=wp.sqrt(rR); dv=F1/(sqL+sqR)
    pL=GM1*(EL-HALF*(ruL*ruL+rvL*rvL+rwL*rwL)/rL)
    pR=GM1*(ER-HALF*(ruR*ruR+rvR*rvR+rwR*rwR)/rR)
    HL=(EL+pL)/rL; HR=(ER+pR)/rR
    u_r=(sqL*(ruL/rL)+sqR*(ruR/rR))*dv
    v_r=(sqL*(rvL/rL)+sqR*(rvR/rR))*dv
    w_r=(sqL*(rwL/rL)+sqR*(rwR/rR))*dv
    H_r=(sqL*HL+sqR*HR)*dv
    q2v=u_r*u_r+v_r*v_r+w_r*w_r
    c2=wp.max((GAMMA-F1)*(H_r-HALF*q2v),TINY); c_r=wp.sqrt(c2)
    NXV=F0; NYV=F0; NZV=F1; LX=F0; LY=NF1; LZ=F0; MX=F1; MY=F0; MZ=F0
    qn=w_r; ql=-v_r; qm=u_r

    if is_smooth:
        sai=F0p6
        ul0=central6_ul(cons[0,ix,iy,face_iz-2],cons[0,ix,iy,face_iz-1],cons[0,ix,iy,face_iz],cons[0,ix,iy,face_iz+1],cons[0,ix,iy,face_iz+2],cons[0,ix,iy,face_iz+3],HALF)
        ur0=central6_ur(cons[0,ix,iy,face_iz-2],cons[0,ix,iy,face_iz-1],cons[0,ix,iy,face_iz],cons[0,ix,iy,face_iz+1],cons[0,ix,iy,face_iz+2],cons[0,ix,iy,face_iz+3],HALF)
        ul1=central6_ul(cons[1,ix,iy,face_iz-2],cons[1,ix,iy,face_iz-1],cons[1,ix,iy,face_iz],cons[1,ix,iy,face_iz+1],cons[1,ix,iy,face_iz+2],cons[1,ix,iy,face_iz+3],HALF)
        ur1=central6_ur(cons[1,ix,iy,face_iz-2],cons[1,ix,iy,face_iz-1],cons[1,ix,iy,face_iz],cons[1,ix,iy,face_iz+1],cons[1,ix,iy,face_iz+2],cons[1,ix,iy,face_iz+3],HALF)
        ul2=central6_ul(cons[2,ix,iy,face_iz-2],cons[2,ix,iy,face_iz-1],cons[2,ix,iy,face_iz],cons[2,ix,iy,face_iz+1],cons[2,ix,iy,face_iz+2],cons[2,ix,iy,face_iz+3],HALF)
        ur2=central6_ur(cons[2,ix,iy,face_iz-2],cons[2,ix,iy,face_iz-1],cons[2,ix,iy,face_iz],cons[2,ix,iy,face_iz+1],cons[2,ix,iy,face_iz+2],cons[2,ix,iy,face_iz+3],HALF)
        ul3=central6_ul(cons[3,ix,iy,face_iz-2],cons[3,ix,iy,face_iz-1],cons[3,ix,iy,face_iz],cons[3,ix,iy,face_iz+1],cons[3,ix,iy,face_iz+2],cons[3,ix,iy,face_iz+3],sai)
        ur3=central6_ur(cons[3,ix,iy,face_iz-2],cons[3,ix,iy,face_iz-1],cons[3,ix,iy,face_iz],cons[3,ix,iy,face_iz+1],cons[3,ix,iy,face_iz+2],cons[3,ix,iy,face_iz+3],sai)
        ul4=central6_ul(cons[4,ix,iy,face_iz-2],cons[4,ix,iy,face_iz-1],cons[4,ix,iy,face_iz],cons[4,ix,iy,face_iz+1],cons[4,ix,iy,face_iz+2],cons[4,ix,iy,face_iz+3],HALF)
        ur4=central6_ur(cons[4,ix,iy,face_iz-2],cons[4,ix,iy,face_iz-1],cons[4,ix,iy,face_iz],cons[4,ix,iy,face_iz+1],cons[4,ix,iy,face_iz+2],cons[4,ix,iy,face_iz+3],HALF)
        e_m2=char_transform(cons,ix,iy,face_iz-2,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_m1=char_transform(cons,ix,iy,face_iz-1,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_0 =char_transform(cons,ix,iy,face_iz  ,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_1 =char_transform(cons,ix,iy,face_iz+1,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_2 =char_transform(cons,ix,iy,face_iz+2,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_3 =char_transform(cons,ix,iy,face_iz+3,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        centL=mp5_ul(e_m2,e_m1,e_0,e_1,e_2); centR=mp5_ur(e_m1,e_0,e_1,e_2,e_3)
        kap=GAMMA-F1
        le0=F1-kap*HALF*q2v/c2; le1=kap*u_r/c2; le2=kap*v_r/c2; le3=kap*w_r/c2; le4=NF1*kap/c2
        dL=centL-(le0*ul0+le1*ul1+le2*ul2+le3*ul3+le4*ul4)
        dR=centR-(le0*ur0+le1*ur1+le2*ur2+le3*ur3+le4*ur4)
        consl[0,ix,iy,face_iz]=ul0+dL;          consr[0,ix,iy,face_iz]=ur0+dR
        consl[1,ix,iy,face_iz]=ul1+dL*u_r;      consr[1,ix,iy,face_iz]=ur1+dR*u_r
        consl[2,ix,iy,face_iz]=ul2+dL*v_r;      consr[2,ix,iy,face_iz]=ur2+dR*v_r
        consl[3,ix,iy,face_iz]=ul3+dL*w_r;      consr[3,ix,iy,face_iz]=ur3+dR*w_r
        consl[4,ix,iy,face_iz]=ul4+dL*HALF*q2v; consr[4,ix,iy,face_iz]=ur4+dR*HALF*q2v
    else:
        c_m2=char_transform(cons,ix,iy,face_iz-2,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_m1=char_transform(cons,ix,iy,face_iz-1,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_0 =char_transform(cons,ix,iy,face_iz  ,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_1 =char_transform(cons,ix,iy,face_iz+1,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_2 =char_transform(cons,ix,iy,face_iz+2,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_3 =char_transform(cons,ix,iy,face_iz+3,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        ul0=weno5_ul(c_m2,c_m1,c_0,c_1,c_2); ur0=weno5_ur(c_m1,c_0,c_1,c_2,c_3)
        c_m2=char_transform(cons,ix,iy,face_iz-2,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_m1=char_transform(cons,ix,iy,face_iz-1,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_0 =char_transform(cons,ix,iy,face_iz  ,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_1 =char_transform(cons,ix,iy,face_iz+1,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_2 =char_transform(cons,ix,iy,face_iz+2,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_3 =char_transform(cons,ix,iy,face_iz+3,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        ul1=weno5_ul(c_m2,c_m1,c_0,c_1,c_2); ur1=weno5_ur(c_m1,c_0,c_1,c_2,c_3)
        c_m2=char_transform(cons,ix,iy,face_iz-2,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_m1=char_transform(cons,ix,iy,face_iz-1,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_0 =char_transform(cons,ix,iy,face_iz  ,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_1 =char_transform(cons,ix,iy,face_iz+1,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_2 =char_transform(cons,ix,iy,face_iz+2,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_3 =char_transform(cons,ix,iy,face_iz+3,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        ul2=weno5_ul(c_m2,c_m1,c_0,c_1,c_2); ur2=weno5_ur(c_m1,c_0,c_1,c_2,c_3)
        c_m2=char_transform(cons,ix,iy,face_iz-2,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_m1=char_transform(cons,ix,iy,face_iz-1,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_0 =char_transform(cons,ix,iy,face_iz  ,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_1 =char_transform(cons,ix,iy,face_iz+1,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_2 =char_transform(cons,ix,iy,face_iz+2,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_3 =char_transform(cons,ix,iy,face_iz+3,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        ul3=weno5_ul(c_m2,c_m1,c_0,c_1,c_2); ur3=weno5_ur(c_m1,c_0,c_1,c_2,c_3)
        c_m2=char_transform(cons,ix,iy,face_iz-2,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_m1=char_transform(cons,ix,iy,face_iz-1,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_0 =char_transform(cons,ix,iy,face_iz  ,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_1 =char_transform(cons,ix,iy,face_iz+1,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_2 =char_transform(cons,ix,iy,face_iz+2,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_3 =char_transform(cons,ix,iy,face_iz+3,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        ul4=weno5_ul(c_m2,c_m1,c_0,c_1,c_2); ur4=weno5_ur(c_m1,c_0,c_1,c_2,c_3)
        consl[0,ix,iy,face_iz]=Rv0(ul0,ul1,ul2,ul3,ul4)
        consl[1,ix,iy,face_iz]=Rv1(ul0,ul1,ul2,ul3,ul4,u_r,c_r,NXV,LX,MX)
        consl[2,ix,iy,face_iz]=Rv2(ul0,ul1,ul2,ul3,ul4,v_r,c_r,NYV,LY,MY)
        consl[3,ix,iy,face_iz]=Rv3(ul0,ul1,ul2,ul3,ul4,w_r,c_r,NZV,LZ,MZ)
        consl[4,ix,iy,face_iz]=Rv4(ul0,ul1,ul2,ul3,ul4,H_r,qn,c_r,ql,qm,q2v)
        consr[0,ix,iy,face_iz]=Rv0(ur0,ur1,ur2,ur3,ur4)
        consr[1,ix,iy,face_iz]=Rv1(ur0,ur1,ur2,ur3,ur4,u_r,c_r,NXV,LX,MX)
        consr[2,ix,iy,face_iz]=Rv2(ur0,ur1,ur2,ur3,ur4,v_r,c_r,NYV,LY,MY)
        consr[3,ix,iy,face_iz]=Rv3(ur0,ur1,ur2,ur3,ur4,w_r,c_r,NZV,LZ,MZ)
        consr[4,ix,iy,face_iz]=Rv4(ur0,ur1,ur2,ur3,ur4,H_r,qn,c_r,ql,qm,q2v)


# =============================================================================
#  S9  HLLC flux + residuals — inlined, SoA
#      One helper macro-func per direction to keep kernels readable
# =============================================================================

@wp.kernel
def flux_x_residual_kernel(
    cons:wp.array(dtype=wp.float64,ndim=4),
    consl:wp.array(dtype=wp.float64,ndim=4), consr:wp.array(dtype=wp.float64,ndim=4),
    resid:wp.array(dtype=wp.float64,ndim=4), dx:wp.float64,
    nx:int,ny:int,nz:int,gp:int
):
    i,j,k = wp.tid()
    if i<1 or i>nx or j<1 or j>ny or k<1 or k>nz: return
    ix=i+gp; iy=j+gp; iz=k+gp

    # --- right face i+1/2 ---
    rL0=consl[0,ix,iy,iz]; rL1=consl[1,ix,iy,iz]; rL2=consl[2,ix,iy,iz]; rL3=consl[3,ix,iy,iz]; rL4=consl[4,ix,iy,iz]
    rR0=consr[0,ix,iy,iz]; rR1=consr[1,ix,iy,iz]; rR2=consr[2,ix,iy,iz]; rR3=consr[3,ix,iy,iz]; rR4=consr[4,ix,iy,iz]
    if is_physical(rL0,rL1,rL2,rL3,rL4)==0 or is_physical(rR0,rR1,rR2,rR3,rR4)==0:
        rL0=cons[0,ix,  iy,iz]; rL1=cons[1,ix,  iy,iz]; rL2=cons[2,ix,  iy,iz]; rL3=cons[3,ix,  iy,iz]; rL4=cons[4,ix,  iy,iz]
        rR0=cons[0,ix+1,iy,iz]; rR1=cons[1,ix+1,iy,iz]; rR2=cons[2,ix+1,iy,iz]; rR3=cons[3,ix+1,iy,iz]; rR4=cons[4,ix+1,iy,iz]
    uL=rL1/rL0; uR=rR1/rR0; vL=rL2/rL0; vR=rR2/rR0; wL=rL3/rL0; wR=rR3/rR0
    pL=GM1*(rL4-HALF*rL0*(uL*uL+vL*vL+wL*wL)); pR=GM1*(rR4-HALF*rR0*(uR*uR+vR*vR+wR*wR))
    HL=(rL4+pL)/rL0; HR=(rR4+pR)/rR0
    sqL=wp.sqrt(rL0); sqR=wp.sqrt(rR0); dv=F1/(sqL+sqR)
    uav=(sqL*uL+sqR*uR)*dv; vav=(sqL*vL+sqR*vR)*dv; wav=(sqL*wL+sqR*wR)*dv; Hav=(sqL*HL+sqR*HR)*dv
    cav=wp.sqrt(wp.max((GAMMA-F1)*(Hav-HALF*(uav*uav+vav*vav+wav*wav)),TINY))
    cLs=wp.sqrt(wp.max(GAMMA*pL/rL0,TINY)); cRs=wp.sqrt(wp.max(GAMMA*pR/rR0,TINY))
    SL=wp.min(uL-cLs,uav-cav); SR=wp.max(uR+cRs,uav+cav)
    den=rL0*(SL-uL)-rR0*(SR-uR); 
    if wp.abs(den)<TINY: den=TINY
    SP=(pR-pL+rL0*uL*(SL-uL)-rR0*uR*(SR-uR))/den
    fL0=rL1; fL1=rL1*uL+pL; fL2=rL1*vL; fL3=rL1*wL; fL4=uL*(rL4+pL)
    fR0=rR1; fR1=rR1*uR+pR; fR2=rR1*vR; fR3=rR1*wR; fR4=uR*(rR4+pR)
    fr0=F0; fr1=F0; fr2=F0; fr3=F0; fr4=F0
    if SL>F0:
        fr0=fL0; fr1=fL1; fr2=fL2; fr3=fL3; fr4=fL4
    elif SL<=F0 and F0<SP:
        fac=rL0*(SL-uL)/(SL-SP); EL_=rL4/rL0+(SP-uL)*(SP+pL/(rL0*(SL-uL)))
        fr0=fL0+SL*(fac-rL0); fr1=fL1+SL*(fac*SP-rL1)
        fr2=fL2+SL*(fac*vL-rL2); fr3=fL3+SL*(fac*wL-rL3); fr4=fL4+SL*(fac*EL_-rL4)
    elif SP<=F0 and F0<=SR:
        fac=rR0*(SR-uR)/(SR-SP); ER_=rR4/rR0+(SP-uR)*(SP+pR/(rR0*(SR-uR)))
        fr0=fR0+SR*(fac-rR0); fr1=fR1+SR*(fac*SP-rR1)
        fr2=fR2+SR*(fac*vR-rR2); fr3=fR3+SR*(fac*wR-rR3); fr4=fR4+SR*(fac*ER_-rR4)
    else:
        fr0=fR0; fr1=fR1; fr2=fR2; fr3=fR3; fr4=fR4

    # --- left face i-1/2 ---
    lL0=consl[0,ix-1,iy,iz]; lL1=consl[1,ix-1,iy,iz]; lL2=consl[2,ix-1,iy,iz]; lL3=consl[3,ix-1,iy,iz]; lL4=consl[4,ix-1,iy,iz]
    lR0=consr[0,ix-1,iy,iz]; lR1=consr[1,ix-1,iy,iz]; lR2=consr[2,ix-1,iy,iz]; lR3=consr[3,ix-1,iy,iz]; lR4=consr[4,ix-1,iy,iz]
    if is_physical(lL0,lL1,lL2,lL3,lL4)==0 or is_physical(lR0,lR1,lR2,lR3,lR4)==0:
        lL0=cons[0,ix-1,iy,iz]; lL1=cons[1,ix-1,iy,iz]; lL2=cons[2,ix-1,iy,iz]; lL3=cons[3,ix-1,iy,iz]; lL4=cons[4,ix-1,iy,iz]
        lR0=cons[0,ix,  iy,iz]; lR1=cons[1,ix,  iy,iz]; lR2=cons[2,ix,  iy,iz]; lR3=cons[3,ix,  iy,iz]; lR4=cons[4,ix,  iy,iz]
    uL=lL1/lL0; uR=lR1/lR0; vL=lL2/lL0; vR=lR2/lR0; wL=lL3/lL0; wR=lR3/lR0
    pL=GM1*(lL4-HALF*lL0*(uL*uL+vL*vL+wL*wL)); pR=GM1*(lR4-HALF*lR0*(uR*uR+vR*vR+wR*wR))
    HL=(lL4+pL)/lL0; HR=(lR4+pR)/lR0
    sqL=wp.sqrt(lL0); sqR=wp.sqrt(lR0); dv=F1/(sqL+sqR)
    uav=(sqL*uL+sqR*uR)*dv; vav=(sqL*vL+sqR*vR)*dv; wav=(sqL*wL+sqR*wR)*dv; Hav=(sqL*HL+sqR*HR)*dv
    cav=wp.sqrt(wp.max((GAMMA-F1)*(Hav-HALF*(uav*uav+vav*vav+wav*wav)),TINY))
    cLs=wp.sqrt(wp.max(GAMMA*pL/lL0,TINY)); cRs=wp.sqrt(wp.max(GAMMA*pR/lR0,TINY))
    SL=wp.min(uL-cLs,uav-cav); SR=wp.max(uR+cRs,uav+cav)
    den=lL0*(SL-uL)-lR0*(SR-uR)
    if wp.abs(den)<TINY: den=TINY
    SP=(pR-pL+lL0*uL*(SL-uL)-lR0*uR*(SR-uR))/den
    fL0=lL1; fL1=lL1*uL+pL; fL2=lL1*vL; fL3=lL1*wL; fL4=uL*(lL4+pL)
    fR0=lR1; fR1=lR1*uR+pR; fR2=lR1*vR; fR3=lR1*wR; fR4=uR*(lR4+pR)
    fl0=F0; fl1=F0; fl2=F0; fl3=F0; fl4=F0
    if SL>F0:
        fl0=fL0; fl1=fL1; fl2=fL2; fl3=fL3; fl4=fL4
    elif SL<=F0 and F0<SP:
        fac=lL0*(SL-uL)/(SL-SP); EL_=lL4/lL0+(SP-uL)*(SP+pL/(lL0*(SL-uL)))
        fl0=fL0+SL*(fac-lL0); fl1=fL1+SL*(fac*SP-lL1)
        fl2=fL2+SL*(fac*vL-lL2); fl3=fL3+SL*(fac*wL-lL3); fl4=fL4+SL*(fac*EL_-lL4)
    elif SP<=F0 and F0<=SR:
        fac=lR0*(SR-uR)/(SR-SP); ER_=lR4/lR0+(SP-uR)*(SP+pR/(lR0*(SR-uR)))
        fl0=fR0+SR*(fac-lR0); fl1=fR1+SR*(fac*SP-lR1)
        fl2=fR2+SR*(fac*vR-lR2); fl3=fR3+SR*(fac*wR-lR3); fl4=fR4+SR*(fac*ER_-lR4)
    else:
        fl0=fR0; fl1=fR1; fl2=fR2; fl3=fR3; fl4=fR4

    resid[0,ix,iy,iz]=resid[0,ix,iy,iz]-(fr0-fl0)/dx
    resid[1,ix,iy,iz]=resid[1,ix,iy,iz]-(fr1-fl1)/dx
    resid[2,ix,iy,iz]=resid[2,ix,iy,iz]-(fr2-fl2)/dx
    resid[3,ix,iy,iz]=resid[3,ix,iy,iz]-(fr3-fl3)/dx
    resid[4,ix,iy,iz]=resid[4,ix,iy,iz]-(fr4-fl4)/dx


@wp.kernel
def flux_y_residual_kernel(
    cons:wp.array(dtype=wp.float64,ndim=4),
    consl:wp.array(dtype=wp.float64,ndim=4), consr:wp.array(dtype=wp.float64,ndim=4),
    resid:wp.array(dtype=wp.float64,ndim=4), dy:wp.float64,
    nx:int,ny:int,nz:int,gp:int
):
    i,j,k = wp.tid()
    if i<1 or i>nx or j<1 or j>ny or k<1 or k>nz: return
    ix=i+gp; iy=j+gp; iz=k+gp

    # right face j+1/2
    rL0=consl[0,ix,iy,iz]; rL1=consl[1,ix,iy,iz]; rL2=consl[2,ix,iy,iz]; rL3=consl[3,ix,iy,iz]; rL4=consl[4,ix,iy,iz]
    rR0=consr[0,ix,iy,iz]; rR1=consr[1,ix,iy,iz]; rR2=consr[2,ix,iy,iz]; rR3=consr[3,ix,iy,iz]; rR4=consr[4,ix,iy,iz]
    if is_physical(rL0,rL1,rL2,rL3,rL4)==0 or is_physical(rR0,rR1,rR2,rR3,rR4)==0:
        rL0=cons[0,ix,iy,  iz]; rL1=cons[1,ix,iy,  iz]; rL2=cons[2,ix,iy,  iz]; rL3=cons[3,ix,iy,  iz]; rL4=cons[4,ix,iy,  iz]
        rR0=cons[0,ix,iy+1,iz]; rR1=cons[1,ix,iy+1,iz]; rR2=cons[2,ix,iy+1,iz]; rR3=cons[3,ix,iy+1,iz]; rR4=cons[4,ix,iy+1,iz]
    uL=rL1/rL0; uR=rR1/rR0; vL=rL2/rL0; vR=rR2/rR0; wL=rL3/rL0; wR=rR3/rR0
    pL=GM1*(rL4-HALF*rL0*(uL*uL+vL*vL+wL*wL)); pR=GM1*(rR4-HALF*rR0*(uR*uR+vR*vR+wR*wR))
    HL=(rL4+pL)/rL0; HR=(rR4+pR)/rR0
    sqL=wp.sqrt(rL0); sqR=wp.sqrt(rR0); dv=F1/(sqL+sqR)
    uav=(sqL*uL+sqR*uR)*dv; vav=(sqL*vL+sqR*vR)*dv; wav=(sqL*wL+sqR*wR)*dv; Hav=(sqL*HL+sqR*HR)*dv
    cav=wp.sqrt(wp.max((GAMMA-F1)*(Hav-HALF*(uav*uav+vav*vav+wav*wav)),TINY))
    cLs=wp.sqrt(wp.max(GAMMA*pL/rL0,TINY)); cRs=wp.sqrt(wp.max(GAMMA*pR/rR0,TINY))
    SL=wp.min(vL-cLs,vav-cav); SR=wp.max(vR+cRs,vav+cav)
    den=rL0*(SL-vL)-rR0*(SR-vR)
    if wp.abs(den)<TINY: den=TINY
    SP=(pR-pL+rL0*vL*(SL-vL)-rR0*vR*(SR-vR))/den
    fL0=rL2; fL1=rL2*uL; fL2=rL2*vL+pL; fL3=rL2*wL; fL4=vL*(rL4+pL)
    fR0=rR2; fR1=rR2*uR; fR2=rR2*vR+pR; fR3=rR2*wR; fR4=vR*(rR4+pR)
    fr0=F0; fr1=F0; fr2=F0; fr3=F0; fr4=F0
    if SL>F0:
        fr0=fL0; fr1=fL1; fr2=fL2; fr3=fL3; fr4=fL4
    elif SL<=F0 and F0<SP:
        fac=rL0*(SL-vL)/(SL-SP); EL_=rL4/rL0+(SP-vL)*(SP+pL/(rL0*(SL-vL)))
        fr0=fL0+SL*(fac-rL0); fr1=fL1+SL*(fac*uL-rL1)
        fr2=fL2+SL*(fac*SP-rL2); fr3=fL3+SL*(fac*wL-rL3); fr4=fL4+SL*(fac*EL_-rL4)
    elif SP<=F0 and F0<=SR:
        fac=rR0*(SR-vR)/(SR-SP); ER_=rR4/rR0+(SP-vR)*(SP+pR/(rR0*(SR-vR)))
        fr0=fR0+SR*(fac-rR0); fr1=fR1+SR*(fac*uR-rR1)
        fr2=fR2+SR*(fac*SP-rR2); fr3=fR3+SR*(fac*wR-rR3); fr4=fR4+SR*(fac*ER_-rR4)
    else:
        fr0=fR0; fr1=fR1; fr2=fR2; fr3=fR3; fr4=fR4

    # left face j-1/2
    lL0=consl[0,ix,iy-1,iz]; lL1=consl[1,ix,iy-1,iz]; lL2=consl[2,ix,iy-1,iz]; lL3=consl[3,ix,iy-1,iz]; lL4=consl[4,ix,iy-1,iz]
    lR0=consr[0,ix,iy-1,iz]; lR1=consr[1,ix,iy-1,iz]; lR2=consr[2,ix,iy-1,iz]; lR3=consr[3,ix,iy-1,iz]; lR4=consr[4,ix,iy-1,iz]
    if is_physical(lL0,lL1,lL2,lL3,lL4)==0 or is_physical(lR0,lR1,lR2,lR3,lR4)==0:
        lL0=cons[0,ix,iy-1,iz]; lL1=cons[1,ix,iy-1,iz]; lL2=cons[2,ix,iy-1,iz]; lL3=cons[3,ix,iy-1,iz]; lL4=cons[4,ix,iy-1,iz]
        lR0=cons[0,ix,iy,  iz]; lR1=cons[1,ix,iy,  iz]; lR2=cons[2,ix,iy,  iz]; lR3=cons[3,ix,iy,  iz]; lR4=cons[4,ix,iy,  iz]
    uL=lL1/lL0; uR=lR1/lR0; vL=lL2/lL0; vR=lR2/lR0; wL=lL3/lL0; wR=lR3/lR0
    pL=GM1*(lL4-HALF*lL0*(uL*uL+vL*vL+wL*wL)); pR=GM1*(lR4-HALF*lR0*(uR*uR+vR*vR+wR*wR))
    HL=(lL4+pL)/lL0; HR=(lR4+pR)/lR0
    sqL=wp.sqrt(lL0); sqR=wp.sqrt(lR0); dv=F1/(sqL+sqR)
    uav=(sqL*uL+sqR*uR)*dv; vav=(sqL*vL+sqR*vR)*dv; wav=(sqL*wL+sqR*wR)*dv; Hav=(sqL*HL+sqR*HR)*dv
    cav=wp.sqrt(wp.max((GAMMA-F1)*(Hav-HALF*(uav*uav+vav*vav+wav*wav)),TINY))
    cLs=wp.sqrt(wp.max(GAMMA*pL/lL0,TINY)); cRs=wp.sqrt(wp.max(GAMMA*pR/lR0,TINY))
    SL=wp.min(vL-cLs,vav-cav); SR=wp.max(vR+cRs,vav+cav)
    den=lL0*(SL-vL)-lR0*(SR-vR)
    if wp.abs(den)<TINY: den=TINY
    SP=(pR-pL+lL0*vL*(SL-vL)-lR0*vR*(SR-vR))/den
    fL0=lL2; fL1=lL2*uL; fL2=lL2*vL+pL; fL3=lL2*wL; fL4=vL*(lL4+pL)
    fR0=lR2; fR1=lR2*uR; fR2=lR2*vR+pR; fR3=lR2*wR; fR4=vR*(lR4+pR)
    fl0=F0; fl1=F0; fl2=F0; fl3=F0; fl4=F0
    if SL>F0:
        fl0=fL0; fl1=fL1; fl2=fL2; fl3=fL3; fl4=fL4
    elif SL<=F0 and F0<SP:
        fac=lL0*(SL-vL)/(SL-SP); EL_=lL4/lL0+(SP-vL)*(SP+pL/(lL0*(SL-vL)))
        fl0=fL0+SL*(fac-lL0); fl1=fL1+SL*(fac*uL-lL1)
        fl2=fL2+SL*(fac*SP-lL2); fl3=fL3+SL*(fac*wL-lL3); fl4=fL4+SL*(fac*EL_-lL4)
    elif SP<=F0 and F0<=SR:
        fac=lR0*(SR-vR)/(SR-SP); ER_=lR4/lR0+(SP-vR)*(SP+pR/(lR0*(SR-vR)))
        fl0=fR0+SR*(fac-lR0); fl1=fR1+SR*(fac*uR-lR1)
        fl2=fR2+SR*(fac*SP-lR2); fl3=fR3+SR*(fac*wR-lR3); fl4=fR4+SR*(fac*ER_-lR4)
    else:
        fl0=fR0; fl1=fR1; fl2=fR2; fl3=fR3; fl4=fR4

    resid[0,ix,iy,iz]=resid[0,ix,iy,iz]-(fr0-fl0)/dy
    resid[1,ix,iy,iz]=resid[1,ix,iy,iz]-(fr1-fl1)/dy
    resid[2,ix,iy,iz]=resid[2,ix,iy,iz]-(fr2-fl2)/dy
    resid[3,ix,iy,iz]=resid[3,ix,iy,iz]-(fr3-fl3)/dy
    resid[4,ix,iy,iz]=resid[4,ix,iy,iz]-(fr4-fl4)/dy


@wp.kernel
def flux_z_residual_kernel(
    cons:wp.array(dtype=wp.float64,ndim=4),
    consl:wp.array(dtype=wp.float64,ndim=4), consr:wp.array(dtype=wp.float64,ndim=4),
    resid:wp.array(dtype=wp.float64,ndim=4), dz:wp.float64,
    nx:int,ny:int,nz:int,gp:int
):
    i,j,k = wp.tid()
    if i<1 or i>nx or j<1 or j>ny or k<1 or k>nz: return
    ix=i+gp; iy=j+gp; iz=k+gp

    # right face k+1/2
    rL0=consl[0,ix,iy,iz]; rL1=consl[1,ix,iy,iz]; rL2=consl[2,ix,iy,iz]; rL3=consl[3,ix,iy,iz]; rL4=consl[4,ix,iy,iz]
    rR0=consr[0,ix,iy,iz]; rR1=consr[1,ix,iy,iz]; rR2=consr[2,ix,iy,iz]; rR3=consr[3,ix,iy,iz]; rR4=consr[4,ix,iy,iz]
    if is_physical(rL0,rL1,rL2,rL3,rL4)==0 or is_physical(rR0,rR1,rR2,rR3,rR4)==0:
        rL0=cons[0,ix,iy,iz  ]; rL1=cons[1,ix,iy,iz  ]; rL2=cons[2,ix,iy,iz  ]; rL3=cons[3,ix,iy,iz  ]; rL4=cons[4,ix,iy,iz  ]
        rR0=cons[0,ix,iy,iz+1]; rR1=cons[1,ix,iy,iz+1]; rR2=cons[2,ix,iy,iz+1]; rR3=cons[3,ix,iy,iz+1]; rR4=cons[4,ix,iy,iz+1]
    uL=rL1/rL0; uR=rR1/rR0; vL=rL2/rL0; vR=rR2/rR0; wL=rL3/rL0; wR=rR3/rR0
    pL=GM1*(rL4-HALF*rL0*(uL*uL+vL*vL+wL*wL)); pR=GM1*(rR4-HALF*rR0*(uR*uR+vR*vR+wR*wR))
    HL=(rL4+pL)/rL0; HR=(rR4+pR)/rR0
    sqL=wp.sqrt(rL0); sqR=wp.sqrt(rR0); dv=F1/(sqL+sqR)
    uav=(sqL*uL+sqR*uR)*dv; vav=(sqL*vL+sqR*vR)*dv; wav=(sqL*wL+sqR*wR)*dv; Hav=(sqL*HL+sqR*HR)*dv
    cav=wp.sqrt(wp.max((GAMMA-F1)*(Hav-HALF*(uav*uav+vav*vav+wav*wav)),TINY))
    cLs=wp.sqrt(wp.max(GAMMA*pL/rL0,TINY)); cRs=wp.sqrt(wp.max(GAMMA*pR/rR0,TINY))
    SL=wp.min(wL-cLs,wav-cav); SR=wp.max(wR+cRs,wav+cav)
    den=rL0*(SL-wL)-rR0*(SR-wR)
    if wp.abs(den)<TINY: den=TINY
    SP=(pR-pL+rL0*wL*(SL-wL)-rR0*wR*(SR-wR))/den
    fL0=rL3; fL1=rL3*uL; fL2=rL3*vL; fL3=rL3*wL+pL; fL4=wL*(rL4+pL)
    fR0=rR3; fR1=rR3*uR; fR2=rR3*vR; fR3=rR3*wR+pR; fR4=wR*(rR4+pR)
    fr0=F0; fr1=F0; fr2=F0; fr3=F0; fr4=F0
    if SL>F0:
        fr0=fL0; fr1=fL1; fr2=fL2; fr3=fL3; fr4=fL4
    elif SL<=F0 and F0<SP:
        fac=rL0*(SL-wL)/(SL-SP); EL_=rL4/rL0+(SP-wL)*(SP+pL/(rL0*(SL-wL)))
        fr0=fL0+SL*(fac-rL0); fr1=fL1+SL*(fac*uL-rL1)
        fr2=fL2+SL*(fac*vL-rL2); fr3=fL3+SL*(fac*SP-rL3); fr4=fL4+SL*(fac*EL_-rL4)
    elif SP<=F0 and F0<=SR:
        fac=rR0*(SR-wR)/(SR-SP); ER_=rR4/rR0+(SP-wR)*(SP+pR/(rR0*(SR-wR)))
        fr0=fR0+SR*(fac-rR0); fr1=fR1+SR*(fac*uR-rR1)
        fr2=fR2+SR*(fac*vR-rR2); fr3=fR3+SR*(fac*SP-rR3); fr4=fR4+SR*(fac*ER_-rR4)
    else:
        fr0=fR0; fr1=fR1; fr2=fR2; fr3=fR3; fr4=fR4

    # left face k-1/2
    lL0=consl[0,ix,iy,iz-1]; lL1=consl[1,ix,iy,iz-1]; lL2=consl[2,ix,iy,iz-1]; lL3=consl[3,ix,iy,iz-1]; lL4=consl[4,ix,iy,iz-1]
    lR0=consr[0,ix,iy,iz-1]; lR1=consr[1,ix,iy,iz-1]; lR2=consr[2,ix,iy,iz-1]; lR3=consr[3,ix,iy,iz-1]; lR4=consr[4,ix,iy,iz-1]
    if is_physical(lL0,lL1,lL2,lL3,lL4)==0 or is_physical(lR0,lR1,lR2,lR3,lR4)==0:
        lL0=cons[0,ix,iy,iz-1]; lL1=cons[1,ix,iy,iz-1]; lL2=cons[2,ix,iy,iz-1]; lL3=cons[3,ix,iy,iz-1]; lL4=cons[4,ix,iy,iz-1]
        lR0=cons[0,ix,iy,iz  ]; lR1=cons[1,ix,iy,iz  ]; lR2=cons[2,ix,iy,iz  ]; lR3=cons[3,ix,iy,iz  ]; lR4=cons[4,ix,iy,iz  ]
    uL=lL1/lL0; uR=lR1/lR0; vL=lL2/lL0; vR=lR2/lR0; wL=lL3/lL0; wR=lR3/lR0
    pL=GM1*(lL4-HALF*lL0*(uL*uL+vL*vL+wL*wL)); pR=GM1*(lR4-HALF*lR0*(uR*uR+vR*vR+wR*wR))
    HL=(lL4+pL)/lL0; HR=(lR4+pR)/lR0
    sqL=wp.sqrt(lL0); sqR=wp.sqrt(lR0); dv=F1/(sqL+sqR)
    uav=(sqL*uL+sqR*uR)*dv; vav=(sqL*vL+sqR*vR)*dv; wav=(sqL*wL+sqR*wR)*dv; Hav=(sqL*HL+sqR*HR)*dv
    cav=wp.sqrt(wp.max((GAMMA-F1)*(Hav-HALF*(uav*uav+vav*vav+wav*wav)),TINY))
    cLs=wp.sqrt(wp.max(GAMMA*pL/lL0,TINY)); cRs=wp.sqrt(wp.max(GAMMA*pR/lR0,TINY))
    SL=wp.min(wL-cLs,wav-cav); SR=wp.max(wR+cRs,wav+cav)
    den=lL0*(SL-wL)-lR0*(SR-wR)
    if wp.abs(den)<TINY: den=TINY
    SP=(pR-pL+lL0*wL*(SL-wL)-lR0*wR*(SR-wR))/den
    fL0=lL3; fL1=lL3*uL; fL2=lL3*vL; fL3=lL3*wL+pL; fL4=wL*(lL4+pL)
    fR0=lR3; fR1=lR3*uR; fR2=lR3*vR; fR3=lR3*wR+pR; fR4=wR*(lR4+pR)
    fl0=F0; fl1=F0; fl2=F0; fl3=F0; fl4=F0
    if SL>F0:
        fl0=fL0; fl1=fL1; fl2=fL2; fl3=fL3; fl4=fL4
    elif SL<=F0 and F0<SP:
        fac=lL0*(SL-wL)/(SL-SP); EL_=lL4/lL0+(SP-wL)*(SP+pL/(lL0*(SL-wL)))
        fl0=fL0+SL*(fac-lL0); fl1=fL1+SL*(fac*uL-lL1)
        fl2=fL2+SL*(fac*vL-lL2); fl3=fL3+SL*(fac*SP-lL3); fl4=fL4+SL*(fac*EL_-lL4)
    elif SP<=F0 and F0<=SR:
        fac=lR0*(SR-wR)/(SR-SP); ER_=lR4/lR0+(SP-wR)*(SP+pR/(lR0*(SR-wR)))
        fl0=fR0+SR*(fac-lR0); fl1=fR1+SR*(fac*uR-lR1)
        fl2=fR2+SR*(fac*vR-lR2); fl3=fR3+SR*(fac*SP-lR3); fl4=fR4+SR*(fac*ER_-lR4)
    else:
        fl0=fR0; fl1=fR1; fl2=fR2; fl3=fR3; fl4=fR4

    resid[0,ix,iy,iz]=resid[0,ix,iy,iz]-(fr0-fl0)/dz
    resid[1,ix,iy,iz]=resid[1,ix,iy,iz]-(fr1-fl1)/dz
    resid[2,ix,iy,iz]=resid[2,ix,iy,iz]-(fr2-fl2)/dz
    resid[3,ix,iy,iz]=resid[3,ix,iy,iz]-(fr3-fl3)/dz
    resid[4,ix,iy,iz]=resid[4,ix,iy,iz]-(fr4-fl4)/dz


# =============================================================================
#  S10  SSP-RK3  +  zero residual
# =============================================================================

@wp.kernel
def rk3_step_kernel(
    cons0:wp.array(dtype=wp.float64,ndim=4),  cons_in:wp.array(dtype=wp.float64,ndim=4),
    resid:wp.array(dtype=wp.float64,ndim=4),  cons_out:wp.array(dtype=wp.float64,ndim=4),
    alpha:wp.float64,beta:wp.float64,dt:wp.float64,
    nx:int,ny:int,nz:int,gp:int
):
    i,j,k = wp.tid()
    if i<1 or i>nx or j<1 or j>ny or k<1 or k>nz: return
    ix=i+gp; iy=j+gp; iz=k+gp
    cons_out[0,ix,iy,iz]=alpha*cons0[0,ix,iy,iz]+beta*(cons_in[0,ix,iy,iz]+dt*resid[0,ix,iy,iz])
    cons_out[1,ix,iy,iz]=alpha*cons0[1,ix,iy,iz]+beta*(cons_in[1,ix,iy,iz]+dt*resid[1,ix,iy,iz])
    cons_out[2,ix,iy,iz]=alpha*cons0[2,ix,iy,iz]+beta*(cons_in[2,ix,iy,iz]+dt*resid[2,ix,iy,iz])
    cons_out[3,ix,iy,iz]=alpha*cons0[3,ix,iy,iz]+beta*(cons_in[3,ix,iy,iz]+dt*resid[3,ix,iy,iz])
    cons_out[4,ix,iy,iz]=alpha*cons0[4,ix,iy,iz]+beta*(cons_in[4,ix,iy,iz]+dt*resid[4,ix,iy,iz])

@wp.kernel
def zero_residual_kernel(resid:wp.array(dtype=wp.float64,ndim=4),nx:int,ny:int,nz:int,gp:int):
    i,j,k = wp.tid()
    if i<1 or i>nx or j<1 or j>ny or k<1 or k>nz: return
    ix=i+gp; iy=j+gp; iz=k+gp
    resid[0,ix,iy,iz]=F0; resid[1,ix,iy,iz]=F0; resid[2,ix,iy,iz]=F0
    resid[3,ix,iy,iz]=F0; resid[4,ix,iy,iz]=F0

# =============================================================================
#  S11  Time step — GPU atomic min, only 8 bytes transferred per step
# =============================================================================

@wp.kernel
def dt_local_kernel(
    u_vel:wp.array(dtype=wp.float64,ndim=3), v_vel:wp.array(dtype=wp.float64,ndim=3),
    w_vel:wp.array(dtype=wp.float64,ndim=3), pres:wp.array(dtype=wp.float64,ndim=3),
    rho:wp.array(dtype=wp.float64,ndim=3),   snd:wp.array(dtype=wp.float64,ndim=3),
    dt_min:wp.array(dtype=wp.float64,ndim=1),
    dx:wp.float64,dy:wp.float64,dz:wp.float64,
    nx:int,ny:int,nz:int,gp:int
):
    i,j,k = wp.tid()
    if i<1 or i>nx or j<1 or j>ny or k<1 or k>nz: return
    ix=i+gp; iy=j+gp; iz=k+gp
    c=wp.sqrt(GAMMA*pres[ix,iy,iz]/rho[ix,iy,iz])
    snd[ix,iy,iz]=c
    sx=wp.abs(u_vel[ix,iy,iz])+c
    sy=wp.abs(v_vel[ix,iy,iz])+c
    sz=wp.abs(w_vel[ix,iy,iz])+c
    dt_cell=wp.min(dx/sx, wp.min(dy/sy, dz/sz))
    wp.atomic_min(dt_min, 0, dt_cell)

# =============================================================================
#  S12  Output
# =============================================================================

def save_npz(rho_np,u_np,v_np,w_np,p_np,x_np,y_np,z_np,step,time_sim):
    fname=f"tgv_{step:06d}.npz"
    np.savez_compressed(fname,
        time=np.float64(time_sim),
        x=x_np[G:G+NX], y=y_np[G:G+NY], z=z_np[G:G+NZ],
        rho=rho_np[G:G+NX,G:G+NY,G:G+NZ],
        p=p_np[G:G+NX,G:G+NY,G:G+NZ],
        u=u_np[G:G+NX,G:G+NY,G:G+NZ],
        v=v_np[G:G+NX,G:G+NY,G:G+NZ],
        w=w_np[G:G+NX,G:G+NY,G:G+NZ])
    print(f"  -> {fname}")

def compute_ke(rho_np,u_np,v_np,w_np):
    r=rho_np[G:G+NX,G:G+NY,G:G+NZ]
    u=u_np[G:G+NX,G:G+NY,G:G+NZ]
    v=v_np[G:G+NX,G:G+NY,G:G+NZ]
    w=w_np[G:G+NX,G:G+NY,G:G+NZ]
    return float(np.sum(r*0.5*(u*u+v*v+w*w)))

def compute_enstrophy(u_np,v_np,w_np,dx_py,dy_py,dz_py):
    u=u_np[G:G+NX,G:G+NY,G:G+NZ]
    v=v_np[G:G+NX,G:G+NY,G:G+NZ]
    w=w_np[G:G+NX,G:G+NY,G:G+NZ]
    dwdy=np.gradient(w,dy_py,axis=1); dvdz=np.gradient(v,dz_py,axis=2)
    dudz=np.gradient(u,dz_py,axis=2); dwdx=np.gradient(w,dx_py,axis=0)
    dvdx=np.gradient(v,dx_py,axis=0); dudy=np.gradient(u,dy_py,axis=1)
    ox=dwdy-dvdz; oy=dudz-dwdx; oz=dvdx-dudy
    return float(np.sum(ox*ox+oy*oy+oz*oz))

def save_png_slice(rho_np,step,time_sim):
    d=rho_np[G:G+NX,G:G+NY,G+NZ//2]
    fig,ax=plt.subplots(figsize=(6,6))
    ax.contourf(d.T,32,cmap='viridis')
    ax.set_title(f'rho z-slice  t={time_sim:.3f}')
    ax.set_aspect('equal')
    fname=f"tgv_{step:06d}.png"
    fig.savefig(fname,dpi=600,bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {fname}")

# =============================================================================
#  S13  Main
# =============================================================================

def main():
    print(f"3D TGV  N={NX}  device={DEVICE}  layout=SoA cons[v,ix,iy,iz]")

    dx_py=(XMAX_PY-XMIN_PY)/NX
    dy_py=(YMAX_PY-YMIN_PY)/NY
    dz_py=(ZMAX_PY-ZMIN_PY)/NZ

    x_np=np.array([XMIN_PY+(i-G-0.5)*dx_py for i in range(GX)],dtype=np.float64)
    y_np=np.array([YMIN_PY+(j-G-0.5)*dy_py for j in range(GY)],dtype=np.float64)
    z_np=np.array([ZMIN_PY+(k-G-0.5)*dz_py for k in range(GZ)],dtype=np.float64)

    x_d=wp.array(x_np,dtype=wp.float64,device=DEVICE)
    y_d=wp.array(y_np,dtype=wp.float64,device=DEVICE)
    z_d=wp.array(z_np,dtype=wp.float64,device=DEVICE)

    # Primitives: (GX,GY,GZ) — AoS fine for scalar fields
    rho_d=wp.zeros((GX,GY,GZ),dtype=wp.float64,device=DEVICE)
    u_d  =wp.zeros((GX,GY,GZ),dtype=wp.float64,device=DEVICE)
    v_d  =wp.zeros((GX,GY,GZ),dtype=wp.float64,device=DEVICE)
    w_d  =wp.zeros((GX,GY,GZ),dtype=wp.float64,device=DEVICE)
    p_d  =wp.zeros((GX,GY,GZ),dtype=wp.float64,device=DEVICE)
    snd_d=wp.zeros((GX,GY,GZ),dtype=wp.float64,device=DEVICE)

    # Conservatives: SoA (5,GX,GY,GZ) — coalesced along z
    cons0 =wp.zeros((5,GX,GY,GZ),dtype=wp.float64,device=DEVICE)
    cons1 =wp.zeros((5,GX,GY,GZ),dtype=wp.float64,device=DEVICE)
    cons2 =wp.zeros((5,GX,GY,GZ),dtype=wp.float64,device=DEVICE)
    consl =wp.zeros((5,GX,GY,GZ),dtype=wp.float64,device=DEVICE)
    consr =wp.zeros((5,GX,GY,GZ),dtype=wp.float64,device=DEVICE)
    resid =wp.zeros((5,GX,GY,GZ),dtype=wp.float64,device=DEVICE)

    der_ux=wp.zeros((GX,GY,GZ),dtype=wp.float64,device=DEVICE)
    der_uy=wp.zeros((GX,GY,GZ),dtype=wp.float64,device=DEVICE)
    der_uz=wp.zeros((GX,GY,GZ),dtype=wp.float64,device=DEVICE)
    der_vx=wp.zeros((GX,GY,GZ),dtype=wp.float64,device=DEVICE)
    der_vy=wp.zeros((GX,GY,GZ),dtype=wp.float64,device=DEVICE)
    der_vz=wp.zeros((GX,GY,GZ),dtype=wp.float64,device=DEVICE)
    der_wx=wp.zeros((GX,GY,GZ),dtype=wp.float64,device=DEVICE)
    der_wy=wp.zeros((GX,GY,GZ),dtype=wp.float64,device=DEVICE)
    der_wz=wp.zeros((GX,GY,GZ),dtype=wp.float64,device=DEVICE)
    ducros=wp.zeros((GX,GY,GZ),dtype=wp.float64,device=DEVICE)

    # GPU dt: single element, atomic min — only 8 bytes per step to CPU
    dt_min=wp.zeros(1, dtype=wp.float64, device=DEVICE)

    # ── Initial conditions ────────────────────────────────────────────────────
    wp.launch(init_cond_kernel, dim=(GX,GY,GZ),
              inputs=[x_d,y_d,z_d,rho_d,u_d,v_d,w_d,p_d,snd_d], device=DEVICE)
    wp.launch(prim_to_cons_kernel, dim=(GX,GY,GZ),
              inputs=[rho_d,u_d,v_d,w_d,p_d,cons0], device=DEVICE)

    rho_np=rho_d.numpy(); u_np=u_d.numpy()
    v_np=v_d.numpy();     w_np=w_d.numpy(); p_np=p_d.numpy()
    ke0=compute_ke(rho_np,u_np,v_np,w_np)
    ent0=compute_enstrophy(u_np,v_np,w_np,dx_py,dy_py,dz_py)
    print(f"t=0  KE={ke0:.6f}  Enstrophy={ent0:.6f}")
    save_npz(rho_np,u_np,v_np,w_np,p_np,x_np,y_np,z_np,step=0,time_sim=0.0)

    fke=open("ke_tgv3d.txt","w")
    fke.write("# time  KE/KE0  Enstrophy/Ent0\n")
    fke.write(f"0.000000  1.000000  1.000000\n"); fke.flush()

    time_sim=0.0; N=1; t0=time.perf_counter()
    # print(f"{'step':>8}  {'time':>12}  {'dt':>12}")
    # if N % 50 == 0:
    #     elapsed = time.perf_counter() - t0
    #     print(f"  step {N:5d}  t={time_sim:.4e}  dt={dt:.3e}  wall={elapsed:.1f}s")

    while time_sim<T_END and N<=NTMAX:

        # ── GPU CFL — atomic min, reset then launch ───────────────────────────
        dt_min.fill_(wp.float64(1.0e30))
        wp.launch(dt_local_kernel, dim=(NX+2,NY+2,NZ+2),
                  inputs=[u_d,v_d,w_d,p_d,rho_d,snd_d,dt_min,
                          wp.float64(dx_py),wp.float64(dy_py),wp.float64(dz_py),NX,NY,NZ,G],
                  device=DEVICE)
        dt = float(CFL_PY * dt_min.numpy()[0])   # only 8 bytes from GPU
        if time_sim+dt>T_END: dt=T_END-time_sim
        time_sim+=dt
        # print(f"{N:>8}  {time_sim:>12.6f}  {dt:>12.6e}")
        if N % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  step {N:5d}  t={time_sim:.4e}  dt={dt:.3e}  wall={elapsed:.1f}s")

        # ── SSP-RK3 ───────────────────────────────────────────────────────────
        rk_stages=[
            (0.0,  1.0,  cons0, cons1),
            (0.75, 0.25, cons1, cons2),
            (1/3,  2/3,  cons2, cons0),
        ]
        for alpha,beta,c_in,c_out in rk_stages:
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
                      inputs=[u_d,v_d,w_d,der_ux,der_uy,der_uz,
                               der_vx,der_vy,der_vz,der_wx,der_wy,der_wz,GX,GY,GZ],
                      device=DEVICE)
            wp.launch(ducros_kernel, dim=(GX-2,GY-2,GZ-2),
                      inputs=[der_ux,der_uy,der_uz,der_vx,der_vy,der_vz,
                               der_wx,der_wy,der_wz,ducros,GX,GY,GZ],
                      device=DEVICE)

            wp.launch(zero_residual_kernel, dim=(NX+2,NY+2,NZ+2),
                      inputs=[resid,NX,NY,NZ,G], device=DEVICE)

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

            wp.launch(rk3_step_kernel, dim=(NX+2,NY+2,NZ+2),
                      inputs=[cons0,c_in,resid,c_out,
                               wp.float64(alpha),wp.float64(beta),wp.float64(dt),
                               NX,NY,NZ,G], device=DEVICE)

        wp.launch(cons_to_prim_kernel, dim=(NX+2,NY+2,NZ+2),
                  inputs=[cons0,rho_d,u_d,v_d,w_d,p_d,NX,NY,NZ,G], device=DEVICE)

        if N%100==0:
            rho_np=rho_d.numpy(); u_np=u_d.numpy()
            v_np=v_d.numpy();     w_np=w_d.numpy()
            ke=compute_ke(rho_np,u_np,v_np,w_np)
            ent=compute_enstrophy(u_np,v_np,w_np,dx_py,dy_py,dz_py)
            fke.write(f"{time_sim:.6f}  {ke/ke0:.10f}  {ent/ent0:.10f}\n"); fke.flush()

        if N%FILE_SAVE==0:
            rho_np=rho_d.numpy(); u_np=u_d.numpy()
            v_np=v_d.numpy();     w_np=w_d.numpy(); p_np=p_d.numpy()
            save_npz(rho_np,u_np,v_np,w_np,p_np,x_np,y_np,z_np,step=N,time_sim=time_sim)
            save_png_slice(rho_np,step=N,time_sim=time_sim)

        if abs(time_sim-T_END)<1.0e-12: break
        N+=1

    wp.synchronize()
    elapsed=time.perf_counter()-t0
    print(f"\nDone. steps={N}  wall={elapsed:.2f}s  ({elapsed/N*1000:.1f} ms/step)")

    rho_np=rho_d.numpy(); u_np=u_d.numpy()
    v_np=v_d.numpy();     w_np=w_d.numpy(); p_np=p_d.numpy()
    save_npz(rho_np,u_np,v_np,w_np,p_np,x_np,y_np,z_np,step=N,time_sim=time_sim)
    save_png_slice(rho_np,step=N,time_sim=time_sim)
    ke=compute_ke(rho_np,u_np,v_np,w_np)
    ent=compute_enstrophy(u_np,v_np,w_np,dx_py,dy_py,dz_py)
    fke.write(f"{time_sim:.6f}  {ke/ke0:.10f}  {ent/ent0:.10f}\n")
    fke.close()

if __name__=="__main__":
    main()
