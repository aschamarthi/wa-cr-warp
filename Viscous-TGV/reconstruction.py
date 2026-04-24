"""
reconstruction.py
-----------------
All pointwise reconstruction functions used by the WCNS kernels:
  - WENO-Z 5th-order  (shock regions) -- Choose whichever you went. I know you guys are clever :)
  - MP5 limiter       (entropy-wave rank-1 correction only) -- Choose whichever you went. I know you guys are clever :)
  - central6          (smooth path, adjustable upwind bias sai) --- sai is just a parameter - eta. I give random names while experimenting
  - 3D left/right Euler eigenvectors (L, R)
  - char_transform    (projects one cell to a single characteristic amplitude)
  - is_physical       (positivity guard for HLLC fallback)

All are @wp.func (device functions inlined into kernels; not launchable directly).
"""

import warp as wp
from constants import (
    GAMMA, GM1, TINY, F0, F1, F2, HALF, QUART, F5_6, F1_3, F0p6,
    NF1, NF1_6, NF3, NF13, F3, F4,
    F7_6, NF7_6, F11_6, F13_12,
    F3_10, F6_10, F1_10, F1_60, F27, F47,
    WENO_EPS, MP5_B2, MP5_ALPH, MP5_EPS,
)

# =============================================================================
#  WENO-Z 5th-order reconstruction
# =============================================================================

@wp.func
def weno5_ul(vm2:wp.float64, vm1:wp.float64, v0:wp.float64,
             v1:wp.float64,  v2:wp.float64) -> wp.float64:
    # Left-biased 5th-order WENO-Z at face i+1/2.
    # Stencil [i-2..i+2] → left-side value.
    # tau = |b0 - b2| is the global smoothness indicator (WENO-Z variant);
    # it sharpens weights near smooth extrema more than classic WENO-JS.
    p0 = F1_3*v0  + F5_6*v1  + NF1_6*v2
    p1 = NF1_6*vm1 + F5_6*v0  + F1_3*v1
    p2 = F1_3*vm2  + NF7_6*vm1 + F11_6*v0
    b0 = F13_12*(v0 -F2*v1 +v2)**wp.float64(2.0) + QUART*(F3*v0-F4*v1+v2)**wp.float64(2.0)
    b1 = F13_12*(vm1-F2*v0 +v1)**wp.float64(2.0) + QUART*(vm1-v1)**wp.float64(2.0)
    b2 = F13_12*(vm2-F2*vm1+v0)**wp.float64(2.0) + QUART*(vm2-F4*vm1+F3*v0)**wp.float64(2.0)
    tau = wp.abs(b0-b2)
    a0 = F3_10*(F1+tau/(WENO_EPS+b0))
    a1 = F6_10*(F1+tau/(WENO_EPS+b1))
    a2 = F1_10*(F1+tau/(WENO_EPS+b2))
    s  = a0+a1+a2
    return (a0*p0+a1*p1+a2*p2)/s

@wp.func
def weno5_ur(vm1:wp.float64, v0:wp.float64, v1:wp.float64,
             v2:wp.float64,  v3:wp.float64) -> wp.float64:
    # Right-biased 5th-order WENO-Z at face i+1/2.
    # Stencil [i-1..i+3] → right-side value.
    # Ideal weights reversed relative to weno5_ul: d0=1/10, d1=6/10, d2=3/10.
    p0 = F11_6*v1  + NF7_6*v2  + F1_3*v3
    p1 = F1_3*v0   + F5_6*v1   + NF1_6*v2
    p2 = NF1_6*vm1 + F5_6*v0   + F1_3*v1
    b0 = F13_12*(v1 -F2*v2+v3)**wp.float64(2.0) + QUART*(F3*v1-F4*v2+v3)**wp.float64(2.0)
    b1 = F13_12*(v0 -F2*v1+v2)**wp.float64(2.0) + QUART*(v0-v2)**wp.float64(2.0)
    b2 = F13_12*(vm1-F2*v0+v1)**wp.float64(2.0) + QUART*(vm1-F4*v0+F3*v1)**wp.float64(2.0)
    tau = wp.abs(b0-b2)
    a0 = F1_10*(F1+tau/(WENO_EPS+b0))
    a1 = F6_10*(F1+tau/(WENO_EPS+b1))
    a2 = F3_10*(F1+tau/(WENO_EPS+b2))
    s  = a0+a1+a2
    return (a0*p0+a1*p1+a2*p2)/s

# =============================================================================
#  MP5 reconstruction (Suresh & Huynh 1997)
#  Used exclusively for the entropy-wave rank-1 correction (Paper 15, Sec. 5).
# =============================================================================

@wp.func
def _minmod2(x:wp.float64, y:wp.float64) -> wp.float64:
    return HALF*(wp.sign(x)+wp.sign(y))*wp.min(wp.abs(x), wp.abs(y))

@wp.func
def _minmod4(w:wp.float64, x:wp.float64, y:wp.float64, z:wp.float64) -> wp.float64:
    sw = wp.sign(w); sx = wp.sign(x); sy = wp.sign(y); sz = wp.sign(z)
    return (wp.float64(0.125)*(sw+sx)
            *wp.abs((sw+sy)*(sw+sz))
            *wp.min(wp.abs(w), wp.min(wp.abs(x), wp.min(wp.abs(y), wp.abs(z)))))

@wp.func
def mp5_ul(vm2:wp.float64, vm1:wp.float64, v0:wp.float64,
           v1:wp.float64,  v2:wp.float64) -> wp.float64:
    # Left-biased MP5 at face i+1/2.  VOR is the unlimited 5th-order value.
    # Returned directly if it lies in [VMIN, VMAX]; otherwise clipped to
    # preserve monotonicity without excess dissipation in smooth regions.
    VOR  = F1_60*(F2*vm2 + NF13*vm1 + F47*v0 + F27*v1 + NF3*v2)
    VMP  = v0 + _minmod2(v1-v0, MP5_ALPH*(v0-vm1))
    if (VOR-v0)*(VOR-VMP) < MP5_EPS:
        return VOR
    DJM1    = vm2 - F2*vm1 + v0
    DJ      = vm1 - F2*v0  + v1
    DJP1    = v0  - F2*v1  + v2
    DM4JPH  = _minmod4(F4*DJ-DJP1, F4*DJP1-DJ, DJ, DJP1)
    DM4JMH  = _minmod4(F4*DJ-DJM1, F4*DJM1-DJ, DJ, DJM1)
    VUL     = v0 + MP5_ALPH*(v0-vm1)
    VAV     = HALF*(v0+v1)
    VMD     = VAV - HALF*DM4JPH
    VLC     = v0 + HALF*(v0-vm1) + MP5_B2*DM4JMH
    VMIN    = wp.max(wp.min(wp.min(v0,v1),VMD), wp.min(wp.min(v0,VUL),VLC))
    VMAX    = wp.min(wp.max(wp.max(v0,v1),VMD), wp.max(wp.max(v0,VUL),VLC))
    return VOR + _minmod2(VMIN-VOR, VMAX-VOR)

@wp.func
def mp5_ur(vm1:wp.float64, v0:wp.float64, v1:wp.float64,
           v2:wp.float64,  v3:wp.float64) -> wp.float64:
    # Right-biased MP5 at face i+1/2.  Mirror of mp5_ul.
    VOR  = F1_60*(NF3*vm1 + F27*v0 + F47*v1 + NF13*v2 + F2*v3)
    VMP  = v1 + _minmod2(v0-v1, MP5_ALPH*(v1-v2))
    if (VOR-v1)*(VOR-VMP) < MP5_EPS:
        return VOR
    DJM1    = vm1 - F2*v0  + v1
    DJ      = v0  - F2*v1  + v2
    DJP1    = v1  - F2*v2  + v3
    DM4JPH  = _minmod4(F4*DJ-DJP1, F4*DJP1-DJ, DJ, DJP1)
    DM4JMH  = _minmod4(F4*DJ-DJM1, F4*DJM1-DJ, DJ, DJM1)
    VUL     = v1 + MP5_ALPH*(v1-v2)
    VAV     = HALF*(v1+v0)
    VMD     = VAV - HALF*DM4JMH
    VLC     = v1 + HALF*(v1-v2) + MP5_B2*DM4JPH
    VMIN    = wp.max(wp.min(wp.min(v1,v0),VMD), wp.min(wp.min(v1,VUL),VLC))
    VMAX    = wp.min(wp.max(wp.max(v1,v0),VMD), wp.max(wp.max(v1,VUL),VLC))
    return VOR + _minmod2(VMIN-VOR, VMAX-VOR)

# =============================================================================
#  6th-order central reconstruction with adjustable upwind bias
# =============================================================================

@wp.func
def central6_ul(vm2:wp.float64, vm1:wp.float64, v0:wp.float64,
                v1:wp.float64,  v2:wp.float64,  v3:wp.float64,
                sai:wp.float64) -> wp.float64:
    # Left-side value at face i+1/2.
    # lb = left-biased 5th-order stencil;  rb = right-biased.
    # sai = 0.5  → pure central   (shear/entropy waves, Paper 11)
    # sai = F0p6 → η*_a = 0.6     (acoustic wave, interior faces, Paper 15)
    # sai = 1.0  → fully upwind   (boundary-adjacent faces)
    lb = F1_60*(F2*vm2 + NF13*vm1 + F47*v0 + F27*v1 + NF3*v2)
    rb = F1_60*(NF3*vm1 + F27*v0 + F47*v1 + NF13*v2 + F2*v3)
    return sai*lb + (F1-sai)*rb

@wp.func
def central6_ur(vm2:wp.float64, vm1:wp.float64, v0:wp.float64,
                v1:wp.float64,  v2:wp.float64,  v3:wp.float64,
                sai:wp.float64) -> wp.float64:
    # Right-side value at face i+1/2.  Roles of lb/rb swapped so that the
    # (ul, ur) pair is symmetric about the face.
    lb = F1_60*(F2*vm2 + NF13*vm1 + F47*v0 + F27*v1 + NF3*v2)
    rb = F1_60*(NF3*vm1 + F27*v0 + F47*v1 + NF13*v2 + F2*v3)
    return (F1-sai)*lb + sai*rb

# =============================================================================
#  3D Euler left eigenvectors L  (rows of L·q)
#
#  Row ordering for a face with outward normal n, tangents l and m:
#    row 0 (Lq0) — left-running  acoustic   λ = u_n - c
#    row 1 (Lq1) — shear wave l             λ = u_n,  tangent l
#    row 2 (Lq2) — shear wave m             λ = u_n,  tangent m
#    row 3 (Lq3) — entropy wave             λ = u_n,  carries density jump
#    row 4 (Lq4) — right-running acoustic   λ = u_n + c
#
#  Inputs: q1=ρ, q2=ρu, q3=ρv, q4=ρw, q5=ρE  (conservative, at one cell)
#  q2v = u²+v²+w²,  qn = u·n,  ql = u·l,  qm = u·m  (Roe averages)
# =============================================================================

@wp.func
def Lq0(q1:wp.float64, q2:wp.float64, q3:wp.float64, q4:wp.float64, q5:wp.float64,
        u:wp.float64, v:wp.float64, w:wp.float64, c:wp.float64, c2:wp.float64,
        nx:wp.float64, ny:wp.float64, nz:wp.float64,
        qn:wp.float64, q2v:wp.float64) -> wp.float64:
    # Left acoustic: combines the pressure fluctuation (κ/2c² terms on ρ and E)
    # with +qn/c to distinguish from the right-running wave (row 4).
    kap = GAMMA-F1
    return (HALF*(kap*HALF*q2v/c2+qn/c)*q1
            -HALF*(kap*u/c2+nx/c)*q2
            -HALF*(kap*v/c2+ny/c)*q3
            -HALF*(kap*w/c2+nz/c)*q4
            +kap/(F2*c2)*q5)

@wp.func
def Lq1(q1:wp.float64, q2:wp.float64, q3:wp.float64, q4:wp.float64, q5:wp.float64,
        lx:wp.float64, ly:wp.float64, lz:wp.float64, ql:wp.float64) -> wp.float64:
    # Shear wave l: projects momentum onto tangent l, no energy-row contribution.
    # The structural absence of q5 reflects the decoupling of shear waves from
    # thermodynamic variables (Paper 14, Sec. 3.3.3; Paper 15, Sec. 1).
    return -ql*q1 + lx*q2 + ly*q3 + lz*q4

@wp.func
def Lq2(q1:wp.float64, q2:wp.float64, q3:wp.float64, q4:wp.float64, q5:wp.float64,
        mx:wp.float64, my:wp.float64, mz:wp.float64, qm:wp.float64) -> wp.float64:
    # Shear wave m: same structure as Lq1 for the second tangent direction.
    return -qm*q1 + mx*q2 + my*q3 + mz*q4

@wp.func
def Lq3(q1:wp.float64, q2:wp.float64, q3:wp.float64, q4:wp.float64, q5:wp.float64,
        u:wp.float64, v:wp.float64, w:wp.float64,
        c2:wp.float64, q2v:wp.float64) -> wp.float64:
    # Entropy wave: extracts the isentropic density perturbation by subtracting
    # the acoustic pressure contribution.  This is the wave targeted by the
    # rank-1 MP5 correction in the central path (Paper 15, Sec. 5).
    kap = GAMMA-F1
    return ((F1-kap*q2v/(F2*c2))*q1
            +kap*u/c2*q2 + kap*v/c2*q3 + kap*w/c2*q4
            -kap/c2*q5)

@wp.func
def Lq4(q1:wp.float64, q2:wp.float64, q3:wp.float64, q4:wp.float64, q5:wp.float64,
        u:wp.float64, v:wp.float64, w:wp.float64, c:wp.float64, c2:wp.float64,
        nx:wp.float64, ny:wp.float64, nz:wp.float64,
        qn:wp.float64, q2v:wp.float64) -> wp.float64:
    # Right acoustic: mirror of Lq0 with -qn/c sign.
    kap = GAMMA-F1
    return (HALF*(kap*HALF*q2v/c2-qn/c)*q1
            -HALF*(kap*u/c2-nx/c)*q2
            -HALF*(kap*v/c2-ny/c)*q3
            -HALF*(kap*w/c2-nz/c)*q4
            +kap/(F2*c2)*q5)

# =============================================================================
#  3D Euler right eigenvectors R  (columns of R)
#
#  Inputs v0..v4: the five characteristic amplitudes from the reconstruction
#  (v0=left-acoustic, v1=shear-l, v2=shear-m, v3=entropy, v4=right-acoustic).
#  Used in the shock path to back-transform WENO-Z char. amplitudes to
#  conservative space.  The entropy column [1,u,v,w,½|u|²] is also the rank-1
#  update direction in the central path.
# =============================================================================

@wp.func
def Rv0(v0:wp.float64, v1:wp.float64, v2:wp.float64,
        v3:wp.float64, v4:wp.float64) -> wp.float64:
    # ρ: shear waves (v1, v2) carry no density perturbation.
    return v0+v3+v4

@wp.func
def Rv1(v0:wp.float64, v1:wp.float64, v2:wp.float64, v3:wp.float64, v4:wp.float64,
        u:wp.float64, c:wp.float64,
        nx:wp.float64, lx:wp.float64, mx:wp.float64) -> wp.float64:
    # ρu: acoustics carry ±c*nx normal momentum; shear waves carry tangential.
    return v0*(u-c*nx) + v1*lx + v2*mx + v3*u + v4*(u+c*nx)

@wp.func
def Rv2(v0:wp.float64, v1:wp.float64, v2:wp.float64, v3:wp.float64, v4:wp.float64,
        vv:wp.float64, c:wp.float64,
        ny:wp.float64, ly:wp.float64, my:wp.float64) -> wp.float64:
    # ρv
    return v0*(vv-c*ny) + v1*ly + v2*my + v3*vv + v4*(vv+c*ny)

@wp.func
def Rv3(v0:wp.float64, v1:wp.float64, v2:wp.float64, v3:wp.float64, v4:wp.float64,
        w:wp.float64, c:wp.float64,
        nz:wp.float64, lz:wp.float64, mz:wp.float64) -> wp.float64:
    # ρw
    return v0*(w-c*nz) + v1*lz + v2*mz + v3*w + v4*(w+c*nz)

@wp.func
def Rv4(v0:wp.float64, v1:wp.float64, v2:wp.float64, v3:wp.float64, v4:wp.float64,
        enth:wp.float64, qn:wp.float64, c:wp.float64,
        ql:wp.float64, qm:wp.float64, q2v:wp.float64) -> wp.float64:
    # ρE: acoustics carry H ± qn·c; entropy carries only kinetic energy ½|u|²
    # (no pressure perturbation at a contact); shear waves carry ql, qm.
    return v0*(enth-qn*c) + v1*ql + v2*qm + v3*(HALF*q2v) + v4*(enth+qn*c)

# =============================================================================
#  Characteristic projection  (one cell, one row of L)
# =============================================================================

@wp.func
def char_transform(cons:wp.array(dtype=wp.float64, ndim=4),
                   kx:int, ky:int, kz:int, row:int,
                   u_r:wp.float64, v_r:wp.float64, w_r:wp.float64,
                   c_r:wp.float64, c2:wp.float64,
                   nx:wp.float64, ny:wp.float64, nz:wp.float64,
                   lx:wp.float64, ly:wp.float64, lz:wp.float64,
                   mx:wp.float64, my:wp.float64, mz:wp.float64,
                   qn:wp.float64, ql:wp.float64, qm:wp.float64,
                   q2v:wp.float64) -> wp.float64:
    # Projects the conservative state at (kx,ky,kz) onto one characteristic
    # amplitude using the Roe-averaged left eigenvector row.
    # row: 0=left-acoustic, 1=shear-l, 2=shear-m, 3=entropy, 4=right-acoustic.
    # Called once per stencil point per wave (30 calls/face for a 6-point stencil
    # across all 5 waves).
    q1 = cons[0,kx,ky,kz]; q2 = cons[1,kx,ky,kz]; q3 = cons[2,kx,ky,kz]
    q4 = cons[3,kx,ky,kz]; q5 = cons[4,kx,ky,kz]
    if   row==0: return Lq0(q1,q2,q3,q4,q5, u_r,v_r,w_r, c_r,c2, nx,ny,nz, qn,q2v)
    elif row==1: return Lq1(q1,q2,q3,q4,q5, lx,ly,lz, ql)
    elif row==2: return Lq2(q1,q2,q3,q4,q5, mx,my,mz, qm)
    elif row==3: return Lq3(q1,q2,q3,q4,q5, u_r,v_r,w_r, c2,q2v)
    else:        return Lq4(q1,q2,q3,q4,q5, u_r,v_r,w_r, c_r,c2, nx,ny,nz, qn,q2v)

# =============================================================================
#  Positivity check
# =============================================================================

@wp.func
def is_physical(c0:wp.float64, c1:wp.float64, c2:wp.float64,
                c3:wp.float64, c4:wp.float64) -> int:
    # Returns 1 if (ρ, ρu, ρv, ρw, ρE) is admissible: ρ > 0 and e > 0.
    # Called by the flux kernels; if the high-order state is inadmissible
    # they fall back to first-order Godunov using cell-centre values.
    if c0 <= F0: return 0
    u_ = c1/c0; v_ = c2/c0; w_ = c3/c0
    if c4/c0 - HALF*(u_*u_+v_*v_+w_*w_) <= F0: return 0
    return 1
