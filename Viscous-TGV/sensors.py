"""
sensors.py
----------
The two-component Ducros shock sensor (Paper 15, Remark 1.1):

  combined = |Δ²p| / |Σp|  ×  div² / (div² + curl² + ε)

The first factor (Jameson pressure indicator) and the second factor
(dilatation-to-vorticity ratio) are multiplied together inside
pressure_sensor_x/y/z.  Using both components prevents the dilatation
ratio alone from misidentifying intense vortical regions as shocks, and
allows a fixed threshold of 0.01 to be used across all test cases without
case-by-case tuning.

ducros_kernel stores only the dilatation/vorticity ratio (second component)
for each cell.  The Jameson pressure term is applied directionally at each
face inside the pressure_sensor functions, which avoids recomputing the
curl for every face in the reconstruction kernels.
"""

import warp as wp
from constants import TINY, HALF

# =============================================================================
#  Jameson pressure indicator × Ducros dilatation/vorticity ratio
#  (one function per sweep direction, evaluated at a single cell index)
# =============================================================================

@wp.func
def pressure_sensor_x(pres:wp.array(dtype=wp.float64, ndim=3),
                      ducros:wp.array(dtype=wp.float64, ndim=3),
                      kx:int, iy:int, iz:int) -> wp.float64:
    F16 = wp.float64(16.0); F30 = wp.float64(30.0)
    aa = wp.abs(-pres[kx-2,iy,iz] + F16*pres[kx-1,iy,iz] - F30*pres[kx,iy,iz]
                +F16*pres[kx+1,iy,iz] - pres[kx+2,iy,iz])
    bb = wp.abs( pres[kx-2,iy,iz] + F16*pres[kx-1,iy,iz] + F30*pres[kx,iy,iz]
                +F16*pres[kx+1,iy,iz] + pres[kx+2,iy,iz])
    return aa/(bb+TINY) * ducros[kx,iy,iz]

@wp.func
def pressure_sensor_y(pres:wp.array(dtype=wp.float64, ndim=3),
                      ducros:wp.array(dtype=wp.float64, ndim=3),
                      ix:int, ky:int, iz:int) -> wp.float64:
    F16 = wp.float64(16.0); F30 = wp.float64(30.0)
    aa = wp.abs(-pres[ix,ky-2,iz] + F16*pres[ix,ky-1,iz] - F30*pres[ix,ky,iz]
                +F16*pres[ix,ky+1,iz] - pres[ix,ky+2,iz])
    bb = wp.abs( pres[ix,ky-2,iz] + F16*pres[ix,ky-1,iz] + F30*pres[ix,ky,iz]
                +F16*pres[ix,ky+1,iz] + pres[ix,ky+2,iz])
    return aa/(bb+TINY) * ducros[ix,ky,iz]

@wp.func
def pressure_sensor_z(pres:wp.array(dtype=wp.float64, ndim=3),
                      ducros:wp.array(dtype=wp.float64, ndim=3),
                      ix:int, iy:int, kz:int) -> wp.float64:
    F16 = wp.float64(16.0); F30 = wp.float64(30.0)
    aa = wp.abs(-pres[ix,iy,kz-2] + F16*pres[ix,iy,kz-1] - F30*pres[ix,iy,kz]
                +F16*pres[ix,iy,kz+1] - pres[ix,iy,kz+2])
    bb = wp.abs( pres[ix,iy,kz-2] + F16*pres[ix,iy,kz-1] + F30*pres[ix,iy,kz]
                +F16*pres[ix,iy,kz+1] + pres[ix,iy,kz+2])
    return aa/(bb+TINY) * ducros[ix,iy,kz]

# =============================================================================
#  Velocity gradient kernel  (2nd-order central, no /ds)
# =============================================================================

@wp.kernel
def velocity_deriv_kernel(
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
    # All 9 velocity gradient components by 2nd-order central differences.
    # /ds is not applied here; it is incorporated inside the ducros and
    # viscous flux kernels.  These gradients are shared between both
    # (a computational advantage of the gradient-based reconstruction
    # family, noted in Paper 10).
    ti, tj, tk = wp.tid()
    ix = ti+1; iy = tj+1; iz = tk+1
    if ix>=gx-1 or iy>=gy-1 or iz>=gz-1: return
    der_ux[ix,iy,iz] = HALF*(-u_vel[ix-1,iy,iz]+u_vel[ix+1,iy,iz])
    der_uy[ix,iy,iz] = HALF*(-u_vel[ix,iy-1,iz]+u_vel[ix,iy+1,iz])
    der_uz[ix,iy,iz] = HALF*(-u_vel[ix,iy,iz-1]+u_vel[ix,iy,iz+1])
    der_vx[ix,iy,iz] = HALF*(-v_vel[ix-1,iy,iz]+v_vel[ix+1,iy,iz])
    der_vy[ix,iy,iz] = HALF*(-v_vel[ix,iy-1,iz]+v_vel[ix,iy+1,iz])
    der_vz[ix,iy,iz] = HALF*(-v_vel[ix,iy,iz-1]+v_vel[ix,iy,iz+1])
    der_wx[ix,iy,iz] = HALF*(-w_vel[ix-1,iy,iz]+w_vel[ix+1,iy,iz])
    der_wy[ix,iy,iz] = HALF*(-w_vel[ix,iy-1,iz]+w_vel[ix,iy+1,iz])
    der_wz[ix,iy,iz] = HALF*(-w_vel[ix,iy,iz-1]+w_vel[ix,iy,iz+1])

# =============================================================================
#  Ducros dilatation/vorticity ratio kernel
# =============================================================================

@wp.kernel
def ducros_kernel(
    der_ux:wp.array(dtype=wp.float64, ndim=3), der_uy:wp.array(dtype=wp.float64, ndim=3),
    der_uz:wp.array(dtype=wp.float64, ndim=3), der_vx:wp.array(dtype=wp.float64, ndim=3),
    der_vy:wp.array(dtype=wp.float64, ndim=3), der_vz:wp.array(dtype=wp.float64, ndim=3),
    der_wx:wp.array(dtype=wp.float64, ndim=3), der_wy:wp.array(dtype=wp.float64, ndim=3),
    der_wz:wp.array(dtype=wp.float64, ndim=3),
    ducros:wp.array(dtype=wp.float64, ndim=3),
    gx:int, gy:int, gz:int
):
    # Stores div²/(div²+curl²+ε) at each interior cell.  This is the second
    # component of the full Ducros sensor; the Jameson pressure factor is
    # applied per-direction inside pressure_sensor_x/y/z.
    ti, tj, tk = wp.tid()
    ix = ti+1; iy = tj+1; iz = tk+1
    if ix>=gx-1 or iy>=gy-1 or iz>=gz-1: return
    div   = der_ux[ix,iy,iz]+der_vy[ix,iy,iz]+der_wz[ix,iy,iz]
    cx    = der_wy[ix,iy,iz]-der_vz[ix,iy,iz]
    cy    = der_uz[ix,iy,iz]-der_wx[ix,iy,iz]
    cz    = der_vx[ix,iy,iz]-der_uy[ix,iy,iz]
    curl2 = cx*cx+cy*cy+cz*cz
    ducros[ix,iy,iz] = div*div/(div*div+curl2+TINY)
