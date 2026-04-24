"""
fluxes.py
---------
HLLC Riemann solver kernels for the three coordinate directions.

Each kernel computes the numerical flux at the right face (i+1/2) and the
left face (i-1/2) of each interior cell and accumulates the flux divergence
into the residual:

    resid[v,i,j,k] -= (F_{i+1/2} - F_{i-1/2}) / ds

Wave speed estimates uses HLLC; the contact speed SP is computed from the left/right states. 
"""

import warp as wp
from constants import GAMMA, GM1, HALF, TINY, F0, F1
from reconstruction import is_physical


# =============================================================================
#  S9-X  HLLC flux divergence — X direction
# =============================================================================

@wp.kernel
def flux_x_residual_kernel(
    cons:wp.array(dtype=wp.float64, ndim=4),
    consl:wp.array(dtype=wp.float64, ndim=4),
    consr:wp.array(dtype=wp.float64, ndim=4),
    resid:wp.array(dtype=wp.float64, ndim=4),
    dx:wp.float64,
    nx:int, ny:int, nz:int, gp:int
):
    i, j, k = wp.tid()
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
    den=rL0*(SL-uL)-rR0*(SR-uR)
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


# =============================================================================
#  S9-Y  HLLC flux divergence — Y direction
# =============================================================================

@wp.kernel
def flux_y_residual_kernel(
    cons:wp.array(dtype=wp.float64, ndim=4),
    consl:wp.array(dtype=wp.float64, ndim=4),
    consr:wp.array(dtype=wp.float64, ndim=4),
    resid:wp.array(dtype=wp.float64, ndim=4),
    dy:wp.float64,
    nx:int, ny:int, nz:int, gp:int
):
    i, j, k = wp.tid()
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


# =============================================================================
#  S9-Z  HLLC flux divergence — Z direction
# =============================================================================

@wp.kernel
def flux_z_residual_kernel(
    cons:wp.array(dtype=wp.float64, ndim=4),
    consl:wp.array(dtype=wp.float64, ndim=4),
    consr:wp.array(dtype=wp.float64, ndim=4),
    resid:wp.array(dtype=wp.float64, ndim=4),
    dz:wp.float64,
    nx:int, ny:int, nz:int, gp:int
):
    i, j, k = wp.tid()
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
