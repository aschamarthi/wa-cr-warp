"""
wcns.py
-------
Wave-appropriate reconstruction kernels for all three face directions.

Each kernel implements the two-path strategy from Paper 15:

  Central path  (combined Ducros sensor < 0.01):
    Physical-space reconstructions via central6 with wave-specific η:
      ρ      (entropy-type)      → HALF  (pure central)
      normal ρq  (acoustic)      → sai   (η*_a = 0.6 interior, 1.0 near boundary)
      tangential ρq (shear × 2)  → HALF
      ρE     (energy)            → HALF
    + Rank-1 entropy correction: MP5 reconstruction of the entropy
      characteristic amplitude drives a rank-1 update along r_entropy,
      eliminating the need for an explicit contact detector (Paper 15, Sec. 5).

  Shock path  (sensor ≥ 0.01):
    Full WENO-Z on all 5 characteristic waves, back-transform via R. Choose whatever you want here.

The "normal" momentum variable differs per sweep direction:
  wcns_x  n=(1,0,0): cons[1]=ρu  is acoustic → gets sai
  wcns_y  n=(0,1,0): cons[2]=ρv  is acoustic → gets sai
  wcns_z  n=(0,0,1): cons[3]=ρw  is acoustic → gets sai
"""

import warp as wp
from constants import (
    F0, F1, F2, HALF, TINY, GAMMA, GM1, F0p6, NF1,
)
from reconstruction import (
    weno5_ul, weno5_ur, mp5_ul, mp5_ur,
    central6_ul, central6_ur,
    char_transform,
    Rv0, Rv1, Rv2, Rv3, Rv4,
)
from sensors import pressure_sensor_x, pressure_sensor_y, pressure_sensor_z


# =============================================================================
#  Shared Roe-average helper (inlined into each kernel below)
# =============================================================================

@wp.func
def _roe_avg(cons:wp.array(dtype=wp.float64, ndim=4),
             iL:int, jL:int, kL:int,
             iR:int, jR:int, kR:int):
    # Roe-averaged primitives from two adjacent cells.
    # Returns (u_r, v_r, w_r, H_r, c_r, c2, rho_geom, p_roe).
    rL  = cons[0,iL,jL,kL]; rR  = cons[0,iR,jR,kR]
    ruL = cons[1,iL,jL,kL]; ruR = cons[1,iR,jR,kR]
    rvL = cons[2,iL,jL,kL]; rvR = cons[2,iR,jR,kR]
    rwL = cons[3,iL,jL,kL]; rwR = cons[3,iR,jR,kR]
    EL  = cons[4,iL,jL,kL]; ER  = cons[4,iR,jR,kR]
    sqL = wp.sqrt(rL); sqR = wp.sqrt(rR); dv = F1/(sqL+sqR)
    pL  = GM1*(EL - HALF*(ruL*ruL+rvL*rvL+rwL*rwL)/rL)
    pR  = GM1*(ER - HALF*(ruR*ruR+rvR*rvR+rwR*rwR)/rR)
    HL  = (EL+pL)/rL; HR = (ER+pR)/rR
    u_r = (sqL*(ruL/rL)+sqR*(ruR/rR))*dv
    v_r = (sqL*(rvL/rL)+sqR*(rvR/rR))*dv
    w_r = (sqL*(rwL/rL)+sqR*(rwR/rR))*dv
    H_r = (sqL*HL+sqR*HR)*dv
    rho_geom = sqL*sqR
    p_roe    = (sqL*pL+sqR*pR)*dv
    c2  = wp.max(GAMMA*p_roe/rho_geom, TINY)
    c_r = wp.sqrt(c2)
    return u_r, v_r, w_r, H_r, c_r, c2


# =============================================================================
#  S6  WCNS X   n=(1,0,0)  l=(0,0,1)  m=(0,-1,0)
# =============================================================================

@wp.kernel
def wcns_x_kernel(
    cons:wp.array(dtype=wp.float64, ndim=4),
    pres:wp.array(dtype=wp.float64, ndim=3),
    ducros:wp.array(dtype=wp.float64, ndim=3),
    consl:wp.array(dtype=wp.float64, ndim=4),
    consr:wp.array(dtype=wp.float64, ndim=4),
    nx:int, ny:int, nz:int, gp:int
):
    face_ix, iy, iz = wp.tid()
    if face_ix<gp-1 or face_ix>nx+gp: return
    if iy<gp+1 or iy>ny+gp: return
    if iz<gp+1 or iz>nz+gp: return

    # Combined two-component Ducros sensor at face and two neighbours.
    sig_max = wp.max(pressure_sensor_x(pres, ducros, face_ix-1, iy, iz),
              wp.max(pressure_sensor_x(pres, ducros, face_ix,   iy, iz),
                     pressure_sensor_x(pres, ducros, face_ix+1, iy, iz)))
    f1idx       = face_ix-gp
    use_central6 = sig_max < wp.float64(0.01)
    # Interior flag: faces within 5 cells of the domain boundary revert to
    # full upwind (sai=1) so the stencil stays inside the ghost layer.
    sai_interior = (f1idx > 5) and (f1idx < nx-5)

    # Roe-averaged state for eigenvector evaluation
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
    rho_geom=sqL*sqR
    p_roe=(sqL*pL+sqR*pR)*dv
    c2=wp.max(GAMMA*p_roe/rho_geom,TINY); c_r=wp.sqrt(c2)
    # X-face geometry: normal n=(1,0,0), tangents l=(0,0,1), m=(0,-1,0)
    # qn = u·n = u,  ql = u·l = w,  qm = u·m = -v
    NXV=F1; NYV=F0; NZV=F0; LX=F0; LY=F0; LZ=F1; MX=F0; MY=NF1; MZ=F0
    qn=u_r; ql=w_r; qm=-v_r

    if use_central6:
        # η*_a for interior faces; full upwind near boundaries
        sai = F0p6 if sai_interior else F1

        # Physical-space central reconstructions — wave-specific η
        # ρ (entropy-type): pure central
        ul0=central6_ul(cons[0,face_ix-2,iy,iz],cons[0,face_ix-1,iy,iz],cons[0,face_ix,iy,iz],cons[0,face_ix+1,iy,iz],cons[0,face_ix+2,iy,iz],cons[0,face_ix+3,iy,iz],HALF)
        ur0=central6_ur(cons[0,face_ix-2,iy,iz],cons[0,face_ix-1,iy,iz],cons[0,face_ix,iy,iz],cons[0,face_ix+1,iy,iz],cons[0,face_ix+2,iy,iz],cons[0,face_ix+3,iy,iz],HALF)
        # ρu (normal momentum, acoustic): upwind bias η*_a
        ul1=central6_ul(cons[1,face_ix-2,iy,iz],cons[1,face_ix-1,iy,iz],cons[1,face_ix,iy,iz],cons[1,face_ix+1,iy,iz],cons[1,face_ix+2,iy,iz],cons[1,face_ix+3,iy,iz],sai)
        ur1=central6_ur(cons[1,face_ix-2,iy,iz],cons[1,face_ix-1,iy,iz],cons[1,face_ix,iy,iz],cons[1,face_ix+1,iy,iz],cons[1,face_ix+2,iy,iz],cons[1,face_ix+3,iy,iz],sai)
        # ρv (tangential, shear wave): pure central
        ul2=central6_ul(cons[2,face_ix-2,iy,iz],cons[2,face_ix-1,iy,iz],cons[2,face_ix,iy,iz],cons[2,face_ix+1,iy,iz],cons[2,face_ix+2,iy,iz],cons[2,face_ix+3,iy,iz],HALF)
        ur2=central6_ur(cons[2,face_ix-2,iy,iz],cons[2,face_ix-1,iy,iz],cons[2,face_ix,iy,iz],cons[2,face_ix+1,iy,iz],cons[2,face_ix+2,iy,iz],cons[2,face_ix+3,iy,iz],HALF)
        # ρw (tangential, shear wave): pure central
        ul3=central6_ul(cons[3,face_ix-2,iy,iz],cons[3,face_ix-1,iy,iz],cons[3,face_ix,iy,iz],cons[3,face_ix+1,iy,iz],cons[3,face_ix+2,iy,iz],cons[3,face_ix+3,iy,iz],HALF)
        ur3=central6_ur(cons[3,face_ix-2,iy,iz],cons[3,face_ix-1,iy,iz],cons[3,face_ix,iy,iz],cons[3,face_ix+1,iy,iz],cons[3,face_ix+2,iy,iz],cons[3,face_ix+3,iy,iz],HALF)
        # ρE (energy): pure central
        ul4=central6_ul(cons[4,face_ix-2,iy,iz],cons[4,face_ix-1,iy,iz],cons[4,face_ix,iy,iz],cons[4,face_ix+1,iy,iz],cons[4,face_ix+2,iy,iz],cons[4,face_ix+3,iy,iz],HALF)
        ur4=central6_ur(cons[4,face_ix-2,iy,iz],cons[4,face_ix-1,iy,iz],cons[4,face_ix,iy,iz],cons[4,face_ix+1,iy,iz],cons[4,face_ix+2,iy,iz],cons[4,face_ix+3,iy,iz],HALF)

        # Rank-1 entropy correction (Paper 15, Sec. 5):
        # Extract entropy characteristic amplitude (row 3) at each stencil
        # point and reconstruct with MP5 → centL/centR.
        e_m2=char_transform(cons,face_ix-2,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_m1=char_transform(cons,face_ix-1,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_0 =char_transform(cons,face_ix  ,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_1 =char_transform(cons,face_ix+1,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_2 =char_transform(cons,face_ix+2,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_3 =char_transform(cons,face_ix+3,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        centL=weno5_ul(e_m2,e_m1,e_0,e_1,e_2); centR=weno5_ur(e_m1,e_0,e_1,e_2,e_3)

        # le0..le4: entropy row of L.
        # dL/dR: error between MP5 target and the entropy amplitude implied
        # by the current central physical-space states.  Applied as a rank-1
        # update along r_entropy = [1, u, v, w, ½|u|²]^T.
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
        # Shock path: full WENO-Z on all 5 characteristic waves, then back-transform.
        # Wave 0 (left acoustic)
        c_m2=char_transform(cons,face_ix-2,iy,iz,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_m1=char_transform(cons,face_ix-1,iy,iz,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_0 =char_transform(cons,face_ix  ,iy,iz,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_1 =char_transform(cons,face_ix+1,iy,iz,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_2 =char_transform(cons,face_ix+2,iy,iz,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_3 =char_transform(cons,face_ix+3,iy,iz,0,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        ul0=weno5_ul(c_m2,c_m1,c_0,c_1,c_2); ur0=weno5_ur(c_m1,c_0,c_1,c_2,c_3)
        # Wave 1 (shear-l)
        c_m2=char_transform(cons,face_ix-2,iy,iz,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_m1=char_transform(cons,face_ix-1,iy,iz,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_0 =char_transform(cons,face_ix  ,iy,iz,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_1 =char_transform(cons,face_ix+1,iy,iz,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_2 =char_transform(cons,face_ix+2,iy,iz,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_3 =char_transform(cons,face_ix+3,iy,iz,1,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        ul1=weno5_ul(c_m2,c_m1,c_0,c_1,c_2); ur1=weno5_ur(c_m1,c_0,c_1,c_2,c_3)
        # Wave 2 (shear-m)
        c_m2=char_transform(cons,face_ix-2,iy,iz,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_m1=char_transform(cons,face_ix-1,iy,iz,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_0 =char_transform(cons,face_ix  ,iy,iz,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_1 =char_transform(cons,face_ix+1,iy,iz,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_2 =char_transform(cons,face_ix+2,iy,iz,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_3 =char_transform(cons,face_ix+3,iy,iz,2,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        ul2=weno5_ul(c_m2,c_m1,c_0,c_1,c_2); ur2=weno5_ur(c_m1,c_0,c_1,c_2,c_3)
        # Wave 3 (entropy)
        c_m2=char_transform(cons,face_ix-2,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_m1=char_transform(cons,face_ix-1,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_0 =char_transform(cons,face_ix  ,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_1 =char_transform(cons,face_ix+1,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_2 =char_transform(cons,face_ix+2,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_3 =char_transform(cons,face_ix+3,iy,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        ul3=weno5_ul(c_m2,c_m1,c_0,c_1,c_2); ur3=weno5_ur(c_m1,c_0,c_1,c_2,c_3)
        # Wave 4 (right acoustic)
        c_m2=char_transform(cons,face_ix-2,iy,iz,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_m1=char_transform(cons,face_ix-1,iy,iz,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_0 =char_transform(cons,face_ix  ,iy,iz,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_1 =char_transform(cons,face_ix+1,iy,iz,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_2 =char_transform(cons,face_ix+2,iy,iz,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        c_3 =char_transform(cons,face_ix+3,iy,iz,4,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        ul4=weno5_ul(c_m2,c_m1,c_0,c_1,c_2); ur4=weno5_ur(c_m1,c_0,c_1,c_2,c_3)
        # Back-transform: multiply by R to recover conservative left/right states.
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
#  S7  WCNS Y   n=(0,1,0)  l=(0,0,1)  m=(1,0,0)
#
#  ρv (cons[2]) is the normal-momentum (acoustic) variable → gets sai=η*_a.
#  cons[1]=ρu and cons[3]=ρw are tangential (shear) → HALF.
# =============================================================================

@wp.kernel
def wcns_y_kernel(
    cons:wp.array(dtype=wp.float64, ndim=4),
    pres:wp.array(dtype=wp.float64, ndim=3),
    ducros:wp.array(dtype=wp.float64, ndim=3),
    consl:wp.array(dtype=wp.float64, ndim=4),
    consr:wp.array(dtype=wp.float64, ndim=4),
    nx:int, ny:int, nz:int, gp:int
):
    ix, face_iy, iz = wp.tid()
    if ix<gp+1 or ix>nx+gp: return
    if face_iy<gp-1 or face_iy>ny+gp: return
    if iz<gp+1 or iz>nz+gp: return

    sig_max=wp.max(pressure_sensor_y(pres,ducros,ix,face_iy-1,iz),
             wp.max(pressure_sensor_y(pres,ducros,ix,face_iy  ,iz),
                    pressure_sensor_y(pres,ducros,ix,face_iy+1,iz)))
    f1idx        = face_iy-gp
    use_central6 = sig_max < wp.float64(0.01)
    sai_interior = (f1idx > 5) and (f1idx < ny-5)

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
    rho_geom=sqL*sqR
    p_roe=(sqL*pL+sqR*pR)*dv
    c2=wp.max(GAMMA*p_roe/rho_geom,TINY); c_r=wp.sqrt(c2)
    # Y-face geometry: normal n=(0,1,0), tangents l=(0,0,1), m=(1,0,0)
    NXV=F0; NYV=F1; NZV=F0; LX=F0; LY=F0; LZ=F1; MX=F1; MY=F0; MZ=F0
    qn=v_r; ql=w_r; qm=u_r

    if use_central6:
        sai = F0p6 if sai_interior else F1
        # ρ: pure central
        ul0=central6_ul(cons[0,ix,face_iy-2,iz],cons[0,ix,face_iy-1,iz],cons[0,ix,face_iy,iz],cons[0,ix,face_iy+1,iz],cons[0,ix,face_iy+2,iz],cons[0,ix,face_iy+3,iz],HALF)
        ur0=central6_ur(cons[0,ix,face_iy-2,iz],cons[0,ix,face_iy-1,iz],cons[0,ix,face_iy,iz],cons[0,ix,face_iy+1,iz],cons[0,ix,face_iy+2,iz],cons[0,ix,face_iy+3,iz],HALF)
        # ρu (tangential, shear): pure central
        ul1=central6_ul(cons[1,ix,face_iy-2,iz],cons[1,ix,face_iy-1,iz],cons[1,ix,face_iy,iz],cons[1,ix,face_iy+1,iz],cons[1,ix,face_iy+2,iz],cons[1,ix,face_iy+3,iz],HALF)
        ur1=central6_ur(cons[1,ix,face_iy-2,iz],cons[1,ix,face_iy-1,iz],cons[1,ix,face_iy,iz],cons[1,ix,face_iy+1,iz],cons[1,ix,face_iy+2,iz],cons[1,ix,face_iy+3,iz],HALF)
        # ρv (normal momentum, acoustic): η*_a
        ul2=central6_ul(cons[2,ix,face_iy-2,iz],cons[2,ix,face_iy-1,iz],cons[2,ix,face_iy,iz],cons[2,ix,face_iy+1,iz],cons[2,ix,face_iy+2,iz],cons[2,ix,face_iy+3,iz],sai)
        ur2=central6_ur(cons[2,ix,face_iy-2,iz],cons[2,ix,face_iy-1,iz],cons[2,ix,face_iy,iz],cons[2,ix,face_iy+1,iz],cons[2,ix,face_iy+2,iz],cons[2,ix,face_iy+3,iz],sai)
        # ρw (tangential, shear): pure central
        ul3=central6_ul(cons[3,ix,face_iy-2,iz],cons[3,ix,face_iy-1,iz],cons[3,ix,face_iy,iz],cons[3,ix,face_iy+1,iz],cons[3,ix,face_iy+2,iz],cons[3,ix,face_iy+3,iz],HALF)
        ur3=central6_ur(cons[3,ix,face_iy-2,iz],cons[3,ix,face_iy-1,iz],cons[3,ix,face_iy,iz],cons[3,ix,face_iy+1,iz],cons[3,ix,face_iy+2,iz],cons[3,ix,face_iy+3,iz],HALF)
        # ρE: pure central
        ul4=central6_ul(cons[4,ix,face_iy-2,iz],cons[4,ix,face_iy-1,iz],cons[4,ix,face_iy,iz],cons[4,ix,face_iy+1,iz],cons[4,ix,face_iy+2,iz],cons[4,ix,face_iy+3,iz],HALF)
        ur4=central6_ur(cons[4,ix,face_iy-2,iz],cons[4,ix,face_iy-1,iz],cons[4,ix,face_iy,iz],cons[4,ix,face_iy+1,iz],cons[4,ix,face_iy+2,iz],cons[4,ix,face_iy+3,iz],HALF)
        e_m2=char_transform(cons,ix,face_iy-2,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_m1=char_transform(cons,ix,face_iy-1,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_0 =char_transform(cons,ix,face_iy  ,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_1 =char_transform(cons,ix,face_iy+1,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_2 =char_transform(cons,ix,face_iy+2,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        e_3 =char_transform(cons,ix,face_iy+3,iz,3,u_r,v_r,w_r,c_r,c2,NXV,NYV,NZV,LX,LY,LZ,MX,MY,MZ,qn,ql,qm,q2v)
        centL=weno5_ul(e_m2,e_m1,e_0,e_1,e_2); centR=weno5_ur(e_m1,e_0,e_1,e_2,e_3)
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
#  S8  WCNS Z   n=(0,0,1)  l=(0,-1,0)  m=(1,0,0)
#
#  ρw (cons[3]) is the normal-momentum (acoustic) variable → gets sai=η*_a.
#  cons[1]=ρu and cons[2]=ρv are tangential (shear) → HALF.
# =============================================================================

@wp.kernel
def wcns_z_kernel(
    cons:wp.array(dtype=wp.float64, ndim=4),
    pres:wp.array(dtype=wp.float64, ndim=3),
    ducros:wp.array(dtype=wp.float64, ndim=3),
    consl:wp.array(dtype=wp.float64, ndim=4),
    consr:wp.array(dtype=wp.float64, ndim=4),
    nx:int, ny:int, nz:int, gp:int
):
    ix, iy, face_iz = wp.tid()
    if ix<gp+1 or ix>nx+gp: return
    if iy<gp+1 or iy>ny+gp: return
    if face_iz<gp-1 or face_iz>nz+gp: return

    sig_max=wp.max(pressure_sensor_z(pres,ducros,ix,iy,face_iz-1),
             wp.max(pressure_sensor_z(pres,ducros,ix,iy,face_iz  ),
                    pressure_sensor_z(pres,ducros,ix,iy,face_iz+1)))
    f1idx        = face_iz-gp
    use_central6 = sig_max < wp.float64(0.01)
    sai_interior = (f1idx > 5) and (f1idx < nz-5)

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
    rho_geom=sqL*sqR
    p_roe=(sqL*pL+sqR*pR)*dv
    c2=wp.max(GAMMA*p_roe/rho_geom,TINY); c_r=wp.sqrt(c2)
    # Z-face geometry: normal n=(0,0,1), tangents l=(0,-1,0), m=(1,0,0)
    NXV=F0; NYV=F0; NZV=F1; LX=F0; LY=NF1; LZ=F0; MX=F1; MY=F0; MZ=F0
    qn=w_r; ql=-v_r; qm=u_r

    if use_central6:
        sai = F0p6 if sai_interior else F1
        # ρ: pure central
        ul0=central6_ul(cons[0,ix,iy,face_iz-2],cons[0,ix,iy,face_iz-1],cons[0,ix,iy,face_iz],cons[0,ix,iy,face_iz+1],cons[0,ix,iy,face_iz+2],cons[0,ix,iy,face_iz+3],HALF)
        ur0=central6_ur(cons[0,ix,iy,face_iz-2],cons[0,ix,iy,face_iz-1],cons[0,ix,iy,face_iz],cons[0,ix,iy,face_iz+1],cons[0,ix,iy,face_iz+2],cons[0,ix,iy,face_iz+3],HALF)
        # ρu (tangential, shear): pure central
        ul1=central6_ul(cons[1,ix,iy,face_iz-2],cons[1,ix,iy,face_iz-1],cons[1,ix,iy,face_iz],cons[1,ix,iy,face_iz+1],cons[1,ix,iy,face_iz+2],cons[1,ix,iy,face_iz+3],HALF)
        ur1=central6_ur(cons[1,ix,iy,face_iz-2],cons[1,ix,iy,face_iz-1],cons[1,ix,iy,face_iz],cons[1,ix,iy,face_iz+1],cons[1,ix,iy,face_iz+2],cons[1,ix,iy,face_iz+3],HALF)
        # ρv (tangential, shear): pure central
        ul2=central6_ul(cons[2,ix,iy,face_iz-2],cons[2,ix,iy,face_iz-1],cons[2,ix,iy,face_iz],cons[2,ix,iy,face_iz+1],cons[2,ix,iy,face_iz+2],cons[2,ix,iy,face_iz+3],HALF)
        ur2=central6_ur(cons[2,ix,iy,face_iz-2],cons[2,ix,iy,face_iz-1],cons[2,ix,iy,face_iz],cons[2,ix,iy,face_iz+1],cons[2,ix,iy,face_iz+2],cons[2,ix,iy,face_iz+3],HALF)
        # ρw (normal momentum, acoustic): η*_a
        ul3=central6_ul(cons[3,ix,iy,face_iz-2],cons[3,ix,iy,face_iz-1],cons[3,ix,iy,face_iz],cons[3,ix,iy,face_iz+1],cons[3,ix,iy,face_iz+2],cons[3,ix,iy,face_iz+3],sai)
        ur3=central6_ur(cons[3,ix,iy,face_iz-2],cons[3,ix,iy,face_iz-1],cons[3,ix,iy,face_iz],cons[3,ix,iy,face_iz+1],cons[3,ix,iy,face_iz+2],cons[3,ix,iy,face_iz+3],sai)
        # ρE: pure central
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
