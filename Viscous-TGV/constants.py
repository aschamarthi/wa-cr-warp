"""
constants.py
------------
CLI argument parsing, domain/run parameters, and all Warp float64 typed
constants shared across every kernel module.

It has been a pain in the neck in the bebinging to store everyhting in float64 as I was learning from Warp's example codes.

wp.init() is called here so that importing this module is sufficient to
initialise the Warp runtime before any @wp.func / @wp.kernel decorator runs.
"""

import argparse, math
import warp as wp

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--cpu",       action="store_true")
parser.add_argument("--n",         type=int,   default=64)
parser.add_argument("--re",        type=float, default=1600.0)
parser.add_argument("--pr",        type=float, default=0.73)
parser.add_argument("--viscosity", default="constant",
                    choices=["constant", "sutherland"])
args, _ = parser.parse_known_args()

DEVICE   = "cpu" if args.cpu else "cuda"
RE_PY    = args.re
PR_PY    = args.pr
USE_SUTH = 1 if args.viscosity == "sutherland" else 0

# Initialise Warp before any constant / kernel declarations.
wp.init()

# ── Domain / run parameters ───────────────────────────────────────────────────
NX = NY = NZ = args.n

# GHOSTP=5 supports the 6-point stencil of the 5th-order WENO-Z / MP5
# reconstructions.  The 8th-order viscous derivative (commented out in
# viscous.py) would also fit within this halo.
GHOSTP = 5
GX = NX + 2*GHOSTP + 1
GY = NY + 2*GHOSTP + 1
GZ = NZ + 2*GHOSTP + 1
G  = GHOSTP

NTMAX     = 800_000
FILE_SAVE = 40_000
CFL_PY    = 0.4
T_END     = 10.0

PI_PY = math.pi
XMIN_PY, XMAX_PY = 0.0, 2.0*PI_PY
YMIN_PY, YMAX_PY = 0.0, 2.0*PI_PY
ZMIN_PY, ZMAX_PY = 0.0, 2.0*PI_PY

# ── Warp float64 constants ────────────────────────────────────────────────────
# Pre-typed constants avoid repeated casts inside GPU kernels.

GAMMA  = wp.constant(wp.float64(5.0/3.0))   # ratio of specific heats (monatomic ideal gas)
GM1    = wp.constant(wp.float64(2.0/3.0))    # gamma - 1
TINY   = wp.constant(wp.float64(1.0e-30))    # floor for divisions / sensor denominators
F0     = wp.constant(wp.float64(0.0))
F1     = wp.constant(wp.float64(1.0))
F2     = wp.constant(wp.float64(2.0))
HALF   = wp.constant(wp.float64(0.5))        # η = 0.5: pure central scheme
QUART  = wp.constant(wp.float64(0.25))
F5_6   = wp.constant(wp.float64(5.0/6.0))
F1_3   = wp.constant(wp.float64(1.0/3.0))

# η*_a = 0.601 — optimised acoustic upwind bias (Paper 15, Sec. 4).
# Applied only to the normal-momentum (acoustic) variable in smooth regions;
# all other waves use HALF.  Faces within 5 cells of the boundary revert to
# F1 (full upwind) to keep the stencil inside the ghost layer.
F0p6   = wp.constant(wp.float64(0.6))

NF1    = wp.constant(wp.float64(-1.0))
NF1_6  = wp.constant(wp.float64(-1.0/6.0))
NF3    = wp.constant(wp.float64(-3.0))
NF13   = wp.constant(wp.float64(-13.0))
F3     = wp.constant(wp.float64(3.0))
F4     = wp.constant(wp.float64(4.0))

# WENO-Z / MP5 stencil coefficients
F7_6   = wp.constant(wp.float64(7.0/6.0))
NF7_6  = wp.constant(wp.float64(-7.0/6.0))
F11_6  = wp.constant(wp.float64(11.0/6.0))
F13_12 = wp.constant(wp.float64(13.0/12.0))

# WENO-Z ideal weights (left-biased: d0=3/10, d1=6/10, d2=1/10)
F3_10  = wp.constant(wp.float64(3.0/10.0))
F6_10  = wp.constant(wp.float64(6.0/10.0))
F1_10  = wp.constant(wp.float64(1.0/10.0))

# Leading coefficient of the 5th-order central stencil (1/60)
F1_60  = wp.constant(wp.float64(1.0/60.0))
F27    = wp.constant(wp.float64(27.0))
F47    = wp.constant(wp.float64(47.0))

WENO_EPS = wp.constant(wp.float64(1.0e-40))  # prevents division by zero in smoothness weights
DTBIG    = wp.constant(wp.float64(1.0e30))   # initial value for GPU atomic-min dt reduction

# 8th-order antisymmetric stencil coefficients (Fortran vf code - I am not using it though, commented out).
# der = +C8_1*f(i-4) -C8_2*f(i-3) +C8_3*f(i-2) -C8_4*f(i-1) + ... (no /dx)
# Retained for reference; the active viscous kernels use 2nd-order stencils.
C8_1 = wp.constant(wp.float64(1.0/280.0))
C8_2 = wp.constant(wp.float64(4.0/105.0))
C8_3 = wp.constant(wp.float64(1.0/5.0))
C8_4 = wp.constant(wp.float64(4.0/5.0))

# ── Viscous / thermodynamic constants ─────────────────────────────────────────
RE     = wp.constant(wp.float64(RE_PY))
PR     = wp.constant(wp.float64(PR_PY))
SUTH_C = wp.constant(wp.float64(110.4/273.15))  # Sutherland reference temperature ratio
F4_3   = wp.constant(wp.float64(4.0/3.0))

# ── MP5 limiter constants (Suresh & Huynh 1997) ───────────────────────────────
# Used exclusively for the entropy-wave rank-1 correction (Paper 15, Sec. 5).
MP5_B2   = wp.constant(wp.float64(4.0/3.0))
MP5_ALPH = wp.constant(wp.float64(4.0))
MP5_EPS  = wp.constant(wp.float64(1.0e-40))
