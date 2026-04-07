# WA-Warp: Wave-Appropriate 3D Compressible Euler Solver on GPU

A high-performance 3D compressible Euler solver implemented in [NVIDIA Warp](https://github.com/NVIDIA/warp), based on the wave-appropriate reconstruction framework of Chamarthi et al. (2023–2026).

![TGV Q-criterion isosurface at t=10, coloured by vorticity magnitude](tgv_t10.00_Q0.3.png)
*Q-criterion isosurface (Q=0.3) coloured by vorticity magnitude |ω| for the inviscid Taylor-Green Vortex at t=10, N=512³.*

---

## Key Features

- **Wave-appropriate reconstruction** — each characteristic wave family (acoustic, entropy, vortical) is treated with its physically appropriate scheme
- **SoA memory layout** `cons[var, ix, iy, iz]` for coalesced GPU memory access. Needed some modifications.
- **SSP-RK3** time integration
- **Ducros sensor** for shock detection and for the contact discontinuity rank-1 correction is used. 

## Numerical Scheme

| Region | Acoustic waves | Entropy wave | Vortical waves |
|--------|---------------|--------------|----------------|
| Smooth | Upwind (`eta=0.6`) | MP5 | Central-6 (`kai=0.5`) |
| Shocked | WENO-Z | WENO-Z | WENO-Z |

In regions of shockwaves one can also do Wave appropriate centralization. Is not included in the current python code.

The wave-appropriate framework decomposes the flow into its five characteristic families and applies the minimum necessary dissipation to each:

- **Acoustic waves** — upwind-biased (η = 0.6) for stability near shocks
- **Entropy wave** — MP5 in smooth regions, WENO-Z (or MP5 or MUSCL) near shocks; rank-1 correction from WA-CR
- **Vortical waves** — central (η = 0.5) to preserve turbulent structures

## Installation

```bash
pip install warp-lang numpy matplotlib
```

## Usage

```bash
# Inviscid Taylor-Green Vortex, 64³
python 3D_TGV_WA.py.py

# 128³
python 3D_TGV_WA.py.py --n 128

# 512³ (A100 recommended)
python 3D_TGV_WA.py.py --n 512

# CPU mode
python 3D_TGV_WA.py.py --n 64 --cpu
```

## Output

Each snapshot is saved as a compressed `.npz` file containing:

```
time, x, y, z, rho, p, u, v, w
```

## Visualization

```bash
# Can use python scripts for plotting but the code does periodically. Can disable it if not required.
```

## Performance (A100 80GB)

| Grid | Memory | Time/step | Steps (t=10) | Wall time |
|------|--------|-----------|--------------|-----------|
| 64³  | ~0.5 GB | ~5 ms | ~1,500 | ~2 min |
| 512³ | ~62 GB | ~1.2 s | ~28,000 | ~9 hr |


The other two codes in the repository siualtes the 2D Riemann problem by using WA-3 or WA-WENO-CR schemes. Takes 11s on A100 using the WA-WENO-CR approach.

## References

1. Chamarthi, Hoffmann, Frankel — *A wave appropriate discontinuity sensor approach for compressible flows*, **Phys. Fluids** 35, 066107 (2023)
2. Hoffmann, Chamarthi, Frankel — *Centralized gradient-based reconstruction for wall-modelled LES of hypersonic boundary layer transition*, **J. Comput. Phys.** (2024)
3. Chamarthi et al. — *Wave-appropriate multidimensional upwinding approach for compressible multiphase flows*, **J. Comput. Phys.** 538, 114157 (2025)
4. Chamarthi — *Physics-appropriate interface capturing reconstruction approach for viscous compressible multicomponent flows*, **Comput. Fluids** 303, 106858 (2025)
5. Chamarthi — *Centralized gradient-based reconstruction: minimum acoustic upwind bias and rank-1 entropy correction*, preprint (2026)

## Author

**Amareshwara Sainadh Chamarthi**
