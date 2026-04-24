"""
diagnostics.py
--------------
Post-processing utilities: file I/O and TGV benchmark diagnostics.

  save_npz         — compressed NumPy snapshot (interior domain only)
  compute_ke       — volume-summed kinetic energy  KE = Σ ½ρ|u|²
  compute_enstrophy — volume-summed enstrophy       Ω = Σ |ω|²
  save_png_slice   — density contour plot at mid-z plane

KE and enstrophy are the two primary diagnostics for the TGV benchmark:
a scheme with low numerical dissipation preserves enstrophy longer and
achieves a higher peak value — the defining figure of merit in Paper rank 1.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
import numpy as np
from constants import NX, NY, NZ, G


def save_npz(rho_np, u_np, v_np, w_np, p_np,
             x_np, y_np, z_np, step, time_sim):
    # Strips ghost layers and saves interior fields as a compressed archive.
    fname = f"tgv_{step:06d}.npz"
    np.savez_compressed(fname,
        time=np.float64(time_sim),
        x=x_np[G:G+NX], y=y_np[G:G+NY], z=z_np[G:G+NZ],
        rho=rho_np[G:G+NX, G:G+NY, G:G+NZ],
        p=p_np[G:G+NX,   G:G+NY, G:G+NZ],
        u=u_np[G:G+NX,   G:G+NY, G:G+NZ],
        v=v_np[G:G+NX,   G:G+NY, G:G+NZ],
        w=w_np[G:G+NX,   G:G+NY, G:G+NZ])
    print(f"  -> {fname}")


def compute_ke(rho_np, u_np, v_np, w_np):
    # Volume-averaged KE = Σ ½ρ|u|² over interior cells (unnormalised).
    # Normalised against the initial value in the time-series output.
    r = rho_np[G:G+NX, G:G+NY, G:G+NZ]
    u = u_np[G:G+NX,   G:G+NY, G:G+NZ]
    v = v_np[G:G+NX,   G:G+NY, G:G+NZ]
    w = w_np[G:G+NX,   G:G+NY, G:G+NZ]
    return float(np.sum(r*0.5*(u*u+v*v+w*w)))


def compute_enstrophy(u_np, v_np, w_np, dx_py, dy_py, dz_py):
    # Volume-averaged enstrophy Ω = Σ (ωx²+ωy²+ωz²), computed with
    # numpy.gradient (2nd-order central). ~!!!!! Could do high order but decided not to
    u = u_np[G:G+NX, G:G+NY, G:G+NZ]
    v = v_np[G:G+NX, G:G+NY, G:G+NZ]
    w = w_np[G:G+NX, G:G+NY, G:G+NZ]
    dwdy = np.gradient(w, dy_py, axis=1); dvdz = np.gradient(v, dz_py, axis=2)
    dudz = np.gradient(u, dz_py, axis=2); dwdx = np.gradient(w, dx_py, axis=0)
    dvdx = np.gradient(v, dx_py, axis=0); dudy = np.gradient(u, dy_py, axis=1)
    ox = dwdy-dvdz; oy = dudz-dwdx; oz = dvdx-dudy
    return float(np.sum(ox*ox+oy*oy+oz*oz))


def save_png_slice(rho_np, step, time_sim):
    # Density contour at the mid-z plane (iz = NZ//2).
    d = rho_np[G:G+NX, G:G+NY, G+NZ//2]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(d.T, 32, cmap='viridis')
    ax.set_title(f'rho z-slice  t={time_sim:.3f}')
    ax.set_aspect('equal')
    fname = f"tgv_{step:06d}.png"
    fig.savefig(fname, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {fname}")
