import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.rc('text', usetex=True)
plt.rc('font', family='arial')
plt.rcParams.update({'font.size': 12})

# ── Load ──────────────────────────────────────────────────────────────────────
fname = 'riem_003125.npz'
d     = np.load(fname)

t     = float(d['time'])
x     = d['x']        # (NX,)
y     = d['y']        # (NY,)
rho   = d['rho']      # (NX, NY)


# ── Schlieren: log(|grad rho| + 1) ───────────────────────────────────────────
drhodx = np.gradient(rho, x, axis=0)
drhody = np.gradient(rho, y, axis=1)
blah   = np.log(np.sqrt(drhodx**2 + drhody**2) + 1.0)

extent = [x.min(), x.max(), y.min(), y.max()]
XX, YY = np.meshgrid(x, y)   # (NY, NX) for contour

# ── Plot ──────────────────────────────────────────────────────────────────────
gs  = gridspec.GridSpec(1, 1)
fig = plt.figure(figsize=(6,6))
ax  = plt.subplot(gs[0])

# ax.imshow(blah.T, vmin=0, vmax=5, cmap=plt.cm.gray_r,
#           origin='lower', extent=extent, aspect='auto')

ax.contour(XX, YY, rho, 32, colors='k', linewidths=0.5)

# ax.set_xlim(0.0, 7.0)
# ax.set_ylim(0.0, 3.0)
ax.set_xlabel(r'\textbf{x}')
ax.set_ylabel(r'\textbf{y}')
# ax.set_title(rf'$t = {t:.3f}$')

fig.tight_layout(pad=0.3)
fig.savefig(fname.replace('.npz', '.png'), dpi=300, bbox_inches='tight', pad_inches=0.05)
print('done')
