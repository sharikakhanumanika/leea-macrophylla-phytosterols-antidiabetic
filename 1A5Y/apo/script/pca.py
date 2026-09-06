import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

def read_xvg(filename):
    data = []

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('@') or line.startswith('#'):
                continue

            parts = line.split()

            if len(parts) >= 2:
                data.append(float(parts[1]))

    return np.array(data)

def get_eigen_percentages(filename):

    eigvals = []

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('@') or line.startswith('#'):
                continue

            parts = line.split()

            if len(parts) >= 2:
                eigvals.append(float(parts[1]))

    eigvals = np.array(eigvals)

    if len(eigvals) == 0:
        return [0, 0, 0]

    perc = eigvals[:3] / np.sum(eigvals) * 100

    return perc

# ---------- Read Data ----------

pc1 = read_xvg("pc1.xvg")
pc2 = read_xvg("pc2.xvg")
pc3 = read_xvg("pc3.xvg")

perc = get_eigen_percentages("eigenval.xvg")

# ---------- Diagnostics ----------

print("PC1:", len(pc1), np.min(pc1), np.max(pc1))
print("PC2:", len(pc2), np.min(pc2), np.max(pc2))
print("PC3:", len(pc3), np.min(pc3), np.max(pc3))

# ---------- Check Length ----------

n = min(len(pc1), len(pc2), len(pc3))

pc1 = pc1[:n]
pc2 = pc2[:n]
pc3 = pc3[:n]

# ---------- Vibrant Color Map ----------

cmap = LinearSegmentedColormap.from_list(
    "trajectory",
    [
        "#0b2c6b",  # deep blue
        "#00bcd4",  # cyan
        "#f4d03f",  # yellow
        "#f39c12",  # orange
        "#c0392b"   # red
    ]
)

trajectory = np.linspace(0, 1, n)

# ---------- Plot ----------

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    pc1,
    pc2,
    pc3,
    c=trajectory,
    cmap=cmap,
    s=12,
    alpha=0.85
)

ax.set_xlabel(
    f'PC1 ({perc[0]:.2f}%)',
    fontsize=14,
    fontweight='bold'
)

ax.set_ylabel(
    f'PC2 ({perc[1]:.2f}%)',
    fontsize=14,
    fontweight='bold'
)

ax.set_zlabel(
    f'PC3 ({perc[2]:.2f}%)',
    fontsize=14,
    fontweight='bold'
)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.grid(True, linestyle='--', alpha=0.4)

cbar = plt.colorbar(
    scatter,
    shrink=0.75,
    pad=0.08
)

cbar.set_label(
    'Simulation Progress',
    fontsize=12,
    fontweight='bold'
)

plt.tight_layout()

plt.savefig(
    "1A5Y.tif",
    dpi=600,
    bbox_inches='tight',
    pil_kwargs={"compression": "tiff_lzw"}
)

print("\nSaved: 1A5Y.tif")