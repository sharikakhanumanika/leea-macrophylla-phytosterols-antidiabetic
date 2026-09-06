"""
Publication-Standard Free Energy Landscape (FEL) Generator
=============================================================
Builds a 3D Gibbs free-energy surface from PC1/PC2 projections (gmx anaeig
output), with:
  - Clean journal-style typography (Times New Roman / serif)
  - Jagged/spiky jet surface with black wireframe edges (classic FEL look)
  - Projected 2D contour "floor"
  - Global energy minimum marked with a star
  - Optional 3D trajectory curve draped over the surface (off by default)
  - High-res PNG (300 dpi) output
  - CSV grid + text summary, saved per system in its own output folder:
      <system_name>/FEL.png
      <system_name>/fel_grid.csv
      <system_name>/fel_summary.txt

Usage:
    python publication_fel.py

Just edit the SYSTEMS list at the bottom with your pc1.xvg / pc2.xvg pairs
and whatever label you want for each system's output folder.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import os

# ------------------------------------------------------------------
# Global publication style (journal-friendly)
# ------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.linewidth': 1.2,
    'mathtext.fontset': 'stix',
    'savefig.dpi': 300,
})

R = 0.008314   # kJ/mol/K
T = 300.0      # K


def read_eigenvalues(eigenval_file):
    """Read gmx anaeig eigenval.xvg and return the eigenvalue array."""
    if not os.path.exists(eigenval_file):
        return None
    vals = []
    with open(eigenval_file, 'r') as f:
        for line in f:
            if not line.startswith(('@', '#')) and line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    vals.append(float(parts[1]))
    return np.array(vals) if vals else None


def pc_variance_percent(eigenvalues, pc1_index=0, pc2_index=1):
    """% of total variance captured by the chosen PC1 / PC2 components."""
    total = np.sum(eigenvalues)
    pct1 = 100.0 * eigenvalues[pc1_index] / total
    pct2 = 100.0 * eigenvalues[pc2_index] / total
    return pct1, pct2


def read_xvg_column(file_path):
    """Read the y-column of an XVG file, convert nm -> Angstrom."""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if not line.startswith(('@', '#')) and line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    data.append(float(parts[1]) * 10.0)  # nm -> Angstrom
    return np.array(data)


def compute_fel(x, y, nbins=50, smooth_sigma=0.8):
    """Histogram -> Gibbs free energy grid, with light smoothing for
    a cleaner publication-style surface (set smooth_sigma=0 to disable)."""
    hist, xedges, yedges = np.histogram2d(x, y, bins=nbins)

    if smooth_sigma > 0:
        hist = gaussian_filter(hist, sigma=smooth_sigma)

    prob = hist / np.sum(hist)
    free_energy = -R * T * np.log(prob + 1e-10)
    free_energy -= np.nanmin(free_energy)

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    return free_energy, x_centers, y_centers


def trajectory_z_values(x, y, free_energy, x_centers, y_centers):
    """Interpolate the free-energy surface at every trajectory (x,y)
    point so the 3D curve sits on top of the surface rather than at z=0."""
    xi, yi = np.meshgrid(x_centers, y_centers)
    points = np.column_stack([xi.ravel(), yi.ravel()])
    values = free_energy.T.ravel()
    z_traj = griddata(points, values, (x, y), method='linear')
    # Fallback for any NaNs at the edges
    if np.any(np.isnan(z_traj)):
        z_nearest = griddata(points, values, (x, y), method='nearest')
        z_traj = np.where(np.isnan(z_traj), z_nearest, z_traj)
    return z_traj


def plot_publication_fel(pc1_file, pc2_file, system_name,
                          eigenval_file=None, pc1_index=0, pc2_index=1,
                          nbins=50, energy_cap=8.0,
                          smooth_sigma=0.0, show_trajectory=False,
                          elev=30, azim=45):

    print(f"Generating publication FEL for {system_name}...")

    p1 = read_xvg_column(pc1_file)
    p2 = read_xvg_column(pc2_file)
    if p1 is None or p2 is None:
        return

    min_len = min(len(p1), len(p2))
    x, y = p1[:min_len], p2[:min_len]

    # ---------------- PCA variance percentages (optional) ----------------
    pct1 = pct2 = None
    if eigenval_file is not None:
        eigenvalues = read_eigenvalues(eigenval_file)
        if eigenvalues is not None and len(eigenvalues) > max(pc1_index, pc2_index):
            pct1, pct2 = pc_variance_percent(eigenvalues, pc1_index, pc2_index)
        else:
            print(f"  Warning: could not read eigenvalues from {eigenval_file}; "
                  f"axis labels will not show % variance.")

    free_energy, x_centers, y_centers = compute_fel(x, y, nbins, smooth_sigma)
    analysis_energy = free_energy.copy()  # uncapped, for stats/CSV

    # ---------------- stats ----------------
    mean_energy = np.nanmean(analysis_energy)
    max_energy = np.nanmax(analysis_energy)
    std_energy = np.nanstd(analysis_energy)

    min_index = np.nanargmin(analysis_energy)
    i_min, j_min = np.unravel_index(min_index, analysis_energy.shape)
    global_min_energy = analysis_energy[i_min, j_min]
    global_pc1 = x_centers[i_min]
    global_pc2 = y_centers[j_min]

    pc1_range = (np.min(x), np.max(x))
    pc2_range = (np.min(y), np.max(y))

    # ---------------- trajectory z-values (before capping) ----------------
    if show_trajectory:
        z_traj = trajectory_z_values(x, y, analysis_energy, x_centers, y_centers)
        z_traj = np.minimum(z_traj, energy_cap) + 0.05  # tiny lift so it's visible

    # ---------------- cap for visualization ----------------
    free_energy_capped = free_energy.copy()
    free_energy_capped[free_energy_capped > energy_cap] = energy_cap

    xi, yi = np.meshgrid(x_centers, y_centers)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        xi, yi, free_energy_capped.T, cmap='jet',
        edgecolor='black', linewidth=0.1, antialiased=False,
        rstride=1, cstride=1, alpha=1.0, zorder=1
    )

    # Projected contour floor
    ax.contourf(xi, yi, free_energy_capped.T, zdir='z', offset=0,
                cmap='jet', levels=25, alpha=1.0)

    # Global minimum marker
    ax.scatter([global_pc1], [global_pc2], [global_min_energy],
               color='white', edgecolor='black', s=80, marker='*',
               depthshade=False, zorder=5, label='Global minimum')

    # ---- optional 3D trajectory curve, colour-graded by simulation time ----
    if show_trajectory:
        points = np.array([x, y, z_traj]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        frame_idx = np.linspace(0, 1, len(x) - 1)
        lc = Line3DCollection(segments, cmap='viridis', array=frame_idx,
                              linewidth=1.3, zorder=6, alpha=0.9)
        ax.add_collection3d(lc)
        cbar_traj = fig.colorbar(lc, ax=ax, shrink=0.5, aspect=12, pad=0.14)
        cbar_traj.set_label('Simulation progression (0 → 100 ns)',
                            fontweight='bold', fontsize=10)

    # Labels (include % variance explained when eigenvalues were provided)
    if pct1 is not None:
        xlabel = rf'PC1 ({pct1:.1f}%) ($\mathrm{{\AA}}$)'
        ylabel = rf'PC2 ({pct2:.1f}%) ($\mathrm{{\AA}}$)'
    else:
        xlabel = r'PC1 ($\mathrm{\AA}$)'
        ylabel = r'PC2 ($\mathrm{\AA}$)'

    ax.set_xlabel(xlabel, fontweight='bold', fontsize=13, labelpad=10)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=13, labelpad=10)
    ax.set_zlabel('Gibbs Free Energy (kJ/mol)', fontweight='bold', fontsize=13, labelpad=10)
    ax.set_zlim(0, energy_cap)

    cbar = fig.colorbar(surf, shrink=0.6, aspect=14, pad=0.1)
    cbar.set_label('Gibbs Free Energy (kJ/mol)', fontweight='bold', fontsize=12)

    ax.view_init(elev=elev, azim=azim)

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor('black')
        pane.fill = False
    ax.grid(False)

    plt.tight_layout()

    out_dir = system_name
    os.makedirs(out_dir, exist_ok=True)

    png_out = os.path.join(out_dir, 'FEL.png')
    plt.savefig(png_out, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------
    # Save grid + summary (same as your original)
    # ------------------------------------------------------------
    grid_file = os.path.join(out_dir, "fel_grid.csv")
    with open(grid_file, "w") as f:
        f.write("PC1_Angstrom,PC2_Angstrom,DeltaG_kJmol\n")
        for i in range(len(x_centers)):
            for j in range(len(y_centers)):
                e = analysis_energy[i, j]
                if np.isnan(e):
                    continue
                f.write(f"{x_centers[i]:.6f},{y_centers[j]:.6f},{e:.6f}\n")

    summary_file = os.path.join(out_dir, "fel_summary.txt")
    with open(summary_file, "w") as f:
        f.write("FREE ENERGY LANDSCAPE SUMMARY\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Frames analysed : {len(x)}\n")
        f.write(f"Histogram bins  : {nbins} x {nbins}\n")
        f.write(f"Smoothing sigma : {smooth_sigma}\n")
        f.write(f"Temperature     : {T:.1f} K\n")
        if pct1 is not None:
            f.write(f"PC1 variance    : {pct1:.2f} %\n")
            f.write(f"PC2 variance    : {pct2:.2f} %\n")
        f.write("\n")
        f.write("GLOBAL MINIMUM\n" + "-" * 27 + "\n")
        f.write(f"DeltaG : {global_min_energy:.4f} kJ/mol\n")
        f.write(f"PC1    : {global_pc1:.4f} \u00c5\n")
        f.write(f"PC2    : {global_pc2:.4f} \u00c5\n\n")
        f.write("FREE ENERGY STATISTICS\n" + "-" * 27 + "\n")
        f.write(f"Mean DeltaG : {mean_energy:.4f} kJ/mol\n")
        f.write(f"Max DeltaG  : {max_energy:.4f} kJ/mol\n")
        f.write(f"SD DeltaG   : {std_energy:.4f} kJ/mol\n\n")
        f.write("PC1 RANGE\n" + "-" * 27 + "\n")
        f.write(f"Minimum : {pc1_range[0]:.4f} \u00c5\n")
        f.write(f"Maximum : {pc1_range[1]:.4f} \u00c5\n")
        f.write(f"Range   : {pc1_range[1]-pc1_range[0]:.4f} \u00c5\n\n")
        f.write("PC2 RANGE\n" + "-" * 27 + "\n")
        f.write(f"Minimum : {pc2_range[0]:.4f} \u00c5\n")
        f.write(f"Maximum : {pc2_range[1]:.4f} \u00c5\n")
        f.write(f"Range   : {pc2_range[1]-pc2_range[0]:.4f} \u00c5\n")

    print(f"Saved: {png_out}, {grid_file}, {summary_file}")
    print(f"FEL analysis completed for {system_name}.\n")


if __name__ == "__main__":
    # ---- Edit this list with your actual pc1.xvg / pc2.xvg / eigenval.xvg files ----
    # eigenval_file is optional — if provided (the eigenval.xvg from
    # `gmx anaeig`), PC1/PC2 axis labels will show the % variance each
    # component explains, e.g. "PC1 (45.3%) (Å)". Leave as None to skip.
    SYSTEMS = [
        # (pc1_file, pc2_file, output_folder_name, eigenval_file)
        ("pc1.xvg", "pc2.xvg", "System_1", "eigenval.xvg"),
        # ("pc1.xvg", "pc2.xvg", "System_2", "eigenval.xvg"),
        # ("pc1.xvg", "pc2.xvg", "System_3", None),  # no % shown
    ]

    for pc1_file, pc2_file, name, eigenval_file in SYSTEMS:
        plot_publication_fel(
            pc1_file, pc2_file, name,
            eigenval_file=eigenval_file,
            pc1_index=0, pc2_index=1,  # which eigenvalues correspond to PC1/PC2
            nbins=50,
            energy_cap=8.0,
            smooth_sigma=0.0,       # set >0 (e.g. 0.8) for a smoother surface
            show_trajectory=False,  # set True to overlay the time-graded path
            elev=30, azim=45,
        )