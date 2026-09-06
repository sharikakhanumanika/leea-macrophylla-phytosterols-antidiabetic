# leea-macrophylla-phytosterols-antidiabetic

**Computational Supporting Data for the *In Silico* Evaluation of Isolated Phytosterols from** ***Leea macrophylla*** **Roots: A Multi-target Antidiabetic Approach**

This repository contains the complete molecular dynamics (MD) simulation and post-simulation analysis data supporting the computational evaluation of phytosterols isolated from *Leea macrophylla* roots against key antidiabetic protein targets.

## Repository Structure

Each target protein has its own directory, identified by its Protein Data Bank (PDB) ID. Each target contains separate **apo** (unbound) and **holo** (ligand-bound) simulation sets.

├── 1A5Y/
│   ├── apo/
│   └── holo/
├── 2QT9/
│   ├── apo/
│   └── holo/
└── 3DZU/
    ├── apo/
    └── holo/

## Simulation Files

Each `apo/` or `holo/` directory contains the files generated during the molecular dynamics simulation workflow:

- `EM.*` — Energy minimization files
- `NVT.*` — NVT equilibration files
- `NPT.*` — NPT equilibration files
- `MD.*` — Production molecular dynamics simulation files
- `topol.top` — System topology file

## Post-Simulation Analysis

The `analysis/` subfolder contains the trajectory-analysis outputs:

- `RMSD_Ca.xvg` — Cα root-mean-square deviation (RMSD)
- `rmsf_ca.xvg` — Cα root-mean-square fluctuation (RMSF)
- `rg.xvg` — Radius of gyration
- `SASA_protein.xvg` — Protein solvent-accessible surface area
- `SASA_ligand.xvg` — Ligand solvent-accessible surface area
- `hbond_count.xvg` — Protein–ligand hydrogen-bond analysis
- `pc1.xvg`, `pc2.xvg`, `pc3.xvg` — Principal component projections
- `eigenval.xvg` — PCA eigenvalues
- `eigenvec.trr` — PCA eigenvectors
- `fel_grid.csv` — Free Energy Landscape (FEL) grid data
- `fel_summary.txt` — FEL analysis summary
- `MM_Results_Data.csv` — MM-PBSA/GBSA binding free-energy results
- `script/fel_analysis.py` — FEL analysis script
- `script/pca.py` — PCA analysis script
- `.xlsx` files — Tabulated versions of the corresponding trajectory analyses

## Computational Workflow

The computational workflow includes:

1. Protein–ligand molecular docking
2. System preparation and solvation
3. Energy minimization
4. NVT equilibration
5. NPT equilibration
6. Production molecular dynamics simulations
7. Trajectory-based structural and conformational analyses
8. Principal Component Analysis (PCA)
9. Free Energy Landscape (FEL) analysis
10. MM-PBSA/GBSA binding free-energy estimation
