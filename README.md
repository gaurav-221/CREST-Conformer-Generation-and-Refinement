# CREST Conformer Generation Pipeline

A publication-grade, fully automated pipeline for conformational analysis of drug-like and bioactive molecules.

---

## Overview

This pipeline accepts molecule libraries via CSV (SMILES + names), applies multi-layer standardization, generates 3D geometries, runs CREST iMTD-GC conformer search, and applies quantum-mechanically motivated ensemble filtering.

### Features

- SMILES standardization (5-stage RDKit)
- ETKDGv3 + MMFF94s 3D geometry generation
- GFN2-xTB tight pre-optimization
- CREST iMTD-GC conformer ensemble search
- Multiple post-CREST filtering strategies:
  - Boltzmann (ΔE or ΔG)
  - Entropy pruning
  - Torsion RMSD diversity
  - Cluster representatives
  - Optional QM refinement (R2SCAN-3c, CENSO)

---

## Prerequisites

### Required Tools

- **Python 3.9+**
- **RDKit** - cheminformatics library
- **xtb** - semi-empirical quantum chemistry program
- **CREST** - conformer ensemble generator

### Install via Conda (Recommended)

```bash
conda install -c conda-forge rdkit xtb crest
```

### Optional (for QM filtering)

- **ORCA** - for R2SCAN-3c QM filtering (commercial license required)
- **CENSO** - Grimme group ensemble refinement pipeline

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Conformer_generation_pipeline.git
cd Conformer_generation_pipeline

# Create conda environment
conda env create -f environment.yml

# Or install Python dependencies
pip install -r requirements.txt
```

---

## Usage

### Prepare Input

Create a CSV file with two columns: SMILES and molecule name.

```csv
Smiles,Molecule Name
COc1cc(NS(c(cc2OC)cc(OC)c2O)(=O)=O)cc(OC)c1O,compound_1
COc1cc(C(Nc(cc2OC)cc(OC)c2O)=O)cc(OC)c1O,compound_2
```

### Run Pipeline

```bash
# Basic usage
python crest_conformer_pipeline_final.py --input mol.csv

# With custom output folder
python crest_conformer_pipeline_final.py --input mol.csv --output ./conformers

# Apply specific filter
python crest_conformer_pipeline_final.py --input mol.csv --filter BOLTZMANN
```

### Filter Options

| Filter | Description |
|--------|-------------|
| `BOLTZMANN` | Classical Boltzmann on GFN2-xTB ΔE |
| `FREE_ENERGY` | Boltzmann on GFN2-xTB ΔG (RRHO + ALPB) |
| `ENTROPY_PRUNING` | Maximum-entropy ensemble pruning |
| `TORSION_RMSD` | Dihedral fingerprint diversity + energy gate |
| `CLUSTER_RMSD` | RMSD-cluster representatives |

---

## Output

The pipeline generates:

- **conformer.xyz** - All conformers in XYZ format
- **conformer_filtered.xyz** - Filtered conformers
- **ensemble_summary.csv** - Summary statistics

---

## License

MIT License — see [LICENSE](LICENSE) file.

---

## Citation

If you use this pipeline in academic work, please cite the original CREST papers:

- Grimme, S. et al. J. Chem. Theory Comput. 2019, 15, 1652-1671
- Grimme, S. et al. J. Chem. Theory Comput. 2021, 17, 4084-4107
