#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  CREST CONFORMER GENERATION & ENSEMBLE FILTERING PIPELINE
  Version 5.0  |  Linux / macOS / Windows-WSL2 compatible
================================================================================

  A publication-grade, fully automated pipeline for conformational analysis of
  drug-like and bioactive molecules.  The pipeline accepts user-supplied
  molecule libraries via CSV (column 1 = SMILES, column 2 = molecule name),
  applies multi-layer SMILES standardisation, generates 3-D starting geometries,
  runs the CREST iMTD-GC conformer search, and applies the user's choice of
  quantum-mechanically motivated ensemble-filtering strategies — including
  state-of-the-art external tools as optional post-CREST refinement layers.

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  PIPELINE WORKFLOW  (v5.0)                                                  │
  │                                                                             │
  │  CSV (SMILES)                                                               │
  │     │                                                                       │
  │     ▼                                                                       │
  │  SMILES standardisation (5-stage RDKit)                                     │
  │     │                                                                       │
  │     ▼                                                                       │
  │  ETKDGv3 + MMFF94s 3-D geometry generation                                 │
  │     │                                                                       │
  │     ▼                                                                       │
  │  GFN2-xTB tight pre-optimisation                                           │
  │     │                                                                       │
  │     ▼                                                                       │
  │  CREST iMTD-GC conformer ensemble search                                   │
  │     │                                                                       │
  │     ▼                                                                       │
  │  ┌──────────────────────────────────────────────────────────────────────┐   │
  │  │  POST-CREST FILTERING  (one or more layers, applied in order)        │   │
  │  │                                                                      │   │
  │  │  Layer A — Built-in Python filters (no external deps beyond xtb):    │   │
  │  │    BOLTZMANN        Classical Boltzmann on GFN2-xTB ΔE               │   │
  │  │    FREE_ENERGY      Boltzmann on GFN2-xTB ΔG (RRHO + ALPB)          │   │
  │  │    ENTROPY_PRUNING  Maximum-entropy ensemble pruning                 │   │
  │  │    TORSION_RMSD     Dihedral fingerprint diversity + energy gate      │   │
  │  │    CLUSTER_REP      RMSD-cluster representatives (energy-weighted)   │   │
  │  │                                                                      │   │
  │  │  Layer B — External QM refinement filters (require extra tools):     │   │
  │  │    XTB_RERANK       xTB re-optimisation with tight/verytight acc     │   │
  │  │                     + recomputed Boltzmann on fresh GFN2 energies    │   │
  │  │    R2SCAN3C         Composite DFT-quality filtering via ORCA         │   │
  │  │                     r²SCAN-3c single points → Boltzmann reranking    │   │
  │  │    CENSO            Grimme-group CENSO pipeline (CREST→DFT-lite)     │   │
  │  │    CONFPASS         CONFPASS intelligent prioritisation              │   │
  │  │    ANI              TorchANI neural-network potential reranking       │   │
  │  └──────────────────────────────────────────────────────────────────────┘   │
  │     │                                                                       │
  │     ▼                                                                       │
  │  Annotated XYZ ensemble  +  CSV/JSON summary                               │
  └─────────────────────────────────────────────────────────────────────────────┘

  ── Multi-filter chaining ─────────────────────────────────────────────────────
  Filters are applied sequentially.  Each filter receives the output of the
  previous one, progressively narrowing the ensemble toward DFT-ready candidates.

  Example:
    --filter1 CLUSTER_REP --filter2 XTB_RERANK --filter3 R2SCAN3C

  The conformer count limit (--nconf) is applied at EACH stage, so the final
  ensemble is always ≤ nconf structures.

  ── Built-in filter methods (no extra software) ──────────────────────────────
  BOLTZMANN       │ Classical Boltzmann weighting on GFN2-xTB energies
  FREE_ENERGY     │ GFN2-xTB ΔG (RRHO thermal + ALPB solvation)
  ENTROPY_PRUNING │ Maximum-entropy ensemble pruning (geometric RSD)
  TORSION_RMSD    │ Dihedral-fingerprint diversity + energy gate
  CLUSTER_REP     │ Energy-weighted RMSD clustering representatives

  ── External QM refinement filters ──────────────────────────────────────────
  XTB_RERANK  │ Re-optimise every conformer with xtb --opt tight --acc 0.2
              │   (or --opt verytight --acc 0.05), recompute GFN2 energies,
              │   rerun Boltzmann on the fresh energy surface.
              │   Requires: xtb ≥ 6.4  (conda install -c conda-forge xtb)
              │   Ref: Grimme, JCTC 2019, 15, 2847
              │
  R2SCAN3C    │ ORCA r²SCAN-3c single-point energies on xTB-optimised
              │   conformer geometries, then Boltzmann reranking.
              │   Requires: ORCA ≥ 5.0  (https://orcaforum.kofo.mpg.de)
              │             Set ORCA_EXE or ensure orca is in PATH.
              │   Ref: Grimme et al., J. Chem. Phys. 2021, 155, 104111
              │
  CENSO       │ Grimme-group CENSO pipeline: multilevel DFT-lite refinement
              │   using xTB + DFT single-points.  CENSO manages its own
              │   screening internally; this pipeline writes the CREST
              │   ensemble, runs censo, and imports the ranked output.
              │   Requires: CENSO (pip install censo)
              │             https://github.com/grimme-lab/censo
              │   Ref: Pracht et al., JCTC 2020, 16, 7044
              │
  CONFPASS    │ CONFPASS intelligent conformer selection: predicts which
              │   conformers are worthwhile for DFT and removes redundancy.
              │   Requires: CONFPASS (pip install confpass)
              │             https://github.com/lamgroup/CONFPASS
              │   Ref: Lam et al., JCIM 2020
              │
  ANI         │ TorchANI neural-network potential (ANI-2x) single-point
              │   energies on conformers, then Boltzmann reranking.
              │   Requires: torchani (pip install torchani)
              │             torch (pip install torch)
              │   Supported elements: H C N O F S Cl
              │   Ref: Smith et al., Nat. Commun. 2019, 10, 2903

  ── SMILES standardisation (chemistry-preserving) ───────────────────────────
  Stage 1 – Strip salts / largest fragment selection
  Stage 2 – Normalise unusual valences (RDKit MolStandardize)
  Stage 3 – Reionise to canonical protonation state
  Stage 4 – Canonical tautomer selection (RDKit TautomerEnumerator)
  Stage 5 – Stereo audit — warn only, never silently change stereo

  ── Input CSV format ────────────────────────────────────────────────────────
  Column 1 = SMILES, column 2 = molecule name.  Override with --smiles_col /
  --name_col (0-based integers or header strings).  Header auto-detected.

  ── Usage examples ──────────────────────────────────────────────────────────
    # List all molecules
    python crest_conformer_pipeline_v5.py --csv mols.csv --list_mols

    # Validate only
    python crest_conformer_pipeline_v5.py --csv mols.csv --validate_only

    # Single filter — xTB re-ranking (best default improvement over v4)
    python crest_conformer_pipeline_v5.py --csv mols.csv \\
        --all --ncores 8 --solvent water --filter1 XTB_RERANK

    # Two-stage: cluster first, then xTB rerank
    python crest_conformer_pipeline_v5.py --csv mols.csv \\
        --all --ncores 8 --filter1 CLUSTER_REP --filter2 XTB_RERANK

    # Publication-grade three-stage pipeline
    python crest_conformer_pipeline_v5.py --csv mols.csv \\
        --all --ncores 16 --solvent water \\
        --filter1 CLUSTER_REP --filter2 XTB_RERANK --filter3 R2SCAN3C \\
        --nconf 20 --ewin 8.0

    # ANI neural-network reranking (fast, no ORCA needed)
    python crest_conformer_pipeline_v5.py --csv mols.csv \\
        --all --ncores 8 --filter1 CLUSTER_REP --filter2 ANI

    # Full CENSO pipeline
    python crest_conformer_pipeline_v5.py --csv mols.csv \\
        --all --ncores 8 --solvent water --filter1 CENSO

    # CONFPASS prioritisation then CENSO
    python crest_conformer_pipeline_v5.py --csv mols.csv \\
        --all --filter1 CONFPASS --filter2 CENSO

    # Legacy mode: single --filter flag (backward compatible with v4)
    python crest_conformer_pipeline_v5.py --csv mols.csv \\
        --all --filter BOLTZMANN

  ── Software dependencies ────────────────────────────────────────────────────
    REQUIRED (always):
      conda install -c conda-forge rdkit xtb crest
      pip install numpy pandas

    OPTIONAL (for Layer B filters):
      XTB_RERANK  : xtb already required above
      R2SCAN3C    : ORCA ≥ 5.0  (download from https://orcaforum.kofo.mpg.de)
      CENSO       : pip install censo
      CONFPASS    : pip install confpass
      ANI         : pip install torchani torch

  ── Output ──────────────────────────────────────────────────────────────────
    <outdir>/
      <id>_<MolName>/
        <MolName>_rdkit.xyz       RDKit ETKDGv3 geometry
        <MolName>_xtb.xyz         GFN2-xTB pre-optimised geometry
        <MolName>_xtb.log         xTB pre-opt log
        <MolName>_crest.log       CREST run log
        crest_conformers.xyz      raw CREST ensemble
        xtb_rerank/               xTB re-opt files (XTB_RERANK filter)
        orca_r2scan3c/            ORCA SP files (R2SCAN3C filter)
        censo_run/                CENSO working directory
        confpass_run/             CONFPASS working directory
      <id>_<MolName>_conformers.xyz   final filtered + annotated ensemble
      conformer_summary.csv           per-molecule statistics table
      conformer_summary.json          machine-readable full results
      validation_report.csv           SMILES validation audit trail
      pipeline.log                    full execution log

  ── Key references ──────────────────────────────────────────────────────────
    [1]  Grimme et al.  JCTC 2019, 15, 2847  (CREST iMTD-GC)
    [2]  Bannwarth et al. JCTC 2019, 15, 1652  (GFN2-xTB)
    [3]  Grimme, Schreiner. ACIE 2018, 57, 4170  (RRHO)
    [4]  Pracht et al. JCTC 2020, 16, 7044  (CREST v2 / CENSO)
    [5]  Grimme et al. JCP 2021, 155, 104111  (r²SCAN-3c)
    [6]  Smith et al. Nat. Commun. 2019, 10, 2903  (ANI)
    [7]  Lam et al. JCIM 2020  (CONFPASS)
    [8]  Weininger, JCICS 1988, 28, 31  (SMILES)
    [9]  RDKit MolStandardize — rdkit.org

  Changelog v5.0:
    - Added five new QM/ML-based Layer B post-CREST filters:
        XTB_RERANK, R2SCAN3C, CENSO, CONFPASS, ANI
    - Added multi-filter chaining via --filter1 / --filter2 / --filter3
    - --filter flag retained for backward compatibility (maps to --filter1)
    - ANI filter with element-support guard and torchani graceful import
    - R2SCAN3C ORCA block generator with basis-set-free composite approach
    - CENSO integration with .censorc config auto-writing
    - CONFPASS integration with CSV export/import
    - Per-filter working subdirectories with complete input/output logs
    - Filter chain provenance recorded in XYZ comment lines and JSON
================================================================================
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import csv
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── RDKit ─────────────────────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import (
        AllChem,
        Descriptors,
        QED,
        rdMolDescriptors,
        MolToSmiles,
        MolFromSmiles,
    )
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.rdMolDescriptors import (
        CalcTPSA,
        CalcNumHBD,
        CalcNumHBA,
        CalcNumRotatableBonds,
    )
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    _RDKIT_VERSION = Chem.rdBase.rdkitVersion
except ImportError:
    sys.exit(
        "\n❌  RDKit not found.\n"
        "   Install: conda install -c conda-forge rdkit\n"
    )

# ── NumPy ─────────────────────────────────────────────────────────────────────
try:
    import numpy as np
except ImportError:
    sys.exit("❌  numpy not found.  pip install numpy")

# ── Logging ───────────────────────────────────────────────────────────────────
_LOG_FORMAT  = "%(asctime)s [%(levelname)-8s] %(message)s"
_LOG_DATEFMT = "%H:%M:%S"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt=_LOG_DATEFMT)
log = logging.getLogger("CREST-Pipeline")
log.info(f"RDKit {_RDKIT_VERSION} loaded")

# ── Platform ──────────────────────────────────────────────────────────────────
_PLATFORM = platform.system()

# ── Physical constants ────────────────────────────────────────────────────────
_HARTREE_TO_KCAL = 627.5094740631
_R_KCAL          = 0.0019872041
_kB_HARTREE      = 3.16681e-6

# ── ANI-supported elements (ANI-2x model) ────────────────────────────────────
_ANI_SUPPORTED_ELEMENTS = {"H", "C", "N", "O", "F", "S", "Cl"}

# ── All valid filter names ─────────────────────────────────────────────────────
_BUILTIN_FILTERS  = {"BOLTZMANN", "FREE_ENERGY", "ENTROPY_PRUNING",
                     "TORSION_RMSD", "CLUSTER_REP"}
_EXTERNAL_FILTERS = {"XTB_RERANK", "R2SCAN3C", "CENSO", "CONFPASS", "ANI"}
_ALL_FILTERS      = _BUILTIN_FILTERS | _EXTERNAL_FILTERS


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY — version-safe attribute setter
# ══════════════════════════════════════════════════════════════════════════════

def _safe_set(obj, attr: str, val) -> None:
    try:
        setattr(obj, attr, val)
    except (AttributeError, TypeError):
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY — molecule embedder (ETKDGv3 → v2 → v1 → bare)
# ══════════════════════════════════════════════════════════════════════════════

def _embed_molecule(mol, random_seed: int = 42, max_attempts: int = 200) -> int:
    for attempt_fn, label in [
        (lambda: AllChem.ETKDGv3(), "ETKDGv3"),
        (lambda: AllChem.ETKDGv2(), "ETKDGv2"),
        (lambda: AllChem.ETKDG(),   "ETKDG"),
    ]:
        try:
            p = attempt_fn()
            _safe_set(p, "randomSeed",              random_seed)
            _safe_set(p, "maxAttempts",             max_attempts)
            _safe_set(p, "numThreads",              1)
            _safe_set(p, "useSmallRingTorsions",    True)
            _safe_set(p, "useMacrocycleTorsions",   False)
            _safe_set(p, "enforceChirality",        True)
            _safe_set(p, "useExpTorsionAnglePrefs", True)
            _safe_set(p, "useBasicKnowledge",       True)
            r = AllChem.EmbedMolecule(mol, p)
            if r >= 0:
                if label != "ETKDGv3":
                    log.warning(f"    Used {label} fallback")
                return r
        except Exception:
            pass
    try:
        r = AllChem.EmbedMolecule(mol)
        if r >= 0:
            log.warning("    Used bare EmbedMolecule fallback")
            return r
    except Exception:
        pass
    return -1


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY — cross-platform binary finder
# ══════════════════════════════════════════════════════════════════════════════

def _find_binary(name: str) -> Optional[str]:
    env_override = os.environ.get(f"{name.upper()}_EXE")
    if env_override and Path(env_override).exists():
        return env_override
    found = shutil.which(name)
    if found:
        return found
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    for candidate in [
        Path(conda_prefix) / "bin" / name,
        Path(conda_prefix) / "bin" / f"{name}.exe",
        Path(sys.prefix)   / "bin" / name,
        Path(sys.prefix)   / "bin" / f"{name}.exe",
        Path("/usr/local/bin") / name,
        Path("/opt/conda/bin") / name,
    ]:
        if candidate.exists():
            return str(candidate)
    return None


def _find_mpirun() -> Optional[str]:
    """
    Locate an MPI launcher (mpirun / mpiexec / orterun) on the system.

    ORCA's MPI build requires mpirun to be in PATH or co-located with the
    orca binary.  This helper searches common installation locations used by
    OpenMPI, MPICH, and Intel MPI so the pipeline can set up the environment
    correctly even when the user has not sourced the MPI environment module.

    Priority (first match wins):
      1. MPIRUN_EXE / MPIEXEC_EXE environment variable overrides
      2. mpirun / mpiexec / orterun already in PATH
      3. Same directory as the orca binary (common with ORCA-bundled OpenMPI)
      4. Common OpenMPI / MPICH system installation prefixes
    """
    # 1 — explicit user override
    for env_var in ("MPIRUN_EXE", "MPIEXEC_EXE"):
        val = os.environ.get(env_var, "")
        if val and Path(val).is_file():
            return val

    # 2 — already on PATH
    for launcher in ("mpirun", "mpiexec", "orterun"):
        found = shutil.which(launcher)
        if found:
            return found

    # 3 — same directory as orca binary (ORCA ships a bundled OpenMPI)
    orca_bin_path = _find_binary("orca")
    if orca_bin_path:
        orca_dir = Path(orca_bin_path).resolve().parent
        for launcher in ("mpirun", "mpiexec", "orterun"):
            candidate = orca_dir / launcher
            if candidate.is_file():
                return str(candidate)

    # 4 — common system OpenMPI / MPICH / Intel-MPI install trees
    _mpi_search_dirs = [
        "/usr/lib/openmpi/bin",
        "/usr/lib64/openmpi/bin",
        "/usr/lib/x86_64-linux-gnu/openmpi/bin",
        "/usr/lib/aarch64-linux-gnu/openmpi/bin",
        "/usr/local/openmpi/bin",
        "/opt/openmpi/bin",
        "/opt/mpi/bin",
        "/opt/intel/mpi/latest/bin",
        "/usr/lib/mpich/bin",
        "/usr/local/mpich/bin",
    ]
    for d in _mpi_search_dirs:
        for launcher in ("mpirun", "mpiexec", "orterun"):
            candidate = Path(d) / launcher
            if candidate.is_file():
                return str(candidate)

    return None


def _build_orca_env(orca_bin: str, mpirun: Optional[str]) -> dict:
    """
    Build an os.environ copy that ensures mpirun (and its shared libraries)
    are resolvable when ORCA is launched as a subprocess.

    For MPI-parallel ORCA builds the mpirun binary and its companion
    libmpi.so must be on PATH / LD_LIBRARY_PATH.  We add the MPI bin/lib
    directories to the child-process environment without mutating the
    parent process.
    """
    env = os.environ.copy()

    # Always ensure the ORCA binary directory is on PATH so ORCA can call its
    # own sibling helper binaries (orca_scf, orca_mp2, …).
    orca_bin_dir = str(Path(orca_bin).resolve().parent)
    if orca_bin_dir not in env.get("PATH", ""):
        env["PATH"] = orca_bin_dir + os.pathsep + env.get("PATH", "")

    if mpirun is None:
        return env

    mpi_bin_dir   = str(Path(mpirun).resolve().parent)
    mpi_prefix    = Path(mpirun).resolve().parent.parent
    mpi_lib_dir   = str(mpi_prefix / "lib")
    mpi_lib64_dir = str(mpi_prefix / "lib64")

    # Prepend MPI launcher directory to PATH
    if mpi_bin_dir not in env.get("PATH", ""):
        env["PATH"] = mpi_bin_dir + os.pathsep + env["PATH"]

    # Prepend MPI library directories to LD_LIBRARY_PATH
    extra_libs = os.pathsep.join(
        p for p in [mpi_lib64_dir, mpi_lib_dir] if Path(p).is_dir()
    )
    if extra_libs:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (extra_libs + os.pathsep + existing).rstrip(os.pathsep)

    return env


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — CHEMISTRY-PRESERVING SMILES STANDARDISER
# ══════════════════════════════════════════════════════════════════════════════

class SMILESStandardiser:
    """
    Five-stage, chemistry-preserving SMILES standardisation.

    Stages
    ------
    S1  Fragment selection — keep largest organic fragment
    S2  Valence normalisation — RDKit MolStandardize Normalizer
    S3  Reionisation — canonical acid/base protonation state
    S4  Canonical tautomer — RDKit TautomerEnumerator (lowest energy)
    S5  Stereochemistry audit — warn on unspecified centres; never mutate

    References
    ----------
    Bento et al. J. Cheminformatics 2020, 12, 51 (ChEMBL standardiser)
    """

    def __init__(self) -> None:
        self._lf    = rdMolStandardize.LargestFragmentChooser()
        self._norm  = rdMolStandardize.Normalizer()
        self._reion = rdMolStandardize.Reionizer()
        self._te    = rdMolStandardize.TautomerEnumerator()
        self._te.SetMaxTautomers(64)

    def standardise(self, smiles: str, name: str) -> Tuple[Optional[str], List[str]]:
        warnings: List[str] = []
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None, [f"[{name}] S0 FAIL: cannot parse SMILES '{smiles}'"]
        original_inchi = _mol_to_inchikey(mol)

        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = self._lf.choose(mol)
            warnings.append(f"[{name}] S1: {len(frags)} fragments — largest selected")

        for fn, tag in [(self._norm.normalize, "S2"), (self._reion.reionize, "S3")]:
            try:
                mol = fn(mol)
            except Exception as exc:
                warnings.append(f"[{name}] {tag} WARN: skipped ({exc})")

        try:
            mol = self._te.Canonicalize(mol)
        except Exception as exc:
            warnings.append(f"[{name}] S4 WARN: tautomer skipped ({exc})")

        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            return None, warnings + [f"[{name}] Sanitisation failed: {exc}"]

        n_usc = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
        n_sc  = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
        if n_usc > 0:
            warnings.append(
                f"[{name}] S5 WARN: {n_usc}/{n_sc} stereocentre(s) unspecified"
            )

        post_inchi = _mol_to_inchikey(mol)
        if (original_inchi and post_inchi and
                original_inchi[:14] != post_inchi[:14]):
            warnings.append(f"[{name}] S4 REVERT: tautomer changed connectivity; reverting")
            mol_orig = Chem.MolFromSmiles(smiles.strip())
            if mol_orig is not None:
                mol = mol_orig

        return Chem.MolToSmiles(mol, canonical=True), warnings


def _mol_to_inchikey(mol) -> Optional[str]:
    try:
        from rdkit.Chem.inchi import MolToInchiKey, MolToInchi
        inchi = MolToInchi(mol)
        return MolToInchiKey(inchi) if inchi else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — CSV MOLECULE LIBRARY READER
# ══════════════════════════════════════════════════════════════════════════════

def _sanitise_name(name: str) -> str:
    safe = re.sub(r"[^\w\-]", "_", name.strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "molecule"


def load_molecules_from_csv(
    csv_path:   Path,
    smiles_col: int = 0,
    name_col:   int = 1,
) -> List[Dict]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        sys.exit(f"❌  CSV file not found: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        raw_rows = list(csv.reader(fh))

    if not raw_rows:
        sys.exit(f"❌  CSV file is empty: {csv_path}")

    def _resolve_col(spec, header_row):
        if isinstance(spec, int):
            return spec
        if header_row:
            s = str(spec).lower()
            for idx, h in enumerate(header_row):
                if h.strip().lower() == s:
                    return idx
        raise ValueError(f"Column '{spec}' not found in CSV header.")

    first_row = raw_rows[0]
    has_header = False
    try:
        test_col = smiles_col if isinstance(smiles_col, int) else 0
        if test_col < len(first_row) and Chem.MolFromSmiles(first_row[test_col].strip()) is None:
            has_header = True
    except Exception:
        has_header = True

    header_row = first_row if has_header else None
    data_rows  = raw_rows[1:] if has_header else raw_rows
    sc = _resolve_col(smiles_col, header_row)
    nc = _resolve_col(name_col,   header_row)

    log.info(
        f"CSV: {csv_path.name}  "
        f"{'header detected' if has_header else 'no header'}  "
        f"{len(data_rows)} data row(s)  SMILES_col={sc}  name_col={nc}"
    )

    molecules:  List[Dict]      = []
    seen_names: Dict[str, int]  = {}
    standardiser = SMILESStandardiser()

    for row_idx, row in enumerate(data_rows, start=1):
        if not any(c.strip() for c in row):
            continue
        try:
            smi_raw = row[sc].strip()
        except IndexError:
            log.warning(f"  Row {row_idx}: SMILES col {sc} out of range — skipped")
            continue
        try:
            name_raw = row[nc].strip()
        except IndexError:
            name_raw = f"mol_{row_idx:04d}"

        if not smi_raw:
            log.warning(f"  Row {row_idx} ({name_raw!r}): empty SMILES — skipped")
            continue

        name_safe = _sanitise_name(name_raw) if name_raw else f"mol_{row_idx:04d}"
        if name_safe in seen_names:
            seen_names[name_safe] += 1
            name_safe = f"{name_safe}_{seen_names[name_safe]}"
            log.warning(f"  Row {row_idx}: duplicate name — renamed to '{name_safe}'")
        else:
            seen_names[name_safe] = 0

        std_smi, std_warnings = standardiser.standardise(smi_raw, name_safe)
        for w in std_warnings:
            log.warning(f"  {w}")
        if std_smi is None:
            log.error(f"  Row {row_idx} ({name_safe}): standardisation failed — skipped")
            continue

        desc   = row[2].strip() if len(row) > 2 else ""
        act    = row[3].strip() if len(row) > 3 else "unknown"
        ref    = row[4].strip() if len(row) > 4 else ""
        series = row[5].strip() if len(row) > 5 else "USER"

        molecules.append({
            "id":                row_idx,
            "name":              name_safe,
            "name_raw":          name_raw,
            "smiles":            std_smi,
            "smiles_raw":        smi_raw,
            "smiles_changed":    std_smi != smi_raw,
            "series":            series or "USER",
            "description":       desc,
            "expected_activity": act,
            "ref":               ref,
        })

    if not molecules:
        sys.exit("❌  No valid molecules could be parsed from the CSV.")
    log.info(f"  Loaded {len(molecules)} molecule(s)")
    return molecules


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — CONFIGURATION DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CRESTConfig:
    """All pipeline parameters in a single, serialisable dataclass."""

    # ── Multi-filter chain ───────────────────────────────────────────────────
    filter_chain:           List[str]       = field(default_factory=lambda: ["BOLTZMANN"])

    # ── Filtering thresholds ──────────────────────────────────────────────────
    boltzmann_threshold:    float           = 0.005
    T_kelvin:               float           = 298.15
    rmsd_cluster_thr:       float           = 0.50
    torsion_diff_thr:       float           = 20.0
    max_conformers_output:  int             = 25

    # ── XTB_RERANK settings ───────────────────────────────────────────────────
    xtb_rerank_accuracy:    str             = "tight"    # "tight" or "verytight"
    xtb_rerank_acc_val:     float           = 0.2        # --acc value (0.2=tight, 0.05=verytight)

    # ── R2SCAN3C / ORCA settings ──────────────────────────────────────────────
    orca_nprocs:            int             = 4
    orca_maxcore_mb:        int             = 4096
    r2scan3c_grid:          str             = "DefGrid2"
    orca_mpi_launcher:      Optional[str]   = None   # None = auto-detect

    # ── CENSO settings ────────────────────────────────────────────────────────
    censo_maxconf:          int             = 20         # max conformers CENSO screens
    censo_solvent:          Optional[str]   = None       # overrides solvent for CENSO

    # ── ANI settings ─────────────────────────────────────────────────────────
    ani_model:              str             = "ANI2x"

    # ── CREST conformer search ────────────────────────────────────────────────
    ewin_kcal:              float           = 6.0
    rmsd_thr:               float           = 0.125
    solvent:                Optional[str]   = None
    crest_version:          int             = 3

    # ── xTB pre-opt ───────────────────────────────────────────────────────────
    xtb_method:             str             = "gfn2"
    xtb_charge:             int             = 0
    xtb_uhf:                int             = 0

    # ── Gaussian / GJF export ─────────────────────────────────────────────────
    make_gjf:               bool            = False   # write per-conformer .gjf files
    gjf_route:              str             = "B3LYP/6-31G opt"
    gjf_mem:                str             = "10GB"
    gjf_nproc:              int             = 4
    gjf_charge:             int             = 0
    gjf_multiplicity:       int             = 1

    # ── RDKit embedding ───────────────────────────────────────────────────────
    random_seed:            int             = 42
    max_embed_attempts:     int             = 200
    ff_steps:               int             = 2000

    # ── Resources ─────────────────────────────────────────────────────────────
    ncores:                 int             = 4
    output_dir:             Path            = Path("crest_output")

    # ── Validation ────────────────────────────────────────────────────────────
    mw_max:                 float           = 900.0
    logp_max:               float           = 7.0
    hbd_max:                int             = 10
    hba_max:                int             = 15
    tpsa_max:               float           = 200.0
    n_phenolic_oh_min:      int             = 0

    def as_dict(self) -> dict:
        return {k: str(v) if isinstance(v, Path) else v
                for k, v in self.__dict__.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — MOLECULE VALIDATOR (7-layer)
# ══════════════════════════════════════════════════════════════════════════════

class MoleculeValidator:
    """
    Seven-layer physico-chemical and structural validation.

    L1 — RDKit sanitization
    L2 — Lipinski MW / logP / HBD / HBA
    L3 — TPSA & rotatable bonds
    L4 — Phenolic OH count (configurable pharmacophore gate)
    L5 — PAINS catalogue (pan-assay interference)
    L6 — Cinnamic/chalcone scaffold detection (warn only)
    L7 — Stereochemistry audit (warn only)
    """

    def __init__(self, cfg: CRESTConfig) -> None:
        self.cfg = cfg
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        self._pains        = FilterCatalog(params)
        self._phenol_sma   = Chem.MolFromSmarts("[OH]c")
        self._catechol_sma = Chem.MolFromSmarts("c1c(O)c(O)ccc1")
        self._cinnamic_sma = Chem.MolFromSmarts("C=CC(=O)")
        self._chalcone_sma = Chem.MolFromSmarts("c/C=C/C=O")

    def validate(self, entry: Dict) -> Tuple[bool, List[str], Dict]:
        smi  = entry["smiles"]
        name = entry["name"]
        msgs:  List[str] = []
        descs: Dict      = {}
        fails: List[str] = []

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False, ["L1 FAIL: cannot parse standardised SMILES"], descs
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            return False, [f"L1 FAIL: sanitization error — {exc}"], descs

        mw   = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd  = CalcNumHBD(mol)
        hba  = CalcNumHBA(mol)
        descs.update(MW=round(mw, 2), LogP=round(logp, 3), HBD=hbd, HBA=hba)

        if mw  > self.cfg.mw_max:  fails.append(f"L2 FAIL: MW {mw:.1f} > {self.cfg.mw_max}")
        if logp> self.cfg.logp_max: msgs.append(f"L2 WARN: LogP {logp:.2f} > {self.cfg.logp_max}")
        if hbd > self.cfg.hbd_max:  fails.append(f"L2 FAIL: HBD {hbd} > {self.cfg.hbd_max}")
        if hba > self.cfg.hba_max:  fails.append(f"L2 FAIL: HBA {hba} > {self.cfg.hba_max}")

        tpsa  = CalcTPSA(mol)
        nrotb = CalcNumRotatableBonds(mol)
        descs.update(TPSA=round(tpsa, 2), RotBonds=nrotb)
        if tpsa > self.cfg.tpsa_max:
            msgs.append(f"L3 WARN: TPSA {tpsa:.1f} > {self.cfg.tpsa_max}")

        n_poh   = len(mol.GetSubstructMatches(self._phenol_sma))
        has_cat = bool(mol.GetSubstructMatches(self._catechol_sma))
        descs.update(PhenolicOH=n_poh, Catechol=has_cat)
        if self.cfg.n_phenolic_oh_min > 0 and n_poh < self.cfg.n_phenolic_oh_min:
            msgs.append(f"L4 WARN: {n_poh} phenolic OH < required {self.cfg.n_phenolic_oh_min}")

        hit = self._pains.GetFirstMatch(mol)
        if hit:
            msgs.append(f"L5 WARN: PAINS — {hit.GetDescription()}")

        descs["HasCinnamicCore"] = bool(mol.GetSubstructMatches(self._cinnamic_sma)) or \
                                   bool(mol.GetSubstructMatches(self._chalcone_sma))

        n_sc  = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
        n_usc = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
        descs.update(StereoCenters=n_sc, UnspecifiedStereoCenters=n_usc)
        if n_usc > 0:
            msgs.append(f"L7 WARN: {n_usc}/{n_sc} stereocentre(s) unspecified")

        descs["QED"]           = round(QED.qed(mol), 4)
        descs["FracCSP3"]      = round(rdMolDescriptors.CalcFractionCSP3(mol), 4)
        descs["AromaticRings"] = rdMolDescriptors.CalcNumAromaticRings(mol)

        return len(fails) == 0, fails + msgs, descs


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — 3-D STRUCTURE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_3d_structure(smi: str, name: str, cfg: CRESTConfig, out_dir: Path) -> Optional[Path]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        log.error(f"  [{name}] 3D-gen: SMILES parse failed")
        return None
    mol = Chem.AddHs(mol)

    if _embed_molecule(mol, cfg.random_seed, cfg.max_embed_attempts) == -1:
        log.error(f"  [{name}] 3D-gen: all embedding attempts failed")
        return None

    try:
        ff_res = AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=cfg.ff_steps)
        if ff_res == 1:
            log.warning(f"  [{name}] MMFF94s did not fully converge")
        elif ff_res == -1:
            AllChem.UFFOptimizeMolecule(mol, maxIters=cfg.ff_steps)
            log.warning(f"  [{name}] MMFF94s unavailable — used UFF fallback")
    except Exception as exc:
        log.warning(f"  [{name}] FF minimisation skipped ({exc})")

    xyz_path = out_dir / f"{name}_rdkit.xyz"
    try:
        conf    = mol.GetConformer()
        n_atoms = mol.GetNumAtoms()
        with open(xyz_path, "w") as fh:
            fh.write(f"{n_atoms}\n")
            fh.write(f"RDKit ETKDGv3+MMFF94s | {name} | SMILES={smi}\n")
            for i in range(n_atoms):
                sym = mol.GetAtomWithIdx(i).GetSymbol()
                pos = conf.GetAtomPosition(i)
                fh.write(f"{sym:<3} {pos.x:14.8f} {pos.y:14.8f} {pos.z:14.8f}\n")
    except Exception as exc:
        log.error(f"  [{name}] XYZ write failed: {exc}")
        return None

    log.info(f"  [{name}] 3D geometry → {xyz_path.name}  ({n_atoms} atoms)")
    return xyz_path


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — xTB PRE-OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════

def run_xtb_preopt(xyz_path: Path, name: str, cfg: CRESTConfig, work_dir: Path) -> Tuple[Path, bool]:
    fallback = work_dir / f"{name}_xtb.xyz"
    log_path = work_dir / f"{name}_xtb.log"
    xtb_bin  = _find_binary("xtb")

    if xtb_bin is None:
        log.warning(f"  [{name}] xTB not found — using RDKit geometry directly")
        shutil.copy(xyz_path, fallback)
        return fallback, False

    solvent_args = ["--alpb", cfg.solvent] if cfg.solvent else []
    cmd = [
        xtb_bin, str(xyz_path.resolve()),
        f"--{cfg.xtb_method}", "--opt", "tight",
        "--charge", str(cfg.xtb_charge), "--uhf", str(cfg.xtb_uhf),
        "--parallel", str(max(1, cfg.ncores)), "--namespace", name,
    ] + solvent_args

    log.info(f"  [{name}] xTB pre-opt: {' '.join(cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(work_dir),
                              capture_output=True, text=True, timeout=900)
        with open(log_path, "w") as fh:
            fh.write(proc.stdout)
            if proc.stderr:
                fh.write("\n--- STDERR ---\n" + proc.stderr)
    except subprocess.TimeoutExpired:
        log.error(f"  [{name}] xTB timeout — using RDKit geometry")
        shutil.copy(xyz_path, fallback)
        return fallback, False
    except Exception as exc:
        log.error(f"  [{name}] xTB error ({exc}) — using RDKit geometry")
        shutil.copy(xyz_path, fallback)
        return fallback, False

    elapsed = time.time() - t0
    if proc.returncode != 0:
        log.warning(f"  [{name}] xTB exit {proc.returncode} ({elapsed:.1f} s) — using RDKit geometry")
        shutil.copy(xyz_path, fallback)
        return fallback, False

    for cand in [work_dir / f"{name}.xtbopt.xyz", work_dir / "xtbopt.xyz"]:
        if cand.exists():
            shutil.copy(cand, fallback)
            log.info(f"  [{name}] xTB pre-opt done ({elapsed:.1f} s)")
            return fallback, True

    log.warning(f"  [{name}] xTB optimised XYZ not found — using RDKit geometry")
    shutil.copy(xyz_path, fallback)
    return fallback, False


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — CREST CONFORMER SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def run_crest(xyz_path: Path, name: str, cfg: CRESTConfig, work_dir: Path) -> Optional[Path]:
    crest_bin = _find_binary("crest")
    if crest_bin is None:
        log.warning(f"  [{name}] CREST binary not found in PATH")
        return None

    xtb_bin  = _find_binary("xtb") or "xtb"
    sol_args = ["--alpb", cfg.solvent] if cfg.solvent else []
    ver_flag = "--v3" if cfg.crest_version == 3 else "--v2"

    cmd = [
        crest_bin, str(xyz_path.resolve()),
        ver_flag, "--xnam", xtb_bin,
        "--T", str(max(1, cfg.ncores)),
        "--ewin", str(cfg.ewin_kcal),
        "--rthr", str(cfg.rmsd_thr),
        "--charge", str(cfg.xtb_charge),
        "--uhf",    str(cfg.xtb_uhf),
        "--quick",
    ] + sol_args

    log.info(f"  [{name}] CREST iMTD-GC: {' '.join(cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(work_dir),
                              capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        log.error(f"  [{name}] CREST timeout (2 h)")
        return None
    except Exception as exc:
        log.error(f"  [{name}] CREST error: {exc}")
        return None

    elapsed = time.time() - t0
    crest_log = work_dir / f"{name}_crest.log"
    with open(crest_log, "w") as fh:
        fh.write(proc.stdout)
        if proc.stderr:
            fh.write("\n--- STDERR ---\n" + proc.stderr)

    if proc.returncode != 0:
        log.error(f"  [{name}] CREST exit {proc.returncode} ({elapsed:.0f} s)")
        return None

    for cand in [work_dir / "crest_conformers.xyz",
                 *work_dir.glob("*.conformers.xyz"),
                 *work_dir.glob("conformers*.xyz")]:
        if Path(cand).exists():
            log.info(f"  [{name}] CREST finished ({elapsed:.0f} s) → {Path(cand).name}")
            return Path(cand)

    log.error(f"  [{name}] CREST ran but conformer file not found")
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — CONFORMER ENSEMBLE DATACLASS & XYZ PARSER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConformerEnsemble:
    """
    Container for a multi-structure XYZ conformer ensemble.

    Attributes
    ----------
    name                  : molecule name
    n_atoms               : atom count (constant across conformers)
    energies_hartree      : active energies (Eₕ) — GFN2-xTB initially;
                            overwritten by QM/ML filters (ORCA, ANI, …)
                            to hold the highest-quality available energy.
    xtb_energies_hartree  : snapshot of the original GFN2-xTB energies
                            taken by the first QM/ML filter that replaces
                            energies_hartree.  Empty for xTB-only runs.
    energy_level          : human-readable label for energies_hartree
                            e.g. "GFN2-xTB", "r2SCAN-3c/DefGrid2", "ANI2x"
    crest_original_rank   : 0-based index of each conformer in the raw
                            CREST ensemble (preserved through sub-ensemble
                            views so the final XYZ always records it).
    free_energies_h       : GFN2-xTB free energies (Eₕ), if available
    coord_blocks          : list of XYZ coordinate line-blocks (strings)
    element_symbols       : list of element symbols from first conformer
    """
    name:               str
    n_atoms:            int              = 0
    energies_hartree:   List[float]      = field(default_factory=list)
    free_energies_h:    List[float]      = field(default_factory=list)
    coord_blocks:       List[List[str]]  = field(default_factory=list)
    element_symbols:    List[str]        = field(default_factory=list)
    # Original GFN2-xTB energies from CREST — never modified after initial parse.
    # This is the ground-truth baseline for all energy comparisons in the XYZ
    # comment block (E_GFN2xTB=...).  Stored separately from xtb_energies_hartree
    # so that chains like XTB_RERANK→R2SCAN3C don't overwrite it with tight-xTB.
    gfn2_energies_hartree: List[float]  = field(default_factory=list)
    # Snapshot of GFN2-xTB taken by the first QM/ML filter that replaces
    # energies_hartree.  Kept for backwards-compatibility with external tools
    # that may reference this field by name.
    xtb_energies_hartree: List[float]   = field(default_factory=list)
    # Label for whatever level energies_hartree currently holds
    energy_level:       str             = "GFN2-xTB"
    # CREST original rank (0-based index in the raw CREST ensemble)
    crest_original_rank: List[int]      = field(default_factory=list)

    @property
    def n_conformers(self) -> int:
        return len(self.energies_hartree)

    @property
    def delta_e_kcal(self) -> List[float]:
        if not self.energies_hartree:
            return []
        E0 = min(self.energies_hartree)
        return [(e - E0) * _HARTREE_TO_KCAL for e in self.energies_hartree]

    @property
    def delta_g_kcal(self) -> List[float]:
        if not self.free_energies_h:
            return self.delta_e_kcal
        G0 = min(self.free_energies_h)
        return [(g - G0) * _HARTREE_TO_KCAL for g in self.free_energies_h]

    def boltzmann_weights(self, T: float = 298.15, use_free_g: bool = False) -> List[float]:
        kBT = _R_KCAL * T
        dE  = self.delta_g_kcal if (use_free_g and self.free_energies_h) else self.delta_e_kcal
        w   = [math.exp(-de / kBT) for de in dE]
        Z   = sum(w)
        return [wi / Z for wi in w] if Z > 0 else [1.0 / len(w)] * len(w)


def parse_crest_xyz(xyz_path: Path, name: str) -> Optional[ConformerEnsemble]:
    """
    Parse a multi-structure XYZ file written by CREST.
    Extracts energy (and optionally free energy) from the comment line.
    """
    ens   = ConformerEnsemble(name=name)
    lines = xyz_path.read_text(errors="replace").splitlines()
    i     = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1; continue
        try:
            n = int(line)
        except ValueError:
            i += 1; continue
        if i + n + 1 >= len(lines):
            break

        comment = lines[i + 1].strip()
        floats  = []
        for tok in comment.split():
            try:
                floats.append(float(tok))
            except ValueError:
                continue

        energy = floats[0] if floats else 0.0
        free_e = floats[1] if len(floats) >= 2 else None

        coords = lines[i + 2 : i + 2 + n]
        if len(coords) < n:
            break

        ens.energies_hartree.append(energy)
        ens.gfn2_energies_hartree.append(energy)   # permanent GFN2-xTB baseline
        ens.xtb_energies_hartree.append(energy)    # snapshot (may be re-set by XTB_RERANK)
        ens.crest_original_rank.append(len(ens.energies_hartree) - 1)
        if free_e is not None:
            ens.free_energies_h.append(free_e)
        ens.coord_blocks.append(coords)
        ens.n_atoms = n

        # Collect element symbols from first conformer
        if not ens.element_symbols:
            for cl in coords:
                parts = cl.split()
                if parts:
                    ens.element_symbols.append(parts[0])

        i += 2 + n

    return ens if ens.n_conformers > 0 else None


def write_xyz_multiframe(
    ens: ConformerEnsemble,
    indices: List[int],
    out_path: Path,
    comment_prefix: str = "",
) -> None:
    """Write a subset of conformers as a multi-frame XYZ file."""
    with open(out_path, "w") as fh:
        for rank, idx in enumerate(indices, 1):
            fh.write(f"{ens.n_atoms}\n")
            fh.write(f"{comment_prefix}conf_{rank}  E={ens.energies_hartree[idx]:.8f}_Eh\n")
            for line in ens.coord_blocks[idx]:
                fh.write(line + "\n")


def parse_xyz_energies_from_comments(xyz_path: Path) -> List[float]:
    """
    Parse energies from the comment lines of a multi-frame XYZ file.
    Looks for patterns like E=-xxx.xxx or just a bare float.
    Returns list of energies in the order found; falls back to 0.0 on failure.
    """
    energies = []
    lines = xyz_path.read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1; continue
        try:
            n = int(line)
        except ValueError:
            i += 1; continue
        if i + 1 >= len(lines):
            break
        comment = lines[i + 1].strip()
        energy  = 0.0
        # Try E= pattern first
        m = re.search(r"E=(-?\d+\.\d+)", comment)
        if m:
            energy = float(m.group(1))
        else:
            for tok in comment.split():
                try:
                    energy = float(tok)
                    break
                except ValueError:
                    continue
        energies.append(energy)
        i += 2 + n
    return energies


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — BUILT-IN FILTERING METHODS (Layer A)
# ══════════════════════════════════════════════════════════════════════════════

"""
BUILT-IN FILTER OVERVIEW
========================
These filters operate purely on the ConformerEnsemble object and the
GFN2-xTB energies already present in the CREST output.  No additional
external calculations are required.

1. BOLTZMANN       — Classical Boltzmann population filter (ΔE)
2. FREE_ENERGY     — Boltzmann on ΔG from CREST RRHO frequencies
3. ENTROPY_PRUNING — Maximum-entropy / information-content pruning
4. TORSION_RMSD    — Dihedral-fingerprint greedy diversity selection
5. CLUSTER_REP     — Cartesian RMSD hierarchical clustering
"""

def _coord_block_to_array(coord_block: List[str]) -> Optional[np.ndarray]:
    coords = []
    for line in coord_block:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue
    return np.array(coords) if coords else None


def _cartesian_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    delta = a - b
    return float(np.sqrt(np.mean(np.einsum("ij,ij->i", delta, delta))))


def _dihedral_from_coords(c: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    b1 = c[j] - c[i]
    b2 = c[k] - c[j]
    b3 = c[l] - c[k]
    n1 = np.cross(b1, b2); n1n = np.linalg.norm(n1)
    n2 = np.cross(b2, b3); n2n = np.linalg.norm(n2)
    if n1n < 1e-8 or n2n < 1e-8:
        return 0.0
    n1 /= n1n; n2 /= n2n
    angle = math.degrees(math.acos(np.clip(np.dot(n1, n2), -1.0, 1.0)))
    if np.dot(np.cross(n1, n2), b2) < 0:
        angle = -angle
    return angle


def _get_rotatable_bond_atom_quads(smi: str, n_atoms: int) -> List[Tuple[int, int, int, int]]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []
    quads: List[Tuple[int, int, int, int]] = []
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondTypeAsDouble() != 1.0:
            continue
        j = bond.GetBeginAtomIdx()
        k = bond.GetEndAtomIdx()
        j_nbrs = [a.GetIdx() for a in mol.GetAtomWithIdx(j).GetNeighbors() if a.GetIdx() != k]
        k_nbrs = [a.GetIdx() for a in mol.GetAtomWithIdx(k).GetNeighbors() if a.GetIdx() != j]
        if j_nbrs and k_nbrs:
            quads.append((j_nbrs[0], j, k, k_nbrs[0]))
    return quads


# ─────────────────────────────────────────────────────────────────────────────

def _filter_boltzmann(ens: ConformerEnsemble, cfg: CRESTConfig, **kw) -> Tuple[List[int], Dict]:
    """
    Classical Boltzmann population filter on raw GFN2-xTB ΔE.

    Retains conformers with fractional Boltzmann population ≥ boltzmann_threshold.
    Reference: standard statistical mechanics weighting.
    """
    pops = ens.boltzmann_weights(T=cfg.T_kelvin, use_free_g=False)
    dE   = ens.delta_e_kcal
    selected = [i for i, p in enumerate(pops) if p >= cfg.boltzmann_threshold
                ][:cfg.max_conformers_output]
    return selected, {
        "filter_method":         "BOLTZMANN",
        "filter_T_K":            cfg.T_kelvin,
        "filter_threshold":      cfg.boltzmann_threshold,
        "pop_top_conformer_pct": round(pops[0] * 100, 2),
        "dE_max_kept_kcal":      round(dE[selected[-1]], 4) if selected else 0.0,
    }


def _filter_free_energy(ens: ConformerEnsemble, cfg: CRESTConfig, **kw) -> Tuple[List[int], Dict]:
    """
    Boltzmann weighting on GFN2-xTB free energies (ΔG).
    Falls back to ΔE if RRHO frequencies are unavailable.

    Reference: Grimme, Schreiner, ACIE 2018, 57, 4170 (RRHO); Bannwarth, JCTC 2019 (ALPB).
    """
    has_g = bool(ens.free_energies_h)
    pops  = ens.boltzmann_weights(T=cfg.T_kelvin, use_free_g=has_g)
    dG    = ens.delta_g_kcal
    if not has_g:
        log.warning(f"  [{ens.name}] FREE_ENERGY: no free energies in CREST output — using ΔE")
    selected = [i for i, p in enumerate(pops) if p >= cfg.boltzmann_threshold
                ][:cfg.max_conformers_output]
    return selected, {
        "filter_method":         "FREE_ENERGY" if has_g else "FREE_ENERGY→BOLTZMANN",
        "filter_T_K":            cfg.T_kelvin,
        "used_free_energy":      has_g,
        "pop_top_conformer_pct": round(pops[0] * 100, 2),
        "dG_max_kept_kcal":      round(dG[selected[-1]], 4) if selected else 0.0,
    }


def _filter_entropy_pruning(ens: ConformerEnsemble, cfg: CRESTConfig, **kw) -> Tuple[List[int], Dict]:
    """
    Maximum-entropy ensemble pruning.
    Iteratively removes conformers that decrease ensemble Shannon entropy
    below a MAD-based threshold.

    Reference: Sitzmann et al. JCIM 2010, 50, 193.
    """
    use_g = bool(ens.free_energies_h)
    pops  = np.array(ens.boltzmann_weights(T=cfg.T_kelvin, use_free_g=use_g))
    keep  = list(range(ens.n_conformers))

    def _entropy(w: np.ndarray) -> float:
        w = w / w.sum()
        return float(-np.sum(w * np.log(w + 1e-300)))

    for _ in range(5 * ens.n_conformers):
        if len(keep) <= max(3, cfg.max_conformers_output):
            break
        w       = pops[keep]
        mean_w  = w.mean()
        mad     = np.abs(w - mean_w).mean()
        cands   = [idx for idx in keep if pops[idx] < mean_w - mad]
        if not cands:
            break
        worst = min(cands, key=lambda idx: _entropy(pops[[k for k in keep if k != idx]]))
        keep.remove(worst)

    keep = sorted(keep, key=lambda idx: ens.energies_hartree[idx])[:cfg.max_conformers_output]
    dE   = ens.delta_e_kcal
    return keep, {
        "filter_method":         "ENTROPY_PRUNING",
        "filter_T_K":            cfg.T_kelvin,
        "used_free_energy":      use_g,
        "ensemble_entropy_bits": round(_entropy(pops[keep]), 4),
        "dE_max_kept_kcal":      round(dE[keep[-1]], 4) if keep else 0.0,
    }


def _filter_torsion_rmsd(ens: ConformerEnsemble, cfg: CRESTConfig,
                         smi: str = "", **kw) -> Tuple[List[int], Dict]:
    """
    Torsion-RMSD diversity + energy gate.
    Greedy selection of maximally diverse conformers based on dihedral fingerprints.

    References:
    Watts et al. JCIM 2010, 50, 534; Friedrich et al. JCTC 2017, 13, 3949.
    """
    quads = _get_rotatable_bond_atom_quads(smi, ens.n_atoms)
    dE    = ens.delta_e_kcal

    if not quads:
        log.warning(f"  [{ens.name}] TORSION_RMSD: no rotatable bonds — using BOLTZMANN")
        return _filter_boltzmann(ens, cfg)

    torsion_fps: List[Optional[np.ndarray]] = []
    for cb in ens.coord_blocks:
        coords = _coord_block_to_array(cb)
        if coords is None:
            torsion_fps.append(None)
            continue
        vq = [q for q in quads if max(q) < len(coords)]
        if not vq:
            torsion_fps.append(None)
            continue
        torsion_fps.append(np.array([_dihedral_from_coords(coords, *q) for q in vq]))

    def _tdist(a, b):
        d = np.abs(a - b)
        return float(np.sqrt(np.mean(np.minimum(d, 360.0 - d) ** 2)))

    energy_gate = sorted(
        [i for i, de in enumerate(dE) if de <= cfg.ewin_kcal and torsion_fps[i] is not None],
        key=lambda i: ens.energies_hartree[i]
    ) or list(range(ens.n_conformers))

    selected = [energy_gate[0]]
    for idx in energy_gate[1:]:
        if len(selected) >= cfg.max_conformers_output:
            break
        fp_i = torsion_fps[idx]
        if fp_i is None:
            continue
        if min(_tdist(fp_i, torsion_fps[s]) for s in selected
               if torsion_fps[s] is not None) >= cfg.torsion_diff_thr:
            selected.append(idx)

    selected = sorted(selected, key=lambda i: ens.energies_hartree[i])
    return selected, {
        "filter_method":        "TORSION_RMSD",
        "n_rotatable_bonds":    len(quads),
        "torsion_diff_thr_deg": cfg.torsion_diff_thr,
        "dE_max_kept_kcal":     round(max(dE[i] for i in selected), 4) if selected else 0.0,
    }


def _filter_cluster_rep(ens: ConformerEnsemble, cfg: CRESTConfig, **kw) -> Tuple[List[int], Dict]:
    """
    Energy-weighted RMSD clustering with lowest-energy cluster-representative selection.
    Uses complete-linkage hierarchical clustering.

    References:
    Saunders, JACS 1987; Torda & van Gunsteren, JCC 1994.
    """
    arrays = [_coord_block_to_array(cb) for cb in ens.coord_blocks]
    dE     = ens.delta_e_kcal
    n      = ens.n_conformers
    thr    = cfg.rmsd_cluster_thr

    dist = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            d = _cartesian_rmsd(arrays[a], arrays[b]) if (
                arrays[a] is not None and arrays[b] is not None) else float("inf")
            dist[a, b] = dist[b, a] = d

    clusters: List[List[int]] = [[i] for i in range(n)]
    while len(clusters) > 1:
        min_d = float("inf")
        best  = (0, 1)
        for ci in range(len(clusters)):
            for cj in range(ci + 1, len(clusters)):
                d = max(dist[i][j] for i in clusters[ci] for j in clusters[cj])
                if d < min_d:
                    min_d = d; best = (ci, cj)
        if min_d > thr:
            break
        ci, cj = best
        merged = clusters[ci] + clusters[cj]
        clusters = [c for idx, c in enumerate(clusters) if idx not in (ci, cj)] + [merged]

    representatives = sorted(
        [min(cl, key=lambda i: ens.energies_hartree[i]) for cl in clusters],
        key=lambda i: ens.energies_hartree[i]
    )[:cfg.max_conformers_output]

    return representatives, {
        "filter_method":      "CLUSTER_REP",
        "n_clusters":         len(clusters),
        "rmsd_cluster_thr_A": thr,
        "dE_max_kept_kcal":   round(max(dE[i] for i in representatives), 4) if representatives else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — EXTERNAL QM/ML FILTERING METHODS (Layer B)
# ══════════════════════════════════════════════════════════════════════════════

"""
EXTERNAL FILTER OVERVIEW
========================
These filters call external tools (xTB, ORCA, CENSO, CONFPASS, TorchANI) to
compute higher-quality energies, then re-apply Boltzmann reranking on the
refined energy surface.

Each filter follows the same contract:
  Input : ConformerEnsemble (pre-filtered subset from Layer A or previous layer)
  Output: (selected_indices_into_ens, stats_dict)

They operate on the FULL coordinate blocks of the input ensemble but update
the energy surface from the external tool.  If the external tool fails or is
not installed, the filter logs a warning and falls back to BOLTZMANN.
"""


def _filter_xtb_rerank(
    ens:      ConformerEnsemble,
    cfg:      CRESTConfig,
    work_dir: Path,
    **kw,
) -> Tuple[List[int], Dict]:
    """
    XTB_RERANK — GFN2-xTB re-optimisation with tighter settings, then Boltzmann.

    Algorithm
    ---------
    1. Write each conformer as a separate XYZ file in work_dir/xtb_rerank/.
    2. Run: xtb conformer_N.xyz --gfn2 --opt {accuracy} --acc {acc_val}
    3. Parse refined energies from xtb output.
    4. Recompute Boltzmann weights on the new energy surface.
    5. Apply Boltzmann threshold, return ranked indices.

    The re-optimisation relaxes geometry artefacts from the CREST search and
    provides a more consistent energy surface for ranking.

    Settings
    --------
    --xtb_rerank_accuracy  : "tight" (--acc 0.2, recommended) or
                             "verytight" (--acc 0.05, slower, more accurate)

    Reference
    ---------
    Grimme, S. J. Chem. Theory Comput. 2019, 15, 2847–2862.
    """
    xtb_bin = _find_binary("xtb")
    if xtb_bin is None:
        log.warning(f"  [{ens.name}] XTB_RERANK: xtb not found — falling back to BOLTZMANN")
        return _filter_boltzmann(ens, cfg)

    rerank_dir = work_dir / "xtb_rerank"
    rerank_dir.mkdir(exist_ok=True)

    refined_energies: List[float] = [None] * ens.n_conformers  # type: ignore

    for idx in range(ens.n_conformers):
        conf_xyz = rerank_dir / f"conf_{idx:04d}.xyz"
        with open(conf_xyz, "w") as fh:
            fh.write(f"{ens.n_atoms}\n")
            fh.write(f"conformer {idx}\n")
            for line in ens.coord_blocks[idx]:
                fh.write(line + "\n")

        sol_args = ["--alpb", cfg.solvent] if cfg.solvent else []
        cmd = [
            xtb_bin, str(conf_xyz),
            "--gfn2", "--opt", cfg.xtb_rerank_accuracy,
            "--acc",  str(cfg.xtb_rerank_acc_val),
            "--charge", str(cfg.xtb_charge),
            "--uhf",    str(cfg.xtb_uhf),
            "--namespace", f"rerank_{idx:04d}",
        ] + sol_args

        try:
            proc = subprocess.run(
                cmd, cwd=str(rerank_dir), capture_output=True, text=True, timeout=600
            )
            log_file = rerank_dir / f"rerank_{idx:04d}.log"
            with open(log_file, "w") as fh:
                fh.write(proc.stdout)
                if proc.stderr:
                    fh.write("\n--- STDERR ---\n" + proc.stderr)

            if proc.returncode == 0:
                # Parse "TOTAL ENERGY" from xTB output (Hartree)
                energy = _parse_xtb_total_energy(proc.stdout)
                if energy is not None:
                    refined_energies[idx] = energy
                else:
                    log.warning(f"  [{ens.name}] XTB_RERANK: could not parse energy for conf {idx}")
                    refined_energies[idx] = ens.energies_hartree[idx]
            else:
                log.warning(f"  [{ens.name}] XTB_RERANK: xtb failed for conf {idx}, using original energy")
                refined_energies[idx] = ens.energies_hartree[idx]
        except Exception as exc:
            log.warning(f"  [{ens.name}] XTB_RERANK: conf {idx} error ({exc})")
            refined_energies[idx] = ens.energies_hartree[idx]

    # Replace energies in a temporary ensemble copy for Boltzmann
    refined = [e if e is not None else ens.energies_hartree[i]
               for i, e in enumerate(refined_energies)]
    E0 = min(refined)
    dE_new = [(e - E0) * _HARTREE_TO_KCAL for e in refined]

    kBT  = _R_KCAL * cfg.T_kelvin
    w    = [math.exp(-de / kBT) for de in dE_new]
    Z    = sum(w)
    pops = [wi / Z for wi in w] if Z > 0 else [1.0 / len(w)] * len(w)

    # Snapshot GFN2-xTB before we replace energies_hartree with tight-xTB
    ens.xtb_energies_hartree = list(ens.energies_hartree)

    ens.energies_hartree = list(refined)
    ens.energy_level     = f"GFN2-xTB/{cfg.xtb_rerank_accuracy}"

    order    = sorted(range(ens.n_conformers), key=lambda i: refined[i])
    selected = [i for i in order if pops[i] >= cfg.boltzmann_threshold
                ][:cfg.max_conformers_output]
    if not selected:
        selected = order[:min(3, len(order))]

    log.info(
        f"  [{ens.name}] XTB_RERANK: {ens.n_conformers} → {len(selected)} conformers "
        f"(accuracy={cfg.xtb_rerank_accuracy}, acc={cfg.xtb_rerank_acc_val})"
    )

    return selected, {
        "filter_method":        "XTB_RERANK",
        "xtb_rerank_accuracy":  cfg.xtb_rerank_accuracy,
        "xtb_acc_value":        cfg.xtb_rerank_acc_val,
        "dE_max_kept_kcal":     round(max(dE_new[i] for i in selected), 4) if selected else 0.0,
        "pop_top_conformer_pct": round(pops[selected[0]] * 100, 2) if selected else 0.0,
    }


def _parse_xtb_total_energy(xtb_stdout: str) -> Optional[float]:
    """Extract GFN2-xTB total energy (Hartree) from xtb stdout."""
    # Primary pattern: "          TOTAL ENERGY             -xx.xxxxxxx Eh"
    for line in reversed(xtb_stdout.splitlines()):
        if "TOTAL ENERGY" in line:
            parts = line.split()
            for tok in parts:
                try:
                    val = float(tok)
                    if val < -1.0:   # sanity: energy is a large negative number
                        return val
                except ValueError:
                    continue
    # Fallback: "total energy  =  -xx.xxxx" pattern
    m = re.search(r"total\s+energy\s*=\s*(-?\d+\.\d+)", xtb_stdout, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _filter_r2scan3c(
    ens:      ConformerEnsemble,
    cfg:      CRESTConfig,
    work_dir: Path,
    smi:      str = "",
    **kw,
) -> Tuple[List[int], Dict]:
    """
    R2SCAN3C — ORCA r²SCAN-3c single-point energies, then Boltzmann reranking.

    The r²SCAN-3c composite method (Grimme, JCP 2021) provides near-DFT accuracy
    at a fraction of the cost of a standard DFT/large-basis calculation.  It
    combines the r²SCAN meta-GGA functional with the mTZVPP basis set and
    D4 dispersion + geometrical counterpoise correction (gCP).

    Algorithm
    ---------
    1. Write xTB-quality geometry for each conformer.
    2. Run ORCA single-point with ! r2SCAN-3c TightSCF DefGrid2.
    3. Parse ORCA total energies.
    4. Boltzmann reranking on ORCA energy surface.

    Requires
    --------
    ORCA ≥ 5.0 binary in PATH (or set ORCA_EXE env variable).
    Download: https://orcaforum.kofo.mpg.de

    Reference
    ---------
    Grimme, S.; Hansen, A.; Ehlert, S.; Mewes, J.-M.
    J. Chem. Phys. 2021, 155, 104111.
    """
    orca_bin = _find_binary("orca")
    if orca_bin is None:
        log.warning(f"  [{ens.name}] R2SCAN3C: ORCA not found — falling back to BOLTZMANN")
        log.warning(f"  [{ens.name}]   Install ORCA from https://orcaforum.kofo.mpg.de")
        log.warning(f"  [{ens.name}]   Set ORCA_EXE=/path/to/orca or add to PATH")
        return _filter_boltzmann(ens, cfg)

    # ── MPI launcher detection ────────────────────────────────────────────────
    # ORCA MPI builds call mpirun internally.  Locate it on the system and
    # inject its directory into the child PATH + LD_LIBRARY_PATH.  If no MPI
    # launcher is found, fall back to nprocs=1 (serial) so the calculation
    # still runs on workstations without a system-wide MPI stack.
    #
    # Override priority:
    #   cfg.orca_mpi_launcher = ""  → force serial (user passed --orca_mpi_launcher none)
    #   cfg.orca_mpi_launcher = str → use that explicit path
    #   cfg.orca_mpi_launcher = None → auto-detect with _find_mpirun()
    if cfg.orca_mpi_launcher == "":
        # User explicitly requested serial mode
        mpirun_path = None
        log.info(f"  [{ens.name}] R2SCAN3C: serial mode requested — nprocs=1")
    elif cfg.orca_mpi_launcher:
        # User supplied explicit path
        p = Path(cfg.orca_mpi_launcher)
        mpirun_path = str(p) if p.is_file() else None
        if mpirun_path:
            log.info(f"  [{ens.name}] R2SCAN3C: using user-specified MPI launcher → {mpirun_path}")
        else:
            log.warning(
                f"  [{ens.name}] R2SCAN3C: --orca_mpi_launcher path not found "
                f"({cfg.orca_mpi_launcher}) — falling back to auto-detect"
            )
            mpirun_path = _find_mpirun()
    else:
        mpirun_path = _find_mpirun()
        if mpirun_path:
            log.info(f"  [{ens.name}] R2SCAN3C: MPI launcher auto-detected → {mpirun_path}")
        else:
            log.warning(
                f"  [{ens.name}] R2SCAN3C: mpirun/mpiexec not found — "
                "running ORCA in serial mode (nprocs=1).  "
                "To enable MPI: set MPIRUN_EXE, install OpenMPI, "
                "or use --orca_mpi_launcher /path/to/mpirun"
            )

    orca_env    = _build_orca_env(orca_bin, mpirun_path)
    orca_nprocs = cfg.orca_nprocs if mpirun_path else 1

    orca_dir = work_dir / "orca_r2scan3c"
    orca_dir.mkdir(exist_ok=True)

    refined_energies: List[Optional[float]] = [None] * ens.n_conformers

    for idx in range(ens.n_conformers):
        # Also write a standalone .xyz for reference / external inspection
        conf_xyz = orca_dir / f"conf_{idx:04d}.xyz"
        with open(conf_xyz, "w") as fh:
            fh.write(f"{ens.n_atoms}\n")
            fh.write(f"conformer {idx}\n")
            for line in ens.coord_blocks[idx]:
                fh.write(line + "\n")

        # ── Build ORCA input ────────────────────────────────────────────────
        # Coordinates are embedded inline (no external xyzfile dependency).
        # This is the only portable approach: ORCA must be invoked by filename
        # alone (not absolute path) when cwd is already the working directory,
        # and an inline * xyz block avoids any relative-path resolution issues
        # with the external .xyz file.

        solv_block = ""
        if cfg.solvent:
            # ORCA CPCM/SMD solvent mapping from xTB/ALPB names
            _orca_solvent_map = {
                "water":       "water",
                "chcl3":       "chloroform",
                "dmso":        "dmso",
                "thf":         "thf",
                "acetonitrile":"acetonitrile",
                "methanol":    "methanol",
            }
            orca_solv  = _orca_solvent_map.get(cfg.solvent, cfg.solvent)
            solv_block = f"\n%cpcm\n  smd true\n  smdsolvent \"{orca_solv}\"\nend\n"

        # Build inline coordinate block
        coord_lines = "\n".join(ens.coord_blocks[idx])
        multiplicity = cfg.xtb_uhf * 2 + 1

        # nprocs line: use serial fallback value when MPI is unavailable
        pal_line = f"%pal nprocs {orca_nprocs} end\n" if orca_nprocs > 1 else ""

        inp_text = (
            f"! r2SCAN-3c TightSCF {cfg.r2scan3c_grid}\n"
            f"{pal_line}"
            f"%maxcore {cfg.orca_maxcore_mb}\n"
            f"{solv_block}"
            f"* xyz {cfg.xtb_charge} {multiplicity}\n"
            f"{coord_lines}\n"
            f"*\n"
        )

        inp_name = f"conf_{idx:04d}.inp"   # filename only — cwd is orca_dir
        inp_file = orca_dir / inp_name
        out_file = orca_dir / f"conf_{idx:04d}.out"
        with open(inp_file, "w") as fh:
            fh.write(inp_text)

        log.info(f"  [{ens.name}] R2SCAN3C: running conf {idx:04d}  "
                 f"({orca_nprocs} core(s))  [{orca_bin} {inp_name}]")

        # Pass the filename only; subprocess cwd=orca_dir means ORCA finds it
        # without absolute-path mangling.  Supply patched env for MPI support.
        cmd = [orca_bin, inp_name]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(orca_dir),
                capture_output=True,
                text=True,
                timeout=3600,
                env=orca_env,
            )
            with open(out_file, "w") as fh:
                fh.write(proc.stdout)
                if proc.stderr:
                    fh.write("\n--- STDERR ---\n" + proc.stderr)

            # Try parsing from stdout first; fall back to .property.txt
            energy = _parse_orca_total_energy(proc.stdout, out_path=out_file)
            if energy is not None:
                refined_energies[idx] = energy
                log.info(
                    f"  [{ens.name}] R2SCAN3C conf {idx:04d}: "
                    f"E = {energy:.10f} Eh"
                )
            else:
                # Surface the actual ORCA error so the user can act on it
                err_snippet = ""
                for line in reversed((proc.stdout + proc.stderr).splitlines()):
                    line = line.strip()
                    if line and not line.startswith("---"):
                        err_snippet = line
                        break
                log.warning(
                    f"  [{ens.name}] R2SCAN3C conf {idx:04d}: "
                    f"energy parse failed — using xTB fallback.  "
                    f"Last ORCA line: {err_snippet!r}  "
                    f"(see {out_file.name})"
                )
                refined_energies[idx] = ens.energies_hartree[idx]
        except subprocess.TimeoutExpired:
            log.warning(
                f"  [{ens.name}] R2SCAN3C conf {idx:04d}: "
                "ORCA timed out (3600 s) — using xTB fallback"
            )
            refined_energies[idx] = ens.energies_hartree[idx]
        except Exception as exc:
            log.warning(
                f"  [{ens.name}] R2SCAN3C conf {idx:04d}: "
                f"subprocess error ({exc}) — using xTB fallback"
            )
            refined_energies[idx] = ens.energies_hartree[idx]

    refined = [e if e is not None else ens.energies_hartree[i]
               for i, e in enumerate(refined_energies)]

    # ── Write ORCA energies back into the ensemble ────────────────────────
    # Snapshot GFN2-xTB before we replace energies_hartree with ORCA r2scan3c.
    # Compare against gfn2_energies_hartree (permanent original) rather than
    # xtb_energies_hartree, so that chains like XTB_RERANK→R2SCAN3C correctly
    # preserve the original GFN2-xTB baseline (not tight-xTB) for comparison.
    if not ens.xtb_energies_hartree or (
        ens.xtb_energies_hartree == ens.gfn2_energies_hartree
    ):
        ens.xtb_energies_hartree = list(ens.energies_hartree)

    ens.energies_hartree = list(refined)   # ensemble now carries ORCA energies
    ens.energy_level     = f"r2SCAN-3c/{cfg.r2scan3c_grid}"

    E0     = min(refined)
    dE_new = [(e - E0) * _HARTREE_TO_KCAL for e in refined]

    kBT  = _R_KCAL * cfg.T_kelvin
    w    = [math.exp(-de / kBT) for de in dE_new]
    Z    = sum(w)
    pops = [wi / Z for wi in w] if Z > 0 else [1.0 / len(w)] * len(w)

    order    = sorted(range(ens.n_conformers), key=lambda i: refined[i])
    selected = [i for i in order if pops[i] >= cfg.boltzmann_threshold
                ][:cfg.max_conformers_output]
    if not selected:
        selected = order[:min(3, len(order))]

    n_orca_ok = sum(1 for e in refined_energies if e is not None)
    log.info(
        f"  [{ens.name}] R2SCAN3C: {ens.n_conformers} → {len(selected)} conformers "
        f"(ORCA r²SCAN-3c/{cfg.r2scan3c_grid}, "
        f"{n_orca_ok}/{ens.n_conformers} SP energies from ORCA)"
    )

    return selected, {
        "filter_method":         "R2SCAN3C",
        "orca_method":           f"r2SCAN-3c/{cfg.r2scan3c_grid}",
        "n_orca_energies_ok":    n_orca_ok,
        "n_orca_fallback_xtb":   ens.n_conformers - n_orca_ok,
        "dE_max_kept_kcal":      round(max(dE_new[i] for i in selected), 4) if selected else 0.0,
        "pop_top_conformer_pct": round(pops[selected[0]] * 100, 2) if selected else 0.0,
    }


def _parse_orca_total_energy(
    orca_stdout: str,
    out_path: Optional[Path] = None,
) -> Optional[float]:
    """
    Extract the final single-point total energy (Hartree) from an ORCA run.

    Handles all known ORCA 5.x and 6.x output formats:

      1. Primary   — "FINAL SINGLE POINT ENERGY   -xxx.xxxxxxxx"
                     (present in every successful SP run; iterated in reverse
                     so the last occurrence is returned for geometry-scan jobs)
      2. Secondary — "Total Energy       =    -xxx.xxxxxxxx Eh"
                     (DFT/HF energy line in the SCF energy print block)
      3. Tertiary  — ORCA 6 .property.txt file alongside the .out file:
                     "SCFEnergy" or "TotalEnergy" key-value pairs
      4. Quaternary— "TOTAL ENERGY     -xxx.xxxxxxxx" regex fallback

    Parameters
    ----------
    orca_stdout : str
        Full text of the ORCA .out file.
    out_path : Path, optional
        Path to the .out file on disk.  If supplied, the co-located
        .property.txt file (ORCA 6) is also searched.

    Returns
    -------
    float or None
        Total energy in Hartree, or None if parsing fails.
    """
    # ── 1. Primary: FINAL SINGLE POINT ENERGY line ───────────────────────────
    lines = orca_stdout.splitlines()
    for line in reversed(lines):
        if "FINAL SINGLE POINT ENERGY" in line:
            parts = line.split()
            for tok in reversed(parts):
                try:
                    val = float(tok)
                    if val < -1.0:
                        return val
                except ValueError:
                    continue

    # ── 2. Secondary: Total Energy = ... Eh (SCF energy print block) ─────────
    for line in reversed(lines):
        if "Total Energy" in line and "Eh" in line:
            m = re.search(r"=\s*(-\d+\.\d+)\s*Eh", line)
            if m:
                try:
                    val = float(m.group(1))
                    if val < -1.0:
                        return val
                except ValueError:
                    pass

    # ── 3. Tertiary: ORCA 6 .property.txt co-located file ────────────────────
    if out_path is not None:
        prop_file = out_path.with_suffix(".property.txt")
        if not prop_file.exists():
            # Also try base stem (e.g. conf_0000.property.txt)
            prop_file = out_path.parent / (out_path.stem + ".property.txt")
        if prop_file.exists():
            try:
                for pline in prop_file.read_text(errors="replace").splitlines():
                    for key in ("SCFEnergy", "TotalEnergy", "Total_Energy"):
                        if key in pline:
                            pm = re.search(r"(-\d+\.\d+)", pline)
                            if pm:
                                val = float(pm.group(1))
                                if val < -1.0:
                                    return val
            except Exception:
                pass

    # ── 4. Quaternary: loose regex over entire output ─────────────────────────
    m = re.search(r"TOTAL\s+ENERGY\s+(-\d+\.\d+)", orca_stdout, re.IGNORECASE)
    if m:
        try:
            val = float(m.group(1))
            if val < -1.0:
                return val
        except ValueError:
            pass

    return None


def _filter_censo(
    ens:      ConformerEnsemble,
    cfg:      CRESTConfig,
    work_dir: Path,
    smi:      str = "",
    **kw,
) -> Tuple[List[int], Dict]:
    """
    CENSO — Grimme-group multilevel DFT-lite conformer screening pipeline.

    CENSO (Conformer Ensemble from Semiempirical Ordering) takes a CREST
    ensemble and applies a cascade of screening steps: GFN2-xTB prescreening,
    DFT-based SP energies (typically r²SCAN-3c or PBE/def2-SV(P)), optional
    COSMO-RS solvation, and thermodynamic property computation.

    Algorithm
    ---------
    1. Write input ensemble as crest_conformers.xyz in censo_run/.
    2. Write a minimal .censorc configuration file.
    3. Run: censo --input crest_conformers.xyz --maxconf N
    4. Parse CENSO's ranking output (anmr_enso.dat or censo.out).
    5. Map ranked conformer indices back to our ensemble.

    Requires
    --------
    pip install censo
    Also requires xTB and (optionally) ORCA for DFT steps.
    GitHub: https://github.com/grimme-lab/censo

    Reference
    ---------
    Pracht, P.; Bohle, F.; Grimme, S.
    J. Chem. Theory Comput. 2020, 16, 7044–7060.
    """
    censo_bin = _find_binary("censo")
    if censo_bin is None:
        # Try python -m censo
        censo_bin = _find_python_module_binary("censo")
    if censo_bin is None:
        log.warning(f"  [{ens.name}] CENSO: censo not found — falling back to BOLTZMANN")
        log.warning(f"  [{ens.name}]   Install: pip install censo")
        log.warning(f"  [{ens.name}]   GitHub:  https://github.com/grimme-lab/censo")
        return _filter_boltzmann(ens, cfg)

    censo_dir = work_dir / "censo_run"
    censo_dir.mkdir(exist_ok=True)

    # Write ensemble as crest_conformers.xyz (CENSO's default input name)
    ens_file = censo_dir / "crest_conformers.xyz"
    with open(ens_file, "w") as fh:
        for idx in range(ens.n_conformers):
            fh.write(f"{ens.n_atoms}\n")
            fh.write(f"{ens.energies_hartree[idx]:.8f}\n")
            for line in ens.coord_blocks[idx]:
                fh.write(line + "\n")

    # Write minimal .censorc
    solvent_censo = cfg.censo_solvent or cfg.solvent or "gas"
    _censo_solv_map = {
        "water": "h2o", "chcl3": "chcl3", "dmso": "dmso",
        "thf": "thf", "acetonitrile": "acetonitrile", "methanol": "methanol",
    }
    solvent_censo = _censo_solv_map.get(solvent_censo, solvent_censo)

    censorc_content = textwrap.dedent(f"""\
        $general
        nconf = {cfg.censo_maxconf}
        charge = {cfg.xtb_charge}
        unpaired = {cfg.xtb_uhf}
        solvent = {solvent_censo}
        prog = xtb
        $end
        $part0
        prog = xtb
        $end
        $part1
        prog = xtb
        $end
    """)
    censorc_path = censo_dir / ".censorc"
    with open(censorc_path, "w") as fh:
        fh.write(censorc_content)

    cmd = [
        censo_bin,
        "--input", str(ens_file),
        "--maxconf", str(min(cfg.censo_maxconf, ens.n_conformers)),
    ]

    log.info(f"  [{ens.name}] CENSO: {' '.join(cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(censo_dir), capture_output=True, text=True, timeout=7200
        )
        censo_log = censo_dir / "censo_pipeline.log"
        with open(censo_log, "w") as fh:
            fh.write(proc.stdout)
            if proc.stderr:
                fh.write("\n--- STDERR ---\n" + proc.stderr)
    except Exception as exc:
        log.warning(f"  [{ens.name}] CENSO: subprocess error ({exc}) — falling back to BOLTZMANN")
        return _filter_boltzmann(ens, cfg)

    elapsed = time.time() - t0

    # Parse CENSO ranking output
    ranked_indices = _parse_censo_ranking(censo_dir, ens.n_conformers)
    if not ranked_indices:
        log.warning(f"  [{ens.name}] CENSO: could not parse ranking output — using BOLTZMANN")
        return _filter_boltzmann(ens, cfg)

    selected = ranked_indices[:cfg.max_conformers_output]
    dE       = ens.delta_e_kcal

    log.info(
        f"  [{ens.name}] CENSO: finished ({elapsed:.0f} s), "
        f"{ens.n_conformers} → {len(selected)} conformers"
    )

    return selected, {
        "filter_method":     "CENSO",
        "censo_maxconf":     cfg.censo_maxconf,
        "censo_solvent":     solvent_censo,
        "dE_max_kept_kcal":  round(max(dE[i] for i in selected if i < len(dE)), 4) if selected else 0.0,
    }


def _parse_censo_ranking(censo_dir: Path, n_total: int) -> List[int]:
    """
    Parse conformer ranking from CENSO output files.
    CENSO writes 'anmr_enso.dat' or 'censo_conformers.xyz' with ranked conformers.
    Returns 0-based indices into the original ensemble.
    """
    ranked: List[int] = []

    # Try anmr_enso.dat first (older CENSO)
    enso_dat = censo_dir / "anmr_enso.dat"
    if enso_dat.exists():
        for line in enso_dat.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts:
                try:
                    conf_id = int(parts[0]) - 1  # CENSO is 1-based
                    if 0 <= conf_id < n_total:
                        ranked.append(conf_id)
                except ValueError:
                    continue
        if ranked:
            return ranked

    # Try censo.out — look for conformer ranking table
    censo_out = censo_dir / "censo.out"
    if censo_out.exists():
        text = censo_out.read_text()
        for m in re.finditer(r"CONF(\d+)", text):
            idx = int(m.group(1)) - 1
            if 0 <= idx < n_total and idx not in ranked:
                ranked.append(idx)
        if ranked:
            return ranked

    # Fallback: parse censo_conformers.xyz if present
    for fname in ["censo_conformers.xyz", "ensemble.xyz"]:
        fpath = censo_dir / fname
        if fpath.exists():
            # These will be a re-ordered subset; map back by energy matching is not
            # straightforward without full parse, so return sequential indices
            energies = parse_xyz_energies_from_comments(fpath)
            if energies:
                # Return indices of conformers sorted by CENSO output order
                return list(range(min(len(energies), n_total)))

    return []


def _find_python_module_binary(module_name: str) -> Optional[str]:
    """Try to locate a Python-installed CLI tool via 'python -m module_name'."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", module_name, "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return f"{sys.executable} -m {module_name}"
    except Exception:
        pass
    return None


def _filter_confpass(
    ens:      ConformerEnsemble,
    cfg:      CRESTConfig,
    work_dir: Path,
    smi:      str = "",
    **kw,
) -> Tuple[List[int], Dict]:
    """
    CONFPASS — Intelligent conformer prioritisation for DFT-ready ensembles.

    CONFPASS uses machine-learning-based scoring to predict which conformers
    are most likely to be relevant for DFT and removes redundant structures
    before expensive calculations.

    Algorithm
    ---------
    1. Write conformers as individual XYZ files.
    2. Write molecule SMILES for CONFPASS.
    3. Run confpass on the ensemble.
    4. Parse the prioritised conformer list from CONFPASS output.

    Requires
    --------
    pip install confpass
    GitHub: https://github.com/lamgroup/CONFPASS

    Reference
    ---------
    Lam, Y. H.; et al. J. Chem. Inf. Model. 2020, 60, 4, 2053–2059.
    """
    try:
        import confpass  # type: ignore
        _confpass_available = True
    except ImportError:
        _confpass_available = False

    if not _confpass_available:
        log.warning(f"  [{ens.name}] CONFPASS: Python package not found — falling back to BOLTZMANN")
        log.warning(f"  [{ens.name}]   Install: pip install confpass")
        log.warning(f"  [{ens.name}]   GitHub:  https://github.com/lamgroup/CONFPASS")
        return _filter_boltzmann(ens, cfg)

    cp_dir = work_dir / "confpass_run"
    cp_dir.mkdir(exist_ok=True)

    # Write conformers as individual XYZs
    xyz_files = []
    for idx in range(ens.n_conformers):
        xyz_f = cp_dir / f"conf_{idx:04d}.xyz"
        with open(xyz_f, "w") as fh:
            fh.write(f"{ens.n_atoms}\n")
            fh.write(f"conformer_{idx}  E={ens.energies_hartree[idx]:.8f}\n")
            for line in ens.coord_blocks[idx]:
                fh.write(line + "\n")
        xyz_files.append(str(xyz_f))

    # Write ensemble as single multi-frame XYZ for CONFPASS
    ens_xyz = cp_dir / "ensemble.xyz"
    with open(ens_xyz, "w") as fh:
        for idx in range(ens.n_conformers):
            fh.write(f"{ens.n_atoms}\n")
            fh.write(f"conformer_{idx}  E={ens.energies_hartree[idx]:.8f}\n")
            for line in ens.coord_blocks[idx]:
                fh.write(line + "\n")

    # Write SMILES file
    smiles_f = cp_dir / "molecule.smi"
    with open(smiles_f, "w") as fh:
        fh.write(f"{smi}\n")

    # Attempt to run CONFPASS via its Python API
    selected_indices: List[int] = []
    try:
        from confpass import prioritize  # type: ignore
        # CONFPASS API: prioritize(xyz_files, smiles, n_keep)
        result = prioritize(
            conformer_files=xyz_files,
            smiles=smi,
            n_keep=cfg.max_conformers_output,
        )
        # result should be a list of indices or filenames
        if isinstance(result, list):
            for item in result:
                if isinstance(item, int):
                    selected_indices.append(item)
                elif isinstance(item, str):
                    m = re.search(r"conf_(\d+)", str(item))
                    if m:
                        selected_indices.append(int(m.group(1)))
    except Exception as exc:
        log.warning(f"  [{ens.name}] CONFPASS API error ({exc}) — trying CLI")
        # Try CLI fallback
        cp_bin = _find_binary("confpass")
        if cp_bin:
            try:
                cmd = [cp_bin, "--input", str(ens_xyz), "--smiles", smi,
                       "--nkeep", str(cfg.max_conformers_output)]
                proc = subprocess.run(cmd, cwd=str(cp_dir),
                                      capture_output=True, text=True, timeout=600)
                with open(cp_dir / "confpass.log", "w") as fh:
                    fh.write(proc.stdout)
                # Parse output for conformer indices
                for line in proc.stdout.splitlines():
                    m = re.search(r"conf[_ ](\d+)", line, re.IGNORECASE)
                    if m:
                        idx = int(m.group(1))
                        if idx not in selected_indices and 0 <= idx < ens.n_conformers:
                            selected_indices.append(idx)
            except Exception as e2:
                log.warning(f"  [{ens.name}] CONFPASS CLI also failed ({e2})")

    if not selected_indices:
        log.warning(f"  [{ens.name}] CONFPASS: no output parsed — falling back to BOLTZMANN")
        return _filter_boltzmann(ens, cfg)

    # Sort by original energy and cap
    selected_indices = sorted(selected_indices, key=lambda i: ens.energies_hartree[i])
    selected = selected_indices[:cfg.max_conformers_output]
    dE = ens.delta_e_kcal

    log.info(
        f"  [{ens.name}] CONFPASS: {ens.n_conformers} → {len(selected)} conformers"
    )

    return selected, {
        "filter_method":     "CONFPASS",
        "dE_max_kept_kcal":  round(max(dE[i] for i in selected), 4) if selected else 0.0,
    }


def _filter_ani(
    ens:      ConformerEnsemble,
    cfg:      CRESTConfig,
    work_dir: Path,
    smi:      str = "",
    **kw,
) -> Tuple[List[int], Dict]:
    """
    ANI — TorchANI neural-network potential energy reranking.

    Uses the ANI-2x model (Smith et al., 2020) to compute single-point energies
    for each conformer.  ANI-2x is trained on DFT/ωB97X/6-31G* data and
    provides near-DFT accuracy for organic molecules containing H, C, N, O, F,
    S, Cl.  Energies are then used for Boltzmann reranking.

    The ANI filter is the fastest "almost-DFT" alternative — no ORCA needed.

    Algorithm
    ---------
    1. Check all elements in ensemble against ANI-2x support list.
    2. Load TorchANI model (ANI-2x) on CPU (or GPU if CUDA available).
    3. Convert coordinate blocks to torch tensors.
    4. Compute single-point energies in batches.
    5. Boltzmann reranking on ANI energy surface.

    Requires
    --------
    pip install torchani torch
    GitHub: https://github.com/aiqm/torchani

    Reference
    ---------
    Smith, J.S.; Zubatyuk, R.; Nebgen, B.; et al.
    Nat. Commun. 2020, 11, 5024.  (ANI-2x)

    Original ANI-1ccx:
    Smith, J.S.; Isayev, O.; Roitberg, A.E.
    Nat. Commun. 2017, 8, 1–9.
    """
    # Guard: check element support
    unsupported = set(ens.element_symbols) - _ANI_SUPPORTED_ELEMENTS
    if unsupported:
        log.warning(
            f"  [{ens.name}] ANI: unsupported element(s) {unsupported} — falling back to BOLTZMANN\n"
            f"  [{ens.name}]   ANI-2x supports: {sorted(_ANI_SUPPORTED_ELEMENTS)}"
        )
        return _filter_boltzmann(ens, cfg)

    # Import TorchANI
    try:
        import torch
        import torchani
        _torch_available = True
    except ImportError:
        _torch_available = False

    if not _torch_available:
        log.warning(f"  [{ens.name}] ANI: torchani / torch not found — falling back to BOLTZMANN")
        log.warning(f"  [{ens.name}]   Install: pip install torchani torch")
        log.warning(f"  [{ens.name}]   GitHub:  https://github.com/aiqm/torchani")
        return _filter_boltzmann(ens, cfg)

    ani_dir = work_dir / "ani_rerank"
    ani_dir.mkdir(exist_ok=True)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"  [{ens.name}] ANI: using device = {device}")

        # Load model
        model_name = cfg.ani_model  # e.g. "ANI2x"
        if hasattr(torchani.models, model_name):
            model = getattr(torchani.models, model_name)().to(device)
        else:
            model = torchani.models.ANI2x().to(device)

        # Build species string (from first conformer element list)
        species_str = "".join(ens.element_symbols)

        refined_energies: List[float] = []

        for idx in range(ens.n_conformers):
            coords_arr = _coord_block_to_array(ens.coord_blocks[idx])
            if coords_arr is None:
                refined_energies.append(ens.energies_hartree[idx])
                continue

            # Convert to Angstrom (CREST already uses Angstrom)
            coords_tensor = torch.tensor(
                coords_arr[None],  # add batch dimension
                dtype=torch.float32, device=device
            )

            # Build species tensor (element indices)
            try:
                species = model.species_to_tensor(species_str).to(device).unsqueeze(0)
                result  = model((species, coords_tensor))
                e_hartree = float(result.energies.item())
                refined_energies.append(e_hartree)
            except Exception as exc:
                log.warning(f"  [{ens.name}] ANI: energy failed for conf {idx} ({exc})")
                refined_energies.append(ens.energies_hartree[idx])

        # Write ANI energies log
        with open(ani_dir / "ani_energies.txt", "w") as fh:
            fh.write("# idx  E_xtb(Eh)  E_ANI(Eh)  dE_xtb(kcal/mol)  dE_ANI(kcal/mol)\n")
            E0_xtb = min(ens.energies_hartree)
            E0_ani = min(refined_energies)
            for i, (e_xtb, e_ani) in enumerate(zip(ens.energies_hartree, refined_energies)):
                fh.write(
                    f"{i:4d}  {e_xtb:.8f}  {e_ani:.8f}  "
                    f"{(e_xtb - E0_xtb)*_HARTREE_TO_KCAL:8.4f}  "
                    f"{(e_ani - E0_ani)*_HARTREE_TO_KCAL:8.4f}\n"
                )

    except Exception as exc:
        log.warning(f"  [{ens.name}] ANI: computation failed ({exc}) — falling back to BOLTZMANN")
        return _filter_boltzmann(ens, cfg)

    # Snapshot GFN2-xTB before we replace with ANI energies
    if not ens.xtb_energies_hartree or (
        ens.xtb_energies_hartree == ens.gfn2_energies_hartree
    ):
        ens.xtb_energies_hartree = list(ens.energies_hartree)

    ens.energies_hartree = list(refined_energies)
    ens.energy_level     = cfg.ani_model

    # Boltzmann reranking on ANI energy surface
    E0     = min(refined_energies)
    dE_new = [(e - E0) * _HARTREE_TO_KCAL for e in refined_energies]
    kBT    = _R_KCAL * cfg.T_kelvin
    w      = [math.exp(-de / kBT) for de in dE_new]
    Z      = sum(w)
    pops   = [wi / Z for wi in w] if Z > 0 else [1.0 / len(w)] * len(w)

    order    = sorted(range(ens.n_conformers), key=lambda i: refined_energies[i])
    selected = [i for i in order if pops[i] >= cfg.boltzmann_threshold
                ][:cfg.max_conformers_output]
    if not selected:
        selected = order[:min(3, len(order))]

    log.info(
        f"  [{ens.name}] ANI ({cfg.ani_model}): {ens.n_conformers} → {len(selected)} conformers"
    )

    return selected, {
        "filter_method":         "ANI",
        "ani_model":             cfg.ani_model,
        "ani_device":            str(device),
        "dE_max_kept_kcal":      round(max(dE_new[i] for i in selected), 4) if selected else 0.0,
        "pop_top_conformer_pct": round(pops[selected[0]] * 100, 2) if selected else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — FILTER DISPATCHER & CHAIN EXECUTOR
# ══════════════════════════════════════════════════════════════════════════════

_BUILTIN_FILTER_FNS = {
    "BOLTZMANN":       _filter_boltzmann,
    "FREE_ENERGY":     _filter_free_energy,
    "ENTROPY_PRUNING": _filter_entropy_pruning,
    "TORSION_RMSD":    _filter_torsion_rmsd,
    "CLUSTER_REP":     _filter_cluster_rep,
}

_EXTERNAL_FILTER_FNS = {
    "XTB_RERANK": _filter_xtb_rerank,
    "R2SCAN3C":   _filter_r2scan3c,
    "CENSO":      _filter_censo,
    "CONFPASS":   _filter_confpass,
    "ANI":        _filter_ani,
}

_ALL_FILTER_FNS = {**_BUILTIN_FILTER_FNS, **_EXTERNAL_FILTER_FNS}


def apply_filter_chain(
    ens:      ConformerEnsemble,
    cfg:      CRESTConfig,
    work_dir: Path,
    smi:      str = "",
) -> Tuple[List[int], List[Dict]]:
    """
    Apply the full filter chain specified in cfg.filter_chain sequentially.

    Each filter in the chain receives the FULL ConformerEnsemble but operates
    only on the indices selected by the previous layer.  A sub-ensemble view
    is created at each step, ensuring that filter logic always sees energies
    relative to the current working set.

    Returns
    -------
    (final_selected_indices_into_original_ens, list_of_per_filter_stats_dicts)
    """
    # Validate chain
    chain = [f.upper() for f in cfg.filter_chain]
    for f in chain:
        if f not in _ALL_FILTER_FNS:
            log.warning(f"Unknown filter '{f}' — replacing with BOLTZMANN")
    chain = [f if f in _ALL_FILTER_FNS else "BOLTZMANN" for f in chain]

    # Start with all conformers
    current_indices = list(range(ens.n_conformers))
    all_stats: List[Dict] = []

    for step, filter_name in enumerate(chain, 1):
        if not current_indices:
            log.warning(f"  [{ens.name}] Filter chain: no conformers remain before step {step}")
            break

        log.info(
            f"  [{ens.name}] Filter chain step {step}/{len(chain)}: "
            f"{filter_name}  ({len(current_indices)} conformers in)"
        )

        # Build a sub-ensemble view for this filter step
        sub = _build_sub_ensemble(ens, current_indices)

        fn = _ALL_FILTER_FNS[filter_name]
        try:
            sub_selected, stats = fn(sub, cfg, work_dir=work_dir, smi=smi)
        except Exception as exc:
            log.warning(
                f"  [{ens.name}] {filter_name} raised exception ({exc}) — "
                f"falling back to BOLTZMANN for this step"
            )
            sub_selected, stats = _filter_boltzmann(sub, cfg)

        stats["chain_step"] = step
        stats["n_in"]       = len(current_indices)
        stats["n_out"]      = len(sub_selected)
        all_stats.append(stats)

        # Map sub-ensemble indices back to original indices
        current_indices = [current_indices[i] for i in sub_selected if i < len(current_indices)]

        log.info(
            f"  [{ens.name}]   → {len(current_indices)} conformers after {filter_name}"
        )

    return current_indices, all_stats


def _build_sub_ensemble(ens: ConformerEnsemble, indices: List[int]) -> ConformerEnsemble:
    """Build a new ConformerEnsemble containing only the conformers at *indices*."""
    sub = ConformerEnsemble(name=ens.name)
    sub.n_atoms           = ens.n_atoms
    sub.element_symbols   = ens.element_symbols
    sub.energy_level      = ens.energy_level
    for i in indices:
        sub.energies_hartree.append(ens.energies_hartree[i])
        sub.gfn2_energies_hartree.append(
            ens.gfn2_energies_hartree[i]
            if i < len(ens.gfn2_energies_hartree)
            else ens.energies_hartree[i]
        )
        sub.xtb_energies_hartree.append(
            ens.xtb_energies_hartree[i]
            if i < len(ens.xtb_energies_hartree)
            else ens.energies_hartree[i]
        )
        sub.crest_original_rank.append(
            ens.crest_original_rank[i]
            if i < len(ens.crest_original_rank)
            else i
        )
        sub.coord_blocks.append(ens.coord_blocks[i])
        if ens.free_energies_h and i < len(ens.free_energies_h):
            sub.free_energies_h.append(ens.free_energies_h[i])
    return sub


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 12 — ENSEMBLE WRITER
# ══════════════════════════════════════════════════════════════════════════════

def write_annotated_ensemble(
    ens:          ConformerEnsemble,
    selected:     List[int],
    filter_stats: List[Dict],
    out_path:     Path,
    cfg:          CRESTConfig,
    mol_name:     str,
) -> Dict:
    """
    Write the final filtered conformer ensemble to *out_path* as a multi-structure
    XYZ file with full provenance annotation in each conformer's comment line.

    Comment line format
    -------------------
    Mol=<name>  Rank=<n>  CRESTrank=<m>
      E_<level>=<val>_Eh  dE_<level>=<val>_kcal/mol  Pop_<level>=<val>%
      [E_GFN2xTB=<val>_Eh  dE_GFN2xTB=<val>_kcal/mol]   (only if QM filter used)
      [dG=<val>_kcal/mol]                                 (only if FREE_ENERGY)
      FilterChain=<chain>

    Every field is machine-parseable: key=value with no spaces inside values.
    """
    # Determine if a QM/ML filter ran (active energy differs from GFN2-xTB)
    has_qm_filter_run = (
        ens.gfn2_energies_hartree
        and ens.gfn2_energies_hartree != ens.energies_hartree
    )
    energy_label = ens.energy_level.replace("/", "_").replace("-", "").replace(" ", "_")

    # Boltzmann populations on the ACTIVE (highest-level) energy surface
    pops  = ens.boltzmann_weights(T=cfg.T_kelvin, use_free_g=bool(ens.free_energies_h))
    dE    = ens.delta_e_kcal
    has_g = bool(ens.free_energies_h)
    dG    = ens.delta_g_kcal if has_g else dE
    chain_str = "→".join(cfg.filter_chain)

    # GFN2-xTB relative energies (original from CREST — always available)
    E0_gfn2  = min(ens.gfn2_energies_hartree) if ens.gfn2_energies_hartree else 0.0
    dE_gfn2  = [(e - E0_gfn2) * _HARTREE_TO_KCAL for e in ens.gfn2_energies_hartree] if ens.gfn2_energies_hartree else []
    kBT      = _R_KCAL * cfg.T_kelvin
    w_gfn2   = [math.exp(-de / kBT) for de in dE_gfn2]
    Z_gfn2   = sum(w_gfn2)
    pops_gfn2 = [wi / Z_gfn2 for wi in w_gfn2] if Z_gfn2 > 0 else [1.0/len(w_gfn2)]*len(w_gfn2)

    with open(out_path, "w") as fh:
        for rank, idx in enumerate(selected, 1):
            fh.write(f"{ens.n_atoms}\n")

            # Core: QM/active-level energy, dE, population
            crest_rank = (
                ens.crest_original_rank[idx] + 1
                if idx < len(ens.crest_original_rank) else "?"
            )
            g_str = f"  dG={dG[idx]:.4f}_kcal/mol" if has_g else ""

            comment = (
                f"Mol={mol_name}  Rank={rank}  CRESTrank={crest_rank}"
                f"  E_{energy_label}={ens.energies_hartree[idx]:.10f}_Eh"
                f"  dE_{energy_label}={dE[idx]:.4f}_kcal/mol"
                f"  Pop_{energy_label}={pops[idx]*100:.2f}%"
            )

            # Secondary: original GFN2-xTB energy (from CREST) for cross-check
            if has_qm_filter_run and idx < len(ens.gfn2_energies_hartree):
                comment += (
                    f"  E_GFN2xTB={ens.gfn2_energies_hartree[idx]:.10f}_Eh"
                    f"  dE_GFN2xTB={dE_gfn2[idx]:.4f}_kcal/mol"
                    f"  Pop_GFN2xTB={pops_gfn2[idx]*100:.2f}%"
                )

            comment += f"{g_str}  FilterChain={chain_str}"
            fh.write(comment + "\n")

            for line in ens.coord_blocks[idx]:
                fh.write(line + "\n")

    # ── Compile final statistics ──────────────────────────────────────────
    n_kept      = len(selected)
    dE_max_kept = round(max(dE[i] for i in selected), 4) if selected else 0.0

    stats: Dict = {
        "n_conformers_raw":      ens.n_conformers,
        "n_conformers_kept":     n_kept,
        "energy_level":          ens.energy_level,
        "dE_max_raw_kcal":       round(max(dE), 4) if dE else 0.0,
        "dE_max_kept_kcal":      dE_max_kept,
        "pop_top_conformer_pct": round(pops[selected[0]] * 100, 2) if selected else 0.0,
        "filter_chain":          chain_str,
        "filter_method_used":    chain_str,
        "filter_steps":          json.dumps(filter_stats, default=str),
        # Rank-1 conformer energies at the active QM level
        "rank1_energy_hartree":  (
            round(ens.energies_hartree[selected[0]], 10) if selected else None
        ),
        "rank1_dE_kcal":         (
            round(dE[selected[0]], 4) if selected and dE else None
        ),
        # Original GFN2-xTB energy of rank-1 conformer (from CREST)
        "rank1_gfn2_energy_hartree": (
            round(ens.gfn2_energies_hartree[selected[0]], 10)
            if selected and ens.gfn2_energies_hartree else None
        ),
    }
    return stats


# ══════════════════════════════════════════════════════════════════════════════
#  GJF EXPORT — Gaussian input file generation
# ══════════════════════════════════════════════════════════════════════════════

def write_gjf_files(
    ens:      ConformerEnsemble,
    selected: List[int],
    cfg:      CRESTConfig,
    mol_dir:  Path,
    mol_name: str,
) -> List[str]:
    """
    Write individual Gaussian .gjf input files for each final ranked conformer.

    One .gjf file is written per conformer at:
      <mol_dir>/g09_<molName>_conf_<rank>.gjf

    Each file contains:
      %MEM=%s
      %NPROC=%d
      %CHK=conf_<rank>.chk
      # %s

      <title line>
      <charge> <multiplicity>

      <atom> <x> <y> <z>
      ...

      (blank line)

    The route line is taken from cfg.gjf_route (default: B3LYP/6-31G opt).
    """
    if not cfg.make_gjf or not selected:
        return []

    gjf_dir = mol_dir / "g09_inputs"
    gjf_dir.mkdir(exist_ok=True)

    written: List[str] = []
    energy_label = ens.energy_level.replace("/", "_").replace("-", "").replace(" ", "_")

    for rank, idx in enumerate(selected, 1):
        base = f"g09_{mol_name}_conf_{rank:03d}"
        gjf_path = gjf_dir / f"{base}.gjf"

        with open(gjf_path, "w") as fh:
            # ── Link 0 commands ─────────────────────────────────────────────
            fh.write(f"%MEM={cfg.gjf_mem}\n")
            fh.write(f"%NPROC={cfg.gjf_nproc}\n")
            fh.write(f"%CHK={base}.chk\n")

            # ── Route line ────────────────────────────────────────────────────
            fh.write(f"# {cfg.gjf_route}\n")
            fh.write("\n")

            # ── Title / comment block ────────────────────────────────────────
            crest_rank = (
                ens.crest_original_rank[idx] + 1
                if idx < len(ens.crest_original_rank) else 0
            )
            fh.write(f"{mol_name}  Rank={rank}  CRESTrank={crest_rank}"
                     f"  E_{energy_label}={ens.energies_hartree[idx]:.10f}_Eh"
                     f"  dE_{energy_label}={ens.delta_e_kcal[idx]:.4f}_kcal/mol"
                     f"  FilterChain={'→'.join(cfg.filter_chain)}\n")
            fh.write("\n")

            # ── Charge & multiplicity ─────────────────────────────────────────
            fh.write(f"{cfg.gjf_charge} {cfg.gjf_multiplicity}\n")

            # ── Coordinates ──────────────────────────────────────────────────
            for line in ens.coord_blocks[idx]:
                fh.write(line + "\n")

            fh.write("\n")

        written.append(str(gjf_path.relative_to(cfg.output_dir)))

    n_total = len(selected)
    log.info(
        f"  [{ens.name}] GJF export: {n_total} file(s) → {gjf_dir.relative_to(cfg.output_dir)/'g09_inputs'}"
    )
    return written


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 13 — PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(mol_entries: List[Dict], cfg: CRESTConfig) -> List[Dict]:
    """
    Execute the full pipeline for every molecule.

    Stages
    ------
    0  SMILES validation (7-layer)
    1  RDKit ETKDGv3 + MMFF94s 3-D structure generation
    2  GFN2-xTB tight geometry optimisation (pre-opt)
    3  CREST iMTD-GC conformer ensemble search
    4  Multi-layer QM/ML ensemble filtering (filter chain)
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(cfg.output_dir / "pipeline.log", mode="w")
    fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    log.addHandler(fh)

    validator = MoleculeValidator(cfg)
    results:  List[Dict] = []

    chain_str = " → ".join(cfg.filter_chain)
    log.info("=" * 80)
    log.info(
        f"  CREST CONFORMER PIPELINE  v5.0  |  {len(mol_entries)} molecule(s)  "
        f"|  Filter chain: {chain_str}"
    )
    log.info(f"  Output : {cfg.output_dir.resolve()}")
    log.info(f"  Solvent: {cfg.solvent or 'gas phase'}  |  T = {cfg.T_kelvin} K")
    log.info(f"  CREST ewin = {cfg.ewin_kcal} kcal/mol  |  ncores = {cfg.ncores}")
    log.info("=" * 80)

    for entry in mol_entries:
        name = entry["name"]
        mid  = entry["id"]
        smi  = entry["smiles"]

        log.info(f"\n{'─'*72}")
        log.info(f"  [{mid:04d}]  {name}  (raw: {entry['name_raw']!r})")
        log.info(f"  Series  : {entry['series']}")
        log.info(f"  SMILES  : {smi}")
        if entry.get("smiles_changed"):
            log.info(f"  Input   : {entry['smiles_raw']}  [standardised]")
        log.info(f"{'─'*72}")

        rec: Dict = {
            "id":                    mid,
            "name":                  name,
            "name_raw":              entry.get("name_raw", name),
            "series":                entry["series"],
            "smiles_input":          entry.get("smiles_raw", smi),
            "smiles_standardised":   smi,
            "smiles_was_changed":    entry.get("smiles_changed", False),
            "description":           entry.get("description", ""),
            "expected_activity":     entry.get("expected_activity", ""),
            "ref":                   entry.get("ref", ""),
            "validation_passed":     False,
            "validation_notes":      "",
            "3d_generated":          False,
            "xtb_preopt_done":       False,
            "crest_done":            False,
            "n_conformers_raw":      0,
            "n_conformers_kept":     0,
            "dE_max_raw_kcal":       0.0,
            "dE_max_kept_kcal":      0.0,
            "pop_top_conformer_pct": 0.0,
            "filter_chain":          chain_str,
            "filter_method_used":    chain_str,
            "output_xyz":            "",
        }

        # ── Stage 0: Validate ──────────────────────────────────────────────
        valid, msgs, descs = validator.validate(entry)
        rec["validation_passed"] = valid
        rec["validation_notes"]  = "; ".join(msgs) if msgs else "OK"
        rec.update(descs)

        if not valid:
            for m in msgs:
                log.warning(f"  [{name}] {m}")
            results.append(rec)
            continue

        for m in [x for x in msgs if "WARN" in x]:
            log.warning(f"  [{name}] {m}")
        log.info(
            f"  [{name}] Validation PASS | MW={descs.get('MW')}  LogP={descs.get('LogP')}  "
            f"QED={descs.get('QED')}  TPSA={descs.get('TPSA')}"
        )

        # ── Per-molecule working directory ─────────────────────────────────
        mol_dir = cfg.output_dir / f"{mid:04d}_{name}"
        mol_dir.mkdir(exist_ok=True)

        # ── Stage 1: 3-D geometry ──────────────────────────────────────────
        xyz3d = generate_3d_structure(smi, name, cfg, mol_dir)
        if xyz3d is None:
            results.append(rec)
            continue
        rec["3d_generated"] = True

        # ── Stage 2: xTB pre-optimisation ─────────────────────────────────
        xyz_opt, xtb_ok = run_xtb_preopt(xyz3d, name, cfg, mol_dir)
        rec["xtb_preopt_done"] = xtb_ok

        # ── Stage 3: CREST conformer search ───────────────────────────────
        conf_xyz = run_crest(xyz_opt, name, cfg, mol_dir)

        if conf_xyz is None:
            log.info(f"  [{name}] CREST not run — storing geometry as single-conformer stub")
            conf_xyz = mol_dir / "crest_conformers.xyz"
            raw_lines = xyz_opt.read_text().splitlines()
            with open(conf_xyz, "w") as fh_stub:
                fh_stub.write(f"{raw_lines[0]}\n")
                fh_stub.write(
                    f"-9999.000000  (RDKit/xTB geometry; CREST not run)  {name}\n"
                )
                for ln in raw_lines[2:]:
                    fh_stub.write(ln + "\n")
        else:
            rec["crest_done"] = True

        # ── Stage 4: Multi-layer QM filtering ────────────────────────────
        ens = parse_crest_xyz(conf_xyz, name)
        if ens is None or ens.n_conformers == 0:
            log.warning(f"  [{name}] No conformers parsed from CREST output")
            results.append(rec)
            continue

        log.info(
            f"  [{name}] Parsed {ens.n_conformers} raw conformers — "
            f"applying filter chain: {chain_str}"
        )

        selected, filter_stats = apply_filter_chain(ens, cfg, mol_dir, smi=smi)

        out_xyz = cfg.output_dir / f"{mid:04d}_{name}_conformers.xyz"
        stats   = write_annotated_ensemble(
            ens, selected, filter_stats, out_xyz, cfg, mol_name=name
        )
        rec.update(stats)
        rec["output_xyz"] = out_xyz.name

        # ── Optional Gaussian GJF export ───────────────────────────────────
        if cfg.make_gjf:
            gjf_paths = write_gjf_files(ens, selected, cfg, mol_dir, name)
            rec["gjf_files"] = gjf_paths

        log.info(
            f"  [{name}] ✓  raw={stats['n_conformers_raw']}  "
            f"kept={stats['n_conformers_kept']}  "
            f"ΔE_max={stats['dE_max_kept_kcal']:.2f} kcal/mol  "
            f"Pop(1)={stats['pop_top_conformer_pct']:.1f}%  "
            f"chain={chain_str}"
        )

        results.append(rec)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 14 — RESULTS EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def write_summary(results: List[Dict], cfg: CRESTConfig) -> None:
    """Write conformer_summary.csv, conformer_summary.json, and terminal table."""
    if not results:
        log.warning("No results to write.")
        return

    csv_path = cfg.output_dir / "conformer_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    log.info(f"\n  CSV summary  → {csv_path}")

    json_path = cfg.output_dir / "conformer_summary.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "pipeline_version": "5.0",
                "config":           cfg.as_dict(),
                "filter_chain":     cfg.filter_chain,
                "n_molecules":      len(results),
                "n_valid":          sum(1 for r in results if r["validation_passed"]),
                "n_crest_complete": sum(1 for r in results if r.get("crest_done")),
                "results":          results,
            },
            fh, indent=2, default=str,
        )
    log.info(f"  JSON summary → {json_path}")
    log.info(f"  Pipeline log → {cfg.output_dir / 'pipeline.log'}")

    # Terminal table
    col = 40
    chain_str = "→".join(cfg.filter_chain)
    log.info(f"\n{'='*106}")
    log.info(
        f"  {'ID':>4}  {'Name':<{col}}  "
        f"{'Valid':^5}  {'CREST':^5}  {'Raw':>4}  {'Kept':>4}  "
        f"{'ΔEmax':>6}  {'Pop1%':>5}  {'Filter Chain'}"
    )
    log.info("─" * 106)
    for r in results:
        nm = r["name"][:col]
        log.info(
            f"  {r['id']:>4}  {nm:<{col}}  "
            f"{'✓' if r['validation_passed'] else '✗':^5}  "
            f"{'✓' if r.get('crest_done') else '─':^5}  "
            f"{r.get('n_conformers_raw',   0):>4}  "
            f"{r.get('n_conformers_kept',  0):>4}  "
            f"{r.get('dE_max_kept_kcal',   0.0):>6.2f}  "
            f"{r.get('pop_top_conformer_pct', 0.0):>5.1f}  "
            f"{chain_str}"
        )
    log.info("=" * 106)

    n_valid = sum(1 for r in results if r["validation_passed"])
    n_crest = sum(1 for r in results if r.get("crest_done"))
    n_3d    = sum(1 for r in results if r.get("3d_generated"))
    log.info(f"\n  Molecules processed  : {len(results)}")
    log.info(f"  Validation passed    : {n_valid}")
    log.info(f"  3D structures built  : {n_3d}")
    log.info(f"  CREST completed      : {n_crest}")
    log.info(f"  Filter chain         : {chain_str}")
    log.info(f"  Output directory     : {cfg.output_dir.resolve()}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 15 — CLI ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def list_molecules_from_entries(molecules: List[Dict]) -> None:
    col = 44
    header = (f"{'ID':>4}  {'Name':<{col}}  {'Raw Name':<{col}}  "
              f"{'Changed':^7}  {'Series':^6}  SMILES")
    print(f"\n{header}")
    print("─" * (len(header) + 20))
    for m in molecules:
        changed = "YES" if m.get("smiles_changed") else "no"
        print(
            f"{m['id']:>4}  {m['name'][:col]:<{col}}  {m['name_raw'][:col]:<{col}}  "
            f"{changed:^7}  {m['series']:^6}  {m['smiles']}"
        )
    print(f"\nTotal: {len(molecules)} molecule(s)\n")


def _valid_filter_type(value: str) -> str:
    v = value.upper()
    if v not in _ALL_FILTERS:
        raise argparse.ArgumentTypeError(
            f"Unknown filter '{value}'.  Valid choices:\n"
            f"  Built-in : {', '.join(sorted(_BUILTIN_FILTERS))}\n"
            f"  External : {', '.join(sorted(_EXTERNAL_FILTERS))}"
        )
    return v


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="crest_conformer_pipeline_v5.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(f"""\
            CREST Conformer Generation & Ensemble Filtering Pipeline — v5.0
            ──────────────────────────────────────────────────────────────────
            Workflow:
              CSV (SMILES + names)
               → SMILES standardisation (5-stage RDKit)
               → ETKDGv3 + MMFF94s 3-D geometry
               → GFN2-xTB tight pre-optimisation
               → CREST iMTD-GC conformer ensemble search
               → Multi-layer QM/ML post-CREST filtering
               → Annotated XYZ + CSV + JSON output

            ── Built-in filters (Layer A — no extra software) ────────────
              BOLTZMANN        Classical Boltzmann on raw GFN2-xTB ΔE
              FREE_ENERGY      Boltzmann on GFN2-xTB ΔG (RRHO + ALPB)
              ENTROPY_PRUNING  Maximum-entropy ensemble pruning
              TORSION_RMSD     Dihedral-fingerprint diversity + energy gate
              CLUSTER_REP      RMSD-cluster representatives

            ── External QM/ML filters (Layer B — require extra tools) ────
              XTB_RERANK   xTB re-opt (tight/verytight) + Boltzmann rerank
                           Requires: xtb (already installed for CREST)
              R2SCAN3C     ORCA r²SCAN-3c SP energies + Boltzmann rerank
                           Requires: ORCA ≥ 5.0 (set ORCA_EXE or PATH)
              CENSO        Grimme CENSO multilevel DFT-lite pipeline
                           Requires: pip install censo
              CONFPASS     CONFPASS ML conformer prioritisation
                           Requires: pip install confpass
              ANI          TorchANI neural-network potential reranking
                           Requires: pip install torchani torch
                           Elements: H C N O F S Cl only

            ── Multi-filter chaining ─────────────────────────────────────
              Use --filter1, --filter2, --filter3 to chain filters.
              Each filter is applied to the output of the previous one.

            ── Examples ──────────────────────────────────────────────────
              # List molecules
              python crest_conformer_pipeline_v5.py --csv mols.csv --list_mols

              # Single filter (xTB reranking — best quick improvement)
              python crest_conformer_pipeline_v5.py --csv mols.csv \\
                  --all --ncores 8 --filter1 XTB_RERANK

              # Two-stage: cluster + xTB rerank
              python crest_conformer_pipeline_v5.py --csv mols.csv \\
                  --all --ncores 8 --filter1 CLUSTER_REP --filter2 XTB_RERANK

              # Publication-grade 3-stage pipeline (recommended)
              python crest_conformer_pipeline_v5.py --csv mols.csv \\
                  --all --ncores 16 --solvent water \\
                  --filter1 CLUSTER_REP --filter2 XTB_RERANK --filter3 R2SCAN3C \\
                  --nconf 20 --ewin 8.0

              # ANI neural-network reranking (fast, no ORCA)
              python crest_conformer_pipeline_v5.py --csv mols.csv \\
                  --all --ncores 8 --filter1 CLUSTER_REP --filter2 ANI

              # Legacy single --filter (backward compatible with v4)
              python crest_conformer_pipeline_v5.py --csv mols.csv \\
                  --all --filter BOLTZMANN
        """),
    )

    # ── Required ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--csv", required=True, metavar="CSV_FILE",
        help="Input CSV: col 1 = SMILES, col 2 = molecule name",
    )
    p.add_argument("--smiles_col", default=0,
                   help="0-based col index or header name for SMILES (default: 0)")
    p.add_argument("--name_col",   default=1,
                   help="0-based col index or header name for molecule name (default: 1)")

    # ── Molecule selection ────────────────────────────────────────────────────
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--all", action="store_true", help="Process all molecules")
    sel.add_argument("--mol_id", nargs="+", type=int, metavar="N",
                     help="Row numbers (1-based) to process")
    sel.add_argument("--mol_names", nargs="+", metavar="NAME",
                     help="Molecule names to process")
    sel.add_argument("--list_mols", action="store_true",
                     help="Print all molecules and exit")
    sel.add_argument("--validate_only", action="store_true",
                     help="Run validation only, write validation_report.csv and exit")

    # ── Filter chain ──────────────────────────────────────────────────────────
    fg = p.add_argument_group("Conformer filtering")
    fg.add_argument(
        "--filter1", metavar="METHOD", type=_valid_filter_type, default=None,
        help="First filter in the chain (primary filter)",
    )
    fg.add_argument(
        "--filter2", metavar="METHOD", type=_valid_filter_type, default=None,
        help="Second filter in the chain (optional refinement)",
    )
    fg.add_argument(
        "--filter3", metavar="METHOD", type=_valid_filter_type, default=None,
        help="Third filter in the chain (optional final refinement)",
    )
    fg.add_argument(
        "--filter", metavar="METHOD", type=_valid_filter_type, default=None,
        help="Legacy single-filter flag (v4 compatibility).  Equivalent to --filter1.",
    )

    # ── Filter thresholds ─────────────────────────────────────────────────────
    fg.add_argument("--boltzmann_thr",    type=float, default=0.005,
                    help="Min Boltzmann population fraction (default: 0.005)")
    fg.add_argument("--rmsd_cluster_thr", type=float, default=0.50,
                    help="RMSD clustering threshold in Å (default: 0.50)")
    fg.add_argument("--torsion_thr",      type=float, default=20.0,
                    help="Min torsion-RMSD in degrees for TORSION_RMSD (default: 20.0)")

    # ── XTB_RERANK options ─────────────────────────────────────────────────────
    xg = p.add_argument_group("XTB_RERANK filter options")
    xg.add_argument(
        "--xtb_rerank_accuracy", default="tight",
        choices=["tight", "verytight"],
        help="xTB optimisation accuracy for XTB_RERANK (default: tight)",
    )

    # ── R2SCAN3C options ───────────────────────────────────────────────────────
    og = p.add_argument_group("R2SCAN3C filter options (ORCA)")
    og.add_argument("--orca_nprocs",   type=int, default=4,
                    help="ORCA parallel processes for R2SCAN3C (default: 4)")
    og.add_argument("--orca_maxcore",  type=int, default=4096,
                    help="ORCA MaxCore (MB) per process (default: 4096)")
    og.add_argument(
        "--orca_mpi_launcher", default=None, metavar="PATH",
        help=(
            "Explicit path to mpirun/mpiexec for the ORCA MPI build "
            "(e.g. /usr/lib/openmpi/bin/mpirun).  "
            "Overrides auto-detection; set to 'none' to force serial mode."
        ),
    )

    # ── CENSO options ─────────────────────────────────────────────────────────
    cg = p.add_argument_group("CENSO filter options")
    cg.add_argument("--censo_maxconf", type=int, default=20,
                    help="Max conformers for CENSO to screen (default: 20)")

    # ── ANI options ───────────────────────────────────────────────────────────
    ag = p.add_argument_group("ANI filter options")
    ag.add_argument("--ani_model", default="ANI2x",
                    choices=["ANI1x", "ANI1ccx", "ANI2x"],
                    help="TorchANI model to use (default: ANI2x)")

    # ── Gaussian GJF export ─────────────────────────────────────────────────
    gg = p.add_argument_group("Gaussian GJF export options")
    gg.add_argument("--gjf", dest="make_gjf", action="store_true",
                    help="Write per-conformer Gaussian .gjf input files for DFT optimisation")
    gg.add_argument("--gjf_route", default="B3LYP/6-31G opt",
                    help="Gaussian route line (default: B3LYP/6-31G opt)")
    gg.add_argument("--gjf_mem", default="10GB",
                    help="Gaussian %MEM value (default: 10GB)")
    gg.add_argument("--gjf_nproc", type=int, default=4,
                    help="Gaussian %NPROC value (default: 4)")
    gg.add_argument("--gjf_charge", type=int, default=0,
                    help="Molecular charge for GJF (default: 0)")
    gg.add_argument("--gjf_multiplicity", type=int, default=1,
                    help="Spin multiplicity for GJF (default: 1)")

    # ── CREST / xTB parameters ────────────────────────────────────────────────
    cp = p.add_argument_group("CREST / xTB parameters")
    cp.add_argument("--ncores",        type=int,   default=4,     help="CPU cores (default: 4)")
    cp.add_argument("--solvent", default=None,
                    choices=["water", "chcl3", "dmso", "thf", "acetonitrile", "methanol"],
                    help="Implicit solvent (ALPB; default: gas phase)")
    cp.add_argument("--ewin",          type=float, default=6.0,   help="CREST energy window kcal/mol (default: 6.0)")
    cp.add_argument("--nconf",         type=int,   default=25,    help="Max conformers to retain (default: 25)")
    cp.add_argument("--rmsd",          type=float, default=0.125, help="CREST RMSD threshold Å (default: 0.125)")
    cp.add_argument("--crest_version", type=int,   default=3,     choices=[2, 3],
                    help="CREST version (default: 3)")
    cp.add_argument("--temperature",   type=float, default=298.15,help="Temperature K (default: 298.15)")
    cp.add_argument("--outdir",        default="crest_output",    help="Output directory (default: crest_output)")
    cp.add_argument("--require_phenol",type=int,   default=0,
                    help="Min phenolic OH groups for validation (default: 0)")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    def _col(spec):
        try:
            return int(spec)
        except (TypeError, ValueError):
            return spec

    # ── Load CSV ───────────────────────────────────────────────────────────────
    molecules = load_molecules_from_csv(
        csv_path   = Path(args.csv),
        smiles_col = _col(args.smiles_col),
        name_col   = _col(args.name_col),
    )

    # ── List mode ──────────────────────────────────────────────────────────────
    if args.list_mols:
        list_molecules_from_entries(molecules)
        return

    # ── Build filter chain ─────────────────────────────────────────────────────
    chain: List[str] = []
    # Legacy --filter flag maps to first position
    if args.filter is not None:
        chain.append(args.filter)
    # New positional flags
    for f in [args.filter1, args.filter2, args.filter3]:
        if f is not None and f not in chain:
            chain.append(f)
    if not chain:
        chain = ["BOLTZMANN"]

    # ── xTB rerank accuracy map ────────────────────────────────────────────────
    _acc_map = {"tight": 0.2, "verytight": 0.05}
    xtb_acc_val = _acc_map.get(args.xtb_rerank_accuracy, 0.2)

    # ── Build configuration ────────────────────────────────────────────────────
    # Resolve --orca_mpi_launcher: "none" → force serial; else use as-is
    _mpi_override = getattr(args, "orca_mpi_launcher", None)
    if isinstance(_mpi_override, str) and _mpi_override.lower() == "none":
        _mpi_override = ""          # empty string = disable MPI in _find_mpirun

    cfg = CRESTConfig(
        filter_chain           = chain,
        boltzmann_threshold    = args.boltzmann_thr,
        rmsd_cluster_thr       = args.rmsd_cluster_thr,
        torsion_diff_thr       = args.torsion_thr,
        max_conformers_output  = args.nconf,
        xtb_rerank_accuracy    = args.xtb_rerank_accuracy,
        xtb_rerank_acc_val     = xtb_acc_val,
        orca_nprocs            = args.orca_nprocs,
        orca_maxcore_mb        = args.orca_maxcore,
        orca_mpi_launcher      = _mpi_override,
        censo_maxconf          = args.censo_maxconf,
        ani_model              = args.ani_model,
        ncores                 = args.ncores,
        solvent                = args.solvent,
        ewin_kcal              = args.ewin,
        rmsd_thr               = args.rmsd,
        crest_version          = args.crest_version,
        T_kelvin               = args.temperature,
        output_dir             = Path(args.outdir),
        n_phenolic_oh_min      = args.require_phenol,
        make_gjf               = args.make_gjf,
        gjf_route              = args.gjf_route,
        gjf_mem                = args.gjf_mem,
        gjf_nproc              = args.gjf_nproc,
        gjf_charge             = args.gjf_charge,
        gjf_multiplicity       = args.gjf_multiplicity,
    )

    # ── Select molecules ───────────────────────────────────────────────────────
    if args.validate_only or args.all:
        selected = molecules[:]
    elif args.mol_id:
        id_set   = set(args.mol_id)
        selected = [m for m in molecules if m["id"] in id_set]
        if not selected:
            sys.exit(f"No molecules matched row IDs: {args.mol_id}")
    elif args.mol_names:
        req = {_sanitise_name(n) for n in args.mol_names}
        selected = [m for m in molecules if m["name"] in req]
        if not selected:
            selected = [m for m in molecules if m["name_raw"] in set(args.mol_names)]
        if not selected:
            sys.exit(
                f"No molecules matched names: {args.mol_names}\n"
                f"Available: {[m['name'] for m in molecules]}"
            )
    else:
        build_arg_parser().print_help()
        sys.exit(
            "\nSpecify a run mode: "
            "--all | --mol_id N ... | --mol_names NAME ... | "
            "--list_mols | --validate_only"
        )

    if not selected:
        sys.exit("No molecules matched the selection.")

    log.info(f"Selected {len(selected)} molecule(s)")
    log.info(f"Filter chain: {' → '.join(chain)}")

    # ── Validate-only mode ─────────────────────────────────────────────────────
    if args.validate_only:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(cfg.output_dir / "pipeline.log", mode="w")
        fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
        log.addHandler(fh)

        validator = MoleculeValidator(cfg)
        rows: List[Dict] = []
        log.info("\n  VALIDATION REPORT")
        log.info("=" * 100)
        for entry in selected:
            valid, msgs, descs = validator.validate(entry)
            status = "✓ PASS" if valid else "✗ FAIL"
            log.info(
                f"  [{entry['id']:04d}] {entry['name']:<44}  {status}  "
                f"MW={descs.get('MW','?'):>7}  LogP={descs.get('LogP','?'):>6}  "
                f"QED={descs.get('QED','?'):>6}  "
                f"Stereo_unspec={descs.get('UnspecifiedStereoCenters','?'):>2}"
            )
            for m in msgs:
                log.info(f"         └─ {m}")
            rows.append({
                "id":             entry["id"],
                "name":           entry["name"],
                "name_raw":       entry["name_raw"],
                "smiles_input":   entry.get("smiles_raw", entry["smiles"]),
                "smiles_std":     entry["smiles"],
                "smiles_changed": entry.get("smiles_changed", False),
                "series":         entry["series"],
                "valid":          valid,
                "notes":          "; ".join(msgs),
                **descs,
            })
        log.info("=" * 100)
        val_csv = cfg.output_dir / "validation_report.csv"
        with open(val_csv, "w", newline="") as fh_csv:
            w = csv.DictWriter(fh_csv, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        log.info(f"\n  Validation CSV → {val_csv}")
        return

    # ── Full pipeline ──────────────────────────────────────────────────────────
    results = run_pipeline(selected, cfg)
    write_summary(results, cfg)


if __name__ == "__main__":
    main()
