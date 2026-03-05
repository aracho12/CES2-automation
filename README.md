# CES2 QM/MM data.file Builder

Automates building LAMMPS data files for CES2 QM/MM simulations — handles electrolyte composition, PACKMOL packing, and LAMMPS input generation.

---

## Quick Start (for collaborators)

### 1. Clone the repository
```bash
git clone https://github.com/aracho12/CES2-automation.git
cd CES2-automation
```

### 2. Set up Python environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Install PACKMOL
PACKMOL must be installed separately and available in your PATH.

**Option A — conda (easiest):**
```bash
conda install -c conda-forge packmol
```

**Option B — from source:**
```bash
git clone https://github.com/m3g/packmol.git
cd packmol
make
# Then add the packmol binary to your PATH
```

Verify it works:
```bash
which packmol   # should print a path
```

### 4. Prepare your config
```bash
cp config_example.yaml config.yaml
# Edit config.yaml with your system settings
```

### 5. Run
```bash
python run_builder.py --config config.yaml
```

**Outputs:**
- `build/` — intermediate files (packmol.inp, xyz files, composition JSON)
- `export/` — final data.file, QM/MM atom ID lists, build summary

---

## Project Structure

```
CES2-automation/
├── ces2_builder/        # Core Python package
│   ├── builder.py       # Main build logic
│   ├── packmol.py       # PACKMOL interface
│   ├── lammps_writer.py # LAMMPS data file writer
│   ├── vasp_io.py       # VASP CONTCAR reader
│   ├── species.py       # Species/molecule handling
│   ├── composition.py   # Electrolyte composition
│   ├── config.py        # Config parsing
│   └── md_workflow.py   # MD relaxation workflow
├── species_db/          # YAML database of ions/molecules
├── config_example.yaml  # Example configuration
├── run_builder.py       # Entry point
└── submit_after_build.sh  # Convenience script for HPC submission
```

---

## Adding a New Species

To add a new ion or molecule, create a YAML file in `species_db/` following the schema in `species_db/SCHEMA.md`.

---

## MD Pre-Relaxation (optional)

If `md_relax.enabled: true` in your config, the builder also generates in `export/`:
- `in.relax` — LAMMPS minimize + short NVT
- `ff/*.in` — placeholder force field include files (fill these in)
- `submit_md.sbatch` / `submit_md.pbs` — HPC job scripts
- `run_md.sh` — local run script

Workflow:
1. Run builder
2. Fill in `export/ff/*.in` with your force field parameters
3. Submit MD job or run locally

Convenience script (submits after build automatically):
```bash
./submit_after_build.sh
```

---

## Requirements

- Python >= 3.9
- PACKMOL (installed separately — see Quick Start)
- Python packages: see `requirements.txt`

---

## Key Changes vs v0.1

- Electrolyte defined by a recipe supporting: water count, multiple salts (concentration + stoichiometry), explicit species counts, counterion pool for charge neutrality
- Any mono/polyatomic ions supported via `species_db/*.yaml`
- Connectivity (bonds/angles) from species YAML, shifted to global indices
- Output `data.file` includes `0 0 0` image flags per atom
