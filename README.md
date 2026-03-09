# CES2 QM/MM Automation

Automated builder for DFT-CES2 QM/MM simulations — generates all required input files (LAMMPS data file, QM/MM LAMMPS input, Quantum ESPRESSO input, and run scripts) from a VASP CONTCAR and a single config file.

---

## What it does

Running `cesbuild` (or `python run_builder.py`) generates a complete, ready-to-run CES2 QM/MM directory from three inputs:

| Input | Description |
|---|---|
| `CONTCAR` | VASP slab structure (QM region) |
| `config.yaml` | System settings (electrolyte recipe, supercell, QE params, …) |
| `species_db/` | YAML database of ions/molecules |

**Output files in `export/`:**

| File | Description |
|---|---|
| `data.file` | LAMMPS data file (slab + electrolyte, full atom_style) |
| `base.in.lammps` | LAMMPS QM/MM input (TIP4P-EW, gridforce/net, bjdisp) |
| `base.pw.in` | QE `pw.x` SCF input with `###qmxyz` / `###dispf` markers |
| `base.pp.in` | QE `pp.x` post-processing input (→ `solute.pot.cube`) |
| `qmmm_dftces2_charging_pts.sh` | CES2 QM/MM outer-loop wrapper script |
| `submit_ces2.sh` | SLURM/PBS batch submission script |
| `qm_atoms.txt` / `mm_atoms.txt` | LAMMPS atom ID lists for QM and MM regions |
| `build_summary.json` | Build metadata and timing |

---

## Quick Start

### 1. Clone and set up Python environment

```bash
git clone https://github.com/aracho12/CES2-automation.git
cd CES2-automation

python -m venv venv
source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
```

### 2. Install PACKMOL

PACKMOL must be installed separately and on your PATH.

```bash
conda install -c conda-forge packmol   # easiest
# or build from source: https://github.com/m3g/packmol
which packmol                          # verify
```

### 3. Prepare your config

```bash
cp config_example.yaml config.yaml
# Edit config.yaml — minimum required: input.vasp_file, cell.supercell,
# electrolyte_recipe, qe.pseudopotentials
```

### 4. Run

```bash
python run_builder.py --config config.yaml
# or with explicit CONTCAR path:
python run_builder.py --config config.yaml --input /path/to/CONTCAR
```

---

## Project Structure

```
CES2-automation/
├── ces2_builder/
│   ├── main.py                  # Orchestrator — calls all writers in order
│   ├── builder.py               # Type registry, LAMMPS data assembly
│   ├── lammps_input_writer.py   # Generates base.in.lammps (TIP4P-EW, gridforce/net)
│   ├── qe_writer.py             # Generates base.pw.in + base.pp.in
│   ├── ces2_script_writer.py    # Generates qmmm wrapper + SLURM submit scripts
│   ├── bjdisp_db.py             # BJ-dispersion parameter DB (QM-MM pair_coeff)
│   ├── lammps_writer.py         # LAMMPS data file writer (reference style)
│   ├── packmol.py               # PACKMOL interface (electrolyte packing)
│   ├── composition.py           # Electrolyte composition from recipe
│   ├── species.py               # Species / molecule data model
│   ├── vasp_io.py               # VASP CONTCAR reader + supercell builder
│   ├── box.py                   # Simulation box geometry
│   ├── config.py                # Config YAML loader
│   └── md_workflow.py           # Optional MD pre-relaxation workflow
├── species_db/
│   ├── SCHEMA.md                # Species YAML format documentation
│   ├── water_tip3p.yaml         # TIP3P/TIP4P water (charges match water_model)
│   ├── Na.yaml, K.yaml, Cl.yaml # Common ion species
│   └── qm_params/
│       └── IrO2.yaml            # QM slab BJ-dispersion parameters (alpha_iso, C6, s)
├── config_example.yaml          # Fully-annotated example config
├── run_builder.py               # CLI entry point
└── requirements.txt
```

---

## Config Overview

`config.yaml` is divided into sections. See `config_example.yaml` for all options with comments.

**Minimum required:**
```yaml
input:
  vasp_file: "CONTCAR"           # path to slab CONTCAR

cell:
  supercell: [3, 6, 1]           # QM slab supercell repetitions

electrolyte_recipe:
  water:
    species_id: "water_tip3p"
    count: 500
  salts:
    - species_ids: ["K", "Cl"]
      concentration_M: 1.0

qe:
  pseudopotentials:
    Ir: "Ir.pbe-n-rrkjus_psl.0.1.UPF"
    O:  "O.pbe-n-kjpaw_psl.0.1.UPF"
```

**Key optional sections:**

| Section | Purpose |
|---|---|
| `ces2.water_model` | `TIP4P` (default, recommended) or `TIP3P` |
| `ces2.md` | LAMMPS MD settings (timestep, n_steps, temperature, …) |
| `ces2.lj_params` | Override per-type LJ epsilon/sigma |
| `ces2_script.slurm` | SLURM job settings for `submit_ces2.sh` |
| `qe.ecutwfc/ecutrho` | QE plane-wave cutoffs (Ry) |
| `qe.k_points` | Monkhorst-Pack k-grid |
| `bjdisp` | Global BJ-dispersion damping params (a1, a2, s8) |

---

## CES2 QM/MM Details

### Water model
- **TIP4P-EW** (default): `pair_style lj/cut/tip4p/long/opt`, `kspace_style pppm/tip4p`, M-site distance 0.125 Å, Smith-Dang ion LJ parameters.
- **TIP3P**: `pair_style lj/cut/long`, `kspace_style pppm`, Joung-Cheatham ion LJ parameters.

### Gridforce/net convention
The `gridforce/net` LAMMPS fix reads LAMMPS cube files written by `pp.x`. Cube index assignment:
- Index 0 → H (water proton group)
- Index 1 → O (water oxygen group)
- Index 2, 3, … → other elements (K, Na, Cl, …) in appearance order

The `#CUBEPOSITION` marker in `base.in.lammps` is replaced at runtime by `qmmm_dftces2_charging_pts.sh` with the actual `grid <cube_files>` command.

### QE runtime markers
`base.pw.in` contains two markers filled by the wrapper script at each QM/MM step:
- `###qmxyz` — current QM atom Cartesian positions (Å)
- `###dispf` — QM-MM dispersion forces from LAMMPS `compute fdisp` (Å, real units)

### Supercell factor
LAMMPS runs the full supercell (e.g., 3×6×1 = 18 unit cells). QE runs one unit cell. The gridforce/net `SC_FACTOR` parameter = rep_x × rep_y × rep_z. QE `nat` = total QM atoms / SC_FACTOR.

### BJ-dispersion (bjdisp)
Per-atom QM parameters (α_iso, C6, s) are stored in `species_db/qm_params/<system>.yaml`. MM species parameters come from their species YAML `atoms[].bjdisp` field. The `bjdisp_db.py` module combines these into `pair_coeff MM_type QM_type bjdisp ...` lines.

---

## Adding a New Ion / Molecule

Create a YAML file in `species_db/` following `species_db/SCHEMA.md`. For ions with BJ-dispersion parameters, add an `atoms[].bjdisp` block:

```yaml
id: Cl
net_charge: -1.0
atoms:
  - element: Cl
    type_label: Cl
    charge: -1.0
    xyz: [0.0, 0.0, 0.0]
    bjdisp:
      alpha_iso: 14.2      # a.u.
      C6: 320.5            # kcal/mol·Å^6
      s: 1.40
```

---

## Adding QM Surface Parameters

For a new QM slab system, add `species_db/qm_params/<SystemName>.yaml`:

```yaml
system: RuO2
types:
  Ru:
    alpha_iso: 28.5        # a.u.
    C6: 3800.0             # kcal/mol·Å^6
    s: 1.40
  O:
    alpha_iso: 3.9
    C6: 142.9
    s: 1.40
```

Then update `qe.pseudopotentials` in your config.yaml.

---

## Requirements

- Python ≥ 3.9
- PACKMOL (separate install — see Quick Start)
- Python packages: `ase`, `numpy`, `pyyaml`, `scipy` (see `requirements.txt`)
- For running simulations: LAMMPS with `gridforce/net` and `bjdisp` pair style, Quantum ESPRESSO ≥ 7.0
