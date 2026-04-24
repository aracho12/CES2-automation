# CES2 QM/MM Automation

Automated builder for DFT-CES2 QM/MM simulations — generates all required input files (LAMMPS data file, QM/MM LAMMPS input, Quantum ESPRESSO input, and run scripts) from a VASP CONTCAR and a single config file.

---

## What it does

Running `python run_builder.py` generates a complete, ready-to-run CES2 QM/MM directory from three inputs:

| Input | Description |
|---|---|
| `CONTCAR` | VASP slab structure (QM region) |
| `config.yaml` | System settings (electrolyte recipe, supercell, QE params, …) |
| `species_db/` | YAML database of ions/molecules and QM surface parameters |

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
| `charged_system_params.json` | Auto-calculated charged system parameters |
| `build_summary.json` | Build metadata, composition, and timing |

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
# electrolyte_recipe, qe section
```

### 4. Run

```bash
# Auto-detect config*.yaml in cwd (works when exactly one config*.yaml exists):
python run_builder.py

# Explicit config path:
python run_builder.py --config config.yaml

# Override CONTCAR path (takes priority over input.vasp_file in config):
python run_builder.py --config config.yaml --input /path/to/CONTCAR
```

---

## Project Structure

```
CES2-automation/
├── builder/
│   ├── __init__.py              # Exports run(config_path, vasp_file)
│   ├── main.py                  # Orchestrator — calls all writers in order
│   ├── config.py                # Config YAML loader (Config dataclass)
│   ├── builder.py               # Type registry, charge distribution, LAMMPS data assembly
│   ├── lammps_input_writer.py   # Generates base.in.lammps (TIP4P-EW, gridforce/net)
│   ├── qe_writer.py             # Generates base.pw.in + base.pp.in
│   ├── ces2_script_writer.py    # Generates qmmm wrapper + SLURM/PBS submit scripts
│   ├── bjdisp_db.py             # BJ-dispersion parameter DB + layer-file parser
│   ├── lammps_writer.py         # LAMMPS data file writer (reference style)
│   ├── packmol.py               # PACKMOL interface (electrolyte packing)
│   ├── composition.py           # Electrolyte composition from recipe + counterion balancing
│   ├── species.py               # Species / molecule data model (AtomDef, Species)
│   ├── vasp_io.py               # VASP CONTCAR reader + supercell builder
│   ├── box.py                   # Simulation box geometry
│   └── md_workflow.py           # Optional MD pre-relaxation workflow (4-stage)
├── species_db/
│   ├── SCHEMA.md                # Species YAML format documentation
│   ├── water_tip4p.yaml         # TIP4P water (recommended for CES2)
│   ├── water_tip3p.yaml         # TIP3P water
│   ├── K_plus.yaml, Na_plus.yaml, ...  # Cation species
│   ├── Cl_minus.yaml, OH_minus.yaml, ... # Anion species
│   ├── TMA_plus.yaml, H3O_plus.yaml     # Polyatomic species
│   └── qm_params/
│       └── IrO2.yaml            # QM slab BJ-dispersion parameters (alpha_iso, C6, s)
├── config_example.yaml          # Fully-annotated example config (all options)
├── run_builder.py               # CLI entry point
└── requirements.txt
```

**Available species in `species_db/`:**
Cations: `K_plus`, `Na_plus`, `Li_plus`, `Cs_plus`, `H3O_plus`, `TMA_plus`.
Anions: `Cl_minus`, `F_minus`, `Br_minus`, `I_minus`, `OH_minus`, `CO3_2minus`, `NO3_minus`, `SO4_2minus`.
Water: `water_tip4p`, `water_tip3p`.

---

## Config Reference

`config.yaml` is divided into hierarchical sections. See `config_example.yaml` for all options with inline comments.

### Minimum required config

```yaml
input:
  vasp_file: "CONTCAR"

cell:
  supercell: [3, 6, 1]

electrolyte_recipe:
  water:
    count: 500
  salts:
    - name: "KOH"
      concentration_M: 1.0
      stoich: { "K_plus": 1, "OH_minus": 1 }

qe:
  pseudo_set: "sssp"             # or provide per-element pseudopotentials
  # pseudopotentials:            # overrides pseudo_set for listed elements
  #   Ir: "Ir.pbe-n-rrkjus_psl.0.1.UPF"
  #   O:  "O.pbe-n-kjpaw_psl.0.1.UPF"
```

### All config sections

| Section | Purpose | Required? |
|---|---|---|
| `project` | `workdir`, `seed` | No (defaults: `"./"`, `1234`) |
| `species_db` | Path to species database directory | No (defaults to bundled `species_db/`) |
| `input` | `vasp_file` — path to CONTCAR | **Yes** |
| `cell` | `supercell: [nx, ny, nz]`, `require_orthogonal` | **Yes** |
| `electrolyte_box` | Simulation box geometry (z_gap, thickness, vacuum, margins) | No (has defaults) |
| `packmol` | PACKMOL binary path, tolerance, maxit | No (defaults: `packmol`, 2.0, 200) |
| `charge_control` | Electrode charging and counterion balancing | No (defaults to neutral) |
| `electrolyte_recipe` | Water count, salts, extras, counterion pool, z-exclusion | **Yes** |
| `output` | `build_dir`, `export_dir` | No (defaults: `build/`, `export/`) |
| `slab` | QM atom type_label overrides, layer-file path | No |
| `ces2` | LAMMPS QM/MM settings (water model, cutoffs, bjdisp, MD params) | No (has defaults) |
| `qe` | Quantum ESPRESSO settings (cutoffs, k-points, pseudopotentials) | **Yes** (at least `pseudo_set` or `pseudopotentials`) |
| `ces2_script` | Script generation (binary paths, SLURM/PBS settings) | No |
| `md_relax` | Optional MD pre-relaxation (4-stage equilibration) | No (disabled by default) |

---

### `electrolyte_recipe` — Electrolyte composition

```yaml
electrolyte_recipe:
  water:
    # species_id is OPTIONAL — auto-derived from ces2.water_model:
    #   TIP4P → water_tip4p (default, recommended)
    #   TIP3P → water_tip3p
    # species_id: "water_tip4p"
    count: 2850
    density_g_per_ml: 1.0        # for volume estimation (default: 1.0)

  salts:
    - name: "KOH"
      concentration_M: 0.1
      stoich: { "K_plus": 1, "OH_minus": 1 }   # species_id → coefficient

    # Multiple salts are supported:
    # - name: "TMACl"
    #   concentration_M: 0.2
    #   stoich: { "TMA_plus": 1, "Cl_minus": 1 }

  # Extra explicit species (added on top of salt-derived counts)
  extras: []
  # - species_id: "OH_minus"
  #   count: 5

  # Counterion pool — auto-adjusts ion counts to satisfy q_target_total.
  # Include at least one cation and one anion for reliable balancing.
  counterion_pool: ["K_plus", "OH_minus"]

  # Z-exclusion: minimum distance from slab surface for packed species
  z_exclusion:
    water_A: 2.5                 # Å for water molecules
    ions_A: 4.0                  # Å for charged species
    neutral_A: 2.0               # Å for neutral non-water molecules
```

**How salt counts are calculated:** The builder converts `concentration_M` to molecule counts using the volume estimated from `water.count` and `water.density_g_per_ml`. Each formula unit generates `stoich[species_id] × n_units` molecules.

---

### `charge_control` — Electrode charging

```yaml
charge_control:
  q_target_total: 0.0            # total MD box charge (0 for neutrality)
  q_electrode_user_value: -1.0   # charge on slab surface [e]
                                 # negative = more electrons (e.g. -1.0 adds 1 e⁻)
  top_layer_tolerance: 0.5       # [Å] atoms within this distance from z_max get charge
  exclude_labels: ["O_ads", "H_ads"]  # type_labels excluded from charging
```

When `q_electrode_user_value ≠ 0`, the builder automatically:
1. Detects the topmost atomic layer (z > z_max − `top_layer_tolerance`).
2. Distributes the electrode charge uniformly across top-layer atoms (excluding any in `exclude_labels`).
3. Adjusts counterion counts via `counterion_pool` to maintain `q_target_total`.
4. Auto-calculates CES2 charged-system parameters: `tot_chg` (QE tot_charge per unit cell = q_electrode / supercell_factor), `mpc_layer` (top-layer z in bohr), `plate_pos` (slab midpoint as fraction of box z). These are written to `charged_system_params.json` and injected into the run scripts.

---

### `slab` — QM atom labeling

Two methods exist for assigning custom `type_label`s to slab atoms (which determine their BJ-dispersion parameters). Both can be combined — explicit overrides take priority.

**Method 1: `type_label_overrides`** — Manual per-atom labeling

```yaml
slab:
  type_label_overrides:
    99:  "O_ads"                 # 1-based PRIMITIVE CELL index (POSCAR order)
    100: "O_ads"
    101: "H_ads"
    102: "H_ads"
```

The builder auto-replicates each primitive index across all supercell copies (ASE repeat ordering). Custom labels must exist in `species_db/qm_params/<system>.yaml`.

**Method 2: `bjparams_layer_file`** — Automatic per-layer labeling

```yaml
slab:
  bjparams_layer_file: "bjparams_layer_avg.dat"   # z-dependent BJ parameters
```

Reads a DFTD3-style layer file and assigns a unique `type_label` per (element, z-layer). This replaces hand-written overrides for systems whose QM atoms have z-dependent parameters. Requires `supercell[2] == 1` (no z-tiling).

**Priority order:** `type_label_overrides` > layer-file label > element symbol.

---

### `electrolyte_box` — Simulation box geometry

```yaml
electrolyte_box:
  z_gap: 0.0                    # [Å] gap between slab top and electrolyte bottom
  thickness: 60.0               # [Å] electrolyte region thickness
  z_margin_top: 2.0             # [Å] margin above electrolyte for packing
  vacuum_z: 20.0                # [Å] vacuum gap above water (for boundary p p f)
  z_buffer_lo: 1.0              # [Å] buffer below slab (zlo = −z_buffer_lo)
```

---

### `ces2` — LAMMPS QM/MM settings

```yaml
ces2:
  water_model: TIP4P             # TIP4P (default, TIP4P-EW) or TIP3P
  lj_cutoff: 15.0                # [Å] real-space LJ cutoff
  coulomb_cutoff: 15.0           # [Å] real-space Coulomb cutoff
  kspace_accuracy: 1.0e-4        # PPPM relative accuracy
  kspace_slab: 3.0               # kspace_modify slab factor (2D correction)
  tip4p_msite: 0.125             # [Å] TIP4P M-site distance (0.125 for TIP4P-EW)
  prefix: "ces2"                 # dump/restart file prefix

  # Optional: load pre-equilibrated dump after read_data
  # initial_dump: "equilibrated.dump"

  # BJ-dispersion global damping parameters
  bjdisp_a1: 1.40
  bjdisp_a2: 0.50
  bjdisp_s8: 2.10

  # LJ parameter overrides: type_label → {epsilon [kcal/mol], sigma [Å]}
  # Built-in defaults for TIP4P-EW water + Smith-Dang ions are used if omitted.
  # lj_params:
  #   MyAtom: { epsilon: 0.200, sigma: 3.500 }

  # MD run settings for the CES2 QM/MM loop
  md:
    timestep_fs: 0.5
    n_steps: 0                   # 0 = single-point; >0 for dynamics
    thermo_every: 100
    dump_every: 1000
    temperature: 300.0           # K
    t_damp_fs: 100.0             # NVT Nose-Hoover damping, fs
    restart_every: 500000
    shake_tol: 1.0e-4
    shake_iter: 20
    shake_maxiter: 500
```

---

### `qe` — Quantum ESPRESSO settings

```yaml
qe:
  prefix: "solute"
  outdir: "./solute"
  pseudo_dir: "/path/to/pseudo"
  ecutwfc: 50.0                  # Ry
  ecutrho: 400.0                 # Ry
  occupations: "smearing"
  smearing: "mv"                 # mv, cold, gaussian
  degauss: 0.0147                # Ry
  k_points: [1, 1, 1, 0, 0, 0]  # Monkhorst-Pack: n1 n2 n3 s1 s2 s3
  emaxpos: 0.8                   # dipole correction position (fraction of cell z)
  edir: 3                        # dipole direction (3 = z)
  conv_thr: 1e-06                # SCF convergence threshold (Ry)
  electron_maxstep: 400
  mixing_beta: 0.3               # charge density mixing
  diagonalization: "cg"          # "cg" (stable) or "david" (faster)

  # Pseudopotential selection — two approaches:
  pseudo_set: "sssp"             # Use built-in SSSP PBE library (covers most elements)
  # Per-element overrides (takes priority over pseudo_set):
  # pseudopotentials:
  #   Ir: "Ir_custom.UPF"
  #   O:  "O_custom.UPF"
```

---

### `ces2_script` — Run script generation

```yaml
ces2_script:
  jobname: "ces2_qmmm"
  dft_ces2_path: "/path/to/dft-ces2"
  qe_binary: "/path/to/pw.x"
  pp_binary: "/path/to/pp.x"
  lmp_binary: "${DFT_CES2_PATH}/MD/lammps/src-ind-cube/lmp_mpi"
  chg2pot_binary: "/path/to/chg2pot"
  mdipc_binary: "/path/to/mdipc"
  chgplate_binary: "/path/to/make_rho_mino"
  np: 24                         # MPI tasks

  n_qmmm_steps: 6               # total QM/MM outer-loop steps
  tot_layer: "4"                 # number of layers for Poisson solver
  mpc_one: "1"                   # 1 = top layer only for point charge scheme
  adsorbate: "0"                 # number of adsorbate atoms (excluded from point charge)

  # Auto-calculated from charge_control (override only if needed):
  # tot_chg:   auto              # QE tot_charge per unit cell
  # mpc_layer: auto              # top-layer z [bohr] for MDIPC
  # plate_pos: auto              # slab midpoint fraction for CHGPLATE

  # QE parallelization flags (pw.x command-line options):
  # npool: 1                     # k-point pools (must divide n_kpoints)
  # ntg: 2                       # FFT task groups
  # ndiag: 9                     # ScaLAPACK procs (must be perfect square)

  # SLURM batch settings:
  slurm:
    account: "members"
    partition: "skylake_24c"
    nodes: 1
    no_requeue: true
    time: "00:00:00"
    comment: "qmmm"

  # Or PBS: (use one or the other)
  # pbs:
  #   queue: "normal"
  #   nodes: 1
  #   ppn: 24
```

---

### `md_relax` — Optional MD pre-relaxation

When `enabled: true`, the builder generates a 4-stage equilibration workflow that runs before QM/MM to relieve PACKMOL packing artifacts.

```yaml
md_relax:
  enabled: true
  lj_cutoff: 10.0               # [Å] lighter cutoff for relax (no bjdisp, no gridforce)
  data_file: "data.file"
  timestep_fs: 1.0
  dump_every: 2000
  write_equilibrated: "equilibrated.data"

  # Stage 0: Soft pushoff — relieve extreme PACKMOL overlaps
  soft_cutoff: 2.0               # [Å] pair_style soft cutoff
  soft_ramp_max: 50.0            # [kcal/mol] ramp A from 0 → this
  soft_steps: 2000
  nve_limit: 0.1                 # [Å/step] max atom displacement

  # Stage 1: CG minimization
  min_etol: 1.0e-4               # energy tolerance
  min_ftol: 1.0e-3               # force tolerance
  min_maxiter: 1000
  min_maxeval: 5000

  # Stage 2: Gradual heating (t_start → t_target)
  t_start: 10.0                  # K
  t_target: 300.0                # K
  tdamp_fs: 100.0                # Nosé-Hoover damping
  heat_steps: 10000

  # Stage 3: NVT equilibration at t_target
  equil_steps: 30000
```

Use `ces2.initial_dump` to start QM/MM from the equilibrated structure.

---

## CES2 QM/MM Details

### Water model

- **TIP4P-EW** (default): `pair_style lj/cut/tip4p/long/opt`, `kspace_style pppm/tip4p`, M-site distance 0.125 Å, Smith-Dang ion LJ parameters.
- **TIP3P**: `pair_style lj/cut/long`, `kspace_style pppm`, Joung-Cheatham ion LJ parameters.

Water species is auto-derived from `ces2.water_model` when `electrolyte_recipe.water.species_id` is omitted: `TIP4P` → `water_tip4p`, `TIP3P` → `water_tip3p`.

### Gridforce/net convention

The `gridforce/net` LAMMPS fix reads LAMMPS cube files written by `pp.x`. Cube index assignment: index 0 → H (water proton group), index 1 → O (water oxygen group), index 2, 3, … → other elements in appearance order. The `#CUBEPOSITION` marker in `base.in.lammps` is replaced at runtime by the wrapper script.

### QE runtime markers

`base.pw.in` contains two markers filled by the wrapper script at each QM/MM step: `###qmxyz` (current QM atom Cartesian positions) and `###dispf` (QM-MM dispersion forces from LAMMPS `compute fdisp`).

### Supercell factor

LAMMPS runs the full supercell (e.g., 3×6×1 = 18 unit cells). QE runs one unit cell. The gridforce/net `SC_FACTOR` parameter = rep_x × rep_y × rep_z. QE `nat` = total QM atoms / SC_FACTOR.

### BJ-dispersion (bjdisp)

Per-atom QM parameters (α_iso, C6, s) come from either `species_db/qm_params/<system>.yaml` (element-based) or from a layer file (`slab.bjparams_layer_file`, z-dependent). MM species parameters come from their species YAML `atoms[].bjdisp` field. The `bjdisp_db.py` module combines these into `pair_coeff MM_type QM_type bjdisp ...` lines.

---

## Adding a New Ion / Molecule

Create a YAML file in `species_db/` following `species_db/SCHEMA.md`. For ions with BJ-dispersion parameters, add an `atoms[].bjdisp` block:

```yaml
id: Cl_minus
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

For polyatomic species, add `connectivity` and `coeffs` sections (see `SCHEMA.md` and existing species like `water_tip4p.yaml` or `OH_minus.yaml`).

---

## Adding QM Surface Parameters

For a new QM slab system, add `species_db/qm_params/<SystemName>.yaml`:

```yaml
system: RuO2
types:
  Ru:
    element: Ru
    alpha_iso: 28.5        # a.u.
    C6: 3800.0             # kcal/mol·Å^6
    s: 1.40
  O:
    element: O
    alpha_iso: 3.9
    C6: 142.9
    s: 1.40
  # Custom labels for adsorbates:
  O_ads:
    element: O
    alpha_iso: 4.5
    C6: 176.0
    s: 1.40
```

Then reference these labels in your config via `slab.type_label_overrides` or `slab.bjparams_layer_file`.

---

## Build Workflow

The builder executes the following steps in order:

1. Load config YAML and set `workdir`/`seed`
2. Load species database (all YAML files in `species_db/`)
3. Read VASP CONTCAR and build supercell
4. Compute simulation box geometry from `electrolyte_box` settings
5. Parse `electrolyte_recipe` → ion/molecule counts from salts + extras
6. Apply `charge_control` → counterion balancing, electrode charge distribution
7. Write per-species XYZ templates for PACKMOL
8. Run PACKMOL → pack electrolyte into box
9. Combine PACKMOL output with slab supercell
10. Build type registry (stable sorting, layer-file and override labels)
11. Assign MM atom types/charges/connectivity
12. Detect top layer and distribute electrode charge (if charged)
13. Auto-calculate charged system parameters (`tot_chg`, `mpc_layer`, `plate_pos`)
14. Write `data.file` (LAMMPS reference style)
15. Generate `base.in.lammps` (with TIP4P/bjdisp/gridforce)
16. Generate `base.pw.in` + `base.pp.in` (QE inputs)
17. Generate `qmmm_dftces2_charging_pts.sh` + `submit_ces2.sh`
18. Optionally generate MD pre-relax bundle (`md_relax.enabled: true`)
19. Write `build_summary.json` and timing data

---

## Requirements

- Python ≥ 3.9
- PACKMOL (separate install — see Quick Start)
- Python packages: `ase`, `numpy`, `pyyaml`, `scipy` (see `requirements.txt`)
- For running simulations: LAMMPS with `gridforce/net` and `bjdisp` pair style, Quantum ESPRESSO ≥ 7.0
