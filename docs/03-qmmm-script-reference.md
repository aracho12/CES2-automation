# 03 — `qmmm_dftces2_charging_pts.sh` Variable Reference

The run script is **generated** by the builder
(`builder/ces2_script_writer.py`); every value in its header comes from
`config.yaml` or from automatic structure analysis. **Prefer changing
`config.yaml` and rebuilding** over editing the script by hand — hand edits are
lost on the next rebuild and are easy to make inconsistent (several variables
must agree with `data.file` / `base.pw.in`).

Column key — **Set from**: the `config.yaml` key that controls the value.
*auto* = computed by the builder from the structure; you normally never touch it.

Variables are grouped in the order they appear in the script.

---

## 1. Loop control

| Variable | Example | Set from | Meaning |
|---|---|---|---|
| `NP` | `256` | `ces2_script.np` | MPI ranks used for `pw.x`, `pp.x`, and the production LAMMPS runs. Must match the cores requested in `submit_ces2.sh`. |
| `QMTYPE` | `"scf"` | `ces2_script.qm_type` | `scf`: single-point QM each step (standard). `opt`: additionally relax QM geometry each step and push the new positions back into LAMMPS. |
| `QMRELAX_HIGH` | `7` | `ces2_script.qm_relax_high` | Only used with `QMTYPE="opt"`. z threshold in Å: atoms with **z below this value are frozen** (`if_pos 0 0 0` is appended in `pw.in`); atoms above it relax. ⚠️ The inline comment in the script states the opposite — trust this table (see the `awk -v zval=` block in the script). |
| `QMMMINISTEP` | `0` | `ces2_script.qmmm_ini_step` | First outer-loop index. `0` starts with the reference QM (`qm_0`, bare neutral slab). **`ces2_resubmit.sh` patches this** when resuming a killed run — don't set it by hand unless you know the run state. |
| `QMMMFINSTEP` | `6` | `ces2_script.n_qmmm_steps` − 1 | Last outer-loop index, **inclusive**. `n_qmmm_steps: 7` → steps 0…6. |
| `QMMAXSTEP` | `"5"` | `ces2_script.qm_max_step` | Max retries of the QM calculation per step (only meaningful for `opt`; for `scf` the loop breaks after the first attempt). |
| `initialqm` | `0` | `ces2_script.initial_qm` | `1` = the QM part of the *current* step is already done: skip `pw.x`/`pp.x` and reuse the cubes in `qm_N/`. Used when resuming a run that died during the MD phase. Patched by `ces2_resubmit.sh`. |
| `skipequil` | `0` | `ces2_script.skip_equil` | `1` = skip the equilibration MD on the **first** resumed step (the run died during averaging, so equilibration is already done). Resets itself to 0 afterwards. Patched by `ces2_resubmit.sh`. |
| `firstrun` | `1` | — | Internal flag: triggers cell-box parsing on the first loop iteration. The comment says it all: *"Should be 1 for all the time."* |

## 2. Geometry & atom bookkeeping

| Variable | Example | Set from | Meaning |
|---|---|---|---|
| `SUPERCELL` | `(3 6 1)` | `cell.supercell` | How many times the QM unit cell is replicated (x y z) to build the MM slab. QE simulates **one unit cell**; LAMMPS simulates the full supercell. |
| `SUPERCELLFACTOR` | `18` | derived | Product of `SUPERCELL`. Used to average the supercell's dispersion forces back onto the unit-cell QM atoms, and to extend `nat` when parsing the slab from `data.file`. |
| `QMLMPTYPE` | `(6 7 8 9 10 11)` | auto (from `slab` labels) | LAMMPS atom-type IDs that belong to the QM slab in `data.file`. Everything else is MM. |
| `NSOLVATOM` | `"6484"` | auto | Number of MM (electrolyte) atoms. Convention: MM atoms have LAMMPS ids `1…NSOLVATOM`; slab atoms come after. The script extracts the slab as `id > NSOLVATOM`. |
| `atoms` | `(H O K H O H Ir …)` | auto | Element symbol for each LAMMPS atom type: index = LAMMPS type − 1. Used to write element names into `pw.in`. |
| `unit_LAMMPS2QE` | `0.001686594` | fixed | Force unit conversion kcal/(mol·Å) → Ry/bohr, applied to the time-averaged dispersion forces (`dispf.ave`) before they are handed to `pw.x` via the `###dispf` marker. |

## 3. Executables & templates

All tool paths come from `ces2_script.*` in `config.yaml`
(`dft_ces2_path`, `qe_binary`, `pp_binary`, `lmp_binary`, `chg2pot_binary`,
`mdipc_binary`, `chgplate_binary`).

| Variable | Meaning |
|---|---|
| `DFT_CES2_PATH` | Root of the DFT-CES2 distribution (LAMMPS fork + cube tools live under it). Exported so child processes see it. |
| `LAMMPS` | The **DFT-CES2 LAMMPS fork** (`src-ind-cube/lmp_mpi`) — a stock LAMMPS will not understand the `grid` command. |
| `QEPW`, `QEPP` | The **patched** QE `pw.x` / `pp.x` (with `dft_ces`, `rho_ces`, `pauli_rep_ces` input keys). |
| `QEPW_FLAGS` | `pw.x` parallelization flags, e.g. `-npool 4 -ntg 2 -ndiag 4` — from `ces2_script.npool/ntg/ndiag`. `npool` must divide the k-point count; `ndiag` must be a perfect square. |
| `CUBEADD`, `CUBEMULTI`, `CUBESUB` | Cube-grid arithmetic (add n cubes / multiply by scalar / subtract). |
| `BLUR` | Gaussian-blurs a cube (used for both MM repulsion and QM density; see §5). |
| `MDDIPOLE` (`dipc2`) | Dipole correction applied to the MM charge density. |
| `CHG2POT` | Solves Poisson: charge-density cube → potential cube. |
| `MDIPC`, `CHGPLATE` | Charged-system tools (point-charge and plate schemes, §6). |
| `LAMMPSIN`, `QMIN`, `QMIN2` | The builder-generated templates `base.in.lammps`, `base.pw.in`, `base.pp.in`. The script copies and `sed`-patches them each step — it never modifies the `base.*` files themselves. |
| `LAMMPSDATA` | `data.file` — the builder-generated LAMMPS data file (starting structure). |
| `LAMMPSRESTART` | LAMMPS restart to resume from. On a fresh run the file doesn't exist and `data.file` is used; during the run the script re-derives it (`ls -lrt *.restart | tail -1`). Patched to the latest restart by `ces2_resubmit.sh`. |

## 4. MD control

| Variable | Example | Set from | Meaning |
|---|---|---|---|
| `MDEQUIL` | `"2000000"` | `ces2_script.md_equil_steps` | Equilibration MD steps per QM/MM step (solvent relaxes in the new QM potential; nothing is averaged). |
| `MDAVERAGE` | `"2000000"` | `ces2_script.md_average_steps` | Averaging MD steps: the run that accumulates the time-averaged solvent charge-density cubes and dispersion forces. `MDEQUIL` must be a divisor of `MDAVERAGE` (the `fix ave/atom` window is set from it). |
| `EQUILTIMESTEP` | `"1.0"` | `ces2_script.equil_timestep_fs` | Timestep (fs) used **only** for the equilibration run; the averaging run keeps the timestep in `base.in.lammps`. Lets you equilibrate faster with a coarser step. |

With 2 M steps each at ~1 fs, one QM/MM step is ≈ 2 ns equilibration + 2 ns
averaging of MD, plus one or two SCF runs.

## 5. MM species & blur parameters

These arrays are all **auto-generated** from the water model and
`electrolyte_recipe` / `species_db`; they must stay mutually consistent
(same length, same species order) and consistent with `base.in.lammps`.
Don't edit them by hand — rebuild instead.

| Variable | Example | Meaning |
|---|---|---|
| `cube_output_MM` | `(hydrogen.cube oxygen.cube potassium.cube h_oh.cube o_oh.cube)` | Per-species time-averaged MM charge-density cubes that LAMMPS writes during the averaging run (ρ_q^B, one per distinct MM site). |
| `MMpartialCharge` | `(0.5242 -1.0484 1.0000 0.0000 -1.0000)` | Representative partial charge of each species in `cube_output_MM` (charge-density cube ÷ charge = number density, needed for the repulsion blur). |
| `MMrepA` | `(H O K H O)` | Element symbol of each species in `cube_output_MM`, used to look up repulsion/polarizability parameters in the tables below. |
| `ATOMrepANAME` / `ATOMrepA` / `ATOMalpha` | fixed tables | Lookup tables per element: Pauli-repulsion strength *repA* (a.u.) and isotropic polarizability α (a.u.). The blur radius is derived from α; the blurred, repA-weighted number density of each MM species becomes its repulsion potential contribution. |
| `TIP4Pinvolved` | `(1 1)` | TIP4P special case. First entry: 1 if a TIP4P-family water is present (its charge sits on the massless M-site, not on O). Second entry: index (0-based) of the TIP4P **oxygen** in `cube_output_MM`. For that species the repulsion is blurred from the separately dumped `rep_TIP4P_oxygen.cube` instead of the charge-density cube. |
| `cube_output_PolarDen` | `(px.cube py.cube pz.cube)` | Time-averaged polarization (induced-dipole) density from the Drude/induction model in LAMMPS; converted to bound charge by `cube_dip_parse` and added into `MOBILE_final.cube`. |
| `cube_coul_QM` | `(solute.pot.cube)` | QM → MM electrostatic potential cube handed to LAMMPS (v^A). |
| `cube_ind_QM` | `(solute.ind.cube)` | QM → MM induction potential cube (uniform-charge scheme; drives the induced dipoles). |
| `cube_rho_QM` | `(solute.rho.cube)` | QM electron density (bookkeeping; input to the blur). |
| `cube_QM_rho_hat` | `(c_rho_hydrogen.cube …)` | Blurred QM density, one per MM species (ρ̂^A): the Pauli-repulsion potential each MM species feels from the QM electrons. |
| `cubeio` / `cubeioequil` | derived | The argument string injected into the LAMMPS `grid` command: number of input cubes, number of output cubes, then the file names. The equilibration run uses `cubeioequil` (inputs only, **no** output averaging); the averaging run uses the full `cubeio`. |

## 6. Dipole correction & charged system

| Variable | Example | Set from | Meaning |
|---|---|---|---|
| `DIPOLECORR` | `"yes"` | `ces2_script.dipole_corr` | Apply a dipole correction to the MM charge density (`dipc2` on `MOBILE_final.cube`) so the slab's periodic images don't interact through the net solvent dipole. |
| `DIPOLEDIR` | `"3"` | `qe.edir` | Direction of the correction (1=x, 2=y, 3=z). Must match QE's `edir` (surface normal). |
| `DIPOLEPOS` | `"0.8"` | `qe.emaxpos` | Fractional position of the sawtooth discontinuity along `DIPOLEDIR` — put it in the vacuum region. Must match QE's `emaxpos`. |
| `TOTCHG` | `"-0.2222…"` | auto from `charge_control` (override: `ces2_script.tot_chg`) | QE `tot_charge` **per unit cell**: total electrode charge ÷ `SUPERCELLFACTOR` (e.g. −4 e over a 3×6×1 supercell → −4/18). Negative = extra electrons. `0` = PZC (potential of zero charge); the charged-system machinery below is then skipped. |
| `STARTINGWFC` / `STARTINGPOT` | derived | QE SCF starting guess, set by the `TOTCHG` conditional on the next line: charged runs start from `atomic`; neutral runs reuse the previous step's wavefunction (`file`, from `solute/solute.save`). `file` is faster but can crash with `pw_readschemafile` if the save directory is stale or incompatible — switching that case to `atomic` is the safe fallback. |

### Point-charge scheme (`MDIPC`) — electrostatics of the counter charge

Used when `TOTCHG ≠ 0`: the solvent-side compensating charge is represented by
point charges on the top slab layer when building the corrected QM potential.

| Variable | Example | Set from | Meaning |
|---|---|---|---|
| `mpclayer` | `"20.56"` | auto (`charged_system_params.json`; override `ces2_script.mpc_layer`) | z position (bohr) of the slab layer that carries the point charges — the top layer. |
| `totlayer` | `"6"` | `ces2_script.tot_layer` | Number of slab layers seen by the Poisson solver. |
| `mpcone` | `"1"` | `ces2_script.mpc_one` | `1` = only the top layer carries the point charge. |
| `adsorbate` | `"6"` | `ces2_script.adsorbate` | Number of adsorbate atoms, excluded from the point-charge layers. |

### Plate scheme (`CHGPLATE` / `make_rho_mino`) — induction

| Variable | Example | Set from | Meaning |
|---|---|---|---|
| `PLATEPOS` | `"12.411039"` | auto (override `ces2_script.plate_pos`) | z position (bohr) of a uniform counter-charge plate, placed **inside the slab** between its bottom and top layers. Used when constructing `solute.ind.cube` (the induction potential must not contain the monopole field of the net charge). |

---

## What `ces2_resubmit.sh` touches

When you resume a walltime-killed run with `tools/ces2_resubmit.sh`, it inspects
`qm_N`/`mm_N` progress and **patches four variables** in the script (leaving a
timestamped `.bak`): `QMMMINISTEP`, `initialqm`, `skipequil`, `LAMMPSRESTART`.
Everything else keeps its built value. See
[05 — Monitoring & restart](05-monitoring-and-restart.md).
