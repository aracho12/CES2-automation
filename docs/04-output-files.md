# 04 — Output File Reference

Everything that appears in `export/` during a DFT-CES2 run, grouped by **when
and by what it is created**. Jump to [What can I delete?](#what-can-i-delete)
for disk management.

A typical `export/` after a few QM/MM steps:

```
export/
├── (build outputs)          data.file, base.*, *.json, qm_atoms.txt, …
├── qmmm.out                 ← main progress log (wrapper stdout)
├── qm_0/ qm_1/ … qm_N/      ← per-step QM archive
├── mm_0/ mm_1/ … mm_N/      ← per-step MD archive
├── solute/                  ← QE scratch (wavefunctions)
├── (scratch cubes)          working copies of the newest step, ~100 files
└── (analysis outputs)       workfunction_*, z_density_*, rdf_*
```

---

## 1. Build time (written once by `run_builder.py`)

| File | Meaning |
|---|---|
| `data.file` | LAMMPS data file: slab (supercell) + packed electrolyte, `full` atom style. The starting structure of the whole run. |
| `base.in.lammps` | LAMMPS input **template**. The script copies it to `in.lammps` / `in.lammps.equil` each step and patches `run`, `read_data`, `timestep`, `grid`, … via `sed`. Never modified in place. |
| `base.pw.in` | QE `pw.x` SCF **template** with `###qmxyz` / `###dispf` placeholder markers, replaced each step by the current slab geometry and averaged dispersion forces. |
| `base.pp.in` | QE `pp.x` **template**; patched per step to dump potential (`plot_num=11`) or density (`plot_num=0`) cubes. |
| `qmmm_dftces2_charging_pts.sh` | The outer-loop wrapper itself ([03 — variable reference](03-qmmm-script-reference.md)). |
| `submit_ces2.sh` | SLURM/PBS submission script; runs the wrapper and redirects its stdout to `qmmm.out`. |
| `qm_atoms.txt` / `mm_atoms.txt` | LAMMPS atom-ID lists of the QM slab and MM electrolyte (used by analysis tools and for sanity checks). |
| `charged_system_params.json` | Auto-computed charging parameters (`tot_chg`, `mpc_layer`, `plate_pos`, …) that were baked into the script — check this before submitting a charged run. |
| `build_summary.json` | Build metadata: composition, box, ion counts, type map. First place to look when something doesn't add up. |
| `combined.xyz` | Slab + electrolyte merged structure for quick visual inspection (OVITO/VESTA). |
| `timings.json` | Builder wall-time bookkeeping. |

## 2. Run time — top level

Top-level files are the **working copies of the current step**; at the end of
each step the important ones are copied into `qm_N/` / `mm_N/`.

### The log you actually read

| File | Written by | Meaning |
|---|---|---|
| `qmmm.out` | wrapper script | Main progress log. Grep for `Starting N QMMM step`, `Running LAMMPS(emxext)` (equilibration), `Running LAMMPS(emdext)` (averaging), `QM calculation done`, `### ERROR`. |
| `<jobname>.oNNN` / `.eNNN` | scheduler (PBS/SLURM) | Batch stdout/stderr — MPI-level errors (crashes, walltime kill) end up here, not in `qmmm.out`. |

### MD stage

| File | Written by | Meaning |
|---|---|---|
| `in.lammps.equil`, `in.lammps` | wrapper (from `base.in.lammps`) | The actual patched inputs of the equilibration / averaging runs of the current step. |
| `data.md` | wrapper (`lmp -r restart`) | LAMMPS data file regenerated from the latest restart — what the MD runs really read. |
| `lammps.equil.out` / `lammps.average.out` | LAMMPS | Thermo logs of the two runs (temperature, energy drift). |
| `ces2.<step>.restart` | LAMMPS | Restart checkpoints (every 500 k steps by default). The wrapper always resumes from the newest one; `ces2_resubmit.sh` relies on them. |
| `ces2.emd.lammpstrj` | LAMMPS | Trajectory of the **averaging** run (1 frame/ps by default). ⚠️ One file per step, archived into `mm_N/`; the timestep labels overlap between consecutive `mm_N` but the frames are **not** continuous across steps — treat each `mm_N` trajectory as an independent 2 ns segment. |
| `dispf.ave` | LAMMPS (`fix ave/atom` dump) | Time-averaged dispersion forces on the slab atoms; converted to Ry/bohr and injected into the next `pw.in` at `###dispf`. |
| `data.solute`, `data.dispforce`, `data.solute.temp*`, `solute.temp` | wrapper (awk) | Scratch used to extract slab positions + averaged forces from `data.file`/`dispf.ave`. Disposable. |

### MM → QM cubes (solvent density pipeline)

| File | Written by | Meaning |
|---|---|---|
| `hydrogen.cube`, `oxygen.cube`, `potassium.cube`, `h_oh.cube`, … | LAMMPS (averaging run) | Time-averaged charge density of each MM species, ρ_q^B (names follow `cube_output_MM`). |
| `px.cube`, `py.cube`, `pz.cube` | LAMMPS | Time-averaged polarization (induced-dipole) density components. |
| `postdipx/y/z.cube`, `mobile.cube` | `cube_dip_parse`, `cube_add` | Intermediates: bound charge from polarization; sum of mobile species charges. Deleted by the script at the end of `post_dipole`. |
| `MOBILE_final.cube` | wrapper (`post_dipole`) | **The** MM charge density handed to `pw.x` (`rho_ces`): mobile charges + polarization bound charge, dipole-corrected if `DIPOLECORR=yes`. |
| `rep_TIP4P_oxygen.cube` | LAMMPS | TIP4P oxygen site density, dumped separately because TIP4P charge sits on the M-site (repulsion must come from O positions). |
| `c_rep_<species>.cube` | wrapper (`Blur_MM`) | Per-species blurred Pauli-repulsion potential from the MM side. |
| `repA.cube` | wrapper (`Blur_MM`) | Sum of all `c_rep_*` — the total MM repulsion potential given to `pw.x` (`pauli_rep_ces`) in the *ortho* run. |
| `dipc.cube` | `dipc2` | Dipole-correction potential for the MM density. |

### QM stage

Each step runs `pw.x` **twice** (when `QMTYPE="scf"`, step > 0):

1. **ortho** — with MM repulsion (`pauli_rep_ces=repA.cube`). The QM electrons
   are pushed out of the solvent region ("orthogonalized"). Its outputs
   (`*_ortho`) are used for **observables** (work function etc.).
2. **non-ortho** — repulsion replaced by `empty.cube`. Its outputs build the
   coupling potentials the **next MM run** will feel.

| File | Written by | Meaning |
|---|---|---|
| `pw.in` / `pw.out` | wrapper / `pw.x` | Patched SCF input & output of the ortho run. |
| `pw.nonortho.in` / `pw.nonortho.out` | wrapper / `pw.x` | Same for the non-ortho run. |
| `pp.pot*.in/out`, `pp.rho*.in/out` | wrapper / `pp.x` | Post-processing runs extracting potential / density cubes. |
| `solute/` (`solute.save`, `solute.xml`) | `pw.x` | QE outdir: wavefunctions & charge density. Enables `startingwfc='file'` restarts. **Large**; safe to delete after the run finishes. |
| `solute.rho_ortho.cube`, `solute.pot_ortho.cube` | `pp.x` (+ wrapper edits) | Electron density / potential of the ortho run. |
| `solute.rho.cube`, `solute.pot.cube` | `pp.x` (+ wrapper edits) | Non-ortho density; and the QM electrostatic potential handed to LAMMPS (v^A: pp output + sawtooth, monopole/dipole-corrected when charged). |
| `solute.ind.cube`, `solute.pot_uni.cube` | wrapper (cube tools) | Induction potential for the MM polarization model (built with the plate scheme when charged); `pot_uni` is its uniform-charge intermediate. |
| `c_rho_<species>.cube` | wrapper (`Blur_QM`) | QM density blurred once per MM species (ρ̂^A) — the repulsion the MM atoms feel from QM electrons. Input cubes to LAMMPS. |
| `empty.cube` | wrapper (`cube_multi × 0`) | An all-zero cube. Placeholder wherever a cube argument is required but no field should act (non-ortho `pauli_rep_ces`, unused LAMMPS slots). |
| `v_saw.cube`, `v_saw_{pw_,}{ortho,nonortho}.cube` | patched `pw.x` (`add_efield.f90`) | Sawtooth dipole-correction potential of each SCF; added into the solute potentials so MM sees the corrected field. |
| `v_hartree.cube` | patched `pw.x` (`v_of_rho.f90`) | Hartree potential of the QM valence electrons (diagnostic). |
| `v_md.cube` | patched `pw.x` (`dft_ces.f90`) | Poisson-solved potential of the MM density (`rho_ces`) on the QM grid — what the QM electrons feel from the solvent (diagnostic). |
| `total_pot_ortho.cube` / `total_pot_nonortho.cube` | wrapper (cube tools) | **Total** electrostatic potential (QM + MM): input for work-function analysis. `workfunction.py` reads `total_pot_ortho.cube`. |
| `ref_solute.rho.cube`, `ref_solute.pot.cube` | step 0 only | Reference density/potential of the bare neutral slab (`qm_0`); later steps subtract these to isolate solvent/charging effects. |

### Charged-system intermediates (only when `TOTCHG ≠ 0`)

| File | Written by | Meaning |
|---|---|---|
| `q.cube`, `V.cube`, `q_mono.cube`, `V_mono.cube` | `make_rho_mino` (CHGPLATE) | Counter-charge plate density/potential and their monopole parts, used to neutralize `solute.ind.cube`. |
| `mdipc.cube` | `mdipc` | Point-charge-scheme corrected charge distribution, used to rebuild `solute.pot.cube` with slab dipole correction. |
| `pot.cube` | `chg2pot` | Generic Poisson-solver output (charge → potential); overwritten many times per step. |
| `subtracted.cube`, `multiplied.cube`, `add.cube`, `blurred.cube` | cube tools | Single-use arithmetic scratch, overwritten constantly. Never analyze these — they are whatever the last operation left behind. |

## 3. Per-step archives: `qm_N/` and `mm_N/`

At the end of each phase the wrapper copies the step's results here. **These
are the files you analyze**; top-level cubes are just the newest working set.

**`qm_N/`** — `pw.in`, `pw.out` (+ `pw.nonortho.out`), `pp.*.out`, and the step's
`solute.rho*/pot*/ind` and `total_pot_*` / `v_saw_*` cubes. `qm_0/` instead
holds the reference `ref_solute.rho.cube` / `ref_solute.pot.cube`.

**`mm_N/`** — `in.lammps`, `in.lammps.equil`, `lammps.equil.out`,
`lammps.average.out`, `ces2.emd.lammpstrj` (that step's 2 ns averaging
trajectory), `dispf.ave`, `MOBILE_final.cube`, `repA.cube`, `empty.cube`, and
the newest `ces2.*.restart`. Analysis tools (`rdf.py`, `z_density.py`, …) are
typically run per-`mm_N`, so their `*.csv`/`*.png` may appear here too.

## 4. Analysis outputs (from `tools/`, run manually or post-run)

| Files | Tool | Meaning |
|---|---|---|
| `workfunction_profile.png`, `workfunction_fermi.png`, `workfunction_rawdata.csv`, `workfunction_summary.txt`, `pot.z.avg` | `tools/workfunction.py` | Planar-averaged `total_pot_ortho.cube` → vacuum level, work function per step. |
| `z_density*_evolution.png/csv`, `z_density_*rawdata.csv` | `tools/z_density.py` | Species density profiles along z and their evolution across `mm_N` steps. |
| `rdf_*.csv/png` | `tools/rdf.py` | Radial distribution functions from `mm_N` trajectories. |

## 5. Resubmission artifacts (from `tools/ces2_resubmit.sh`)

| File | Meaning |
|---|---|
| `qmmm_dftces2_charging_pts.sh.bak.<timestamp>` | Backup of the wrapper before each patch. |
| `ces2_resubmit_<timestamp>.log` | What the resubmit tool detected (last completed step, phase) and what it changed. |

---

## What can I delete?

**Keep** (needed for analysis / resume):

- `qm_N/`, `mm_N/` — the actual results
- `qmmm.out`, `build_summary.json`, `charged_system_params.json`
- the **newest** `ces2.*.restart` (needed to resume or extend)
- `base.*`, `data.file`, the wrapper & submit scripts
- analysis outputs

**Safe to delete once the run is finished** (largest first):

- `solute/` — QE wavefunctions, often the single biggest item
- older `ces2.*.restart` files (keep the newest)
- all top-level `*.cube` — working copies; everything analysis-worthy was
  archived into `qm_N/`/`mm_N`
- `data.md`, `data.solute*`, `data.dispforce`, `solute.temp`, `pw.in`,
  `pp.*.in` — regenerated every step

**Never delete while the job is still running** — the top-level cubes and
restarts *are* the loop's state.
