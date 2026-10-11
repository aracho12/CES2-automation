# 06 — Analysis & Operations Tools

Everything lives in `tools/`. Python tools run with the builder's environment
(`numpy`, `matplotlib`, `ase`; `rdf.py` needs `freud` and `seaborn` — on a
cluster without compilers `pip install --only-binary=:all: seaborn`).
Run `<tool> --help` for the full option list; this page says what each one is
for and what to watch out for.

## 1. Operations (submit, resume, protect)

| Tool | Purpose | Notes |
|---|---|---|
| `ces2_preflight.sh <sys_or_export>…` | **Pre-submit audit.** Host reality (binaries, pseudo_dir, NP vs allocation, scheduler header), charge three-consistency, k-mesh, step-state traps, restarts, seed files | exit 2 = do not submit. Read-only. [02 §3](02-first-run.md), [08](08-new-charge-state-checklist.md) |
| `ces2_resubmit.sh` | **Resume a killed chain.** Detects the last completed phase, patches `QMMMINISTEP`/`initialqm`/`skipequil`/`LAMMPSRESTART`, submits | run inside `export/`; self-chaining is PBS-only, on SLURM `sbatch` yourself. [05 §3](05-monitoring-and-restart.md) |
| `ces2_backup_save.sh backup\|restore\|list <export> [label]` | **Snapshot the volatile QE `.save`** (converged charge density = SCF seed) to `<export>/_save_backup/<label>/` | refuses while a live job writes the directory. [05 §6.5](05-monitoring-and-restart.md) |
| `configure.sh [--master\|--variant]` | Interactive wizard for config YAMLs (master = structure/physics, variant = salt/charge/job name) | [README §Master/Variant](../README.md) |
| `extract_bjparams.py` | BJ dispersion parameters from a bulk reference calculation (build-time helper) | |
| `surface_charge.py <export\|build_summary.json\|data.file> --charge-e N` | Surface charge density σ (µC cm⁻²) from the box area | `--surfaces 2` if the charge is shared by two faces |

A project-level progress monitor (per-system phase, SCF iteration / MD step
%, elapsed, scheduler state) is not yet in `tools/`; the parsing rules it
should implement are in [05 §2.2](05-monitoring-and-restart.md).

## 2. QM-side analysis

| Tool | Purpose | Inputs → outputs |
|---|---|---|
| `workfunction.py <qm_N or export>` | Planar-average `total_pot_ortho.cube` → vacuum level, Fermi level, **Φ**, `U_SHE = Φ − 4.44 V`, sawtooth amplitude | `workfunction_summary.txt`, `workfunction_rawdata.csv`, `workfunction_profile.png`, `pot.z.avg` |
| `wf_convergence.py SYSDIR… -o OUTDIR [--metric vrhe\|phi\|vshe] [--ph 13] [--she 4.44] [--recompute]` | **Step-convergence check**: Φ / V_SHE / V_RHE vs `qm_N` for many systems, one panel per charge state; reuses cached `qm_N/workfunction.out`, computes missing ones | `wf_convergence_<metric>.png`, `wf_convergence.csv`, `convergence_summary.txt` (Δ of the last two steps; `|Δ| < 0.05 V` flagged converging) |

Interpretation: shrinking step-to-step drift (e.g. 0.1–0.3 V at `qm_1→2`,
0.02–0.08 V at `qm_2→3`) is healthy; compare same-step values across systems
and trends within a system. The first QM step after a chain (re)start is not
representative ([05 §7](05-monitoring-and-restart.md)). `V_RHE = V_SHE +
0.059·pH` with the pH you pass.

## 3. MM-side analysis

| Tool | Purpose | Notes |
|---|---|---|
| `traj_multi_analysis.py TRJ… --cation X -o OUT [--lammps-input in.lammps] [--cutoff r_c] [--layer-mode peak\|min] [--layer-z z] [--max-lag-ps 500] [--rebuild-cache]` | **One-pass 5-in-1**: parses each trajectory once into `<trj>.multi.npz`, then z-density, cation–Ow RDF (→ `r_c`), coordination ACF, first-layer occupancy ACF, water orientation | re-runs with new cutoffs/boundaries are seconds; pool all `mm_N` of a case in ONE call |
| `z_density.py TRJ [--elements …]` | ρ(z) per element/type label and its evolution across `mm_N` | |
| `rdf.py TRJ` | RDFs (freud backend), default pairs Ow–Ow, Hw–Ow, cation–Ow | needs `freud`, `seaborn` |
| `water_orientation.py TRJ` | P1(z), P2(z) of the water dipole and O–H vectors | |
| `plot_temperature.py LOG [--avg-window]` | T(t) from LAMMPS thermo output; DOF-corrected for frozen QM atoms and SHAKE; sweeps `mm_N` | |
| `lammpstrj_to_traj.py TRJ [-o x.traj]` | LAMMPS dump → ASE `.traj` for `ase gui` | type→element map from `in.lammps` or `--types` |
| `plot_setting.py` | shared matplotlib defaults (font sizes, line widths) imported by the plotting tools | |

### Guardrails for MM analysis (each one cost a retracted figure)

1. **`mm_N` segments are not continuous in time.** Each
   `mm_N/ces2.emd.lammpstrj` is the averaging run of one step (its last
   2 M steps by default, 1 ps/frame). Timestep labels overlap between
   segments; they are separate runs. Time-correlation functions use lags
   **within** a segment only (`traj_multi_analysis.py` enforces this; do not
   undo it by concatenating files). Pooling segments as independent origins
   is fine.
2. **Put the TIP4P water charge on the M-site** (`O + 0.125 Å` along the
   bisector for TIP4P-Ew; the value is the last argument of
   `pair_style lj/cut/tip4p/long` in `in.lammps`) in any electrostatics
   computed from trajectories — Poisson profiles, site potentials, fields,
   dipoles. Putting it on O over-weights the water dipole/quadrupole and
   flipped the sign of a cation-trend conclusion once. OH⁻ carries its full
   charge on O in the default species.
3. **Sharp z* cuts bias large ions.** A first-layer boundary at the water-O
   peak sits ~0.3 Å too low for Cs⁺ relative to K⁺ purely from ionic radius,
   so "who adsorbs more" can flip with a 0.2 Å shift. For adsorption
   amounts integrate each cation's **own** first z-peak to its own first
   minimum (available from the npz cache) and report the boundary
   sensitivity.
4. **Occupancy `tau_int` is cutoff-dependent.** The tool integrates the
   occupancy ACF up to `--max-lag-ps` without subtracting the uncorrelated
   baseline `<h> = frac_in`, so for high-occupancy cations it grows with the
   cutoff. For residence times use the normalized
   `δC = (C − <h>)/(1 − <h>)` and quote its 1/e time, or compare
   `tau_cont`. (Same caveat for the coordination `tau_int`.)
5. **Sanity anchors** for a 0.5 M alkali hydroxide at an oxide surface: water
   first peak ≈ slab top + 1.8 Å; `r_c` Li ≈ 2.6, Na ≈ 3.2, K ≈ 3.5,
   Cs ≈ 3.9 Å; bulk `n_coord` Li ≈ 3–4 … Cs ≈ 7–8. If these are off,
   suspect the type map (`--lammps-input`, group comments in `in.lammps`)
   before the physics.
6. **Adsorption statistics are sensitive to the QM k-mesh** even when the
   work function is not: compare adsorption only between runs with the same
   mesh ([09 A9](09-failure-catalog.md)).
7. On a shared login node pin BLAS threads (`OMP_NUM_THREADS` etc.) — the
   npz analyses are numpy-heavy — or submit them through the scheduler.

## 4. Typical workflow after a chain finishes

1. `wf_convergence.py` over the dataset → which systems are converged.
2. For converged systems: `workfunction.py` on the last `qm_N` for the
   quantities you will quote; `traj_multi_analysis.py` over the equilibrated
   `mm_N` segments (drop the first segment after any restart).
3. Keep plotted numbers in CSVs next to the figures; record which `qm_N`/`mm_N`
   each number came from — provenance is the first thing a reviewer (or your
   future self) asks for.
