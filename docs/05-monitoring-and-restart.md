# 05 — Monitoring & Restart

<!-- TODO: fill in. Outline below. -->

## Reading `qmmm.out`

<!-- TODO: the phase markers and what a healthy step looks like:
     "######### Starting N QMMM step #########"
     "### QM calculation done"
     "### Running LAMMPS(emxext) for equilibration ..."   ← equilibration
     "### Running LAMMPS(emdext) for averaging ..."        ← averaging
     "### ERROR: ..." lines and preflight_check failures. -->

## Where progress lives on disk

<!-- TODO: qm_N/mm_N appearance order; how to tell which phase a killed run
     died in (mm_N empty vs. populated, newest restart timestep vs. MDEQUIL/MDAVERAGE). -->

## Resuming a killed run: `tools/ces2_resubmit.sh`

<!-- TODO: run inside export/; what it detects; the four variables it patches
     (QMMMINISTEP, initialqm, skipequil, LAMMPSRESTART — see 03); .bak and log
     files; when NOT to trust it (hand-edited scripts, partial cube writes). -->

## Seeding charged SCF from a converged charge density (speed + reliability)

**Always run charged systems (`TOTCHG ≠ 0`) with a charge-density seed.** An
`atomic`-superposition start on a charged slab oscillates or diverges for
hundreds of SCF iterations — observed repeatedly in production (q-20: ~1 iter/hr,
K diverging, Na hung; q-12: 170 iterations / 75 h plateaued at 1e-3 vs thr 1e-4).
A seeded SCF converges in **~10–16 iterations**.

**New builds:** set `ces2_script.qm_startingpot: "file"` in the config (see
`config_example.yaml`). **Existing scripts:** patch the `TOTCHG` conditional in
`qmmm_dftces2_charging_pts.sh` (the `STARTINGWFC`/`STARTINGPOT` line, ~line 54)
so the charged branch reads `STARTINGPOT="file"` (keep `STARTINGWFC="atomic"`).
For a hard SCF also drop `mixing_beta` 0.3 → 0.1 in `base.pw.in`.

**① Required files** — place all three in `<export>/solute/solute.save/`:

| file | size | if missing |
|---|---|---|
| `charge-density.dat` | ~180M | `pw_readschemafile: xml data file not found` |
| `data-file-schema.xml` | ~150–220K | same (holds nelec / grid metadata) |
| `paw.txt` | ~420K | `read_scf: Reading PAW becsum` crash (PAW setups → mandatory) |

`wfc*.dat` (~19G) are **not needed** — `STARTINGWFC` stays `atomic`; reading
another charge state's wavefunctions risks Cholesky errors.

**② Seed-source rules:**
- The density must come from a **converged** run (check `JOB DONE`) — never
  from an oscillating/diverged one.
- **Never seed a charged run from a neutral (qm_0 / PZC) density** — observed
  to converge badly. Charged runs need a charged density: the same system's
  previous step, **a neighbouring charge state** (QE renormalizes nelec on
  read — a converged q-20 can seed q-21), or **another cation's** converged
  density (the QM slab is cation-independent — verified Na→Cs/K/Li).
- Same QE version on both ends.

**③ `.save` is volatile — back it up:** the driver copies only cubes into
`qm_N/`, and the next pw.x run overwrites `solute/solute.save/`. Snapshot a
converged density with `tools/ces2_backup_save.sh backup <export> <label>`
(→ `<export>/_save_backup/<label>/`; `restore` to bring it back).

**④ Verify after (re)submit:** `pw.out` must print
`The initial density is read from file` and converge within a few dozen
iterations. If you see `Initial potential from superposition of free atoms`,
the seed silently fell back to atomic — stop and fix before burning walltime.

## Common failures

<!-- TODO:
     - PBS/SLURM walltime kill → resubmit (the standard case)
     - pw.x "pw_readschemafile" crash on resume of charged/neutral runs:
       stale solute.save with startingwfc='file' → set STARTINGWFC/STARTINGPOT
       to 'atomic' for the resumed step (or clear solute/)
     - LAMMPS "Non-numeric pressure" / lost atoms after bad packing → md_relax
     - preflight_check FAILED → wrong binary paths in ces2_script config
     - QM not converging on a charged system → almost always an atomic start;
       use the charge-density seed above (+ mixing_beta 0.1)
     - QEPW_FLAGS: leave empty — npool/ntg/ndiag gave unreliable results with
       the CES2-patched pw.x (see 03 §3) -->
