# 08 — New Charge State / Clone Checklist (read BEFORE building or submitting)

Hard-won on 2026-08-12: a cloned charge state ran an entire MD as the WRONG
system because derived files resurrected the old state. Follow this checklist
mechanically. The golden rule:

> **"I edited the file" ≠ "the run will use it."**
> The driver REGENERATES `in.lammps`/`in.lammps.equil` (from
> `base.in.lammps`), `data.md` (from a restart if one exists, else
> `data.file`), and `pw.in`/`pw.nonortho.in` (from `base.pw.in` + script
> variables) at runtime. Only the sources of truth matter.

## Sources of truth (the ONLY files you edit)

`base.pw.in`, `base.in.lammps`, `base.pp.in`, `data.file`,
`qmmm_dftces2_charging_pts.sh`, `submit_ces2.sh`.
Everything else (in.lammps*, data.md, ces2.*.restart, root cubes,
trajectories, solute.save) is derived or run state — when cloning to a new
charge, DELETE derived artifacts so nothing stale leaks.

## The three-consistency rule (charge state N)

All three must encode the SAME N, in these places:

| # | quantity | where | value |
|---|---|---|---|
| 1 | QM charge | script `TOTCHG` | −N/SUPERCELLFACTOR (e.g. −18/18 = −1.0) → sed'd into pw.in each step |
| 2 | MM solution net charge | `data.file` ion counts | n(cation) − n(OH⁻) = +N |
| 3 | Electrode point charges | **`base.in.lammps`** `set group top charge` (and mirrored in data.file slab charges) | −N/(top-layer atom count, here 288) |

A mismatch does NOT crash — it silently runs a different physical system
(monopole-shifted potentials; WF garbage with offset ∝ mismatch).

## Build paths

- **Preferred: fresh build** with `run_builder.py` (charge_control does 1–3
  consistently). See 02.
- **Clone shortcut** (copy a neighbor-charge export): edit the sources of
  truth for the new N; adjust ion count by DELETING bulk-most cations from
  `data.file` (atoms + their Velocities lines + header count) or rebuild;
  rescale slab charges in data.file; then delete ALL of:
  `ces2.*.restart`, `data.md`, `in.lammps`, `in.lammps.equil`, root
  ensemble cubes (`c_rho_*` etc.), root trajectories. A surviving restart
  resurrects the old box via restart→data.md conversion.
- **Contamination quarantine:** if any one file copied from the source
  turns out to be foreign (another machine's scheduler header, binaries,
  `pseudo_dir`, `NP`, flags; a different force-field snapshot), audit
  **every** other file from that source before trusting it — a symptom-level
  fix leaves the rest armed ([10 §2](10-automation-safety.md)).
- Extending a **finished** chain of the same system is a different
  operation (four header lines, restart naming): [05 §5](05-monitoring-and-restart.md).

## qm_0 reuse (always do this — saves half a day per system)

qm_0 is the NEUTRAL reference: identical across charge states AND cations.
Copy any completed system's `qm_0/` archive (must contain
`ref_solute.rho.cube`, `ref_solute.pot.cube`) into the new export and set
`QMMMINISTEP=0`, **`initialqm=1`**. The chain then starts at mm_0.

### `initialqm` semantics (verified in driver source)

`initialqm=1` = **"skip the QM of the INISTEP step; resume with its MM"**
(pw.x is gated on `initialqm==0`; the flag auto-resets after the first loop
pass). Consequences:
- INISTEP=0 + initialqm=1 → skip qm_0, run mm_0 first. ✓ (reuse trick)
- INISTEP=N + initialqm=1 → skip qm_N — only valid if qm_N's outputs
  (archive + root coupling cubes) already exist and are current.
- INISTEP=FINSTEP + initialqm=1 → **the job runs NO new QM at all.** Trap.
- `skipequil=1` makes the first MM skip equilibration (resume-averaging
  path) — set 0 for any new box.

## Seed the first charged SCF (REQUIRED when qm_0's QM is skipped)

With the qm_0-skip, nothing populates `solute/solute.save` before qm_1, and
the charged branch uses `startingpot='file'` → plant a seed BEFORE submit:
copy the 3-file set (`charge-density.dat`, `data-file-schema.xml`,
`paw.txt`) from a **converged** neighbor-charge save into
`solute/solute.save/`. Rules (also in 05): source's last pw run must say
JOB DONE (an unconverged/killed save is disqualifying — check!);
neighbor charge ladder preferred; cross-cation is fine (the QM slab has no
cations); NEVER seed a heavily charged state from a neutral density;
never copy wfc*.dat.

## Pre-submit audit (run it, don't trust memory)

One line per system: TOTCHG | data.file "ncat-nOH=net" & slab Q |
base.in.lammps top charge ×288 | INI/FIN/initialqm/skipequil | seed 3/3
and seed `nelec` = nelec_neutral − TOTCHG ([05 §6.2](05-monitoring-and-restart.md)) |
qm_0 present | zero ces2.*.restart & data.md | k-mesh in `base.pw.in` =
campaign standard | force-field fingerprint (e.g. `grep "bond_coeff 2"
base.in.lammps`) = campaign snapshot | `QEPW_FLAGS=""` | submit: right
scheduler header for THIS machine, job name, partition/queue, any site node
exclusions (see your machine profile, 07).

`tools/ces2_preflight.sh` automates everything in this line except the seed
charge and the force-field fingerprint; run it anyway, then check those two
by hand.

## Post-start LIVE verification (within the first hours — mandatory)

1. Regenerated `in.lammps`: `grep "set group top charge"` → new value?
2. `data.md` header atom count == data.file's?
3. Live trajectory frame: count cations/OH⁻ (must match the new box).
4. When qm_1 starts: `pw.in` has the right `tot_charge`; `pw.out` says
   "The initial density is read from file"; SCF converging in tens of
   iterations (atomic-fallback = hundreds = your seed didn't load).

If any check fails: scancel immediately, fix sources of truth, delete
derived artifacts, resubmit. Do not wait for qm_2 to "self-heal" — a wrong
MM box never heals.
