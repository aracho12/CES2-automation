# 05 — Monitoring & Restart

How to tell what a running (or dead) chain is doing, how to resume it, and how
to seed the charged SCF so it converges. Every rule here was paid for with
walltime; the incident behind each one is in [09 — Failure catalog](09-failure-catalog.md).

Protocols are written as **signal → check → decision → action → verify** so
they can be followed mechanically (by a person or by an automation agent — see
[10 — Automation safety](10-automation-safety.md)).

---

## 1. Reading `qmmm.out`

The driver prints a fixed set of markers. A healthy step `N` (`QMTYPE="scf"`,
`N > 0`) looks like this:

```
######### Starting N QMMM step #########
   (pw.x ortho  → pw.out)        no marker while running; watch pw.out
   (pw.x non-ortho → pw.nonortho.out)
### QM calculation done
   (pp.x + cube post-processing → root solute.*.cube, c_rho_*.cube)
### Running LAMMPS(emxext) for equilibration N QMMM iterations
### Running LAMMPS(emdext) for averaging solvent charge density N QMMM iterations
   (archive → qm_N/, mm_N/)
######### Starting N+1 QMMM step #########
```

| Marker | Meaning |
|---|---|
| `######### Starting N QMMM step #########` | outer-loop iteration `N` began |
| `### QM calculation done` | `pw.x` printed `JOB DONE` (checked on the **ortho** `pw.out` only — the non-ortho leg is never checked, see §6.4) |
| `### QM calculation aborted, check output file` | no `JOB DONE` in `pw.out` → read `pw.out` tail and the scheduler `.e` file |
| `### Running LAMMPS(emxext) …` | equilibration MD launched (`in.lammps.equil`, log `lammps.equil.out`) |
| `### Running LAMMPS(emdext) …` | averaging MD launched (`in.lammps`, log `lammps.average.out`, trajectory `ces2.emd.lammpstrj`) |
| `### ERROR: LAMMPS … failed with exit code` / `produced no restart files` / `No restart file found` / `Could not parse timestep` / `restart-to-data conversion failed` | MD-phase failure; the driver stops. See [09](09-failure-catalog.md) |
| `Using data.file directly` (no restart found) | **only acceptable at the very first MD of a fresh build.** Anywhere else it means the MD silently rewound to the build structure (§5.3) |

`qmmm.out` is the driver's stdout as redirected by `submit_ces2.sh`; MPI-level
crashes and walltime kills land in the scheduler's `.o`/`.e` files, not here.
`initialqm=1` suppresses the whole QM block of the first step, so a resumed
job legitimately goes straight from `Starting N` to `Running LAMMPS(emxext)`.

## 2. Where progress lives on disk

### 2.1 Phase detection

| Phase of step `N` | What exists / is being written |
|---|---|
| QM ortho running | root `pw.in` regenerated (mtime > job start), `pw.out` growing; `qm_N/` absent |
| QM non-ortho running | `pw.nonortho.in` / `pw.nonortho.out` growing |
| QM post-processing | `pp.*.out`, root `solute.pot.cube`, `solute.ind.cube`, `c_rho_*.cube` being rewritten |
| MD equilibration | `qm_N/` archived; `lammps.equil.out` growing; `ces2.<step>.restart` files appearing (equil numbering restarts from the equilibration's own step count) |
| MD averaging | `lammps.average.out` growing; `ces2.emd.lammpstrj` growing; restarts continue with the averaging run's step numbers |
| step done | `mm_N/` populated (`lammps.*.out`, trajectory, `MOBILE_final.cube`, newest restart) |

### 2.2 Progress numbers

| Phase | Progress | Where |
|---|---|---|
| QM | SCF iteration = number of `estimated scf accuracy` lines in `pw.out`; compare the latest value with `conv_thr` from `pw.in`; done when `convergence has been achieved in N iterations` → `JOB DONE` | `pw.out` / `pw.nonortho.out` |
| MD | current `Step` line of the thermo output vs `MDEQUIL` (equilibration) or `MDAVERAGE` (averaging); `Loop time` / `Total wall time` when finished | `lammps.equil.out` / `lammps.average.out` |
| elapsed (QM) | while running: `mtime(pw.out) − mtime(pw.in)`; when finished: the `electrons : … WALL` line, or `Program PWSCF … starts on` minus `This run was terminated on:` | `pw.out` |

> ⚠ **Do not use the `PWSCF : … WALL` summary line as elapsed time.** In this
> patched QE 6.3 build it is inflated 2–4× (e.g. 55 h printed for a 13.5 h
> run). Use the start/terminate timestamps or the `electrons` clock.

### 2.3 The stale-file rule (mandatory for every monitor)

The driver **reuses the root working files** (`pw.in`, `pw.out`,
`pw.nonortho.out`, `lammps.equil.out`, `lammps.average.out`, root cubes): each
step overwrites them, and after a job ends they keep the last state. **Right
after a new job starts, every root file is still the previous job's.** Two real
false alarms came from this: a k-mesh check that read yesterday's `pw.out`, and
a hang monitor that read a month-old `lammps.equil.out`.

Rules:
1. Trust a root file only if `mtime(file) > job start time`
   (`squeue -o %V` / `scontrol show job <id> | grep StartTime`, or PBS `qstat -f`).
2. For QM output use the regeneration order: the driver writes `pw.in` first, so
   this step's output satisfies `mtime(pw.out) > mtime(pw.in) ≥ mtime(qm_{N-1}/)`.
3. Final judgements come from the archives (`qm_N/pw.out`, `mm_N/lammps.*.out`)
   — archives are immutable.

### 2.4 Hang vs. slow

| Signal | Verdict | Check |
|---|---|---|
| `lammps.*.out` ends at `Finding SHAKE clusters ...`, **no thermo step line**, file mtime frozen for > 15–20 min, job still `R` | **LAMMPS launch hang** (node/partition hardware or MPI fabric; it never recovers) | `stat -c %y lammps.*.out` vs `date`; compare node with the machine profile's exclusion list |
| `pw.out` mtime frozen for hours during iteration 1 of a large charged SCF | usually **not** a hang — the first iteration of a ~10⁶-point grid can take 3 h+ | `sstat -j <id>.batch --format=AveCPU` (SLURM) shows full CPU use; `pw.out` grows at the next iteration |
| `pw.out` writes nothing for > 1 day and QE scratch (`solute/solute.save/`) has no new files | pw.x hung (rare; MPI fabric) | kill, resubmit from the last archive |
| SCF accuracy flat at a few 10⁻⁴ Ry for 40+ iterations | **noise-floor wall**, not slow convergence — see §6.3 | plot `estimated scf accuracy` vs iteration |

Hang handling: `scancel`/`qdel`, record node + phase in the machine profile
([07 §1](07-server-profiles-and-migration.md)), add the node (or its whole
partition — in one deployment five sibling nodes failed identically) to the
submit-script exclusion list, and resume with the state table in §4. A hang at
launch costs nothing but queue time: the QM result of that step is already in
`qm_N/`, so resume with `initialqm=1`.

## 3. Resuming a killed run: `tools/ces2_resubmit.sh`

Run it **inside `export/`** (the directory holding
`qmmm_dftces2_charging_pts.sh`; a system folder that only wraps `export/` will
fail with "script not found").

What it does:
1. Inspects `qm_N/`, `mm_N/`, `ces2.*.restart` and `pw.out` to find the last
   completed phase.
2. Patches **four** driver variables, leaving
   `qmmm_dftces2_charging_pts.sh.bak.<timestamp>` and
   `ces2_resubmit_<timestamp>.log`: `QMMMINISTEP`, `initialqm`, `skipequil`,
   `LAMMPSRESTART` (meanings: [03 §1](03-qmmm-script-reference.md), [08](08-new-charge-state-checklist.md#initialqm-semantics-verified-in-driver-source)).
3. Submits. On **PBS** it also self-chains (`qsub … depend=afterany:$PBS_JOBID`).
   On **SLURM** the chaining is a no-op: patch with the tool, then
   `sbatch submit_ces2.sh` yourself (or chain with `--dependency=afterok:<id>`).
4. §3.5 of the tool checks whether the driver actually uses a `file` SCF
   restart and, if so, requires `solute/solute.save/` to exist.

When **not** to trust it blindly:
- a hand-edited driver (it patches by variable name; anything else you changed
  stays);
- a kill **during QM post-processing**: root `solute.pot.cube`/`solute.ind.cube`
  may be half-written or stale, and `initialqm=1` would feed them to the next
  MD. Rerun that step's QM instead (`initialqm=0`; it reconverges from the
  save in a few iterations). `ces2_preflight.sh` warns about this case;
- restarts copied with `rsync`/`cp` without `-p`: mtimes are scrambled and the
  "newest restart" fallback can pick the wrong file (§5.2);
- after a batch of PBS resubmits, check for **duplicate** held children
  (`qstat -u $USER`, two jobs with the same `afterany` parent) and delete the
  later one — two resumes racing in one `export/` corrupt it.

Always run `tools/ces2_preflight.sh` after the patch and before submitting.

## 4. Which state am I in? (resume table)

| Last thing that completed | `QMMMINISTEP` | `initialqm` | `skipequil` | `LAMMPSRESTART` |
|---|---|---|---|---|
| `qm_N` archived, `mm_N` empty/absent (died in equilibration) | N | 1 | 0 | newest restart of `mm_{N-1}` (the equilibration will restart from it) |
| died in averaging of step N (equil restarts exist, no `mm_N`) | N | 1 | 1 | newest **equilibration** restart of step N |
| `mm_N` archived (died in QM of step N+1) | N+1 | 0 | 0 | newest restart (the one in `mm_N/`) |
| QM of step N+1 converged (`qm_{N+1}/` exists) but post-processing/MD never ran | N+1 | 1 | 0 | newest restart — only if root `solute.pot/ind.cube` are from that QM; otherwise `initialqm=0` |

`skipequil` resets itself to 0 after the first MD; `initialqm` resets to 0
after the first loop pass. Both are "first step of this invocation only".

## 5. Extending a finished chain (adding QM/MM steps)

A chain that ran to `QMMMFINSTEP` can be extended, but a **new job re-reads the
driver header from the top**, so the resume variables do not carry over.

### 5.1 The four lines

```
QMMMINISTEP=<N+1>                 # first new step
QMMMFINSTEP=<target>              # last new step (inclusive)
LAMMPSRESTART="<newest existing ces2.*.restart>"   # §5.3
initialqm=0                       # a NEW step must run its QM
```
plus `skipequil=0` (and `skiportho=0` if the driver has it). Then
`tools/ces2_preflight.sh` → submit.

### 5.2 Which restart

`ls -1rt *.restart | tail -1` is right when the files were produced in place
(equilibration numbering → averaging numbering follows mtime). After a copy
or `rsync`, sort by **step number**, pick the highest, and `touch` it so the
driver's mtime-based fallback agrees; move older restarts out of `export/`.
Preflight fails when the highest-numbered and newest-mtime restarts differ.

### 5.3 The `data.file` rewind trap

The driver header sets `LAMMPSRESTART` to a name that often does not exist in
a given export (e.g. `ini.restart`). Inside one job the loop re-derives the
newest restart at the end of each step, so a chain *within* a job is fine. A
**new** job uses the header value: if that file is missing the driver prints
`Using data.file directly` and the MD **silently restarts from the build
structure** — the liquid loses all its equilibration and the step is wasted.
Preflight fails when `INISTEP > 0` and no `ces2.*.restart` exists; it cannot
know which one you meant, so set line 34 explicitly.

### 5.4 `initialqm` when extending

`initialqm=1` means "skip the QM of the first step of this invocation". With
`INISTEP == FINSTEP` it makes the job run **no QM at all** (preflight: FAIL);
with `INISTEP > 0` it reuses whatever root `solute.pot/ind.cube` are lying
around (preflight: WARN). For an extension always `initialqm=0`.

### 5.5 Never patch a live driver

bash reads the script lazily; editing `qmmm_dftces2_charging_pts.sh` while the
job runs corrupts execution. Wait for the job to end (or kill it), then patch
→ preflight → submit.

### 5.6 Driver variants

Not every deployment's driver has the same end-of-loop logic. One variant
exits **before** the MD leg when `qmmmstep -ge QMMMFINSTEP`, so `FIN=N` stops
after `qm_N` (no `mm_N`); the other runs `qm_N` **and** `mm_N`. Check your
driver (`grep -n 'QMMMFINSTEP' qmmm_dftces2_charging_pts.sh`) before deciding
`FIN`, and record the answer in the machine profile ([07 §1](07-server-profiles-and-migration.md)).

## 6. Seeding charged SCF from a converged charge density

**Always run charged systems (`TOTCHG ≠ 0`) with a charge-density seed.** An
`atomic`-superposition start on a charged slab oscillates or diverges for
hundreds of SCF iterations — observed repeatedly in production (a −20 e
electrode: ~1 iteration/hour, one system diverging to ~200 Ry, one hung; a
−12 e electrode: 170 iterations / 75 h plateaued at 1e-3 vs thr 1e-4). A
seeded SCF converges in **~10–20 iterations**.

**New builds:** set `ces2_script.qm_startingpot: "file"` in the config (see
`config_example.yaml`). **Existing scripts:** patch the `TOTCHG` conditional in
`qmmm_dftces2_charging_pts.sh` (the `STARTINGWFC`/`STARTINGPOT` line, ~line 54)
so the charged branch reads `STARTINGPOT="file"` (keep `STARTINGWFC="atomic"`).
For a hard SCF also lower `mixing_beta` in `base.pw.in` (0.3 → 0.1; 0.05 for
the hardest cases, with `mixing_ndim 16`) and keep `mixing_mode='local-TF'`.

### 6.1 Required files

Place all three in `<export>/solute/solute.save/`:

| file | size | if missing |
|---|---|---|
| `charge-density.dat` | ~180M | `pw_readschemafile: xml data file not found` |
| `data-file-schema.xml` | ~150–220K | same (holds nelec / grid metadata) |
| `paw.txt` | ~420K | `read_scf: Reading PAW becsum` crash (PAW setups → mandatory). An **absent** file gives the same message as a corrupt one (QE opens an empty file and hits EOF). Its byte size differs between pw.x builds (3 values/line vs one long record) — a provenance marker, not a validity test. |

`wfc*.dat` (~19G) are **not needed** — `STARTINGWFC` stays `atomic`; reading
another charge state's wavefunctions risks Cholesky errors.

### 6.2 Seed-source rules, in order of quality

1. The density must come from a **converged** run (`JOB DONE` in the source
   `pw.out`) — never from an oscillating, diverged or killed one.
2. **Same charge, same slab** is best — including a different cation: the QM
   slab contains no electrolyte, so a converged density from any cation's run
   of the same charge state seeds another cation's (verified repeatedly).
3. **Same cation, neighbouring charge (±1–2 e)** is next. QE renormalizes
   `nelec` on read.
4. A **charge ladder further away (±4 e)** often works for an easy SCF but is
   the usual cause of the noise-floor wall (§6.3) on a hard one.
5. **Never seed a charged run from a neutral (qm_0 / PZC) density** — it
   converges badly.
6. Same QE version on both ends (schema compatibility).

**The seed's charge state is checked by preflight**: it compares
`<tot_charge>` in the seed's `data-file-schema.xml` with this run's `TOTCHG`
(PASS when equal; WARN for a neighbouring charge within `seed_gap_warn_e`,
default 2 e total; WARN "far seed" beyond that; FAIL for a neutral density
under a charged run) and reports `nelec` against
`nelec(neutral slab) − TOTCHG` when `qm_0/pw.out` is present. By hand:
`grep 'tot_charge\|nelec' <outdir>/<prefix>.save/data-file-schema.xml`.

### 6.3 The noise-floor wall (seed too far away)

Signal: `estimated scf accuracy` sits flat or wanders at **3·10⁻⁴ – 7·10⁻⁴ Ry
for 40+ iterations** (tens of minutes per iteration on one node for a
~10⁶-point grid) while the total energy jitters by ~1 mRy. This is not slow
convergence; it never gets below the floor. Remedy, in the order that has
worked:

1. **Swap the seed** for a closer one (§6.2 rule 2, then 3). This fixed every
   occurrence so far; mixing changes alone did not.
2. `mixing_beta 0.05`, `mixing_ndim 16` together with the new seed.
3. `mixing_mode='local-TF'` (standard for long-wavelength sloshing in a slab +
   vacuum cell) — usually already on in the templates.

Park the stalled save (`mv solute/solute.save solute/solute.save.stalled_<tag>`)
before planting the new seed so nothing is lost.

### 6.4 Non-ortho leg stuck at the floor: `solute.EXIT`

The driver checks `JOB DONE` only on the **ortho** `pw.out`; the non-ortho
density only feeds the next MD's field, so 10⁻⁴ vs a few 10⁻⁴ does not matter
there. If the non-ortho SCF is grinding at its floor: `touch <export>/solute.EXIT`
(file name = `<prefix>.EXIT`). QE 6.3 checks for it at the top of every
iteration, writes the xml + density, and exits cleanly; the driver continues
with `pp.x` → MD. **Never** do this on the ortho leg (work function, energies),
and **delete the file at once if the SCF converges on its own first** — a
leftover `solute.EXIT` kills the next `pw.x`.

### 6.5 `.save` is volatile — back it up

The driver copies only cubes into `qm_N/`; the next `pw.x` run overwrites
`solute/solute.save/`. Snapshot a converged density with
`tools/ces2_backup_save.sh backup <export> <label>`
(→ `<export>/_save_backup/<label>/`; `restore` to bring it back; refused while
a live job is writing the directory). Do this **before** deleting `solute/`
to free disk (see [04 — What can I delete](04-output-files.md#what-can-i-delete)).

### 6.6 Verify after (re)submit

`pw.out` must print `The initial density is read from file` and converge
within a few dozen iterations. `Initial potential from superposition of free
atoms` means the seed silently fell back to atomic — stop and fix before
burning walltime. (With `initialqm=1` the first step has no QM, so look at the
*next* step's `pw.out`.)

## 7. Judging convergence of the outer loop

The loop has no automatic criterion; adopt a step when:

- the work function / `V_RHE` changes by **< 0.05 V between successive QM
  steps** (`tools/wf_convergence.py` tabulates this per system);
- the **first QM step after any chain (re)start is excluded** from averages
  (its MM field came from a restart boundary);
- a series that is still drifting monotonically is **extended, not averaged**
  — a mean over a drifting series is biased.

Every QM step of one dataset must share the same k-mesh, pseudopotentials,
QE flags and force field. Declare them once in the dataset's `campaign.yaml`
(template `campaign_example.yaml`) so `ces2_preflight.sh` enforces them on
every submit; `pw.out`'s `number of k points` line is the quick after-the-fact
fingerprint for the mesh (see [04](04-output-files.md#qm-stage)).

## 8. Common failures

The full signature → cause → fix catalog is [09](09-failure-catalog.md). The
ones you will meet first:

- walltime kill → §3/§4 (the standard case);
- `pw_readschemafile` / `Reading PAW becsum` → §6.1 (seed files);
- SCF not converging on a charged system → §6 (atomic start, far seed);
- LAMMPS stuck at `Finding SHAKE clusters` → §2.4 (node hang);
- `Using data.file directly` after a restart → §5.3;
- preflight FAIL on binaries/pseudo_dir/NP → a foreign machine profile leaked
  into the export ([07 §4](07-server-profiles-and-migration.md), [08](08-new-charge-state-checklist.md)).
