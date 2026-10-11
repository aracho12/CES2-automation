# 09 — Failure Catalog (signature → cause → fix → verify)

Every entry is something that actually happened in production. Look up the
**signature** you see (log line, file state, number), apply the fix, and run
the verification before calling it solved. Add a row for every new incident
([10 §6](10-automation-safety.md)).

Conventions: `root` = `export/` working files; `archive` = `qm_N/`, `mm_N/`.
Resume variables are explained in [05 §4](05-monitoring-and-restart.md) and
[08](08-new-charge-state-checklist.md).

---

## A. QM (pw.x) failures

### A1. `Error in routine pw_readschemafile (1): xml data file not found`
- **Where:** `pw.out` tail; `qmmm.out` says `### QM calculation aborted`.
- **Cause:** `startingpot='file'` (or `startingwfc='file'`) but
  `solute/solute.save/` is missing or lacks `data-file-schema.xml` /
  `charge-density.dat`. Classic after a migration (the `.save` was not copied),
  after `solute/` was deleted to save disk, or on a **neutral** run whose driver
  uses `file`/`file` while the resume skipped the step that would have written
  the save.
- **Fix:** provide the 3-file seed ([05 §6.1](05-monitoring-and-restart.md))
  from a converged run of the same charge; or, for neutral runs only, set the
  neutral branch of the driver's `STARTINGWFC/STARTINGPOT` line to `atomic`
  (slower first SCF, no file dependency). Note that a rebuilt export loses
  such a driver patch — re-check the line after every rebuild.
- **Verify:** `pw.out` prints `The initial density is read from file` (seed) or
  `Initial potential from superposition of free atoms` (intended atomic start).

### A2. `read_scf: Reading PAW becsum` (crash right after the density read)
- **Cause:** `paw.txt` missing from `solute/solute.save/`. QE opens an empty
  file and hits EOF, which prints the same message as a parse error. The file's
  size differs between pw.x builds (424 kB vs 453 kB for the same content) —
  that is formatting, not corruption; cross-machine `paw.txt` reads fine.
- **Fix:** copy `paw.txt` from the same source as the density. All three seed
  files travel together.
- **Verify:** as A1.

### A3. Charged SCF oscillates or diverges from the start (hundreds of iterations, accuracy ≫ 1 Ry)
- **Signature:** `estimated scf accuracy` of 1–200 Ry after 20+ iterations,
  ~1 iteration/hour, `pw.out` still growing after days.
- **Cause:** atomic-superposition start on a charged slab (`STARTINGPOT="atomic"`
  in the charged branch) — the driver's original default.
- **Fix:** `STARTINGPOT="file"` + seed ([05 §6](05-monitoring-and-restart.md)),
  `mixing_beta` 0.3 → 0.1. Kill the running job first; its outdir is useless.
- **Verify:** seed-read line + convergence in ~10–20 iterations.

### A4. Noise-floor wall: accuracy flat at 3–7·10⁻⁴ Ry for 40+ iterations
- **Signature:** `estimated scf accuracy` wanders between ~3·10⁻⁴ and 7·10⁻⁴
  from iteration ~25 onward, total energy jitter ~1 mRy; `conv_thr` 10⁻⁴ never
  reached. Seed **was** read (`initial density is read from file`).
- **Cause:** seed density from a charge state too far away (a 4 e gap on the
  charge ladder in every case seen; the seed xml's `tot_charge`/`nelec`
  belong to the other state). Mixing tweaks alone did not fix it in any case
  so far. Preflight now warns on such a seed.
- **Fix:** swap the seed for the same charge state (any cation) or ±1–2 e
  same cation; add `mixing_beta 0.05`, `mixing_ndim 16`. Park the stalled
  save first. ([05 §6.3](05-monitoring-and-restart.md))
- **Verify:** `grep nelec solute/solute.save/data-file-schema.xml` equals
  `nelec_neutral − TOTCHG`; convergence in < 30 iterations.

### A5. Non-ortho leg stalls after a fresh chain restart
- **Signature:** ortho `pw.out` has `JOB DONE`; `pw.nonortho.out` oscillates
  for hundreds of iterations.
- **Cause:** inside a running chain the non-ortho SCF seeds from the previous
  non-ortho state; a restarted chain hands it only the ortho-embedding
  density ([07 §5](07-server-profiles-and-migration.md)).
- **Fix:** lower `mixing_beta`; or, since the non-ortho density only feeds the
  next MD's field, `touch <export>/solute.EXIT` to stop at the current
  accuracy ([05 §6.4](05-monitoring-and-restart.md)). Never on the ortho leg.
- **Verify:** `pp.x` runs, `qmmm.out` proceeds to `Running LAMMPS(emxext)`;
  `solute.EXIT` is gone.

### A6. `Initial potential from superposition of free atoms` where a seed was intended
- **Cause:** `STARTINGPOT` still `atomic` in the branch that ran (driver line
  not patched, or `qm_startingpot` not set in the config), or the save was
  overwritten by an unconverged run.
- **Fix:** stop the job (it will crawl, A3); fix the driver line; replant the
  seed from `_save_backup/`.

### A7. `PWSCF : … WALL` claims more hours than the job's walltime
- **Cause:** timer bug in this patched QE 6.3 build; the summary is inflated
  2–4×. Not a hang, not I/O overhead.
- **Fix:** nothing to fix in the run. For timing use `Program PWSCF … starts
  on` → `This run was terminated on:`, or the `electrons … WALL` line.

### A8. `K-points division: npool = 4` with inconsistent results / `QEPW_FLAGS` set
- **Cause:** `-npool/-ntg/-ndiag` with the CES2-patched pw.x gave unreliable
  results (a whole PZC dataset had to be rerun flag-off).
- **Fix:** `QEPW_FLAGS=""` (leave `ces2_script.npool/ntg/ndiag` unset).
  Preflight warns when flags are present.
- **Verify:** `pw.out` header shows `npool = 1` (or no division line).

### A9. Different `number of k points` between steps of one chain
- **Signature:** `grep 'number of k points' qm_*/pw.out` shows two values
  across the steps of one chain (the count itself depends on mesh and cell
  symmetry; what matters is that it changes).
- **Cause:** `base.pw.in` edited or rebuilt with a different mesh mid-chain;
  a config copied from another campaign.
- **Impact:** work-function offsets of tens of mV; adsorption statistics in
  the MM shift much more (the local surface potential changes). Mixed-mesh
  chains must not be pooled.
- **Fix:** standardize the mesh in the config and rebuild/extend the chain
  from the first mismatched step; keep the old steps in a sub-folder with a
  README. Preflight warns on deviation from the campaign standard.

### A10. nspin = 2 and the DFT-CES Pauli energy
- **Signature:** a spin-polarized run prints a CES repulsion (Pauli) energy
  far from the nspin=1 value at the same geometry/field, and the total-energy
  difference is implausible (hundreds of meV).
- **Cause:** the patched `pw.x` evaluates the ρ̂·repA term for the spin-up
  density only (spin-down contribution dropped) when `nspin=2`.
- **Fix:** recompute the Pauli term from both spin densities and correct
  `E_tot` by the difference. Use nspin=1 unless the moment matters.

### A11. MPI collective crash (`MPID_nem_tmi_pending_ssend_dequeue`, `MPI_Gatherv`) after a healthy SCF
- **Cause:** interconnect/MPI transient, not the input.
- **Fix:** resubmit as-is (resume state per [05 §4](05-monitoring-and-restart.md)).

---

## B. MD (LAMMPS) failures

### B1. Launch hang at `Finding SHAKE clusters ...`
- **Signature:** last line of `lammps.equil.out`/`lammps.average.out` is
  `Finding SHAKE clusters ...`, **no `Step` line**, file mtime frozen > 20
  min, job state still running. Can occur at any LAMMPS launch of the job
  (equilibration or averaging), probabilistically, on specific nodes; `pw.x`
  on the same nodes is unaffected.
- **Cause:** node/partition hardware or MPI fabric (one deployment: a whole
  5-node partition).
- **Fix:** kill; add the node **set** (not just the one node — two jobs
  excluding one bad node each landed on each other's) to the submit script
  exclusion (`#SBATCH --exclude=`, or PBS equivalent); resume. The QM of the
  step is already archived, so `initialqm=1` (+ `skipequil=1` if the
  equilibration had finished). QM-only jobs may still use those nodes.
- **Verify:** within minutes of the new launch `lammps.*.out` shows thermo
  `Step` lines; mtime advancing.
- **Do not confuse with:** a slow first SCF iteration (A7-style patience);
  check `sstat`/`top` on the node before declaring a hang.

### B2. DFT-CES grid misparse (wrong Pauli energy, ×10³–10⁴)
- **Signature:** `lammps.*.out` contains `Grid file will be parsed:
  solute.ind.cube` **instead of** `DFT-CES: solute.ind.cube with jellium
  potential has been parsed`; the repulsion/Pauli energy per box is ~10⁵–10⁶
  kcal/mol instead of ~10²; forces nonsense.
- **Cause:** the `grid` command's `netcharge_flag` is never initialized in the
  Grid constructor; whether `solute.ind.cube` is treated as the induction
  potential depends on freed memory. Triggered reliably by launching through
  `xargs bash -c` with exported functions (large environment); direct
  `mpirun -np N lmp_mpi` launches have been fine (campaign audit: 623/623 OK).
- **Fix:** launch LAMMPS directly; wrap launches with a signature check that
  retries on the bad line; discard any MM output produced under the bad line.
- **Verify:** the `jellium potential has been parsed` line is present in
  every MM log you analyze.

### B3. `Using data.file directly` / `read_data data.file` on a resumed or extended run
- **Cause:** `LAMMPSRESTART` in the driver header names a file that does not
  exist (`ini.restart`, a restart that was moved, a stale name like
  `ces2.3000000.restart` after the chain advanced) → silent fallback to the
  build structure; the liquid loses all equilibration ([05 §5.3](05-monitoring-and-restart.md)).
- **Fix:** kill; set `LAMMPSRESTART` to the real newest restart; discard the
  MD output of that step; resubmit.
- **Verify:** `lammps.equil.out` shows `read_restart`/restart-to-data
  conversion of the intended file; the first thermo `Step` equals the
  restart's step; ion count in the first trajectory frame matches the box.

### B4. Wrong restart picked after a copy (mtime scrambled)
- **Signature:** preflight `restart mtime scrambled: highest step X but
  newest mtime Y`; or an MD that starts from an older step than expected.
- **Cause:** `cp`/`rsync` without preserving times; the driver's fallback is
  mtime-based.
- **Fix:** keep only the intended restart in `export/` (move the rest to a
  sub-folder), `touch` it, name it explicitly in `LAMMPSRESTART`.

### B5. `### ERROR: LAMMPS equilibration failed` / `produced no restart files`
- **Cause:** MD died early: lost atoms, non-numeric pressure, overlapping
  molecules after a bad packing, or a data file inconsistent with
  `base.in.lammps` (types, bonds).
- **Fix:** read the `lammps.equil.out` tail and the scheduler `.e` file. For
  packing problems rebuild with `md_relax` enabled ([README](../README.md));
  for type/bond mismatches rebuild — never hand-edit `data.file` types.

### B6. The MD runs the wrong physical system (no error at all)
- **Signature:** work function offset by hundreds of mV to volts with no
  other symptom; ion count in the trajectory ≠ build; `set group top charge`
  in the regenerated `in.lammps` ≠ expected.
- **Cause:** the three-consistency rule broken (`TOTCHG` vs ion imbalance vs
  electrode point charges), typically after cloning a neighbouring charge
  state and editing derived files, or a surviving restart resurrecting the
  old box ([08](08-new-charge-state-checklist.md)).
- **Fix:** kill; fix the sources of truth; delete every derived artifact;
  preflight; resubmit; live-verify. Nothing downstream "heals" it.

---

## C. Chain / scheduler / state failures

### C1. Walltime kill mid-step
- **Signature:** scheduler reports timeout; `qmmm.out` ends without a marker;
  root files partially written.
- **Fix:** `tools/ces2_resubmit.sh` in `export/` → preflight → submit
  ([05 §3](05-monitoring-and-restart.md)). On SLURM submit manually.

### C2. Job finishes immediately, no new QM (`initialqm=1` with `INISTEP == FINSTEP`)
- **Cause:** `initialqm=1` skips the QM of the first step; if that is also the
  last step nothing is computed. Preflight: FAIL.
- **Fix:** `initialqm=0` for a new step; `FINSTEP > INISTEP` for a resume
  that should continue.

### C3. `initialqm=1` at step N > 0 feeds stale coupling cubes to the MD
- **Cause:** a kill during QM post-processing left old root
  `solute.pot.cube`/`solute.ind.cube`; the skip-QM resume reuses them as-is.
  Preflight: WARN.
- **Fix:** rerun the QM (`initialqm=0`; a few iterations from the save) or
  verify the cubes' mtimes postdate the archived `qm_N/pw.out`.

### C4. Duplicate chained jobs on PBS
- **Signature:** two held jobs with the same `depend=afterany:<id>` after a
  batch resubmit.
- **Fix:** `qdel` the later one before the parent finishes.

### C5. Jobs appear that nobody submitted
- **Cause:** a background auto-submit loop from an old shell session
  (`while true; … sbatch …; sleep 300`) still alive weeks later, submitting
  with whatever driver was on disk.
- **Fix:** `pgrep -af 'sbatch|qsub'` across all sessions, kill the loop and
  its parent shell; cancel the rogue jobs; audit their exports for driver
  changes made since. Rule: feeders are one-shot or carry a kill switch
  ([10 §4](10-automation-safety.md)).

### C6. False alarm from a stale root file
- **Signature:** a check reads `pw.out`/`lammps.*.out` right after a new job
  started and sees yesterday's content (wrong k-mesh, "frozen" MD).
- **Fix:** compare mtime with the job start before reading
  ([05 §2.3](05-monitoring-and-restart.md)).

### C7. Foreign machine profile in a copied export
- **Signature:** preflight FAILs on binaries / `pseudo_dir` / NP vs
  allocation / scheduler header; or an instant crash at the first launch
  (`invalid partition`, binary not found, pseudopotential not found).
- **Cause:** a config or export copied from another cluster or era carries its
  paths, core count and flags into the driver and `base.pw.in`.
- **Fix:** patch the whole list in [07 §4](07-server-profiles-and-migration.md);
  then **audit every sibling file from the same source** — one foreign file
  means the rest are suspect.

### C8. Force-field drift between builds of one campaign
- **Signature:** `grep "bond_coeff 2" base.in.lammps` (or any pinned
  parameter) differs between two exports of the same campaign; a rebuilt
  system's first step disagrees with its siblings by more than step noise.
- **Cause:** the shared `species_db/` moved on (e.g. the hydroxide model was
  updated) while the campaign was built with the earlier values.
- **Fix:** pin the force field per campaign: point `species_db:` in every
  config at a frozen copy (record its git hash), and check the fingerprint on
  every hand-back/rebuild. Keep mixed-FF data out of one dataset.

### C9. Patch script ran in the wrong directory
- **Signature:** stray `qm_0/`, `solute/` or deleted files in the project
  root; the intended export untouched.
- **Cause:** `cd <export>` failed (not built yet) and the following relative
  `rm`/`cp`/`mkdir` ran where the script was started.
- **Fix:** restore from backups; write scripts with `cd … || exit 1` and
  absolute paths ([10 §5](10-automation-safety.md)).

### C10. Monitor/waiter never finishes
- **Cause:** `until ! pgrep -f <script>` matches the waiter's own command
  line.
- **Fix:** wait on the PID (`kill -0`) or on an output artifact.
