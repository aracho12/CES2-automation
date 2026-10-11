# 10 — Automation Safety

Rules for anything that acts on a CES2 run without a human watching each
command: shell loops, cron jobs, resubmit scripts, and LLM agents driving the
tools. Each rule exists because the opposite happened in production (dates in
parentheses; details in [09](09-failure-catalog.md)).

The design principle behind all of them: **judgement may be automated;
execution goes through the scripted tools; every state change is verified
from the live run, not from the intent.**

## 1. Gates

1. **Preflight before every submit.** `tools/ces2_preflight.sh <export>`;
   exit 2 = do not submit, no exceptions. It catches foreign machine profiles,
   charge inconsistency, missing seeds, the `initialqm`/`FINSTEP` traps and
   scrambled restarts (2026-08-13: 8 FAILs on a contaminated export that had
   looked fine to the eye).
2. **Live verification after every submit**, within the first hours
   ([08 — Post-start live verification](08-new-charge-state-checklist.md#post-start-live-verification-within-the-first-hours--mandatory)):
   regenerated `in.lammps` electrode charge, `data.md` atom count, ion count
   in the live trajectory, `tot_charge` in `pw.in`, the seed-read line in
   `pw.out`. Report success only after these pass.
3. **Never claim that a wrong state will "self-heal" downstream** without
   tracing the data flow. A wrong MM box never heals; a stale coupling cube
   propagates into the next MD (2026-08-12).

## 2. Sources of truth and derived files

4. Edit **only** runtime sources (`base.*`, `data.file`, the driver, the
   submit script); then **delete every derived artifact** (`ces2.*.restart`,
   `data.md`, `in.lammps*`, root cubes, trajectories) when cloning a system,
   so nothing stale leaks (2026-08-12: a surviving restart resurrected the
   previous charge state's box). "I edited the file" ≠ "the run will use it"
   ([08](08-new-charge-state-checklist.md)).
5. **Back up before overwriting.** Converged `solute.save` →
   `tools/ces2_backup_save.sh`; driver → `.bak.<timestamp>` (the resubmit tool
   does this); anything else → a dated copy.
6. **Never edit a live driver in place** — bash reads the script lazily. Kill
   or wait, then patch → preflight → submit ([05 §5.5](05-monitoring-and-restart.md)).
7. **Contamination quarantine:** when one file copied from a source turns out
   to be foreign (a PBS header on a SLURM machine, a `pseudo_dir` from another
   cluster), audit **every** sibling from that source before trusting the rest
   (2026-08-13: a "fresh" build carried another server's binaries, NP and
   flags in its driver). Symptom-level fixes leave the mines armed.

## 3. Reading state

8. **Mtime against job start, always** ([05 §2.3](05-monitoring-and-restart.md)).
   Root `pw.out`/`lammps.*.out` are the previous job's until overwritten.
   Two false alarms on 2026-08-21 came from ignoring this.
9. **Hang = frozen mtime + the launch signature**, not "it has been a while"
   ([05 §2.4](05-monitoring-and-restart.md)). A 3-hour first SCF iteration is
   normal on a large charged grid.
10. **Elapsed time from timestamps**, never from the `PWSCF … WALL` summary
    (timer bug, [04](04-output-files.md#qm-stage)).
11. **Final judgements from archives** (`qm_N/`, `mm_N/`), not from root
    working files.

## 4. Submitting

12. **No long-lived queue-feeder loops.** A `while true; … sbatch …; sleep`
    loop left in a background shell kept submitting jobs for weeks after its
    session was forgotten, using whatever driver was on disk at the time
    (2026-08-19 → found 2026-09-07; two "rogue" jobs). Feeders must be
    **one-shot**, or carry a kill-switch file they check every iteration, and
    every status check should list live submitters across all sessions
    (`pgrep -af sbatch`, `pgrep -af qsub`).
13. **One resume per export at a time.** After batch resubmits on PBS, dedupe
    held `afterany` children; two resumes racing in one `export/` corrupt it.
14. **Respect the machine profile**: node/partition exclusions, per-user job
    caps, shared-partition etiquette ([07 §1](07-server-profiles-and-migration.md)).
    An automation that fills every free slot is a nuisance to the other users
    of the cluster.
15. **Human approval for the irreversible**: cancelling a running job,
    deleting archives or `_save_backup/`, overwriting a seed. Everything else
    (patch, preflight, submit, monitor, resubmit on walltime) can be automated
    once the checks above are in place.

## 5. Writing scripts and waiters

16. **Absolute paths; abort if `cd` fails.** A patch script run on a
    not-yet-built export failed its `cd` and executed `rm -rf qm_0; mkdir
    solute` in the project root (2026-10-08). `set -e` does not save you if
    the `cd` is in a subshell or followed by `;`. Pattern:
    `cd "$E" || { echo "no $E"; exit 1; }` and `set -euo pipefail` at the top.
17. **Background commands do not inherit the foreground `cd`.** Pass the
    directory explicitly.
18. **Waiters by PID or artifact, never by `pgrep -f <pattern>`:** the
    waiter's own command line contains the pattern, so
    `until ! pgrep -f foo.py` never exits (four stuck monitors, 2026-09-29).
    Use `while kill -0 $PID; do sleep 30; done`, or poll for the output file.
    If `pgrep` is unavoidable, use a self-excluding pattern (`'[f]oo.py'`).
19. **Launch LAMMPS directly** (`mpirun -np N lmp_mpi …`), never through
    `xargs bash -c` with exported functions: the DFT-CES grid reader has an
    uninitialized flag that misparses `solute.ind.cube` under a large
    environment ([09](09-failure-catalog.md)). Check the parse signature line
    in the log before trusting any MM output.
20. **Pin threads on shared login nodes** (`OMP_NUM_THREADS`,
    `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`) before running numpy analysis
    there; anything longer than minutes goes through the scheduler.
21. **Idempotent patches:** a script that is run twice must not apply a
    change twice (sed on an already-patched line, a second `exit 0` inserted
    into the driver). Grep for the intended end state first.

## 6. Keeping the knowledge current

22. **Every incident becomes a catalog row** in [09](09-failure-catalog.md):
    signature, cause, fix, verification. A protocol that lives only in one
    person's memory (or one agent's session) is lost at the next hand-over.
23. **Machine-checkable beats prose.** When a rule can be expressed as a check
    (k-mesh in `base.pw.in`, `bond_coeff` of the pinned force field, nelec of
    the seed), add it to `ces2_preflight.sh` rather than to a checklist.
