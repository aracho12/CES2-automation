# 07 — Running Across Machines: Profiles, Migration, Hand-off

The pipeline is machine-agnostic, but every deployment differs in scheduler,
core counts, walltime, and binary paths. This doc gives the GENERAL rules; a
concrete two-machine example (the original deployment) is at the end, marked
site-specific — treat it as a template, not as defaults.

## 1. Make a machine profile before you build anything

For each cluster you use, record (once) the answers to:

| item | why it matters |
|---|---|
| Scheduler type & commands (PBS `qsub/qstat/qdel` vs SLURM `sbatch/squeue/scancel`) | submit script header format; chaining mechanism |
| Cores per job you will use | `NP` in the qmmm script must match |
| Walltime limit | if finite, chains die mid-step — plan resubmits (05) |
| Paths: patched pw.x / pp.x, CES2 LAMMPS (`lmp_mpi`), pseudo_dir | 5 path variables in the script + `base.pw.in` |
| Required modules / environment | submit script preamble |
| Per-user job/CPU limits, unhealthy nodes to exclude | submit headers; queue planning |
| Does `tools/ces2_resubmit.sh` self-chaining work? (needs PBS `depend=`) | on SLURM it is a no-op — chain with `--dependency=afterok:<id>` |
| Driver variant: does `QMMMFINSTEP=N` run `mm_N` after `qm_N`, or exit before the MD leg? | decides what `FIN` to set when extending ([05 §5.6](05-monitoring-and-restart.md)) |
| Login-node limits (per-user CPU cap, thread defaults) | pin `OMP_NUM_THREADS` for on-node analysis; submit anything heavier |
| Nodes/partitions with known launch hangs | the exclusion set for every MD-containing job ([09 B1](09-failure-catalog.md)) |

Keep one healthy `submit_ces2.sh` per machine as the template; for a new
system copy it and change only the job name. **Never reuse a submit header
across scheduler types** (a PBS header on a SLURM machine fails with
confusing errors, e.g. "invalid partition").

Timing anchors are worth recording too (per QM step, per MD step at your core
count) — every restart/hand-off decision below is a time trade-off.

## 2. Why split a chain across machines

qm_1 (the first charged SCF) is the most expensive QM of a chain; later steps
seed from their predecessor and are much cheaper. If you have short-lived
access to a large allocation, converging qm_1 there and continuing the
MD-heavy remainder on your workhorse machine is the highest-value split.

### Resource sizing (measured once, ~100-atom QM cell, ~8300 MM atoms)

The two phases scale very differently, so "more nodes" mostly buys wall
time on the QM side and wastes node-hours on the MM side:

| Phase | 1 node → 2 → 4 → 8 nodes (32 c each) | Verdict |
|---|---|---|
| MM (LAMMPS) | parallel efficiency 100 → 85 → 65 → **46 %** (node-h per step ×2.2 at 8 nodes) | 2 nodes is the sweet spot when the walltime allows (~36 h per 2 M + 2 M steps) |
| QM (pw.x SCF) | 4 → 8 nodes ≈ 79 % efficient (1.6× speed-up) | 4 nodes unless a charged `qm_1` must fit a short walltime |

A QM@4 / MM@2 phase split (separate jobs alternating through the
`initialqm` resume mechanism) saved ~32–36 % node-hours in a pilot at the
cost of twice the queue waits. On a single-node-per-job cluster none of this
applies: one node runs everything and a step takes ~2 days.

## 3. Hand-off recipe: qm_1 converged on machine A → continue on machine B

Copy into the machine-B export (typically ~1 GB; NEVER the multi-GB wfc):

1. `qm_1/` archive directory (outputs + cubes).
2. `solute/solute.save/` **3-file set**: `charge-density.dat`,
   `data-file-schema.xml`, **`paw.txt`** (for PAW setups; missing paw.txt =
   `read_scf: Reading PAW becsum` crash). Do NOT copy `wfc*.dat`.
3. Root **coupling cubes from qm_1 post-processing** (`solute.pot.cube`,
   `solute.ind.cube`, plus `v_saw*`/`total_pot*` if present). Verify their
   mtimes postdate the qm_1 pw run — a job killed during post-processing
   leaves stale ones; in that case rerun qm_1's QM (`initialqm=0`; it
   re-converges from the save in a few iterations).
4. `qm_0/` archive (neutral reference — see 08 §qm_0 reuse).
5. The machine-B export must already contain ITS OWN correct MM box files
   (`data.file`, `base.in.lammps`; see 08). Never copy MM box files across
   charge states or compositions.

Then set `QMMMINISTEP=1`, `QMMMFINSTEP=<target>`, **`initialqm=1`** ("the QM
of step INISTEP is done — resume with its MM"; semantics in 08),
`skipequil=0`, and submit with machine-B's template.

Sanity after start: the driver log should go straight to the MM
equilibration; no pw.x run before mm_1.

## 4. General migration patch list (moving any run state)

Every machine-specific string lives in the export copy. Patch, in one pass:
- `NP` (core count) in the qmmm script
- the 5 binary/tool paths in the qmmm script
- `pseudo_dir` in `base.pw.in` (**check every migration** — a foreign
  pseudo_dir is a classic silent trap)
- submit script → target machine's template (+ any node exclusions/limits)
- `LAMMPSRESTART` filename AND **touch the intended restart file** — copying
  scrambles mtimes and resume logic may pick the wrong (newest) file
- `QMMMINISTEP`/`QMMMFINSTEP`/`initialqm`/`skipequil` per the actual state
  (table in 08), `STARTINGPOT` per the seed rules (05)

Then run the pre-submit audit and post-start live verification from 08 —
migrations are where silent-wrong-system errors happen.

## 5. Fresh-restart non-ortho hazard (any machine)

Inside a running chain each step's non-ortho SCF seeds from the previous
non-ortho state and converges normally. A chain RESTARTED at step N hands
the non-ortho only the ortho-embedding density, which can oscillate for
hundreds of iterations on heavily charged systems. Mitigations, in order:
lower `mixing_beta` (0.1 → 0.05); set a leash (e.g. must pass 1e-2 by ~60
iterations / converge within a fixed wall-clock budget) and kill if missed.
The ortho result and its work function are already valid — weigh whether the
rest of the step justifies the queue slot. Never let it grind to
`electron_maxstep` unattended.

---

## Appendix (site-specific example): the original nurion/tars deployment

| | nurion | tars |
|---|---|---|
| Scheduler | PBS | SLURM |
| Sizing | `select=8:ncpus=32` (256c) | 1 node × 24c |
| Walltime | 48 h | none |
| pw.x/pp.x | `/home01/x3379a03/ces_share/qe_test/bin/` | `/home/mino/02.programs/qe_CES2_mino_241105/bin/` |
| LAMMPS (`$DFT_CES2_PATH/MD/lammps/src-ind-cube/lmp_mpi`) | `DFT_CES2_PATH=/home01/x3379a03/ces_share/dft-ces2-bash-test` | `DFT_CES2_PATH=/home/jthlol/dft-ces2-bash-test` |
| pseudo_dir | `/home01/x3379a03/pseudo` | `/home/aracho/bin/qe/pseudo` |
| Modules | `intel/18.0.3 impi/18.0.3` | system default |
| Quirks | resubmit self-chaining OK; driver exits **before** the MD leg at `qmmmstep ≥ QMMMFINSTEP` (set `FIN=N+1` to get `mm_N`) | MD jobs exclude node16–20 (whole `haswell_24c_ib_sm61` partition: LAMMPS launch hangs; pw.x fine there); driver runs `qm_N` **and** `mm_N` at `FIN=N`; members QOS max 6 jobs/user; login node capped at 2 cores/user; shared `debug`/`nehalem` partitions used only with permission |
| Step timing (100-atom QM cell) | qm_1 cold: hours–1 day | ortho ~14 h + nonortho ~13 h; MD ~18 h (~1.9 d/step) |
