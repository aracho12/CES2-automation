# 02 — Your First Run

The golden path from a relaxed slab to a converged QM/MM chain, with the
checks that make the difference between a result and a wasted week. Commands
assume the repo at `~/bin/CES2-automation` and a Python env with the builder
dependencies ([main README §Quick Start](../README.md)).

```
build → inspect export → preflight → submit → live-verify (first hours)
      → monitor → resume on walltime → analyze → judge convergence
```

## 0. Prerequisites

- [ ] A built DFT-CES2 stack on this machine: the `dft-ces2` LAMMPS fork
      (`src-ind-cube/lmp_mpi`), patched QE `pw.x`/`pp.x`, cube tools. Record
      their paths in a **machine profile** ([07 §1](07-server-profiles-and-migration.md)).
- [ ] A relaxed VASP `CONTCAR` of the slab, with the adsorbate/surface atoms
      you will label in `slab:` ([README §slab](../README.md)).
- [ ] Decide the **campaign standards** once and write them into a
      `campaign.yaml` at the top of the dataset (template:
      `campaign_example.yaml`): k-mesh, QE flags (keep `QEPW_FLAGS` empty),
      the force-field snapshot (`species_db` path or git hash, plus a
      fingerprint preflight can test), `NP`, site rules for the submit
      script. Every system of a dataset must share them
      ([05 §7](05-monitoring-and-restart.md)); preflight enforces the file.

## 1. Build the run directory

Start from `config_example.yaml` (every key is commented) or from a sibling
system's config. Keys that bite on a first build:

| key | rule |
|---|---|
| `project.workdir` | absolute path on **this** machine; `export/` = `workdir/export` |
| `species_db` | pin to the campaign's frozen force-field copy; delete the key only if you really want the repo default |
| `ces2_script.qm_startingpot` | `"file"` for any charged system (`charge_control` ≠ 0) — see §4 and [05 §6](05-monitoring-and-restart.md) |
| `ces2_script.np` | must equal the cores the submit script requests |
| `ces2_script.dft_ces2_path`, `qe_binary`, `pp_binary`, `lmp_binary`, … | this machine's binaries (a copied config silently carries another machine's) |
| `qe.pseudo_dir` | this machine's pseudopotential directory |
| `ces2_script.n_qmmm_steps` | steps 0…N−1; 4–6 is a typical first chain |

```bash
cd <system_folder>
PYTHONPATH=~/bin/CES2-automation python3 ~/bin/CES2-automation/run_builder.py --config config.yaml
```

What success looks like: `export/` contains `data.file`, `base.in.lammps`,
`base.pw.in`, `base.pp.in`, `qmmm_dftces2_charging_pts.sh`, `submit_ces2.sh`,
`build_summary.json`, `charged_system_params.json`, `combined.xyz`
([04 §1](04-output-files.md)).

Inspect before going further:

```bash
cd export
python3 -c "import json;d=json.load(open('build_summary.json'));print(d['composition'] if 'composition' in d else d)"
grep -A1 K_POINTS base.pw.in          # campaign k-mesh?
grep 'bond_coeff 2' base.in.lammps    # pinned force-field fingerprint (e.g. OH- r0)
grep -n 'TOTCHG=\|^NP=\|QEPW_FLAGS=' qmmm_dftces2_charging_pts.sh
cat charged_system_params.json        # tot_chg, plate/point-charge positions
```
Open `combined.xyz` in OVITO/VESTA once: slab orientation, water box height,
ions not inside the slab.

## 2. Reuse a neutral reference and plant the seed (charged systems)

Two things save days on every charged system:

1. **`qm_0` reuse.** The neutral reference QM is identical for every charge
   state and cation of the same slab. Copy a completed system's `qm_0/`
   (must contain `ref_solute.rho.cube`, `ref_solute.pot.cube`) into
   `export/` and set `QMMMINISTEP=0`, `initialqm=1` so the chain starts at
   `mm_0` ([08 §qm_0 reuse](08-new-charge-state-checklist.md)).
2. **SCF seed.** With `STARTINGPOT="file"` the first charged SCF needs a
   converged charge density in `export/solute/solute.save/`
   (`charge-density.dat`, `data-file-schema.xml`, `paw.txt`) from the same
   charge state (any cation) or a neighbouring one
   ([05 §6](05-monitoring-and-restart.md)). Without it the first SCF takes
   days instead of hours.

For a truly first system (nothing to reuse) run the chain from step 0 with
the charged branch temporarily on `atomic`, accept a long `qm_1`, and back up
its `.save` with `tools/ces2_backup_save.sh` — that becomes the seed for
everything after.

## 3. Preflight (mandatory)

```bash
/bin/bash ~/bin/CES2-automation/tools/ces2_preflight.sh <system_folder_or_export>
```
exit 0 = PASS, 1 = warnings, **2 = FAIL → do not submit**. It checks: binaries
and pseudopotentials exist on this host, `NP` vs the scheduler request and
the scheduler type of the submit header, charge three-consistency read from
the generated files (`TOTCHG` × supercell vs the net charge of the MM atoms
in `data.file` vs the electrode charge in `data.file` vs `set group top
charge` × the atoms inside `region toplayer`), the seed 3-file set **and the
seed's charge state** (`tot_charge` in its xml vs this run's `TOTCHG`),
`INISTEP`/`FINSTEP`/`initialqm`/`skipequil` traps, `qm_0` reference, restart
presence and mtime order. Read the warnings — they are the cases it cannot
decide for you.

**Campaign standards.** Put a `campaign.yaml` (copy `campaign_example.yaml`
from the repo root) in the dataset's top folder: preflight finds it by
searching upward from `export/` and additionally checks the k-mesh, `NP`,
`QEPW_FLAGS`, any force-field fingerprints (regex + expected value in a
generated file, e.g. the OH⁻ bond length of the pinned species DB) and
required substrings of the submit script (node exclusions). Without the file
those values are reported but not judged.

## 4. Submit

```bash
cd export
sbatch submit_ces2.sh        # SLURM
qsub   submit_ces2.sh        # PBS
```
Note the job id. Typical cost per QM/MM step for a ~100-atom QM cell with
~8000 MM atoms: hours to a day of QM plus a day of MD on one 24-core node
([07 Appendix](07-server-profiles-and-migration.md) has measured anchors).

## 5. Live verification — within the first hours

Do this for **every** submit, not just the first ([08](08-new-charge-state-checklist.md#post-start-live-verification-within-the-first-hours--mandatory)).
Only trust root files newer than the job start ([05 §2.3](05-monitoring-and-restart.md)).

```bash
grep 'set group top charge' in.lammps            # regenerated → expected electrode charge?
head -5 data.md | grep atoms                     # == data.file atom count?
tail -2 qmmm.out                                 # at the expected phase?
grep -c 'Finding SHAKE' lammps.equil.out; stat -c %y lammps.equil.out   # launched and moving?
```
When the first charged QM starts:
```bash
grep tot_charge pw.in                            # right charge per unit cell?
grep -m1 'initial density\|superposition' pw.out # must be "read from file"
grep -c 'estimated scf accuracy' pw.out          # iterations so far
```
If anything is wrong: kill, fix the sources of truth, delete derived
artifacts, preflight, resubmit. It will not fix itself.

## 6. Monitor

- `qmmm.out` markers tell the phase ([05 §1](05-monitoring-and-restart.md)).
- `qm_N/` and `mm_N/` appearing is the real progress; root cubes are the
  working set of the newest step only ([04 §3](04-output-files.md)).
- SCF: count `estimated scf accuracy` lines, compare the latest with
  `conv_thr`. MD: the thermo `Step` line vs `MDEQUIL`/`MDAVERAGE`.
- Hang vs slow: [05 §2.4](05-monitoring-and-restart.md). Do not kill a slow
  first SCF iteration.

## 7. Resume after walltime

```bash
cd export
/bin/bash ~/bin/CES2-automation/tools/ces2_resubmit.sh   # detects the phase, patches 4 driver vars, leaves .bak + log
/bin/bash ~/bin/CES2-automation/tools/ces2_preflight.sh .
sbatch submit_ces2.sh                                     # SLURM (PBS: the tool already qsub'd and chained)
```
To add steps to a finished chain, patch the four header lines yourself
([05 §5](05-monitoring-and-restart.md)) — the `data.file` rewind trap lives
there.

## 8. First analysis

```bash
# work function / electrode potential per step, convergence table
python3 ~/bin/CES2-automation/tools/wf_convergence.py <system_folder> -o wf_convergence
# one step in detail (vacuum level, Fermi, dipole amplitude)
python3 ~/bin/CES2-automation/tools/workfunction.py export/qm_3
# density profiles and the 5-in-1 trajectory analysis of the MM segments
python3 ~/bin/CES2-automation/tools/z_density.py export/mm_3/ces2.emd.lammpstrj
python3 ~/bin/CES2-automation/tools/traj_multi_analysis.py export/mm_2/ces2.emd.lammpstrj export/mm_3/ces2.emd.lammpstrj \
        --cation K --lammps-input export/in.lammps -o multi/K
```
Each `mm_N` trajectory is an independent 2 ns (or longer) segment — never
concatenate them in time ([06](06-analysis-tools.md)).

## 9. Judge convergence, then stop

Adopt a chain when successive QM steps differ by **< 0.05 V** in `V_RHE`
and the density profiles are stable; exclude the first step after any
(re)start; extend rather than average a drifting series
([05 §7](05-monitoring-and-restart.md)). Before deleting `solute/` to free
disk, back up the converged `.save` — it is the seed for the next system.
