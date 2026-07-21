# Example: Cs_q-12_0.5M — charged IrO2(4OH) / 0.5 M CsOH QM/MM run

A complete, real production system from the cation-effects dataset
(IrO2 4OH-terminated surface, 0.5 M CsOH, electrode charge −12 e over a
3×6×1 supercell → QE `tot_charge = −12/18 = −0.667` per unit cell).

This replaces the previous synthetic examples: everything here is taken from
the actual run (server *tars*, SLURM, 1 node × 24 cores), with the driver reset
to fresh-run state (`QMMMINISTEP=0`, `LAMMPSRESTART="ini.restart"`).

## Layout

| path | what it is |
|---|---|
| `config_Cs_q-12_0.5M.yaml` | Build config. Edit `project.workdir` (absolute path on your machine) and the `ces2_script.*` binary paths, then build. |
| `CONTCAR` | Input slab structure (VASP format, IrO2 + 4 OH adsorbates). |
| `build/` | Builder intermediates: packmol input/log, supercell, per-species xyz, `box_meta.json`, `composition_final.json`, … Kept when `output.keep_build: true`. |
| `export/` | What the builder writes for the run (the run directory). Runtime products (cubes, trajectories, `qm_N/`/`mm_N/`) are not included here. |

`export/` contents: `base.pw.in` / `base.pp.in` (QE templates),
`base.in.lammps` (LAMMPS template), `data.file` (starting structure),
`qmmm_dftces2_charging_pts.sh` (QM/MM driver), `submit_ces2.sh` (SLURM batch),
`qm_atoms.txt` / `mm_atoms.txt`, `build_summary.json`,
`charged_system_params.json` (auto-derived tot_chg/mpc_layer/plate_pos),
`timings.json`.

## Reproduce

```bash
PYTHONPATH=~/bin/CES2-automation python3 ~/bin/CES2-automation/run_builder.py \
    --config config_Cs_q-12_0.5M.yaml
cd <workdir>/export && sbatch submit_ces2.sh     # SLURM (tars)
```

Coordinates will differ from this export (packmol packing is seeded but
environment-dependent); topology, inputs, and scripts should match.

## Two practices this example bakes in (see docs/03 §3, docs/05)

1. **Charged SCF is seeded from a converged charge density** —
   `ces2_script.qm_startingpot: "file"` → driver sets `STARTINGPOT="file"` for
   the charged branch. An atomic start on a charged slab oscillates for
   hundreds of SCF iterations (days); a seeded one converges in ~10–16.
   Place a converged `charge-density.dat` + `data-file-schema.xml` + `paw.txt`
   in `export/solute/solute.save/` before the charged steps (protocol:
   docs/05 "Seeding charged SCF").
2. **No QE parallelization flags** — `npool`/`ntg`/`ndiag` are left unset
   (`QEPW_FLAGS=""`): they gave unreliable results with the CES2-patched pw.x.
