# 06 — Analysis Tools

<!-- TODO: fill in per tool: purpose, minimal command, inputs it expects
     (which files from 04), outputs, common flags. -->

All tools live in `tools/`. Most operate on an `export/` run directory or on
individual `mm_N` / `qm_N` archives.

| Tool | Purpose |
|---|---|
| `workfunction.py` | Planar-average `total_pot_ortho.cube` → vacuum level & work function; compare/diff across runs. |
| `z_density.py` | Species number-density profiles along z; evolution across `mm_N`. |
| `rdf.py` | Radial distribution functions from `mm_N` trajectories (needs `seaborn`; install with `pip install --only-binary=:all: seaborn` on the cluster). |
| `water_orientation.py` | Water dipole P1/P2 and O–H orientation vs. z. |
| `surface_charge.py` | Surface charge density from `build_summary.json` / `data.file`. |
| `acf_coordination.py` | Ion coordination autocorrelation. ⚠️ Each `mm_N` trajectory is an independent 2 ns segment (frames are *not* continuous across steps) — compute time-correlations per segment only. |
| `z_density.py`, `lammpstrj_to_traj.py`, `plot_temperature.py`, … | <!-- TODO --> |
| `extract_bjparams.py` | BJ dispersion parameters from a bulk reference calculation (build-time helper). |
| `ces2_resubmit.sh` | Not analysis — see [05](05-monitoring-and-restart.md). |

<!-- TODO: worked example — from a finished run to a work-function-vs-step
     convergence plot and a z-density figure. -->
