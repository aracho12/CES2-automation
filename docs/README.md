# DFT-CES2 Documentation

Documentation for running DFT-CES2 QM/MM simulations with this repository's
builder and tools. Written for people running DFT-CES2 for the first time.

> For **building** input files (config.yaml, species DB, builder usage), see the
> [main README](../README.md). These pages cover what happens **after** the build:
> what the generated run script does, what every variable and output file means,
> and how to monitor, restart, and analyze a run.

## Reading order (first-timers)

| Page | What it covers |
|---|---|
| [01 — Overview](01-overview.md) | What DFT-CES2 is; the QM/MM outer loop; what `qm_N` / `mm_N` mean |
| [02 — Your first run](02-first-run.md) | End-to-end walkthrough: build → submit → monitor → analyze |

## Reference (look things up as needed)

| Page | What it covers |
|---|---|
| [03 — `qmmm_dftces2_charging_pts.sh` variable reference](03-qmmm-script-reference.md) | Every variable in the generated run script, with the `config.yaml` key that controls it |
| [04 — Output file reference](04-output-files.md) | Every file that appears in `export/` during a run: who writes it, what it contains, whether you can delete it |
| [05 — Monitoring & restart](05-monitoring-and-restart.md) | Reading `qmmm.out`, restart files, `ces2_resubmit.sh`, common crashes |
| [06 — Analysis tools](06-analysis-tools.md) | The `tools/` scripts: work function, z-density, RDF, water orientation, … |
| [Glossary](glossary.md) | SAW potential, blur, PZC, ortho vs. non-ortho, and other jargon |
- [07 — Server profiles & migration (nurion↔tars)](07-server-profiles-and-migration.md)
- [08 — New charge state / clone checklist](08-new-charge-state-checklist.md)
