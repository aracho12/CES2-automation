# DFT-CES2 Documentation

Documentation for running DFT-CES2 QM/MM simulations with this repository's
builder and tools. Written for people running DFT-CES2 for the first time —
and for automation that has to follow the same protocols.

> For **building** input files (config.yaml, species DB, builder usage), see the
> [main README](../README.md). These pages cover what happens **after** the build:
> what the generated run script does, what every variable and output file means,
> and how to monitor, restart, and analyze a run.

## Reading order (first-timers)

| Page | What it covers |
|---|---|
| [01 — Overview](01-overview.md) | What DFT-CES2 is; the QM/MM outer loop; what `qm_N` / `mm_N` mean |
| [02 — Your first run](02-first-run.md) | The golden path: build → preflight → submit → live-verify → monitor → resume → analyze → judge convergence |

## Reference (look things up as needed)

| Page | What it covers |
|---|---|
| [03 — `qmmm_dftces2_charging_pts.sh` variable reference](03-qmmm-script-reference.md) | Every variable in the generated run script, with the `config.yaml` key that controls it |
| [04 — Output file reference](04-output-files.md) | Every file that appears in `export/` during a run: who writes it, what it contains, whether you can delete it; log fingerprints |
| [05 — Monitoring & restart](05-monitoring-and-restart.md) | Reading `qmmm.out`, phase detection, the stale-file rule, hang vs slow, `ces2_resubmit.sh`, extending a chain, SCF seeding, convergence criteria |
| [06 — Analysis & operations tools](06-analysis-tools.md) | The `tools/` scripts: preflight, resubmit, save backup, work function, convergence, 5-in-1 trajectory analysis, guardrails |
| [07 — Server profiles & migration](07-server-profiles-and-migration.md) | Machine profiles, moving a chain between clusters, hand-off recipe, resource sizing |
| [08 — New charge state / clone checklist](08-new-charge-state-checklist.md) | Sources of truth vs derived files, the three-consistency rule, `initialqm` semantics, pre-submit audit, live verification |
| [09 — Failure catalog](09-failure-catalog.md) | Signature → cause → fix → verify for every failure seen in production |
| [10 — Automation safety](10-automation-safety.md) | Rules for scripts, loops and agents that act on runs unattended |
| [Glossary](glossary.md) | SAW potential, blur, PZC, ortho vs. non-ortho, and other jargon |
