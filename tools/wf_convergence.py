#!/usr/bin/env python3
"""
wf_convergence.py — work-function convergence vs QM/MM step (qm_1..qm_N).

For each system dir, finds every canonical qm_N/ holding a total_pot_ortho.cube,
reuses the cached qm_N/workfunction.out when present (compute_vrhe.sh convention)
or runs tools/workfunction.py to create it, then plots V_RHE (or Phi / V_SHE)
vs step — one panel per charge state, colored by cation — and writes a CSV.
The step-to-step delta of the last two points is the convergence indicator.

Works for any surface dataset (2oh2o / 4o / 4oh / ...) — pass the system dirs.

Usage:
  wf_convergence.py SYSDIR [SYSDIR ...] -o OUTDIR
      [--metric vrhe|phi|vshe]  (default vrhe)
      [--ph 13.0] [--she 4.44]
      [--recompute]             ignore cached workfunction.out
Example:
  wf_convergence.py 2oh2o/*_0.5M -o 2oh2o/wf_convergence
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

WF_TOOL = Path.home() / "bin/CES2-automation/tools/workfunction.py"
CUBE = "total_pot_ortho.cube"
QM_RE = re.compile(r"^qm_(\d+)$")          # canonical only (skips *.prev*, *_ref)
PHI_RE = re.compile(r"Work function.*=\s*([-\d.]+)\s*eV")
SHE_RE = re.compile(r"U vs SHE \[4\.44 V\]\s*=\s*([+\-\d.]+)")

# --- figure formatting convention (2026-08-07, project standard) ---
import matplotlib as _mpl
_mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans",
                        "Nimbus Sans", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": True,   # dual-axis: right spine stays
    "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "xtick.major.size": 6.0, "ytick.major.size": 6.0,
    "xtick.minor.size": 3.5, "ytick.minor.size": 3.5,
    "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "xtick.minor.width": 0.8, "ytick.minor.width": 0.8,
})

CAT_COLORS = {"Li": "#CA4420", "Na": "#EDAF3C", "K": "#419395", "Cs": "#15389D"}


def parse_wfout(text: str):
    phi = PHI_RE.search(text)
    she = SHE_RE.search(text)
    return (float(phi.group(1)) if phi else None,
            float(she.group(1)) if she else None)


def wf_for_step(export: Path, qmdir: Path, recompute: bool):
    """Return (phi, vshe) for one qm_N, computing+caching if needed."""
    wfout = qmdir / "workfunction.out"
    if wfout.exists() and not recompute:
        phi, she = parse_wfout(wfout.read_text())
        if phi is not None and she is not None:
            return phi, she
    # compute (must run from export/ so workfunction.py finds step context)
    res = subprocess.run(
        [sys.executable, str(WF_TOOL), "--cube", f"{qmdir.name}/{CUBE}"],
        cwd=export, capture_output=True, text=True)
    log = res.stdout + res.stderr
    wfout.write_text(log)
    for art in ("workfunction_profile.png", "workfunction_fermi.png",
                "workfunction_rawdata.csv", "workfunction_summary.txt"):
        src = export / art
        if src.exists():
            (qmdir / art).write_bytes(src.read_bytes())
    return parse_wfout(log)


def sys_meta(sysdir: Path):
    """(label, cation, q) from a system dir name like Cs_q-20_0.5M."""
    name = sysdir.name
    cat = name.split("_")[0]
    m = re.search(r"q-(\d+)", name)
    q = int(m.group(1)) if m else -1
    return name, cat, q


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sysdirs", nargs="+", type=Path)
    ap.add_argument("-o", "--outdir", type=Path, required=True)
    ap.add_argument("--metric", choices=["vrhe", "phi", "vshe"], default="vrhe")
    ap.add_argument("--ph", type=float, default=13.0)
    ap.add_argument("--she", type=float, default=4.44)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    shift = 0.059 * args.ph

    rows = []       # (label, cation, q, step, phi, vshe, vrhe)
    for sd in args.sysdirs:
        export = sd / "export" if (sd / "export").is_dir() else sd
        if not export.is_dir():
            print(f"SKIP (no export): {sd}")
            continue
        label, cat, q = sys_meta(sd if (sd / "export").is_dir() else sd.parent)
        steps = []
        for d in sorted(export.iterdir()):
            m = QM_RE.match(d.name)
            if m and (d / CUBE).exists() and (d / CUBE).stat().st_size > 0:
                steps.append((int(m.group(1)), d))
        steps = [(n, d) for n, d in sorted(steps) if n > 0]   # skip neutral qm_0
        if not steps:
            print(f"PASS (no completed qm_N cubes): {label}")
            continue
        for n, d in steps:
            phi, she = wf_for_step(export, d, args.recompute)
            if phi is None or she is None:
                print(f"WARN parse failed: {label} qm_{n}")
                continue
            rows.append((label, cat, q, n, phi, she, she + shift))
            print(f"{label} qm_{n}: phi={phi:.4f}  V_SHE={she:+.4f}  "
                  f"V_RHE={she + shift:.4f}")

    if not rows:
        sys.exit("no data")

    # CSV
    csv = args.outdir / "wf_convergence.csv"
    with open(csv, "w") as f:
        f.write("system,cation,q,step,phi_eV,V_SHE,V_RHE\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    # convergence table (delta of last two steps per system)
    print("\n=== step-convergence (last two steps) ===")
    bysys = {}
    for r in rows:
        bysys.setdefault(r[0], []).append(r)
    conv_lines = []
    for label, rs in bysys.items():
        rs = sorted(rs, key=lambda r: r[3])
        if len(rs) >= 2:
            dv = rs[-1][6] - rs[-2][6]
            line = (f"{label}: qm_{rs[-2][3]}->qm_{rs[-1][3]}  "
                    f"dV_RHE={dv:+.4f} V {'(converging)' if abs(dv) < 0.05 else ''}")
        else:
            line = f"{label}: only qm_{rs[0][3]} — need next step"
        conv_lines.append(line)
        print(line)
    (args.outdir / "convergence_summary.txt").write_text("\n".join(conv_lines) + "\n")

    # plot: left axis = metric, right axis = step-to-step delta of the metric
    if HAVE_MPL:
        plt.rcParams["legend.frameon"] = False       # project style: no legend box
        qvals = sorted({r[2] for r in rows})
        icol = {"vrhe": 6, "phi": 4, "vshe": 5}[args.metric]
        ylab = {"vrhe": r"V$_{\mathrm{RHE}}$ (V)", "phi": r"$\Phi$ (eV)",
                "vshe": r"V$_{\mathrm{SHE}}$ (V)"}[args.metric]
        dlab = {"vrhe": r"$\Delta$V$_{\mathrm{RHE}}$ (V)",
                "phi": r"$\Delta\Phi$ (eV)",
                "vshe": r"$\Delta$V$_{\mathrm{SHE}}$ (V)"}[args.metric]
        # layout: up to 3 panels in one row; 4+ wraps into 2 columns (e.g. 2x2)
        ncol = len(qvals) if len(qvals) <= 3 else (len(qvals) + 1) // 2
        nrow = -(-len(qvals) // ncol)
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(max(5.0 * ncol, 8.0), 4.4 * nrow),
                                 squeeze=False)
        flat = [a for row in axes for a in row]
        for a in flat[len(qvals):]:
            a.set_visible(False)
        for ax, q in zip(flat, qvals):
            ax2 = ax.twinx()
            ax2.axhspan(-0.05, 0.05, color="gray", alpha=0.10, zorder=0)
            ax2.axhline(0, color="gray", lw=0.8, ls=":", zorder=0)
            for label, rs in bysys.items():
                if rs[0][2] != q:
                    continue
                rs = sorted(rs, key=lambda r: r[3])
                cat = rs[0][1]
                col = CAT_COLORS.get(cat, "gray")
                steps_x = [r[3] for r in rs]
                vals = [r[icol] for r in rs]
                ax.plot(steps_x, vals, "o-", color=col, label=cat, zorder=3,
                        markeredgecolor="black", markeredgewidth=0.7)
                if len(rs) >= 2:                       # deltas on right axis
                    dx = steps_x[1:]
                    dv = [b - a for a, b in zip(vals, vals[1:])]
                    ax2.plot(dx, dv, "x--", color=col, alpha=0.55,
                             lw=1.0, zorder=2)
            ax.set_xlabel("QM/MM step (qm$_N$)")
            ax.set_ylabel(ylab)
            ax2.set_ylabel(dlab)
            dmax = 0.4
            ax2.set_ylim(-dmax, dmax)
            ax.set_title(f"q-{q}")
            allsteps = sorted({r[3] for rs in bysys.values() for r in rs
                               if rs[0][2] == q})
            if len(allsteps) == 1:      # single-step panel: keep integer axis
                ax.set_xlim(allsteps[0] - 0.5, allsteps[0] + 0.5)
                ax.set_xticks(allsteps)  # FixedLocator; no MaxNLocator override
            else:
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.legend(fontsize=8)
        # dataset (parent dir) also labels the figure title
        dataset = Path(args.sysdirs[0]).resolve().parent.name
        _surf = {"2oh2o": "IrO$_2$(2OH+2O)", "4o": "IrO$_2$(4O)",
                 "4oh": "IrO$_2$(4OH)"}.get(dataset, dataset)
        fig.suptitle(f"{_surf} — WF convergence vs step  (solid: {ylab.split(' ')[0]},  "
                     f"dashed x: step delta,  band: |Δ|<0.05)")
        fig.tight_layout()
        png = args.outdir / f"{dataset}_wf_convergence_{args.metric}.png"
        fig.savefig(png, dpi=150)
        print(f"\nsaved {png}\nsaved {csv}")


if __name__ == "__main__":
    main()
