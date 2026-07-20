#!/usr/bin/env python3
"""
traj_multi_analysis.py — ONE-PASS multi-analysis of CES2 MM trajectories.

Parses each ces2.emd.lammpstrj ONCE (the parse is the bottleneck: ~1 GB ASCII
per segment), caches the needed arrays to <trj>.multi.npz, then computes FIVE
analyses from the cache (re-runs with different cutoffs are near-instant):

  1. z-density profiles           (cation / Ow / Hw, Gaussian-smoothed)
  2. RDF cation-Ow                (xy-periodic minimum image; used for r_c)
  3. Coordination ACF             (cation-Ow pairs within r_c: C_int, C_cont)
  4. First-layer occupancy ACF    (NEW: cation z inside the first water layer
                                   -> 1 else 0; boundary = z of the FIRST PEAK
                                   of the water-O z-density near the surface
                                   by default; also C_int, C_cont, <n_in>)
  5. Water orientation vs z       (dipole P1(z), P2(z); dipole = O->mid(H,H))

Segments (mm_N) are NOT continuous in time: profiles/RDF pool all frames, but
ACF time-origins never cross a segment boundary (per-segment lags only).

Usage:
  traj_multi_analysis.py TRJ [TRJ ...] --cation Na -o outdir
      [--lammps-input PATH]   in.lammps for the type map (default: sibling)
      [--cutoff R]            coordination cutoff Å (default: first RDF min)
      [--layer-mode peak|min] first-layer boundary from water profile (peak)
      [--layer-z Z]           override boundary with absolute z (Å)
      [--max-lag-ps T]        ACF max lag (default 500 ps)
      [--rebuild-cache]       ignore existing .multi.npz caches
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# LAMMPS type fallbacks (2oh2o builds): 1 Hw, 2 Ow, 3 cation, 4 H_oh, 5 O_oh,
# 6..11 QM slab. Overridden by the in.lammps group comments when parseable.
FALLBACK_LABELS = {1: "Hw", 2: "Ow", 3: "cation", 4: "H_oh", 5: "O_oh"}
SLAB_TYPES_FALLBACK = set(range(6, 12))
TIMESTEP_FS = 1.0          # MD timestep (fs) in the emd (averaging) run
ANG = "Å"


# ──────────────────────────── type map ────────────────────────────────────
def parse_type_label_map(lammps_input: Path) -> Dict[int, str]:
    """Type→label map from `group ... type ...  # label` comments (rdf.py style)."""
    pat = re.compile(r"^\s*group\s+\S+\s+type\s+([\d\s]+)(?:#\s*(.+))?$", re.I)
    label_map: Dict[int, str] = {}
    if not lammps_input or not lammps_input.exists():
        return label_map
    for line in open(lammps_input):
        m = pat.match(line)
        if not m:
            continue
        tids = [int(x) for x in m.group(1).split()]
        text = (m.group(2) or "")
        text = text.split(":", 1)[1] if ":" in text else text
        text = re.sub(r"\([^)]*\)", " ", text)
        toks = [t for t in re.split(r"[\s,]+", text.strip())
                if t and t.lower() not in {"all", "water", "qm", "slab", "atoms"}]
        if len(toks) >= len(tids):
            for tid, lab in zip(tids, toks):
                label_map[tid] = lab
        elif len(tids) == 1 and toks:
            label_map[tids[0]] = toks[0]
    return label_map


def resolve_types(label_map: Dict[int, str], cation: str
                  ) -> Tuple[int, int, int, List[int]]:
    """Return (cation_type, Ow_type, Hw_type, slab_types)."""
    cat_t = ow_t = hw_t = None
    slab: List[int] = []
    for tid, lab in label_map.items():
        low = lab.lower()
        if low == cation.lower():
            cat_t = tid
        elif low in {"ow"}:
            ow_t = tid
        elif low in {"hw"}:
            hw_t = tid
        elif low.startswith(("ir", "o_ads", "o_bulk", "o_surf", "h_ads")):
            slab.append(tid)
    if cat_t is None:
        cat_t = 3
        print(f"WARN: cation '{cation}' not in type map — falling back to type 3")
    if ow_t is None:
        ow_t = 2
    if hw_t is None:
        hw_t = 1
    if not slab:
        slab = sorted(SLAB_TYPES_FALLBACK)
    return cat_t, ow_t, hw_t, slab


# ──────────────────────────── 1-pass parser ───────────────────────────────
def scan_segment(trj: Path, cat_t: int, ow_t: int, hw_t: int,
                 slab_types: List[int], rebuild: bool) -> dict:
    """Parse one lammpstrj ONCE -> npz cache. Returns dict of arrays.

    Cached: timesteps[F], box_z[2], cat[F,Nc,3], ow[F,No,3], hw[F,Nh,3],
            slab_top_z (scalar; slab is frozen in the MM run), box_xy[2].
    """
    cache = trj.with_suffix(trj.suffix + ".multi.npz")
    if cache.exists() and not rebuild:
        d = dict(np.load(cache))
        d["_cache"] = str(cache)
        return d

    timesteps: List[int] = []
    cat_f: List[np.ndarray] = []
    ow_f: List[np.ndarray] = []
    hw_f: List[np.ndarray] = []
    slab_top = None
    box_xy = box_z = None

    with open(trj) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            ts = int(f.readline())
            f.readline()                     # ITEM: NUMBER OF ATOMS
            natoms = int(f.readline())
            f.readline()                     # ITEM: BOX BOUNDS
            xlo, xhi = map(float, f.readline().split()[:2])
            ylo, yhi = map(float, f.readline().split()[:2])
            zlo, zhi = map(float, f.readline().split()[:2])
            cols = f.readline().split()[2:]  # ITEM: ATOMS id type xu yu zu ...
            it, ix, iy, iz = (cols.index(c) for c in ("type", "xu", "yu", "zu"))
            if box_xy is None:
                box_xy = np.array([xhi - xlo, yhi - ylo])
                box_z = np.array([zlo, zhi])
            # fast block parse: one big fromstring on the whole frame block
            block = "".join(f.readline() for _ in range(natoms))
            arr = np.fromstring(block, sep=" ").reshape(natoms, len(cols))
            types = arr[:, it].astype(int)
            xyz = arr[:, [ix, iy, iz]].astype(np.float32)
            cat_f.append(xyz[types == cat_t])
            ow_f.append(xyz[types == ow_t])
            hw_f.append(xyz[types == hw_t])
            if slab_top is None:
                m = np.isin(types, slab_types)
                slab_top = float(xyz[m][:, 2].max()) if m.any() else float(zlo)
            timesteps.append(ts)

    out = {
        "timesteps": np.array(timesteps, dtype=np.int64),
        "cat": np.stack(cat_f).astype(np.float32),
        "ow": np.stack(ow_f).astype(np.float32),
        "hw": np.stack(hw_f).astype(np.float32),
        "slab_top_z": np.float64(slab_top),
        "box_xy": box_xy.astype(np.float64),
        "box_z": box_z.astype(np.float64),
    }
    np.savez_compressed(cache, **out)
    out["_cache"] = str(cache)
    return out


# ──────────────────────────── analyses ────────────────────────────────────
def gaussian_smooth(y: np.ndarray, dz: float, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return y
    n = max(1, int(3 * sigma / dz))
    x = np.arange(-n, n + 1) * dz
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return np.convolve(y, k, mode="same")


def z_profiles(segs: List[dict], dz: float = 0.1, sigma: float = 0.3):
    """Number-density profiles vs z (absolute Å) pooled over all segments."""
    zlo = min(s["box_z"][0] for s in segs)
    zhi = max(s["box_z"][1] for s in segs)
    edges = np.arange(zlo, zhi + dz, dz)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    area = segs[0]["box_xy"][0] * segs[0]["box_xy"][1]
    prof = {}
    nfr = sum(len(s["timesteps"]) for s in segs)
    for key in ("cat", "ow", "hw"):
        h = np.zeros(len(ctr))
        for s in segs:
            h += np.histogram(s[key][:, :, 2].ravel(), bins=edges)[0]
        dens = h / (nfr * area * dz)               # atoms / Å^3
        prof[key] = gaussian_smooth(dens, dz, sigma)
    return ctr, prof


def first_layer_boundary(ctr, ow_prof, slab_top, mode="peak"):
    """z* from the water-O profile above the surface: first PEAK (default)
    or the first MINIMUM after that peak."""
    m = ctr > slab_top + 0.5                       # look above the slab only
    z, y = ctr[m], ow_prof[m]
    pk = None
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1] and y[i] > 0.5 * y.max():
            pk = i
            break
    if pk is None:
        pk = int(np.argmax(y))
    if mode == "peak":
        return float(z[pk])
    for i in range(pk + 1, len(y) - 1):            # mode == "min"
        if y[i] <= y[i - 1] and y[i] <= y[i + 1]:
            return float(z[i])
    return float(z[pk])


def rdf_cat_ow(segs: List[dict], rmax: float = 8.0, dr: float = 0.05):
    """Cation-Ow g(r), xy minimum image (z is non-periodic)."""
    edges = np.arange(0, rmax + dr, dr)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    hist = np.zeros(len(ctr))
    n_pairs = 0
    vol = None
    for s in segs:
        Lx, Ly = s["box_xy"]
        if vol is None:
            zspan = s["ow"][:, :, 2].max() - s["ow"][:, :, 2].min()
            vol = Lx * Ly * max(zspan, 1.0)
        for fi in range(len(s["timesteps"])):
            c, o = s["cat"][fi], s["ow"][fi]
            if len(c) == 0:
                continue
            d = o[None, :, :] - c[:, None, :]
            d[..., 0] -= Lx * np.round(d[..., 0] / Lx)
            d[..., 1] -= Ly * np.round(d[..., 1] / Ly)
            r = np.linalg.norm(d, axis=-1).ravel()
            hist += np.histogram(r, bins=edges)[0]
            n_pairs += len(c) * len(o) / vol
    shell = 4 * np.pi * ctr**2 * dr
    g = np.divide(hist, shell * max(n_pairs, 1e-30),
                  out=np.zeros_like(hist), where=shell > 0)
    return ctr, g


def first_min_after_peak(r, g):
    pk = int(np.argmax(g))
    for i in range(pk + 1, len(g) - 1):
        if g[i] <= g[i - 1] and g[i] <= g[i + 1]:
            return float(r[i])
    return float(r[min(pk + 20, len(r) - 1)])


def acf_from_bool(h_segs: List[np.ndarray], max_lag: int):
    """Pooled INTERMITTENT ACF  C(t) = <h(0)h(t)> / <h>  from per-segment
    boolean arrays [F, N].  Origins never cross a segment boundary.
    (The continuous/survival ACF is computed by acf_continuous().)"""
    lags = np.arange(max_lag + 1)
    ci = np.zeros(len(lags))
    norm = np.zeros(len(lags))
    for h in h_segs:
        hb = h.astype(bool)
        F = hb.shape[0]
        for lag in lags:
            if lag >= F:
                break
            a, b = hb[:F - lag], hb[lag:]
            ci[lag] += (a & b).sum()
            norm[lag] += a.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        c_int = np.divide(ci, norm, out=np.zeros_like(ci), where=norm > 0)
    return lags, c_int, norm


def acf_continuous(h_segs: List[np.ndarray], max_lag: int):
    """Continuous (uninterrupted) survival ACF, vectorized via run lengths.

    A run of length L contributes (L - k) origins that survive >= k frames
    (k < L).  With RL[L] = number of runs of length L:
        surv[k] = sum_{L>k} RL[L] * (L - k)
    computed with cumulative sums in O(Lmax + max_lag)."""
    lags = np.arange(max_lag + 1)
    run_lengths: List[np.ndarray] = []
    for h in h_segs:
        hb = h.astype(bool)
        # separate columns with a guaranteed False so runs never span atoms
        x = np.vstack([hb, np.zeros((1, hb.shape[1]), dtype=bool)])
        flat = x.T.ravel().astype(np.int8)
        d = np.diff(np.concatenate(([0], flat, [0])))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        if len(starts):
            run_lengths.append(ends - starts)
    if not run_lengths:
        return lags, np.zeros(len(lags))
    L = np.concatenate(run_lengths)
    Lmax = int(L.max())
    RL = np.bincount(L, minlength=Lmax + 1).astype(np.float64)
    Ls = np.arange(Lmax + 1, dtype=np.float64)
    tail0 = np.cumsum((RL)[::-1])[::-1]          # sum_{L>=k} RL[L]
    tail1 = np.cumsum((RL * Ls)[::-1])[::-1]     # sum_{L>=k} RL[L]*L
    surv = np.zeros(len(lags))
    for k in lags:
        if k + 1 <= Lmax:
            surv[k] = (tail1[k + 1] - 0) - k * (tail0[k + 1] - 0)
        elif k == 0:
            surv[0] = tail1[0]
    n0 = surv[0] if surv[0] > 0 else 1.0
    return lags, surv / n0


def residence_time(lags_ps, c):
    """Integral of C(t) up to where it decays (simple trapezoid)."""
    return float(np.trapz(c, lags_ps))


def water_orientation(segs: List[dict], edges: np.ndarray):
    """Dipole P1/P2 vs z.  Dipole = O -> midpoint(H1,H2).

    Waters are dumped as consecutive O,H,H molecules (ids sorted), so water j
    pairs with Hw[2j], Hw[2j+1] — O(N) per frame.  Validated on the first
    frame of each segment (max O-H < 1.3 Å); falls back to a nearest-H search
    if the ordering ever breaks."""
    ctr = 0.5 * (edges[:-1] + edges[1:])
    s1 = np.zeros(len(ctr)); s2 = np.zeros(len(ctr)); cnt = np.zeros(len(ctr))

    def dip_indexed(o, hh, Lx, Ly):
        h1, h2 = hh[0::2][:len(o)], hh[1::2][:len(o)]
        v = 0.5 * (h1 + h2) - o
        v[:, 0] -= Lx * np.round(v[:, 0] / Lx)
        v[:, 1] -= Ly * np.round(v[:, 1] / Ly)
        return v

    def dip_nearest(o, hh, Lx, Ly):
        d = hh[None, :, :] - o[:, None, :]
        d[..., 0] -= Lx * np.round(d[..., 0] / Lx)
        d[..., 1] -= Ly * np.round(d[..., 1] / Ly)
        idx = np.argsort((d ** 2).sum(-1), axis=1)[:, :2]
        return 0.5 * np.take_along_axis(d, idx[:, :, None], axis=1).sum(1)

    for s in segs:
        Lx, Ly = s["box_xy"]
        o0, h0 = s["ow"][0], s["hw"][0]
        v0 = dip_indexed(o0.copy(), h0, Lx, Ly)
        # sanity: indexed O-H distances must look like real bonds
        b1 = h0[0::2][:len(o0)] - o0
        indexed_ok = np.abs(np.linalg.norm(b1, axis=1)).max() < 1.3
        dip = dip_indexed if indexed_ok else dip_nearest
        if not indexed_ok:
            print("WARN: O,H,H ordering broken — using nearest-H pairing (slow)")
        for fi in range(len(s["timesteps"])):
            o, hh = s["ow"][fi], s["hw"][fi]
            v = dip(o.copy(), hh, Lx, Ly)
            cosz = v[:, 2] / np.maximum(np.linalg.norm(v, axis=1), 1e-12)
            bi = np.digitize(o[:, 2], edges) - 1
            ok = (bi >= 0) & (bi < len(ctr))
            np.add.at(s1, bi[ok], cosz[ok])
            np.add.at(s2, bi[ok], 1.5 * cosz[ok] ** 2 - 0.5)
            np.add.at(cnt, bi[ok], 1)
    with np.errstate(invalid="ignore"):
        p1 = np.divide(s1, cnt, out=np.zeros_like(s1), where=cnt > 0)
        p2 = np.divide(s2, cnt, out=np.zeros_like(s2), where=cnt > 0)
    return ctr, p1, p2, cnt


# ──────────────────────────── main ────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trajs", nargs="+", type=Path)
    ap.add_argument("--cation", required=True)
    ap.add_argument("-o", "--outdir", type=Path, required=True)
    ap.add_argument("--lammps-input", type=Path, default=None)
    ap.add_argument("--cutoff", type=float, default=None)
    ap.add_argument("--layer-mode", choices=["peak", "min"], default="peak")
    ap.add_argument("--layer-z", type=float, default=None)
    ap.add_argument("--max-lag-ps", type=float, default=500.0)
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    lmp_in = args.lammps_input or (args.trajs[0].parent / "in.lammps")
    if not lmp_in.exists():
        lmp_in = args.trajs[0].parent.parent / "in.lammps"
    label_map = parse_type_label_map(lmp_in)
    cat_t, ow_t, hw_t, slab_types = resolve_types(label_map, args.cation)
    print(f"[types] cation({args.cation})={cat_t} Ow={ow_t} Hw={hw_t} "
          f"slab={slab_types}  (map from {lmp_in if label_map else 'fallback'})")

    # ---- 1-pass scan per segment (cached) --------------------------------
    segs = []
    for t in args.trajs:
        print(f"[scan] {t}")
        segs.append(scan_segment(t, cat_t, ow_t, hw_t, slab_types,
                                 args.rebuild_cache))
        print(f"       frames={len(segs[-1]['timesteps'])} cache={segs[-1]['_cache']}")

    # frame spacing (ps)
    ts = segs[0]["timesteps"]
    dt_ps = (ts[1] - ts[0]) * TIMESTEP_FS / 1000.0 if len(ts) > 1 else 1.0
    max_lag = int(args.max_lag_ps / dt_ps)
    slab_top = float(segs[0]["slab_top_z"])

    # ---- 1) z-density ----------------------------------------------------
    ctr, prof = z_profiles(segs)
    np.savetxt(args.outdir / "zdensity.csv",
               np.column_stack([ctr, prof["cat"], prof["ow"], prof["hw"]]),
               delimiter=",", header="z_A,cation_per_A3,Ow_per_A3,Hw_per_A3",
               comments="")

    # ---- first-layer boundary -------------------------------------------
    if args.layer_z is not None:
        zstar = args.layer_z
        src = "user --layer-z"
    else:
        zstar = first_layer_boundary(ctr, prof["ow"], slab_top, args.layer_mode)
        src = f"water-O first {args.layer_mode} (slab_top={slab_top:.2f})"
    print(f"[layer] boundary z* = {zstar:.3f} {ANG}  ({src})")

    # ---- 2) RDF + cutoff -------------------------------------------------
    r, g = rdf_cat_ow(segs)
    rc = args.cutoff or first_min_after_peak(r, g)
    np.savetxt(args.outdir / "rdf_cation_Ow.csv", np.column_stack([r, g]),
               delimiter=",", header="r_A,g_r", comments="")
    print(f"[rdf] first-shell cutoff r_c = {rc:.3f} {ANG}"
          f"{' (user)' if args.cutoff else ' (first min)'}")

    # ---- 3) coordination ACF (per cation-Ow PAIR within r_c) -------------
    coord_segs = []      # boolean [F, Nc*No]: pair (i,j) in first shell
    occ_segs = []        # boolean [F, Nc]   : cation inside first layer
    coordno = []
    for s in segs:
        Lx, Ly = s["box_xy"]
        F, Nc = s["cat"].shape[0], s["cat"].shape[1]
        No = s["ow"].shape[1]
        hp = np.zeros((F, Nc * No), dtype=np.int8)
        for fi in range(F):
            c, o = s["cat"][fi], s["ow"][fi]
            d = o[None, :, :] - c[:, None, :]
            d[..., 0] -= Lx * np.round(d[..., 0] / Lx)
            d[..., 1] -= Ly * np.round(d[..., 1] / Ly)
            hp[fi] = (np.linalg.norm(d, axis=-1) < rc).reshape(-1)
        coord_segs.append(hp)
        coordno.append(hp.sum(axis=1).mean() / max(Nc, 1))
        occ_segs.append((s["cat"][:, :, 2] <= zstar).astype(np.int8))
    cn = float(np.mean(coordno))

    lags, ci_coord, _ = acf_from_bool(coord_segs, max_lag)
    _, cc_coord = acf_continuous(coord_segs, max_lag)
    lags_ps = lags * dt_ps
    tau_c_int = residence_time(lags_ps, ci_coord)
    tau_c_cont = residence_time(lags_ps, cc_coord)
    np.savetxt(args.outdir / "acf_coordination.csv",
               np.column_stack([lags_ps, ci_coord, cc_coord]), delimiter=",",
               header=(f"t_ps,C_intermittent,C_continuous (r_c={rc:.3f}A; "
                       f"coord_no={cn:.2f}; tau_int={tau_c_int:.1f}ps; "
                       f"tau_cont={tau_c_cont:.1f}ps)"),
               comments="")
    print(f"[coord] n_coord={cn:.2f}  tau_int={tau_c_int:.1f} ps  "
          f"tau_cont={tau_c_cont:.1f} ps")

    # ---- 4) FIRST-LAYER OCCUPANCY ACF (the new analysis) -----------------
    frac_in = float(np.mean([h.mean() for h in occ_segs]))
    n_in = float(np.mean([h.sum(axis=1).mean() for h in occ_segs]))
    _, ci_occ, _ = acf_from_bool(occ_segs, max_lag)
    _, cc_occ = acf_continuous(occ_segs, max_lag)
    tau_int = residence_time(lags_ps, ci_occ)
    tau_cont = residence_time(lags_ps, cc_occ)
    hdr = (f"t_ps,C_intermittent,C_continuous "
           f"(z*={zstar:.3f}A [{src}]; frac_in={frac_in:.4f}; "
           f"n_in={n_in:.3f}; tau_int={tau_int:.1f}ps; tau_cont={tau_cont:.1f}ps)")
    np.savetxt(args.outdir / "acf_layer_occupancy.csv",
               np.column_stack([lags_ps, ci_occ, cc_occ]), delimiter=",",
               header=hdr, comments="")
    print(f"[occupancy] <in-layer fraction>={frac_in:.4f}  <n_in>={n_in:.3f}  "
          f"tau_int={tau_int:.1f} ps  tau_cont={tau_cont:.1f} ps")

    # ---- 5) water orientation -------------------------------------------
    dzo = 0.25
    edges = np.arange(ctr[0], ctr[-1] + dzo, dzo)
    zo, p1, p2, cnt = water_orientation(segs, edges)
    np.savetxt(args.outdir / "water_orientation.csv",
               np.column_stack([zo, cnt, p1, p2]), delimiter=",",
               header="z_A,count,P1_dipole,P2_dipole", comments="")

    # ---- plots -----------------------------------------------------------
    if HAVE_MPL:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        ax = axes[0, 0]
        ax.plot(ctr, prof["ow"], label="Ow")
        ax.plot(ctr, prof["cat"], label=args.cation)
        ax.axvline(zstar, ls="--", c="k", lw=1, label=f"z*={zstar:.2f}")
        ax.axvline(slab_top, ls=":", c="gray", lw=1)
        ax.set_xlabel(f"z ({ANG})"); ax.set_ylabel(f"density (1/{ANG}$^3$)")
        ax.legend(); ax.set_title("z-density")
        ax = axes[0, 1]
        ax.plot(r, g); ax.axvline(rc, ls="--", c="k", lw=1)
        ax.set_xlabel(f"r ({ANG})"); ax.set_ylabel("g(r)")
        ax.set_title(f"RDF {args.cation}-Ow (r_c={rc:.2f})")
        ax = axes[1, 0]
        ax.plot(lags_ps, ci_occ, label="C_int (layer)")
        ax.plot(lags_ps, cc_occ, label="C_cont (layer)")
        ax.plot(lags_ps, ci_coord, "--", label="C_int (coord)")
        ax.plot(lags_ps, cc_coord, "--", label="C_cont (coord)")
        ax.set_xlabel("t (ps)"); ax.set_ylabel("C(t)"); ax.legend()
        ax.set_title(f"ACF (occupancy: frac={frac_in:.3f}, "
                     f"tau_cont={tau_cont:.0f} ps)")
        ax = axes[1, 1]
        ax.plot(zo, p1, label="P1 dipole")
        ax.plot(zo, p2, label="P2 dipole")
        ax.axvline(zstar, ls="--", c="k", lw=1)
        ax.set_xlabel(f"z ({ANG})"); ax.set_ylabel("P1 / P2"); ax.legend()
        ax.set_title("water orientation")
        fig.suptitle(f"{args.cation}: multi-analysis "
                     f"({sum(len(s['timesteps']) for s in segs)} frames, "
                     f"{len(segs)} segment(s))")
        fig.tight_layout()
        fig.savefig(args.outdir / "multi_analysis.png", dpi=150)
        print(f"[plot] {args.outdir/'multi_analysis.png'}")

    print(f"[done] outputs in {args.outdir}")


if __name__ == "__main__":
    main()
