#!/bin/bash
# ces2_preflight.sh — pre-submit audit for a CES2 export directory.
#
# Catches the classes of landmine seen in production:
#   (1) machine-profile contamination: driver / base.pw.in / submit script
#       carrying paths, NP or scheduler directives from ANOTHER machine
#       -> instant crash at the first QM or MD launch.
#   (2) charge/state inconsistency: TOTCHG vs the net charge of the MM
#       solution vs the electrode point charges (all read from the generated
#       files), a missing or wrong-charge SCF seed, stale restarts,
#       initialqm / FINSTEP traps.
#   (3) campaign-standard drift (optional): k-mesh, QE flags, NP, force-field
#       fingerprints and submit-script requirements declared in a campaign
#       file, so that every system of a dataset is built the same way.
#
# Usage:  ces2_preflight.sh [EXPORT_DIR|SYSTEM_DIR ...]     (default: .)
# Exit:   0 = all PASS,  1 = warnings only,  2 = at least one FAIL.
# Read-only: never modifies files. Run it in every export before sbatch/qsub.
#
# Campaign file (optional): $CES2_CAMPAIGN, else the first campaign.yaml or
# campaign.json found in the export directory or up to 4 parent directories.
# Keys are documented in campaign_example.yaml (repo root). Without a
# campaign file the standards checks are skipped and the values are reported.
#
# Needs: bash, python3 (PyYAML only for a .yaml campaign file; .json needs
# nothing). Override the interpreter with PYBIN=/path/to/python3.

PYBIN="${PYBIN:-python3}"

case "${1:-}" in -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0;; esac

find_campaign() {
  # $1 = export dir. Prints the campaign file path or nothing.
  if [ -n "${CES2_CAMPAIGN:-}" ]; then
    [ -f "$CES2_CAMPAIGN" ] && echo "$CES2_CAMPAIGN"; return
  fi
  local d; d=$(cd "$1" && pwd)
  local i
  for i in 0 1 2 3 4; do
    [ -f "$d/campaign.yaml" ] && { echo "$d/campaign.yaml"; return; }
    [ -f "$d/campaign.json" ] && { echo "$d/campaign.json"; return; }
    d=$(dirname "$d")
  done
}

check_export() {
  local E="$1"
  local S="$E/qmmm_dftces2_charging_pts.sh"
  local nfail=0 nwarn=0

  say()  { printf '  %s\n' "$1"; }
  pass() { say "[PASS] $1"; }
  warn() { say "[WARN] $1"; nwarn=$((nwarn+1)); }
  fail() { say "[FAIL] $1"; nfail=$((nfail+1)); }

  echo "=== preflight: $E ==="
  if [ ! -f "$S" ]; then fail "driver script not found: $S"; RET=2; return; fi

  # ---------- parse driver ----------
  local V; V() { grep -m1 "^$1=" "$S" | sed 's/^[^=]*=//; s/ *#.*//; s/^"//; s/"$//'; }
  local DCP; DCP=$(grep -m1 '^export DFT_CES2_PATH=' "$S" | cut -d'"' -f2)
  local expand; expand() { echo "$1" | sed "s|\${DFT_CES2_PATH}|$DCP|"; }
  local TOTCHG NP INI FIN IQM SKEQ FLAGS
  TOTCHG=$(V TOTCHG); NP=$(V NP); FLAGS=$(V QEPW_FLAGS)
  INI=$(V QMMMINISTEP); FIN=$(V QMMMFINSTEP); IQM=$(V initialqm); SKEQ=$(V skipequil)

  # ---------- 1. host reality: binaries ----------
  local name bin miss=0
  for name in QEPW QEPP LAMMPS CHG2POT MDIPC CHGPLATE; do
    bin=$(expand "$(V $name)")
    [ -z "$bin" ] && continue
    if [ -x "$bin" ]; then :; else fail "$name not executable on this host: $bin"; miss=1; fi
  done
  [ $miss -eq 0 ] && pass "driver binaries all exist & executable"

  # ---------- 2. pseudo_dir + UPFs ----------
  local PD
  PD=$(grep -m1 'pseudo_dir' "$E/base.pw.in" 2>/dev/null | sed "s/.*= *'//; s/'.*//")
  if [ -z "$PD" ]; then warn "no pseudo_dir found in base.pw.in"
  elif [ ! -d "$PD" ]; then fail "pseudo_dir missing on this host: $PD"
  else
    local upf bad=0
    for upf in $(awk '/ATOMIC_SPECIES/{f=1;next} f&&NF>=3{print $3} f&&NF==0{exit}' "$E/base.pw.in"); do
      [ -f "$PD/$upf" ] || { fail "pseudopotential missing: $PD/$upf"; bad=1; }
    done
    [ $bad -eq 0 ] && pass "pseudo_dir + all UPFs present: $PD"
  fi

  # ---------- 3. NP vs scheduler allocation ----------
  local SUB="$E/submit_ces2.sh"
  if [ -f "$SUB" ]; then
    local host_sched="none"
    command -v sbatch >/dev/null 2>&1 && host_sched="slurm"
    [ "$host_sched" = none ] && command -v qsub >/dev/null 2>&1 && host_sched="pbs"
    local has_sb has_pbs
    has_sb=$(grep -c '^#SBATCH' "$SUB"); has_pbs=$(grep -c '^#PBS' "$SUB")
    if [ "$host_sched" = slurm ] && [ "$has_sb" -eq 0 ]; then
      fail "host is SLURM but submit script has no #SBATCH lines (foreign template?)"
    elif [ "$host_sched" = pbs ] && [ "$has_pbs" -eq 0 ]; then
      fail "host is PBS but submit script has no #PBS lines (foreign template?)"
    else
      local ntask
      ntask=$(grep -m1 -E '^#SBATCH (-n|--ntasks)[ =]' "$SUB" | grep -oE '[0-9]+' | head -1)
      if [ -z "$ntask" ]; then
        # PBS: select=<nodes>:ncpus=<c>[:mpiprocs=<m>] -> nodes*mpiprocs (or ncpus)
        ntask=$(grep -m1 -E '^#PBS -l .*select=' "$SUB" | sed -E 's/.*select=([0-9]+):ncpus=([0-9]+)(:mpiprocs=([0-9]+))?.*/\1 \2 \4/' \
                | awk 'NF>=2{m=($3!=""?$3:$2); print $1*m}')
      fi
      if [ -n "$ntask" ] && [ -n "$NP" ]; then
        if [ "$ntask" = "$NP" ]; then pass "NP=$NP == scheduler tasks ($ntask)"
        else fail "NP=$NP != scheduler tasks $ntask (mpirun will over/under-subscribe)"; fi
      else
        warn "could not compare NP=$NP vs submit allocation"
      fi
    fi
  else
    warn "no submit_ces2.sh in export"
  fi

  # ---------- 4–6. charge consistency, campaign standards, SCF seed (python) ----------
  local CAMP; CAMP=$(find_campaign "$E")
  [ -n "$CAMP" ] && say "[INFO] campaign file: $CAMP"
  local pyout
  pyout=$("$PYBIN" - "$E" "$S" "$TOTCHG" "$CAMP" "$NP" "$FLAGS" <<'PYEOF'
import sys, re, os, json, math
E, S, totchg_s, camp, NP, FLAGS = sys.argv[1:7]
out = []
P = lambda m: out.append("[PASS] " + m)
W = lambda m: out.append("[WARN] " + m)
F = lambda m: out.append("[FAIL] " + m)
I = lambda m: out.append("[INFO] " + m)
def done():
    print("\n".join("  " + l for l in out)); sys.exit(0)

drv = open(S).read()
def var(name):
    m = re.search(r'^%s=("?)([^"#\n]*)\1' % re.escape(name), drv, re.M)
    return m.group(2).strip() if m else None
def readf(name):
    p = os.path.join(E, name)
    return open(p).read() if os.path.isfile(p) else None
def num_eq(a, b, tol=1e-6):
    try:
        fa, fb = float(a), float(b)
        return abs(fa - fb) <= tol * max(1.0, abs(fb))
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()

# ---- 4. charge three-consistency, from the generated files ----
try:
    totchg = float(totchg_s)
except ValueError:
    F("TOTCHG unparseable: %r" % totchg_s); done()
scf = None
m = re.search(r'^SUPERCELL=\(([^)]*)\)', drv, re.M)
if m:
    try: scf = math.prod(int(x) for x in m.group(1).split())
    except ValueError: scf = None
if scf is None:
    try: scf = int(float(var("SUPERCELLFACTOR") or ""))
    except ValueError: scf = None
N = None
if scf is None:
    W("cannot read SUPERCELL/SUPERCELLFACTOR from the driver — charge checks skipped")
else:
    N = -totchg * scf          # net charge of the MM solution = -(electrode charge)
    I("TOTCHG=%g per QM cell x supercell %d -> electrode %+g e, solution %+g e" % (totchg, scf, -N, N))

qm_types = set()
m = re.search(r'^QMLMPTYPE=\(([^)]*)\)', drv, re.M)
if m:
    qm_types = {int(x) for x in m.group(1).split() if x.isdigit()}
nsolv = var("NSOLVATOM")
nsolv = int(float(nsolv)) if nsolv else None
elem = []
m = re.search(r'^atoms=\(([^)]*)\)', drv, re.M)
if m: elem = m.group(1).split()

atoms = []   # (id, type, q, x, y, z)
data = readf("data.file")
if data is None:
    F("data.file missing")
else:
    lines = data.split("\n")
    try:
        i0 = next(k for k, l in enumerate(lines) if l.strip().startswith("Atoms"))
        started = False
        for l in lines[i0 + 1:]:
            s = l.split()
            if not s:
                if started: break
                continue
            if not s[0].lstrip("-").isdigit(): break
            started = True
            if len(s) < 7: continue
            atoms.append((int(s[0]), int(s[2]), float(s[3]), float(s[4]), float(s[5]), float(s[6])))
    except StopIteration:
        F("data.file has no Atoms section")

if atoms and N is not None:
    if qm_types:
        is_slab = lambda a: a[1] in qm_types
    elif nsolv is not None:
        is_slab = lambda a: a[0] > nsolv
    else:
        is_slab = None
        W("neither QMLMPTYPE nor NSOLVATOM in driver — cannot split MM/slab atoms")
    if is_slab:
        mm = [a for a in atoms if not is_slab(a)]
        sl = [a for a in atoms if is_slab(a)]
        if nsolv is not None and len(mm) != nsolv:
            W("NSOLVATOM=%d but data.file has %d non-slab atoms" % (nsolv, len(mm)))
        q_mm = sum(a[2] for a in mm); q_sl = sum(a[2] for a in sl)
        # composition report
        bs = readf("build_summary.json")
        comp = None
        if bs:
            try: comp = json.loads(bs).get("counts")
            except ValueError: comp = None
        if comp:
            I("MM composition (build_summary.json): " + ", ".join("%s=%s" % kv for kv in comp.items()))
        else:
            from collections import Counter
            c = Counter(a[1] for a in mm)
            I("MM atoms by type: " + ", ".join("t%d%s=%d" % (t, "(%s)" % elem[t-1] if t-1 < len(elem) else "", n)
                                              for t, n in sorted(c.items())))
        if abs(q_mm - N) < 0.05:
            P("MM solution net charge %+.3f e == +N (TOTCHG x supercell)" % q_mm)
        else:
            F("MM solution net charge %+.3f e != expected %+g e from TOTCHG — wrong ion counts/charges in data.file" % (q_mm, N))
        if abs(q_sl + N) < 0.05:
            P("electrode charge in data.file %+.3f e == -N" % q_sl)
        else:
            F("electrode (slab) charge in data.file %+.3f e != expected %+g e" % (q_sl, -N))
        # top-layer point charges set at runtime by base.in.lammps
        bil = readf("base.in.lammps") or ""
        mq = re.search(r'^\s*set\s+group\s+top\s+charge\s+(\S+)', bil, re.M)
        mr = re.search(r'^\s*region\s+toplayer\s+block\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', bil, re.M)
        n_top = None; src = ""
        if mr:
            try:
                x0, x1, y0, y1, z0, z1 = (float(v) if v.upper() not in ("INF", "EDGE") else None for v in mr.groups())
                def inside(a):
                    return all(lo is None or lo <= v for lo, v in ((x0, a[3]), (y0, a[4]), (z0, a[5]))) and \
                           all(hi is None or v <= hi for hi, v in ((x1, a[3]), (y1, a[4]), (z1, a[5])))
                n_top = sum(1 for a in sl if inside(a)); src = "region toplayer in base.in.lammps"
            except ValueError:
                n_top = None
        if n_top is None:
            cp = readf("charged_system_params.json")
            if cp:
                try: n_top = json.loads(cp).get("n_top_atoms"); src = "charged_system_params.json"
                except ValueError: pass
        if mq is None:
            if abs(N) < 1e-9: I("neutral run: no 'set group top charge' needed")
            else: F("charged run but no 'set group top charge' line in base.in.lammps")
        elif n_top:
            q_top = float(mq.group(1))
            if abs(q_top * n_top + N) < 0.02:
                P("top-layer charge %.6g x %d atoms (%s) = %+.3f == -N" % (q_top, n_top, src, q_top * n_top))
            else:
                F("top-layer charge %.6g x %d atoms (%s) = %+.3f != expected %+g" % (q_top, n_top, src, q_top * n_top, -N))
        else:
            W("cannot count top-layer atoms (no 'region toplayer' or charged_system_params.json) — check 'set group top charge' x n_top = %+g by hand" % -N)

# ---- 5. campaign standards ----
camp_d = {}
if camp:
    txt = open(camp).read()
    try:
        if camp.endswith(".json"):
            camp_d = json.loads(txt)
        else:
            try:
                import yaml
                camp_d = yaml.safe_load(txt) or {}
            except ImportError:
                camp_d = json.loads(txt)   # a JSON-formatted .yaml still works
    except Exception as ex:
        W("campaign file unreadable (%s) — standards checks skipped" % ex); camp_d = {}
    if not isinstance(camp_d, dict):
        W("campaign file is not a mapping — standards checks skipped"); camp_d = {}

pw = readf("base.pw.in") or ""
mk = re.search(r'K_POINTS[^\n]*\n\s*([0-9]+)\s+([0-9]+)\s+([0-9]+)', pw)
kmesh = " ".join(mk.groups()) if mk else None
if "kmesh" in camp_d:
    want = " ".join(str(camp_d["kmesh"]).replace("x", " ").split())
    if kmesh == want: P("k-mesh %s == campaign standard" % kmesh)
    else: F("k-mesh [%s] != campaign standard [%s]" % (kmesh, want))
elif kmesh:
    I("k-mesh %s (no campaign file: not checked)" % kmesh)

if "qepw_flags" in camp_d:
    want = str(camp_d["qepw_flags"] or "").strip()
    if (FLAGS or "").strip() == want: P('QEPW_FLAGS="%s" == campaign' % FLAGS)
    else: F('QEPW_FLAGS="%s" != campaign "%s"' % (FLAGS, want))
elif (FLAGS or "").strip():
    W('QEPW_FLAGS="%s" set — parallel flags gave unreliable results with the CES2-patched pw.x' % FLAGS)

if "np" in camp_d:
    if num_eq(NP, camp_d["np"]): P("NP=%s == campaign" % NP)
    else: W("NP=%s != campaign %s" % (NP, camp_d["np"]))

for fp in camp_d.get("fingerprints", []) or []:
    try:
        fname, pat, exp = fp["file"], fp["pattern"], fp["expect"]
    except (KeyError, TypeError):
        W("fingerprint entry needs file/pattern/expect: %r" % (fp,)); continue
    label = fp.get("about") or "%s: %s" % (fname, pat)
    txt = readf(fname)
    if txt is None:
        F("fingerprint '%s': file %s missing" % (label, fname)); continue
    mm_ = re.search(pat, txt, re.M)
    if not mm_:
        F("fingerprint '%s': pattern not found in %s" % (label, fname)); continue
    got = mm_.group(1) if mm_.groups() else mm_.group(0)
    if num_eq(got, exp): P("fingerprint '%s' = %s" % (label, got))
    else: F("fingerprint '%s' = %s, campaign expects %s" % (label, got, exp))

sub = readf("submit_ces2.sh")
for needle in camp_d.get("submit_must_contain", []) or []:
    if sub is None:
        F("submit_must_contain '%s': no submit_ces2.sh" % needle)
    elif str(needle) in sub:
        P("submit script contains '%s'" % needle)
    else:
        F("submit script lacks '%s' (campaign requirement)" % needle)

# ---- 6. SCF seed ----
mpre = re.search(r"prefix\s*=\s*'([^']*)'", pw); mout = re.search(r"outdir\s*=\s*'([^']*)'", pw)
prefix = mpre.group(1) if mpre else "solute"; outdir = mout.group(1) if mout else "./" + prefix
save = os.path.normpath(os.path.join(E, outdir, prefix + ".save"))
charged = abs(totchg) > 1e-9
mb = re.search(r'TOTCHG != 0.*?then(.*?);\s*else(.*?);\s*fi', drv, re.S)
branch = (mb.group(1) if charged else mb.group(2)) if mb else ""
def opt(name):
    mm_ = re.search(r'%s="?([a-z]+)"?' % name, branch)
    return mm_.group(1) if mm_ else None
spot, swfc = opt("STARTINGPOT"), opt("STARTINGWFC")
if not mb:
    W("cannot find the STARTINGWFC/STARTINGPOT line in the driver — seed check skipped")
else:
    I("%s run: STARTINGWFC=%s STARTINGPOT=%s (save dir %s)" % ("charged" if charged else "neutral", swfc, spot, os.path.relpath(save, E)))
    if spot == "file" or swfc == "file":
        need = ["charge-density.dat", "data-file-schema.xml"]
        upfs = " ".join(re.findall(r'^\s*\S+\s+\S+\s+(\S+\.[uU][pP][fF])', pw, re.M))
        paw = "paw" in upfs.lower()
        if paw or not upfs: need.append("paw.txt")
        missing = [f for f in need if not os.path.getsize(os.path.join(save, f)) > 0] if os.path.isdir(save) else need
        if missing:
            F("seed file(s) missing/empty in %s: %s" % (os.path.relpath(save, E), ", ".join(missing)))
        else:
            P("seed files present (%s)" % ", ".join(need))
        if swfc == "file" and not (os.path.isdir(save) and any(f.startswith("wfc") for f in os.listdir(save))):
            W("STARTINGWFC=file but no wfc*.dat in the save (QE will fail or fall back)")
        xmlp = os.path.join(save, "data-file-schema.xml")
        if os.path.isfile(xmlp):
            x = open(xmlp).read()
            mt = re.search(r'<tot_charge>([^<]*)</tot_charge>', x); mn = re.search(r'<nelec>([^<]*)</nelec>', x)
            seed_tc = float(mt.group(1)) if mt else None
            seed_ne = float(mn.group(1)) if mn else None
            gap_warn = float(camp_d.get("seed_gap_warn_e", 2.0))
            if seed_tc is None:
                W("seed xml has no <tot_charge> — cannot verify the seed's charge state")
            else:
                d_unit = seed_tc - totchg
                d_tot = d_unit * (scf or 1)
                if abs(d_tot) < 0.01:
                    P("seed charge state matches (tot_charge %g per cell%s)" % (seed_tc, ", nelec %g" % seed_ne if seed_ne else ""))
                elif charged and abs(seed_tc) < 1e-9:
                    F("seed is a NEUTRAL density (tot_charge 0) for a charged run — converges badly; use a density of the same or a neighbouring charge state")
                elif abs(d_tot) <= gap_warn:
                    W("seed from a neighbouring charge state (%+.2f e total vs this run); QE renormalizes nelec — usually fine" % d_tot)
                else:
                    W("seed charge far from this run (%+.2f e total): noise-floor SCF wall likely — prefer a same-charge density (any cation) or within %g e" % (d_tot, gap_warn))
            q0 = readf(os.path.join("qm_0", "pw.out"))
            if q0 and seed_ne is not None:
                m0 = re.search(r'number of electrons\s*=\s*([0-9.]+)', q0)
                if m0:
                    exp_ne = float(m0.group(1)) - totchg
                    I("nelec check: seed %.3f vs expected %.3f (= neutral %.2f - TOTCHG)" % (seed_ne, exp_ne, float(m0.group(1))))
    elif charged:
        W("charged run with STARTINGPOT=%s (atomic start: expect a very slow or oscillating SCF — seed it, see docs/05 §6)" % spot)
done()
PYEOF
)
  printf '%s\n' "$pyout"
  nfail=$((nfail + $(printf '%s' "$pyout" | grep -c '\[FAIL\]')))
  nwarn=$((nwarn + $(printf '%s' "$pyout" | grep -c '\[WARN\]')))

  # ---------- 7. step-state sanity ----------
  say "[INFO] steps: INISTEP=$INI FINSTEP=$FIN initialqm=$IQM skipequil=$SKEQ"
  [ -n "$INI" ] && [ -n "$FIN" ] && [ "$INI" -gt "$FIN" ] && fail "INISTEP > FINSTEP"
  [ "$IQM" = 1 ] && [ "$INI" = "$FIN" ] && fail "INISTEP==FINSTEP with initialqm=1: job would do NO new QM"
  if [ "$INI" = 0 ] && [ "$IQM" = 1 ]; then
    # at step 0 the driver rebuilds root grid cubes from qm_0/ unconditionally
    # (MM-prep else-branch: cp qm_0/*.cube . ; solute.pot/ind from ref_solute.pot)
    # so only the qm_0/ archive is load-bearing (checked below).
    say "[INFO] qm_0-skip at step 0: driver rebuilds root grid cubes from qm_0/ (safe)"
  fi
  if [ -n "$INI" ] && [ "$INI" -gt 0 ] && [ "$IQM" = 1 ]; then
    # skip-QM resume at step N>0 reuses ROOT solute.pot/ind cubes as-is
    # (no ref_solute.* in qm_N archives to rebuild from): a kill during QM
    # post-processing leaves stale ones behind.
    if [ -f "$E/solute.pot.cube" ]; then
      warn "initialqm=1 resume at step $INI reuses ROOT solute.pot/ind.cube as-is — verify they came from qm_$INI's completed postprocessing (stale-cube hazard)"
    else
      fail "initialqm=1 at step $INI but no root solute.pot.cube to reuse"
    fi
  fi
  # charged QM steps subtract against qm_0/ref_solute.* explicitly
  if [ -f "$E/qm_0/ref_solute.rho.cube" ] && [ -f "$E/qm_0/ref_solute.pot.cube" ]; then
    pass "qm_0/ref_solute.{rho,pot}.cube present (charged-QM reference)"
  else
    fail "qm_0/ref_solute.{rho,pot}.cube missing — charged QM subtraction will crash"
  fi
  if [ -n "$INI" ] && [ "$INI" -gt 0 ]; then
    if [ "$IQM" = 0 ]; then
      ls "$E"/c_rho_*.cube >/dev/null 2>&1 \
        && pass "continuation QM input: c_rho_* ensemble cubes present" \
        || fail "INISTEP=$INI starts with QM but no c_rho_*.cube ensemble at root"
    fi
    local nres
    nres=$(ls "$E"/ces2.*.restart 2>/dev/null | wc -l)
    if [ "$nres" -eq 0 ]; then
      fail "INISTEP=$INI but no ces2.*.restart — MD would silently reset to initial structure"
    elif [ "$nres" -gt 1 ]; then
      local bynum bymtime
      bynum=$(ls "$E"/ces2.*.restart | sed 's/.*ces2\.//; s/\.restart//' | sort -n | tail -1)
      bymtime=$(ls -t "$E"/ces2.*.restart | head -1 | sed 's/.*ces2\.//; s/\.restart//')
      if [ "$bynum" = "$bymtime" ]; then
        pass "restarts: newest mtime == highest step ($bynum)"
      else
        fail "restart mtime scrambled: highest step $bynum but newest mtime $bymtime — mtime-latest fallback would resume from the WRONG restart"
      fi
    else
      pass "single restart present: $(basename "$(ls "$E"/ces2.*.restart)")"
    fi
    local LR; LR=$(V LAMMPSRESTART)
    if [ -n "$LR" ] && [ ! -f "$E/$LR" ]; then
      warn "LAMMPSRESTART=\"$LR\" does not exist — driver falls back to the newest root restart, or to data.file if none (docs/05 §5.3)"
    fi
  fi

  # ---------- 8. stale derived artifacts ----------
  local a
  for a in data.md in.lammps in.lammps.equil; do
    [ -f "$E/$a" ] && say "[INFO] $a present (driver regenerates it — stale copy is inert but confusing)"
  done

  # ---------- summary ----------
  if [ $nfail -gt 0 ]; then
    echo "  >>> RESULT: FAIL ($nfail fail, $nwarn warn) — do NOT submit"
    [ "$RET" -lt 2 ] && RET=2
  elif [ $nwarn -gt 0 ]; then
    echo "  >>> RESULT: PASS with $nwarn warning(s)"
    [ "$RET" -lt 1 ] && RET=1
  else
    echo "  >>> RESULT: PASS"
  fi
}

RET=0
[ $# -eq 0 ] && set -- .
for d in "$@"; do
  # accept either the system dir or the export dir itself
  if [ -d "$d/export" ]; then check_export "$d/export"; else check_export "$d"; fi
  echo
done
exit $RET
