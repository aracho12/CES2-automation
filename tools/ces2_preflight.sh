#!/bin/bash
# ces2_preflight.sh — pre-submit audit for a CES2 export directory.
#
# Catches the two classes of landmine seen in production:
#   (1) machine-profile contamination: driver/base.pw.in carrying paths,
#       NP, or scheduler directives from ANOTHER server (config copied from
#       a foreign-era template) -> instant crash at the first QM/MD launch.
#   (2) charge/state inconsistency: TOTCHG vs ion imbalance vs electrode
#       charge, missing SCF seed, stale restarts that resurrect old state.
#
# Usage:  ces2_preflight.sh [EXPORT_DIR ...]     (default: .)
# Exit:   0 = all PASS,  1 = warnings only,  2 = at least one FAIL.
# Run it in every export before sbatch/qsub. Read-only: never modifies files.

PYBIN="${PYBIN:-python3}"

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
  local TOTCHG NP INI FIN IQM SKEQ
  TOTCHG=$(V TOTCHG); NP=$(V NP)
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
  local FLAGS; FLAGS=$(V QEPW_FLAGS)
  [ -n "$FLAGS" ] && warn "QEPW_FLAGS=\"$FLAGS\" set — verify npool/ndiag fit the allocation"

  # ---------- 4. charge three-consistency ----------
  local pyout
  pyout=$("$PYBIN" - "$E" "$TOTCHG" "$S" <<'PYEOF'
import sys, re
E, totchg_s, S = sys.argv[1], sys.argv[2], sys.argv[3]
out = []
try:
    totchg = float(totchg_s)
except Exception:
    print("  [FAIL] TOTCHG unparseable: %r" % totchg_s); sys.exit(0)
N = -totchg * 18.0
frac = abs(N - round(N)) > 1e-3
# cation / OH-O types from MMrepA element list (fallback 3/5)
cat_t, oho_t = 3, 5
m = re.search(r'^MMrepA=\(([^)]*)\)', open(S).read(), re.M)
if m:
    els = m.group(1).split()
    for i, e in enumerate(els, 1):
        if e in ("Li", "Na", "K", "Cs"):
            cat_t = i
    o_idx = [i for i, e in enumerate(els, 1) if e == "O" and i > cat_t]
    if o_idx: oho_t = o_idx[-1]
try:
    lines = open(f"{E}/data.file").read().split("\n")
    i = next(k for k, l in enumerate(lines) if l.strip().startswith("Atoms"))
    from collections import Counter
    c = Counter()
    for l in lines[i+1:]:
        s = l.split()
        if not s:
            if c: break
            continue
        if s[0].isdigit(): c[int(s[2])] += 1
    net = c.get(cat_t, 0) - c.get(oho_t, 0)
    if abs(net - N) < 0.5 or (frac and net == int(-(-N // 1))):  # frac: ceil
        out.append(f"[PASS] ion imbalance +{net} consistent with TOTCHG={totchg_s} (N={N:g})")
    else:
        out.append(f"[FAIL] ion imbalance +{net} (cat t{cat_t}={c.get(cat_t,0)}, OH t{oho_t}={c.get(oho_t,0)}) vs TOTCHG -> N={N:g}")
except Exception as ex:
    out.append(f"[WARN] data.file ion check failed: {ex}")
try:
    q = None
    for l in open(f"{E}/base.in.lammps"):
        mm = re.search(r"set\s+group\s+top\s+charge\s+(\S+)", l)
        if mm: q = float(mm.group(1)); break
    if q is None:
        out.append("[WARN] no 'set group top charge' in base.in.lammps")
    else:
        best = [(abs(q*n + N), n) for n in (288, 144, 96, 576)]
        d, n = min(best)
        if d < 0.02:
            out.append(f"[PASS] electrode charge {q:.6g} x {n} = {q*n:+.3f} == -N")
        else:
            out.append(f"[FAIL] electrode charge {q:.6g}: no top-atom count in (288,144,96,576) gives -N={-N:g} (closest x{n}={q*n:+.3f})")
except Exception as ex:
    out.append(f"[WARN] base.in.lammps check failed: {ex}")
print("\n".join("  " + l for l in out))
PYEOF
)
  printf '%s\n' "$pyout"
  nfail=$((nfail + $(printf '%s' "$pyout" | grep -c '\[FAIL\]')))
  nwarn=$((nwarn + $(printf '%s' "$pyout" | grep -c '\[WARN\]')))

  # ---------- 5. step-state sanity ----------
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
    # (no ref_solute.* in qm_N archives to rebuild from) — the 8/12 Li
    # incident: a kill during QM postprocessing leaves stale ones behind.
    if [ -f "$E/solute.pot.cube" ]; then
      warn "initialqm=1 resume at step $INI reuses ROOT solute.pot/ind.cube as-is — verify they came from qm_$INI's completed postprocessing (stale-cube hazard)"
    else
      fail "initialqm=1 at step $INI but no root solute.pot.cube to reuse"
    fi
  fi
  # charged QM steps subtract against qm_0/ref_solute.* explicitly (driver l.455/464)
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
  fi

  # ---------- 6. SCF seed for charged systems ----------
  local charged
  charged=$("$PYBIN" -c "print(1 if abs(float('$TOTCHG'))>1e-9 else 0)" 2>/dev/null)
  if [ "$charged" = 1 ]; then
    local sp
    sp=$(grep -m1 'TOTCHG != 0' "$S" | grep -oE 'then [^;]*; STARTINGPOT="[a-z]*"' | grep -oE '"[a-z]*"$' | tr -d '"')
    if [ "$sp" = file ]; then
      local f missing=0
      for f in charge-density.dat data-file-schema.xml paw.txt; do
        [ -s "$E/solute/solute.save/$f" ] || { fail "seed file missing/empty: solute/solute.save/$f"; missing=1; }
      done
      [ $missing -eq 0 ] && pass "charged run: STARTINGPOT=file + seed 3/3 present"
      say "[INFO] seed provenance is NOT checkable here — confirm source SCF was JOB DONE"
    else
      warn "charged run but STARTINGPOT=$sp (atomic start: expect very slow/oscillating SCF)"
    fi
  fi

  # ---------- 7. stale derived artifacts ----------
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
