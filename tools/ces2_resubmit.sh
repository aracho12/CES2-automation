#!/usr/bin/env bash
#
# ces2_resubmit.sh — Auto-detect DFT-CES2 QM/MM progress, patch config, and resubmit.
#
# Run inside a CES2 calculation directory containing:
#   - qmmm_dftces2_charging_pts.sh
#   - submit_ces2.sh
#   - qm_*/  and  mm_*/  subdirs from previous runs (if any)
#
# Behavior:
#   1. Scan qm_N/pw.out for "JOB DONE"  → last completed QM step
#   2. Scan mm_N/*.restart               → last completed MM step
#   3. Patch qmmm_dftces2_charging_pts.sh with the right QMMMINISTEP / initialqm
#   4. Patch submit_ces2.sh to skip relax stages that already produced dumps
#   5. qsub submit_ces2.sh (unless --dry-run)
#
# Usage:
#   ces2_resubmit.sh                       # detect, patch, qsub (one-shot)
#   ces2_resubmit.sh --dry-run             # detect+patch only, no qsub
#   ces2_resubmit.sh --continue            # also inject SIGTERM trap into
#                                          # submit_ces2.sh so PBS walltime
#                                          # kill auto-resubmits the chain
#   ces2_resubmit.sh --continue --max=N    # cap chain retries (default 5)
#   ces2_resubmit.sh --no-continue         # remove the trap (stop the chain)

set -euo pipefail

DRYRUN=0
CONTINUE=0
NO_CONTINUE=0
MAX_RESUBMIT=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)         DRYRUN=1; shift ;;
    --continue)        CONTINUE=1; shift ;;
    --no-continue)     NO_CONTINUE=1; shift ;;
    --max=*)           MAX_RESUBMIT="${1#--max=}"; shift ;;
    --max)             MAX_RESUBMIT="${2:?--max needs an integer}"; shift 2 ;;
    -h|--help)
      sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -e 's/^# \{0,1\}//' -e '/^set -euo/d'
      exit 0 ;;
    *)
      echo "ERROR: unknown option: $1" >&2; exit 2 ;;
  esac
done

if (( CONTINUE == 1 && NO_CONTINUE == 1 )); then
  echo "ERROR: --continue and --no-continue are mutually exclusive." >&2
  exit 2
fi
if ! [[ "$MAX_RESUBMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --max must be a positive integer, got '$MAX_RESUBMIT'" >&2
  exit 2
fi

# Absolute path to this script — embedded into the trap block so the chain
# survives different working directories on compute nodes.
SELF_ABS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"

# ---- Log all output to file (and still print to terminal) ----
LOGFILE="ces2_resubmit_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

QMMM="qmmm_dftces2_charging_pts.sh"
SUB="submit_ces2.sh"

for f in "$QMMM" "$SUB"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: $f not found in $(pwd)" >&2
    exit 1
  fi
done

# ---------- 1. Detect last completed qm_N ----------
last_qm=-1
for d in qm_*/; do
  [ -d "$d" ] || continue
  n=${d%/}; n=${n#qm_}
  [[ "$n" =~ ^[0-9]+$ ]] || continue
  if [ -f "${d}pw.out" ] && grep -q "JOB DONE" "${d}pw.out"; then
    (( n > last_qm )) && last_qm=$n
  fi
done

# ---------- 2. Detect last completed mm_N ----------
last_mm=-1
for d in mm_*/; do
  [ -d "$d" ] || continue
  n=${d%/}; n=${n#mm_}
  [[ "$n" =~ ^[0-9]+$ ]] || continue
  if ls "${d}"*.restart >/dev/null 2>&1; then
    (( n > last_mm )) && last_mm=$n
  fi
done

echo "==[ State detected ]=="
echo "   last completed qm_N : $last_qm"
echo "   last completed mm_N : $last_mm"

# ---------- 3. Decide QMMMINISTEP / initialqm ----------
if (( last_qm == -1 && last_mm == -1 )); then
  qmmmini=0;  initialqm=0
  reason="fresh start"
elif (( last_qm >= 0 && last_mm == last_qm - 1 )); then
  # qm_N done, mm_N not → resume mm_N (skip re-running qm_N)
  qmmmini=$last_qm
  initialqm=1
  reason="qm_$last_qm done, mm_$last_qm pending"
elif (( last_qm >= 0 && last_qm == last_mm )); then
  # Both qm_N and mm_N done → start qm_(N+1)
  qmmmini=$(( last_qm + 1 ))
  initialqm=0
  reason="qm_$last_qm/mm_$last_mm done, start qm_$qmmmini"
else
  echo "ERROR: inconsistent state (last_qm=$last_qm, last_mm=$last_mm)." >&2
  echo "       expected last_mm == last_qm or last_qm - 1." >&2
  exit 1
fi

# ---------- 3.5 Sanity-check QE restart files when qmmmini > 0 ----------
# When qmmmini > 0, qmmm_dftces2_charging_pts.sh injects
#   startingpot = 'file'
#   startingwfc = 'file'
# into pw.in, which makes QE try to read <outdir>/<prefix>.xml (or
# <outdir>/<prefix>.save/data-file-schema.xml). If those files are gone
# (e.g. scratch cleanup, manual deletion), QE aborts with
#   "Error in routine pw_readschemafile (1): xml data file not found".
# Detect that case and fall back to a fresh start from qm_0.
if (( qmmmini > 0 )); then
  qe_in=""
  for cand in base.pw.in pw.in; do
    [ -f "$cand" ] && { qe_in="$cand"; break; }
  done
  if [ -n "$qe_in" ]; then
    qe_prefix=$(grep -E "^[[:space:]]*prefix[[:space:]]*=" "$qe_in" | head -1 \
      | sed -E "s/.*=[[:space:]]*['\"]([^'\"]+)['\"].*/\1/")
    qe_outdir=$(grep -E "^[[:space:]]*outdir[[:space:]]*=" "$qe_in" | head -1 \
      | sed -E "s/.*=[[:space:]]*['\"]([^'\"]+)['\"].*/\1/")
    qe_outdir="${qe_outdir%/}"
    if [ -n "$qe_prefix" ] && [ -n "$qe_outdir" ]; then
      xml_a="${qe_outdir}/${qe_prefix}.xml"
      xml_b="${qe_outdir}/${qe_prefix}.save/data-file-schema.xml"
      if [ ! -f "$xml_a" ] && [ ! -f "$xml_b" ]; then
        echo "   ! QE restart missing under '$qe_outdir' (prefix='$qe_prefix')"
        echo "     neither $xml_a nor $xml_b exists."
        echo "     Falling back to fresh start from qm_0 (cannot resume SCF without it)."
        qmmmini=0
        initialqm=0
        reason="$reason; QE restart missing → fresh start from qm_0"
      fi
    else
      echo "   ! Could not parse prefix/outdir from $qe_in — skipping QE restart check."
    fi
  fi
fi

echo "   → $reason"
echo "   → QMMMINISTEP=$qmmmini, initialqm=$initialqm"

# Also sanity-check we have the cube inputs needed for qm_(qmmmini) when qmmmini>0.
# The script at line ~245 copies mm_$((qmmmini-1))/{MOBILE_final,empty,repA}.cube
if (( qmmmini > 0 )); then
  prev="mm_$((qmmmini-1))"
  for c in MOBILE_final.cube empty.cube repA.cube; do
    if [ ! -f "$prev/$c" ]; then
      echo "WARNING: $prev/$c missing — qm_$qmmmini will fail to start."
    fi
  done
fi

# ---------- 4. Patch qmmm script ----------
ts=$(date +%Y%m%d_%H%M%S)
cp "$QMMM" "${QMMM}.bak.${ts}"
sed -i.tmp "s|^QMMMINISTEP=.*|QMMMINISTEP=$qmmmini # no of initial QMMM step|" "$QMMM"
sed -i.tmp "s|^initialqm=.*|initialqm=$initialqm #1, when the initial qm has been done.|" "$QMMM"
rm -f "${QMMM}.tmp"

# ---------- 5. Patch submit script (skip relax if it contains relax steps) ----------
# Only patch when the submit file actually has raw `mpirun ... in.relax_*` lines.
# Some workflows already strip them, in which case there is nothing to do.
if grep -q '^mpirun -np \$NP \$LMP -in in\.relax_' "$SUB"; then
  cp "$SUB" "${SUB}.bak.${ts}"
  # Idempotent guards: wrap each `mpirun ... in.relax_X` with `[ -f X.dump ] ||`.
  # If already wrapped (from a prior resubmit) the regex won't match, so no change.
  # Use `#` delimiter so `||` in the pattern is literal (no escaping needed in BRE).
  sed -i.tmp \
    's#^mpirun -np \$NP \$LMP -in in\.relax_min || true$#[ -f minimized.dump ] || mpirun -np $NP $LMP -in in.relax_min || true#' \
    "$SUB"
  sed -i.tmp \
    's#^mpirun -np \$NP \$LMP -in in\.relax_heat || true$#[ -f heated.dump ] || mpirun -np $NP $LMP -in in.relax_heat || true#' \
    "$SUB"
  sed -i.tmp \
    's#^mpirun -np \$NP \$LMP -in in\.relax_equil || true$#[ -f equilibrated.dump ] || mpirun -np $NP $LMP -in in.relax_equil || true#' \
    "$SUB"
  rm -f "${SUB}.tmp"
  sub_patched=1
else
  sub_patched=0
fi

echo
echo "==[ Patched ]=="
echo "   $QMMM (backup: ${QMMM}.bak.${ts})"
grep -E "^(QMMMINISTEP|QMMMFINSTEP|initialqm|skipequil|firstrun)=" "$QMMM" | sed 's/^/     /' || true
if (( sub_patched == 1 )); then
  echo "   $SUB  (backup: ${SUB}.bak.${ts})"
  grep -n 'in\.relax_' "$SUB" | sed 's/^/     /' || true
else
  echo "   $SUB  (no relax lines found — left as-is)"
fi

# ---------- 5.5 Auto-resubmit chain (--continue / --no-continue) ----------
# When --continue is given, inject a SIGTERM trap into submit_ces2.sh that
# fires when PBS sends SIGTERM (typically on walltime exceeded) and re-runs
# this script with --continue, perpetuating the chain. A counter file caps
# total retries; on clean script exit the counter is reset.
TRAP_BEGIN='# >>> CES2_AUTORESUBMIT >>> (managed by ces2_resubmit.sh, do not edit by hand)'
TRAP_END='# <<< CES2_AUTORESUBMIT <<<'

remove_trap_block() {
  # Strip any existing trap block from $SUB (idempotent).
  if grep -qF "$TRAP_BEGIN" "$SUB"; then
    sed -i.tmp "\|^${TRAP_BEGIN}\$|,\|^${TRAP_END}\$|d" "$SUB"
    rm -f "${SUB}.tmp"
    return 0
  fi
  return 1
}

inject_trap_block() {
  # Always strip first so re-injection picks up new --max etc.
  remove_trap_block || true

  # Build the block in a temp file. Single-quoted heredoc keeps $vars literal
  # in the output; the path/max are substituted via printf afterwards.
  local tmpf
  tmpf="$(mktemp)"
  cat >"$tmpf" <<EOF
${TRAP_BEGIN}
__CES2_RESUBMIT_SCRIPT="${SELF_ABS}"
__CES2_MAX_RESUBMIT=${MAX_RESUBMIT}
__CES2_COUNT_FILE=".resubmit_count"
__cesresub_via_term=0
__cesresub_term_handler() {
    __cesresub_via_term=1
    local count
    count=\$(cat "\$__CES2_COUNT_FILE" 2>/dev/null || echo 0)
    if [ "\$count" -lt "\$__CES2_MAX_RESUBMIT" ]; then
        echo "[\$(date)] SIGTERM (likely walltime) — auto-resubmit \$((count+1))/\$__CES2_MAX_RESUBMIT" >&2
        echo \$((count+1)) > "\$__CES2_COUNT_FILE"
        bash "\$__CES2_RESUBMIT_SCRIPT" --continue --max="\$__CES2_MAX_RESUBMIT" \\
            >> resubmit_chain.log 2>&1 || true
    else
        echo "[\$(date)] Hit MAX_RESUBMIT=\$__CES2_MAX_RESUBMIT — chain stopped." >&2
    fi
    exit 143
}
__cesresub_exit_handler() {
    if [ "\$__cesresub_via_term" -eq 0 ]; then
        rm -f "\$__CES2_COUNT_FILE"
    fi
}
trap __cesresub_term_handler TERM
trap __cesresub_exit_handler EXIT
${TRAP_END}
EOF

  # Insert the block right after "cd \$curr_dir" (or "cd \${PBS_O_WORKDIR}").
  local anchor_line
  anchor_line=$(grep -nE '^[[:space:]]*cd[[:space:]]+(\$curr_dir|\$\{?PBS_O_WORKDIR\}?)[[:space:]]*$' "$SUB" | head -1 | cut -d: -f1)
  if [ -z "$anchor_line" ]; then
    echo "ERROR: could not find 'cd \$curr_dir' anchor in $SUB — trap NOT injected." >&2
    rm -f "$tmpf"
    return 1
  fi
  sed -i.tmp "${anchor_line}r ${tmpf}" "$SUB"
  rm -f "${SUB}.tmp" "$tmpf"
  return 0
}

if (( CONTINUE == 1 )); then
  echo
  echo "==[ Auto-resubmit chain: ON (--continue) ]=="
  if [ -z "${sub_patched_for_chain:-}" ]; then
    cp -p "$SUB" "${SUB}.bak.chain.${ts}" 2>/dev/null || true
  fi
  if inject_trap_block; then
    echo "   trap injected into $SUB  (max retries: $MAX_RESUBMIT)"
    echo "   counter file: .resubmit_count  (auto-reset on clean exit)"
    echo "   chain log:    resubmit_chain.log"
  else
    echo "   trap NOT injected (see error above)."
    exit 1
  fi
elif (( NO_CONTINUE == 1 )); then
  echo
  echo "==[ Auto-resubmit chain: OFF (--no-continue) ]=="
  if remove_trap_block; then
    echo "   trap removed from $SUB."
  else
    echo "   no trap was present in $SUB."
  fi
  rm -f .resubmit_count
fi

# ---------- 6. Submit ----------
if (( DRYRUN == 1 )); then
  echo
  echo "Dry run: skipping qsub."
  exit 0
fi

if ! command -v qsub >/dev/null 2>&1; then
  echo
  echo "WARNING: qsub not found on this machine. Patches applied; submit manually on the HPC."
  exit 0
fi

echo
echo "==[ Submitting ]=="
qsub "$SUB"
