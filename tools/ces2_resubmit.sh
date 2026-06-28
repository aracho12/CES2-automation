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
#      (and skipequil=1 when a pending mm_N was killed during its averaging phase,
#       so the already-finished equilibration is not needlessly rerun).
#      Also point LAMMPSRESTART at the latest *.restart (recovering it from
#      mm_(N-1)/ when the work dir was purged) so a resumed MM step continues the
#      MD trajectory instead of restarting from the initial structure. This is
#      needed for qmmm scripts generated before the runtime resume-safety patch,
#      which on a fresh resubmit reset LAMMPSRESTART to its init value
#      "ini.restart" and silently discarded MD progress.
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
#   ces2_resubmit.sh --patch-only          # detect+patch only; NO qsub, NO trap.
#                                          # Used by submit_ces2.sh's depend=
#                                          # afterany self-chain at job startup.
#                                          # Writes .ces2_chain_done if all
#                                          # QMMM steps are complete.

set -euo pipefail

DRYRUN=0
CONTINUE=0
NO_CONTINUE=0
PATCH_ONLY=0
MAX_RESUBMIT=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)         DRYRUN=1; shift ;;
    --continue)        CONTINUE=1; shift ;;
    --no-continue)     NO_CONTINUE=1; shift ;;
    --patch-only)      PATCH_ONLY=1; shift ;;
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
if (( PATCH_ONLY == 1 )) && (( CONTINUE == 1 || NO_CONTINUE == 1 || DRYRUN == 1 )); then
  echo "ERROR: --patch-only cannot be combined with --continue/--no-continue/--dry-run." >&2
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
  # A qm_N step is COMPLETE only when its ortho QE run finished (qm_N/pw.out
  # "JOB DONE"). For N>0 the step then runs a *non-ortho* QE pass, which is what
  # actually produces the solute.pot.cube / solute.ind.cube fed to the next MM
  # step (copied into qm_N/ only at the very end). qm_0 has no non-ortho pass
  # (skipped), so require it only for N>0. Without this, a job killed after the
  # ortho stage copied pw.out into qm_N/ but before non-ortho finished would be
  # mis-detected as "qm_N done", jump to mm_N with initialqm=1, and silently
  # reuse the stale qm_0 cubes — giving a wrong potential from mm_N onward.
  [ -f "${d}pw.out" ] && grep -q "JOB DONE" "${d}pw.out" || continue
  if (( n > 0 )); then
    [ -f "${d}pw.nonortho.out" ] && grep -q "JOB DONE" "${d}pw.nonortho.out" || continue
  fi
  (( n > last_qm )) && last_qm=$n
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
# skipequil: when resuming a pending mm_N whose equilibration already finished
# (job killed during the averaging phase), skip the multi-million-step
# equilibration on the resumed step. Decided in §3.4 below. mm_pending marks the
# "qm_N done, mm_N pending" branch that §3.4 inspects.
skipequil=0
mm_pending=0
if (( last_qm == -1 && last_mm == -1 )); then
  qmmmini=0;  initialqm=0
  reason="fresh start"
elif (( last_qm >= 0 && last_mm == last_qm - 1 )); then
  # qm_N done, mm_N not → resume mm_N (skip re-running qm_N)
  qmmmini=$last_qm
  initialqm=1
  mm_pending=1
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
# The QE restart (solute/ save dir) is only read when qmmm_dftces2_charging_pts.sh
# injects startingwfc/startingpot = 'file'. The script does that in the TOTCHG==0
# else-branch *only if* that branch still assigns STARTINGPOT="file"; charged runs
# (TOTCHG != 0) always inject 'atomic'. The option-B fix patches the neutral branch
# to 'atomic' too, so detect the ACTUAL assignment in $QMMM rather than assuming
# TOTCHG==0 always means file-restart — otherwise a patched neutral run with no
# solute/ save dir would be forced into a spurious (and expensive) fall-back to qm_0.
totchg=$(grep -E '^[[:space:]]*TOTCHG=' "$QMMM" | head -1 \
  | sed -E "s/^[[:space:]]*TOTCHG=[\"']?([^\"'[:space:]#]*).*/\1/")
if [ "$(awk -v c="$totchg" 'BEGIN{print (c+0==0)?1:0}')" = "1" ] \
   && grep -q 'STARTINGPOT="file"' "$QMMM"; then
  uses_file_restart=1   # TOTCHG==0 and the script still injects startingpot='file'
else
  uses_file_restart=0   # charged run, or neutral branch patched to 'atomic'
fi
if (( qmmmini > 0 )) && [ "$uses_file_restart" = "1" ]; then
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
        mm_pending=0
        reason="$reason; QE restart missing → fresh start from qm_0"
      fi
    else
      echo "   ! Could not parse prefix/outdir from $qe_in — skipping QE restart check."
    fi
  fi
elif (( qmmmini > 0 )); then
  echo "   QE startingwfc/startingpot = 'atomic' (TOTCHG=${totchg:-?}) — solute/ QE save dir not needed; skipping QE restart check."
fi

# ---------- 3.4 Detect kill-during-averaging → skip redundant equilibration ----------
# Each MM step runs two LAMMPS phases in the work dir, in order:
#   1) equilibration (emxext) → lammps.equil.out, writes *.restart
#   2) averaging     (emdext) → lammps.average.out, then results copied to mm_N/
# mm_N only counts as "done" once averaging copies its restart into mm_N/. So in
# the "mm_N pending" branch the job may have been killed *during averaging*, with
# equilibration already finished. Re-running the multi-million-step equilibration
# would waste hours, so detect that case and set skipequil=1 for the resumed step.
# Signals are content-based — LAMMPS prints "Total wall time" only on a clean run
# end — so they survive rsync'd / touched dirs where mtimes are unreliable.
if (( mm_pending == 1 )); then
  equil_done=0
  avg_done=0
  if [ -f lammps.equil.out ]   && grep -q "Total wall time" lammps.equil.out;   then equil_done=1; fi
  if [ -f lammps.average.out ] && grep -q "Total wall time" lammps.average.out; then avg_done=1;   fi
  if [ -f qmmm.out ]; then
    last_phase=$(grep -aE 'Running LAMMPS\((emxext|emdext)\)' qmmm.out | tail -1 || true)
    [ -n "$last_phase" ] && echo "   last work-dir LAMMPS phase: $last_phase"
  fi
  if (( equil_done == 1 && avg_done == 0 )); then
    if ls *.restart >/dev/null 2>&1; then
      skipequil=1
      reason="$reason; equilibration complete, killed during averaging → skip equil"
    else
      echo "   ! equilibration looks complete but no *.restart in work dir —"
      echo "     cannot skip equil safely; leaving skipequil=0 (equilibration will rerun)."
    fi
  fi
fi

echo "   → $reason"
echo "   → QMMMINISTEP=$qmmmini, initialqm=$initialqm, skipequil=$skipequil"

# ---------- 3.6 Completion detection ----------
# Read QMMMFINSTEP from qmmm script. If the next step to run is past the final
# step, the simulation is done — write a marker so submit_ces2.sh's chain step
# can stop the depend=afterany chain.
qmmmfin=$(grep -E "^[[:space:]]*QMMMFINSTEP=" "$QMMM" | head -1 \
          | sed -E 's/^[[:space:]]*QMMMFINSTEP=([0-9]+).*/\1/' || true)
is_complete=0
if [ -n "$qmmmfin" ] && (( qmmmini > qmmmfin )); then
  is_complete=1
  echo "   → All QMMM steps complete (next would be qm_$qmmmini > QMMMFINSTEP=$qmmmfin)."
  : > .ces2_chain_done
  echo "   → Wrote .ces2_chain_done marker."
fi

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

# ---------- 3.7 Resolve LAMMPSRESTART for the resumed step ----------
# Old qmmm scripts (generated before the runtime "resume safety" patch) do NOT
# re-derive $LAMMPSRESTART at runtime: on a fresh resubmit the variable resets to
# its init value "ini.restart", so a step>0 equilibration would restart from the
# initial structure (ini.restart/data.file) and silently discard MD progress.
# Patch the qmmm script's LAMMPSRESTART= line to point at the latest restart so
# the resumed MM step continues the existing trajectory. Harmless for new scripts:
# their runtime block re-derives LAMMPSRESTART for step>0 regardless.
pick_latest_restart() {
  # Print the basename of the most-recently-modified *.restart in directory $1
  # (default "."). Selection is by mtime (`ls -t`), NOT the embedded timestep:
  # pre-patch qmmm scripts do not accumulate restarts — they reuse the same
  # 0..N timestep range every MM step — so the filename timestep is not monotonic
  # across steps and mtime is the only reliable "latest". When recovering a
  # restart by hand, copy (or `touch`) the intended one last so it is newest.
  local dir="${1:-.}"
  local latest
  latest=$(ls -1t "$dir"/*.restart 2>/dev/null | head -n 1 || true)
  [ -n "$latest" ] && basename "$latest"
  # Always succeed: an empty result is normal (no restart yet) and must not trip
  # `set -e` when captured via  rfile=$(pick_latest_restart ...).
  return 0
}

restart_patch=""
if (( qmmmini == 0 )); then
  # Fresh start: reset to the init value in case a prior resubmit patched it.
  restart_patch="ini.restart"
  echo "   LAMMPSRESTART → ini.restart (fresh start)"
else
  rfile=$(pick_latest_restart ".")
  if [ -z "$rfile" ]; then
    # Work dir purged (scratch cleanup / rsync to a fresh dir): recover the latest
    # restart archived under the previous step's mm_(N-1)/ and copy it back so both
    # `-r $LAMMPSRESTART` and the script's `ls *.restart` timestep parse see it.
    prev_mm="mm_$((qmmmini-1))"
    cand=$(pick_latest_restart "$prev_mm")
    if [ -n "$cand" ]; then
      cp "$prev_mm/$cand" ./
      rfile="$cand"
      echo "   restart: none in work dir; recovered $cand from $prev_mm/"
    fi
  fi
  if [ -n "$rfile" ]; then
    restart_patch="$rfile"
    echo "   LAMMPSRESTART → $rfile"
  else
    echo "   ! WARNING: no *.restart in work dir or mm_$((qmmmini-1))/ —"
    echo "     cannot point LAMMPSRESTART at an MD restart; the resumed MM step"
    echo "     may fall back to the initial structure (ini.restart/data.file)."
  fi
fi

# ---------- 4. Patch qmmm script ----------
ts=$(date +%Y%m%d_%H%M%S)
cp "$QMMM" "${QMMM}.bak.${ts}"
sed -i.tmp "s|^QMMMINISTEP=.*|QMMMINISTEP=$qmmmini # no of initial QMMM step|" "$QMMM"
sed -i.tmp "s|^initialqm=.*|initialqm=$initialqm #1, when the initial qm has been done.|" "$QMMM"
# Always set skipequil explicitly (0 or 1) so a value left over from a prior
# resubmit can't leak into this run.
sed -i.tmp "s|^skipequil=.*|skipequil=$skipequil #1, skip equil on first MM step (resume from averaging)|" "$QMMM"
# Point LAMMPSRESTART at the resumed step's restart (see §3.7). Only patch when we
# resolved a target and the script actually has a LAMMPSRESTART= line.
if [ -n "$restart_patch" ] && grep -q '^LAMMPSRESTART=' "$QMMM"; then
  sed -i.tmp "s|^LAMMPSRESTART=.*|LAMMPSRESTART=\"$restart_patch\" # set by ces2_resubmit.sh: resume from latest restart|" "$QMMM"
fi
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
grep -E "^(QMMMINISTEP|QMMMFINSTEP|initialqm|skipequil|firstrun|LAMMPSRESTART)=" "$QMMM" | sed 's/^/     /' || true
if (( sub_patched == 1 )); then
  echo "   $SUB  (backup: ${SUB}.bak.${ts})"
  grep -n 'in\.relax_' "$SUB" | sed 's/^/     /' || true
else
  echo "   $SUB  (no relax lines found — left as-is)"
fi

# ---------- 5.4 Patch-only short-circuit ----------
# In --patch-only mode (used by submit_ces2.sh's depend=afterany self-chain),
# we stop here: NO trap injection, NO qsub. The caller decides what to do
# next based on the .ces2_chain_done marker (if any) and its own counter.
if (( PATCH_ONLY == 1 )); then
  echo
  echo "==[ Patch-only mode: skipping trap injection and qsub ]=="
  if (( is_complete == 1 )); then
    echo "   .ces2_chain_done present — caller should stop the chain."
  fi
  exit 0
fi

# ---------- 5.5 Auto-resubmit chain (--continue / --no-continue) ----------
# When --continue is given, inject a depend=afterany self-chain block into
# submit_ces2.sh. Each chain step:
#   1) calls ces2_resubmit.sh --patch-only  (state detect + sed patch, no qsub)
#   2) if .ces2_chain_done marker exists → exits cleanly (chain end)
#   3) pre-queues the NEXT submit_ces2.sh as a held job (depend=afterany on
#      $PBS_JOBID) — runs after THIS job ends, no matter how (clean exit,
#      walltime SIGTERM, crash). qsub of the next step happens BEFORE qmmm
#      starts, so PBS's kill_delay can never interrupt the chaining itself.
#   4) runs qmmm
# A counter file (.resubmit_count) caps consecutive chain steps as a safety net;
# the marker file is the normal stopping condition.
TRAP_BEGIN='# >>> CES2_AUTORESUBMIT >>> (managed by ces2_resubmit.sh, do not edit by hand)'
TRAP_END='# <<< CES2_AUTORESUBMIT <<<'

remove_trap_block() {
  # Strip any existing CES2_AUTORESUBMIT block from $SUB (idempotent).
  # Handles both legacy SIGTERM-trap blocks and the current depend-chain blocks.
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

  # Build the block in a temp file. Unquoted heredoc → ${SELF_ABS} and
  # ${MAX_RESUBMIT} get substituted now; \$foo stays literal for runtime.
  local tmpf
  tmpf="$(mktemp)"
  cat >"$tmpf" <<EOF
${TRAP_BEGIN}
# depend=afterany self-chain. See ces2_resubmit.sh §5.5 for design notes.
__CES2_RESUBMIT_SCRIPT="${SELF_ABS}"
__CES2_MAX_RESUBMIT=${MAX_RESUBMIT}
__CES2_COUNT_FILE=".resubmit_count"
__CES2_DONE_MARKER=".ces2_chain_done"
__CES2_LOG="resubmit_chain.log"
__CES2_THIS_SCRIPT="\${curr_dir}/submit_ces2.sh"

__ces_ts()  { date '+%Y-%m-%d %H:%M:%S'; }
__ces_log() { echo "[\$(__ces_ts)] [job=\${PBS_JOBID:-?}] \$*" >> "\$__CES2_LOG"; }

__ces_log "Chain step start (count=\$(cat "\$__CES2_COUNT_FILE" 2>/dev/null || echo 0)/\${__CES2_MAX_RESUBMIT})"

# (1) Patch state in place (sed only, no qsub).
if [ -x "\$__CES2_RESUBMIT_SCRIPT" ]; then
    if ! bash "\$__CES2_RESUBMIT_SCRIPT" --patch-only >> "\$__CES2_LOG" 2>&1; then
        __ces_log "ERROR: ces2_resubmit.sh --patch-only failed — chain stopped."
        rm -f "\$__CES2_COUNT_FILE"
        exit 1
    fi
else
    __ces_log "WARN: \$__CES2_RESUBMIT_SCRIPT not found — running without state patch"
fi

# (2) Done?  Stop the chain.
if [ -f "\$__CES2_DONE_MARKER" ]; then
    __ces_log "Simulation complete (marker present) — chain stopped."
    rm -f "\$__CES2_COUNT_FILE"
    exit 0
fi

# (3) Counter cap, then pre-queue the next step.
__ces_count=\$(cat "\$__CES2_COUNT_FILE" 2>/dev/null || echo 0)
if [ "\$__ces_count" -ge "\$__CES2_MAX_RESUBMIT" ]; then
    __ces_log "Hit MAX_RESUBMIT=\${__CES2_MAX_RESUBMIT} — chain stopped."
    rm -f "\$__CES2_COUNT_FILE"
elif [ -n "\$PBS_JOBID" ]; then
    __ces_next=\$((__ces_count + 1))
    __ces_next_jid=\$(qsub -W depend=afterany:\${PBS_JOBID} "\$__CES2_THIS_SCRIPT" 2>>"\$__CES2_LOG")
    if [ \$? -eq 0 ] && [ -n "\$__ces_next_jid" ]; then
        echo "\$__ces_next" > "\$__CES2_COUNT_FILE"
        __ces_log "Pre-queued next chain step \${__ces_next}/\${__CES2_MAX_RESUBMIT} → \${__ces_next_jid}"
    else
        __ces_log "WARN: failed to pre-queue next chain step"
    fi
else
    __ces_log "WARN: PBS_JOBID empty — skipping pre-queue"
fi
${TRAP_END}
EOF

  # Insert the block right after "cd \$curr_dir" (or "cd \${PBS_O_WORKDIR}").
  local anchor_line
  anchor_line=$(grep -nE '^[[:space:]]*cd[[:space:]]+(\$curr_dir|\$\{?PBS_O_WORKDIR\}?)[[:space:]]*$' "$SUB" | head -1 | cut -d: -f1)
  if [ -z "$anchor_line" ]; then
    echo "ERROR: could not find 'cd \$curr_dir' anchor in $SUB — chain block NOT injected." >&2
    rm -f "$tmpf"
    return 1
  fi
  sed -i.tmp "${anchor_line}r ${tmpf}" "$SUB"
  rm -f "${SUB}.tmp" "$tmpf"
  return 0
}

if (( CONTINUE == 1 )); then
  echo
  echo "==[ Auto-resubmit chain: ON (--continue, depend=afterany) ]=="
  cp -p "$SUB" "${SUB}.bak.chain.${ts}" 2>/dev/null || true
  if inject_trap_block; then
    echo "   chain block injected into $SUB  (max retries: $MAX_RESUBMIT)"
    echo "   marker file:  .ces2_chain_done  (written by --patch-only when complete)"
    echo "   counter file: .resubmit_count   (safety cap on consecutive steps)"
    echo "   chain log:    resubmit_chain.log"
    # Fresh chain → reset counter and stale marker so the new run starts clean.
    rm -f .resubmit_count .ces2_chain_done
  else
    echo "   chain block NOT injected (see error above)."
    exit 1
  fi
elif (( NO_CONTINUE == 1 )); then
  echo
  echo "==[ Auto-resubmit chain: OFF (--no-continue) ]=="
  if remove_trap_block; then
    echo "   chain block removed from $SUB."
  else
    echo "   no chain block was present in $SUB."
  fi
  rm -f .resubmit_count .ces2_chain_done
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
