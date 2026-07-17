#!/bin/bash
# ces2_backup_save.sh
# -----------------------------------------------------------------------------
# Snapshot / restore a CES2 QM run's QE save directory (outdir/prefix.save),
# i.e. the converged charge-density.dat + wfc*.dat.  The driver
# (qmmm_dftces2_charging_pts.sh) does NOT archive this .save into qm_$step and
# OVERWRITES it on every subsequent pw.x run, so a converged density is easily
# lost.  This tool preserves one before it is clobbered, and restores it so a
# restart can seed from it (startingpot='file'/startingwfc='file').
#
# Layout assumed:  <export>/solute/solute.save/   (prefix='solute', outdir='./solute')
# Backups go to:   <export>/_save_backup/<label>/solute.save/  + manifest.txt
#
# Usage:
#   ces2_backup_save.sh backup  <export_dir> [label]     # default label: qm_ortho_converged
#   ces2_backup_save.sh restore <export_dir> <label>     # copy backup back into outdir
#   ces2_backup_save.sh list    <export_dir>             # show existing backups
# -----------------------------------------------------------------------------
set -euo pipefail

CP="/bin/cp"                       # avoid any cp alias
PREFIX="solute"                    # QE prefix (matches pw.in)
OUTDIR_SUB="solute"                # QE outdir subdir (matches pw.in outdir='./solute')
FREEZE_WIN_MIN=5                   # save must be untouched this many minutes to be "safe"

die(){ echo "ERROR: $*" >&2; exit 1; }

mode="${1:-}"; export_dir="${2:-}"
[ -n "$mode" ] && [ -n "$export_dir" ] || die "usage: $0 {backup|restore|list} <export_dir> [label]"
export_dir="$(cd "$export_dir" && pwd)" || die "no such export dir: $export_dir"

save_dir="$export_dir/$OUTDIR_SUB/$PREFIX.save"
bak_root="$export_dir/_save_backup"

case "$mode" in
  list)
    echo "Backups under $bak_root:"
    [ -d "$bak_root" ] && ls -1 "$bak_root" | sed 's/^/  /' || echo "  (none)"
    ;;

  backup)
    label="${3:-qm_ortho_converged}"
    [ -d "$save_dir" ] || die "save dir not found: $save_dir"
    [ -s "$save_dir/charge-density.dat" ] || die "no charge-density.dat in $save_dir (nothing to seed from)"

    # Safety: refuse to copy a .save that is being actively written (live SCF).
    recent=$(find "$save_dir" -type f -newermt "$FREEZE_WIN_MIN minutes ago" 2>/dev/null | wc -l)
    [ "$recent" -eq 0 ] || die "$save_dir changed within last ${FREEZE_WIN_MIN} min -> job still writing it; refusing (kill/settle first)"

    dest="$bak_root/$label/$PREFIX.save"
    [ -e "$dest" ] && die "backup already exists: $dest (pick another label or remove it)"
    mkdir -p "$bak_root/$label"

    src_du=$(du -sb "$save_dir" | cut -f1)
    cd_mtime=$(stat -c '%y' "$save_dir/charge-density.dat" | cut -d. -f1)
    echo "Backing up:"
    echo "  src   : $save_dir"
    echo "  dest  : $dest"
    echo "  size  : $(du -sh "$save_dir" | cut -f1)   charge-density.dat mtime: $cd_mtime"
    $CP -a "$save_dir" "$dest"

    # Verify byte-size and file count match.
    dst_du=$(du -sb "$dest" | cut -f1)
    sc=$(find "$save_dir" -type f | wc -l); dc=$(find "$dest" -type f | wc -l)
    [ "$src_du" = "$dst_du" ] || die "size mismatch after copy (src=$src_du dst=$dst_du)"
    [ "$sc" = "$dc" ] || die "file-count mismatch after copy (src=$sc dst=$dc)"

    {
      echo "# CES2 save backup manifest"
      echo "label            : $label"
      echo "source_save      : $save_dir"
      echo "backup_save      : $dest"
      echo "charge-density_mtime : $cd_mtime"
      echo "bytes            : $src_du"
      echo "files            : $sc"
      echo "backed_up_from_host : $(hostname)"
    } > "$bak_root/$label/manifest.txt"

    echo "OK: verified ($sc files, $src_du bytes)."
    echo "Manifest -> $bak_root/$label/manifest.txt"
    echo
    echo "To seed a restart later:  $0 restore $export_dir $label"
    ;;

  restore)
    label="${3:-}"; [ -n "$label" ] || die "restore needs a <label> (see: $0 list $export_dir)"
    src="$bak_root/$label/$PREFIX.save"
    [ -d "$src" ] || die "backup not found: $src"
    mkdir -p "$export_dir/$OUTDIR_SUB"
    dst="$export_dir/$OUTDIR_SUB/$PREFIX.save"
    if [ -e "$dst" ]; then
      ts=$(stat -c '%Y' "$dst")
      mv "$dst" "$dst.pre_restore.$ts"
      echo "moved existing outdir save aside -> $dst.pre_restore.$ts"
    fi
    $CP -a "$src" "$dst"
    echo "restored: $src -> $dst"
    echo "Now ensure pw.in/pw.nonortho.in use startingpot='file' (and optionally startingwfc='file')."
    ;;

  *) die "unknown mode '$mode' (use backup|restore|list)";;
esac
