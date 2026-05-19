#!/usr/bin/env bash
# Interactive wizard to configure a CES2 build YAML.
#
# Usage:
#   tools/configure.sh [OPTIONS] [path/to/config.yaml]
#
# Without options, runs variant mode and prompts for: salt, concentration,
# electrode charge, job name, output dirs (and only edits those run-variant keys).
#
# Recommended workflow:
#   1) Make a reusable structural/physical master setup:
#        tools/configure.sh --master --out configs/masters/IrO2_2OH_2O_TIP4P_50A.yaml config_example.yaml
#   2) Derive run variants from that master by changing only electrolyte/charge:
#        tools/configure.sh --variant --out configs/variants/IrO2_2OH_2O_LiOH_1M_qm1.yaml configs/masters/IrO2_2OH_2O_TIP4P_50A.yaml
#
# Main modes:
#   --master        Structural/physical setup: bjparams source/file,
#                   water model, electrolyte box, water.count, adsorbate count,
#                   and optional pre-relaxation/parallel settings.
#                   At the qm_params/layer-file prompt, type '?' to list
#                   available DB entries that can be copied into the answer.
#   --variant       Run-condition setup: salt, concentration, electrode charge,
#                   job name, output dirs, and optional QMMM length.
#                   This is the default mode for backward compatibility.
#   --out PATH      Copy input config to PATH before editing. This is the
#                   recommended way to create master/variant config files.
#   --validate      Validate the selected config and exit without editing.
#
# Optional section flags — each opens an extra interactive prompt block that
# shows the current YAML value as the default, so hitting <Enter> keeps it.
# Sections that aren't requested are NOT touched at all:
#
#   --box           Expert override: prompt for electrolyte_box.thickness,
#                   vacuum_z [Å] outside --master.
#                   Always shows the current water.count and the auto count
#                   that would be computed for the (new) thickness, and lets
#                   you keep, switch to auto, or enter an explicit integer.
#                   Default answer is "auto" only when the thickness changed
#                   AND the existing count is an explicit integer (i.e. now
#                   stale); otherwise "keep" to avoid silent rewrites.
#   --prerelax      Pre-relaxation pipeline: md_relax.enabled (on/off) and
#                   ces2.initial_dump (auto-set when on, removed when off)
#   --parallel      QE parallelization flags: ces2_script.{npool, ntg, ndiag}
#                   (pw.x mpirun options; see config_example.yaml comments)
#   --qe            Review QE setup: k_points, pseudopotential directory/set,
#                   cutoffs, smearing, and SCF controls.
#   --lammps        Review production LAMMPS/QM-MM MD setup: timestep, run
#                   length, thermo/dump/restart cadence, and temperature.
#   --qmmm          Edit QMMMFINSTEP in ./qmmm_dftces2_charging_pts.sh (if
#                   present in CWD) and keep YAML's ces2_script.n_qmmm_steps
#                   in sync (n_qmmm_steps = QMMMFINSTEP + 1).
#   -h, --help      Show this help and exit.
#
# Combine freely:
#   tools/configure.sh --master --parallel --out configs/masters/base.yaml config_example.yaml
#   tools/configure.sh --variant --qmmm --out configs/variants/LiOH_1M_qm1.yaml configs/masters/base.yaml
#
# Backup: writes .<basename>.bak alongside the original before editing.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

print_help() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -e 's/^# \{0,1\}//' -e '/^set -euo/d'
}

# ---- parse CLI flags ----
DO_BOX=0        # 1 => prompt for electrolyte_box.{thickness, vacuum_z}
DO_PRERELAX=0   # 1 => prompt for md_relax.enabled (and sync ces2.initial_dump)
DO_PARALLEL=0   # 1 => prompt for ces2_script.{npool, ntg, ndiag}
DO_QMMM=0       # 1 => prompt for QMMMFINSTEP (and sync ces2_script.n_qmmm_steps)
DO_QE=0         # 1 => prompt for qe setup
DO_LAMMPS=0     # 1 => prompt for ces2.md setup
MODE="variant"  # variant (default) or master
MASTER_MODE_REQUESTED=0
DO_VALIDATE=0
OUT_CONFIG=""
CONFIG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --master)           MODE="master"; MASTER_MODE_REQUESTED=1; DO_BOX=1; DO_QE=1; DO_LAMMPS=1; shift ;;
        --variant)          MODE="variant"; shift ;;
        --out)              OUT_CONFIG="${2:-}"; if [[ -z "$OUT_CONFIG" ]]; then echo "ERROR: --out requires a path" >&2; exit 2; fi; shift 2 ;;
        --validate)         DO_VALIDATE=1; shift ;;
        --box)              DO_BOX=1; shift ;;
        --prerelax)         DO_PRERELAX=1; shift ;;
        --parallel)         DO_PARALLEL=1; shift ;;
        --qe)               DO_QE=1; shift ;;
        --lammps)           DO_LAMMPS=1; shift ;;
        --qmmm)             DO_QMMM=1; shift ;;
        -h|--help)          print_help; exit 0 ;;
        --)                 shift; CONFIG="${1:-}"; break ;;
        -*)                 echo "ERROR: unknown option: $1" >&2; print_help; exit 2 ;;
        *)                  CONFIG="$1"; shift ;;
    esac
done

CONFIG="${CONFIG:-${REPO_ROOT}/config_example.yaml}"
CURRENT_DIR="$(pwd)"

# These get populated by interactive prompts only when their section flag is set.
# Empty value at python-apply time means "do not touch".
THICKNESS=""
VACUUM_Z=""
WATER_COUNT=""   # set by --box block when thickness change makes explicit count stale
PRERELAX=""
NPOOL=""
NTG=""
NDIAG=""
QMMMFINSTEP=""
BJ_SOURCE=""
QM_PARAMS_FILE=""
BJ_LAYER_FILE=""
WATER_MODEL=""
OUTPUT_BUILD_DIR=""
OUTPUT_EXPORT_DIR=""
QE_PREFIX=""
QE_OUTDIR=""
QE_PSEUDO_DIR=""
QE_PSEUDO_SET=""
QE_PSEUDOS=""
QE_ECUTWFC=""
QE_ECUTRHO=""
QE_KPOINTS=""
QE_OCCUPATIONS=""
QE_SMEARING=""
QE_DEGAUSS=""
QE_CONV_THR=""
QE_ELECTRON_MAXSTEP=""
QE_MIXING_BETA=""
QE_DIAGONALIZATION=""
LMP_TIMESTEP=""
LMP_N_STEPS=""
LMP_THERMO_EVERY=""
LMP_DUMP_EVERY=""
LMP_TEMPERATURE=""
LMP_RESTART_EVERY=""

# Path to the qmmm script in the current working directory (if any).
QMMM_SCRIPT_PATH="./qmmm_dftces2_charging_pts.sh"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

echo
echo "Current working directory:"
echo "  $CURRENT_DIR"

# ---- ensure ruamel.yaml is importable ----
if ! python3 -c "import ruamel.yaml" 2>/dev/null; then
    echo "ruamel.yaml is required but not installed."
    read -r -p "Install it now with 'pip3 install --user ruamel.yaml'? [Y/n] " ans
    ans="${ans:-Y}"
    if [[ "$ans" =~ ^[Yy] ]]; then
        pip3 install --user ruamel.yaml
    else
        echo "Aborting." >&2
        exit 1
    fi
fi

validate_config() {
    CONFIG="$CONFIG" python3 <<'PY'
import os
from ruamel.yaml import YAML

yaml = YAML()
with open(os.environ["CONFIG"]) as f:
    cfg = yaml.load(f)
if not isinstance(cfg, dict):
    raise SystemExit("ERROR: config root must be a YAML mapping")

print(f"validated -> {os.environ['CONFIG']}")
PY
}

if [[ "$DO_VALIDATE" == "1" ]]; then
    validate_config
    exit 0
fi

# ---- read current values for default-prompts ----
read_defaults() {
    CONFIG="$CONFIG" REPO_ROOT="$REPO_ROOT" python3 <<'PY'
import os
from ruamel.yaml import YAML
yaml = YAML()
with open(os.environ["CONFIG"]) as f:
    d = yaml.load(f)

def g(*keys, default=""):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return "" if cur is None else cur

salts = (d.get("electrolyte_recipe") or {}).get("salts") or []
salt_name = salts[0]["name"] if salts else "water"
salt_conc = salts[0].get("concentration_M", 0.1) if salts else 0.1
qcharge   = g("charge_control", "q_electrode_user_value", default=0.0)
adsorb    = g("ces2_script", "adsorbate", default="0")
jobname   = g("ces2_script", "jobname", default="ces2_qmmm")
# Current values for fields gated behind CLI flags (shown as defaults in section prompts).
cur_thick   = g("electrolyte_box", "thickness", default="?")
cur_vacz    = g("electrolyte_box", "vacuum_z",  default="?")
cur_relax   = g("md_relax", "enabled", default=False)
cur_initdmp = g("ces2", "initial_dump", default="")
cur_npool   = g("ces2_script", "npool", default="")
cur_ntg     = g("ces2_script", "ntg",   default="")
cur_ndiag   = g("ces2_script", "ndiag", default="")
cur_nsteps  = g("ces2_script", "n_qmmm_steps", default="")
cur_bjsrc   = g("slab", "bjparams_source", default="yaml")
cur_qmfile  = g("slab", "qm_params_file", default="")
cur_layer   = g("slab", "bjparams_layer_file", default="")
cur_wmodel  = g("ces2", "water_model", default="TIP4P")
cur_build   = g("output", "build_dir", default="build")
cur_export  = g("output", "export_dir", default="export")
cur_qe_prefix = g("qe", "prefix", default="solute")
cur_qe_outdir = g("qe", "outdir", default="./solute")
cur_qe_pdir   = g("qe", "pseudo_dir", default="./pseudo")
cur_qe_pset   = g("qe", "pseudo_set", default="sssp")
cur_qe_pseudos = (d.get("qe") or {}).get("pseudopotentials") or {}
cur_qe_pseudos_s = ",".join(f"{k}={v}" for k, v in cur_qe_pseudos.items())
cur_qe_ecutwfc = g("qe", "ecutwfc", default=50.0)
cur_qe_ecutrho = g("qe", "ecutrho", default=400.0)
cur_qe_kpts    = " ".join(str(x) for x in (g("qe", "k_points", default=[1,1,1,0,0,0]) or [1,1,1,0,0,0]))
cur_qe_occ     = g("qe", "occupations", default="smearing")
cur_qe_smear   = g("qe", "smearing", default="mv")
cur_qe_degauss = g("qe", "degauss", default=0.02)
cur_qe_conv    = g("qe", "conv_thr", default=1e-8)
cur_qe_emax    = g("qe", "electron_maxstep", default=400)
cur_qe_mix     = g("qe", "mixing_beta", default=0.3)
cur_qe_diag    = g("qe", "diagonalization", default="cg")
cur_lmp_timestep = g("ces2", "md", "timestep_fs", default=1.0)
cur_lmp_nsteps   = g("ces2", "md", "n_steps", default=0)
cur_lmp_thermo   = g("ces2", "md", "thermo_every", default=100)
cur_lmp_dump     = g("ces2", "md", "dump_every", default=1000)
cur_lmp_temp     = g("ces2", "md", "temperature", default=300.0)
cur_lmp_restart  = g("ces2", "md", "restart_every", default=500000)
# water block (used by --box water.count helper)
cur_wcount    = g("electrolyte_recipe", "water", "count",            default="auto")
cur_wrho      = g("electrolyte_recipe", "water", "density_g_per_ml", default=1.0)
cur_wuf       = g("electrolyte_recipe", "water", "packmol_underfill", default=0.97)

# Slab Lx, Ly via builder.vasp_io (best-effort — needed to compute a suggested
# auto water count for the new thickness).  Empty strings on any failure so the
# shell can fall back to a non-numeric prompt.
slab_lx, slab_ly = "", ""
try:
    import sys
    sys.path.insert(0, os.environ["REPO_ROOT"])
    from builder.vasp_io import read_vasp, make_supercell  # type: ignore
    cfg_dir   = os.path.dirname(os.path.abspath(os.environ["CONFIG"]))
    workdir   = (d.get("project") or {}).get("workdir", "./") or "./"
    vasp_file = (d.get("input") or {}).get("vasp_file", "CONTCAR") or "CONTCAR"
    candidates = [
        os.path.join(os.getcwd(), workdir, vasp_file),
        os.path.join(cfg_dir,    workdir, vasp_file),
        os.path.join(cfg_dir,             vasp_file),
        vasp_file,
    ]
    poscar = next((c for c in candidates if os.path.isfile(c)), None)
    if poscar is not None:
        slab    = read_vasp(poscar)
        rep     = tuple(int(x) for x in (d.get("cell") or {}).get("supercell", [1, 1, 1]))
        slab_sc = make_supercell(slab, rep)
        cell    = slab_sc.cell.array
        slab_lx = f"{float(cell[0][0]):.6f}"
        slab_ly = f"{float(cell[1][1]):.6f}"
except Exception:
    pass

print(f"{salt_name}|{salt_conc}|{qcharge}|{adsorb}|{jobname}|{cur_thick}|{cur_vacz}|{cur_relax}|{cur_initdmp}|{cur_npool}|{cur_ntg}|{cur_ndiag}|{cur_nsteps}|{cur_bjsrc}|{cur_qmfile}|{cur_layer}|{cur_wmodel}|{cur_build}|{cur_export}|{cur_wcount}|{cur_wrho}|{cur_wuf}|{slab_lx}|{slab_ly}|{cur_qe_prefix}|{cur_qe_outdir}|{cur_qe_pdir}|{cur_qe_pset}|{cur_qe_pseudos_s}|{cur_qe_ecutwfc}|{cur_qe_ecutrho}|{cur_qe_kpts}|{cur_qe_occ}|{cur_qe_smear}|{cur_qe_degauss}|{cur_qe_conv}|{cur_qe_emax}|{cur_qe_mix}|{cur_qe_diag}|{cur_lmp_timestep}|{cur_lmp_nsteps}|{cur_lmp_thermo}|{cur_lmp_dump}|{cur_lmp_temp}|{cur_lmp_restart}")
PY
}

IFS='|' read -r CUR_SALT CUR_CONC CUR_CHARGE CUR_ADSORB CUR_JOB CUR_THICK CUR_VACZ CUR_RELAX CUR_INITDMP CUR_NPOOL CUR_NTG CUR_NDIAG CUR_NSTEPS CUR_BJSRC CUR_QMFILE CUR_LAYER CUR_WMODEL CUR_BUILD CUR_EXPORT CUR_WCOUNT CUR_WRHO CUR_WUF SLAB_LX SLAB_LY CUR_QE_PREFIX CUR_QE_OUTDIR CUR_QE_PDIR CUR_QE_PSET CUR_QE_PSEUDOS CUR_QE_ECUTWFC CUR_QE_ECUTRHO CUR_QE_KPTS CUR_QE_OCC CUR_QE_SMEAR CUR_QE_DEGAUSS CUR_QE_CONV CUR_QE_EMAX CUR_QE_MIX CUR_QE_DIAG CUR_LMP_TIMESTEP CUR_LMP_NSTEPS CUR_LMP_THERMO CUR_LMP_DUMP CUR_LMP_TEMP CUR_LMP_RESTART <<<"$(read_defaults)"

# Try to read QMMMFINSTEP from the qmmm script in CWD (if present).
CUR_QMMMFINSTEP=""
if [[ -f "$QMMM_SCRIPT_PATH" ]]; then
    CUR_QMMMFINSTEP="$(grep -E '^[[:space:]]*QMMMFINSTEP=' "$QMMM_SCRIPT_PATH" | head -1 \
        | sed -E 's/^[[:space:]]*QMMMFINSTEP=([0-9]+).*/\1/')"
fi

ask() {
    # ask "prompt" "default" -> echoes user reply (or default if blank)
    local prompt="$1" default="$2" reply
    read -r -p "$prompt [$default]: " reply
    echo "${reply:-$default}"
}

list_qm_params_files() {
    local dir file base found
    dir="${REPO_ROOT}/species_db/qm_params"
    found=0

    echo "Available QM parameter YAML files:" >&2
    if [[ -d "$dir" ]]; then
        while IFS= read -r file; do
            base="$(basename -- "$file")"
            echo "  ${base%.yaml}" >&2
            found=1
        done < <(find "$dir" -maxdepth 1 -type f -name '*.yaml' -print | sort)
    fi
    if [[ "$found" == "0" ]]; then
        echo "  (none found in $dir)" >&2
    fi
}

list_qm_layer_files() {
    local dir file base shown found
    found=0

    echo "Available BJ layer .dat files:" >&2
    for dir in \
        "${REPO_ROOT}/species_db/qm_params/layer_files" \
        "${REPO_ROOT}/species_db/qm_params" \
        "$CURRENT_DIR" \
        "$(dirname -- "$CONFIG")"
    do
        [[ -d "$dir" ]] || continue
        while IFS= read -r file; do
            base="$(basename -- "$file")"
            case "$dir" in
                "${REPO_ROOT}/species_db/qm_params/layer_files"|\
                "${REPO_ROOT}/species_db/qm_params")
                    shown="${base%.dat}"
                    ;;
                *)
                    shown="$file"
                    ;;
            esac
            echo "  $shown" >&2
            found=1
        done < <(find "$dir" -maxdepth 1 -type f -name '*.dat' -print | sort)
    done
    if [[ "$found" == "0" ]]; then
        echo "  (none found in bundled DB, current directory, or config directory)" >&2
    fi
}

ask_qm_params_file() {
    local default="$1" reply
    while true; do
        reply="$(ask "  qm_params_file ('?' to list)" "$default")"
        if [[ "$reply" == "?" ]]; then
            list_qm_params_files
            continue
        fi
        echo "$reply"
        return 0
    done
}

ask_qm_layer_file() {
    local default="$1" reply
    while true; do
        reply="$(ask "  bjparams_layer_file ('?' to list)" "$default")"
        if [[ "$reply" == "?" ]]; then
            list_qm_layer_files
            continue
        fi
        echo "$reply"
        return 0
    done
}

sanitize_jobname_part() {
    printf '%s' "$1" | sed -E 's/[^A-Za-z0-9._+-]+/_/g; s/^_+//; s/_+$//'
}

suggest_jobname_from_pwd() {
    local dir="$1" trimmed part count last n start i suggestion sep sanitized
    trimmed="${dir%/}"
    [[ -n "$trimmed" ]] || return 0

    IFS='/' read -r -a part <<< "$trimmed"
    count="${#part[@]}"
    [[ "$count" -gt 0 ]] || return 0

    last="${part[$((count - 1))]}"
    if [[ "$last" == "water" ]]; then
        n=3
    elif [[ "$last" =~ ^[0-9]+([.][0-9]+)?[Mm]$ ]]; then
        n=4
    else
        n=4
    fi
    (( count < n )) && n="$count"
    start=$((count - n))

    suggestion=""
    sep=""
    for ((i=start; i<count; i++)); do
        sanitized="$(sanitize_jobname_part "${part[$i]}")"
        [[ -n "$sanitized" ]] || continue
        suggestion="${suggestion}${sep}${sanitized}"
        sep="_"
    done
    printf '%s' "$suggestion"
}

SUGGESTED_JOB="$(suggest_jobname_from_pwd "$CURRENT_DIR")"

echo
echo "=========================================="
echo "  CES2 build config wizard"
echo "  mode  : $MODE"
echo "  target: $CONFIG"
echo "  pwd   : $CURRENT_DIR"
echo "=========================================="
echo

if [[ "$MODE" == "master" ]]; then
    echo "Master setup: structural/physical choices that should stay fixed across variants."
    echo
    echo "BJ parameter source:"
    echo "  1) yaml       (species_db/qm_params/<System>.yaml)"
    echo "  2) layer_file (species_db/qm_params/layer_files/<name>.dat or project file)"
    bj_default="$CUR_BJSRC"
    bj_choice="$(ask "Choose bjparams_source (number or name)" "$bj_default")"
    case "$(printf '%s' "$bj_choice" | tr '[:upper:]' '[:lower:]')" in
        1|yaml) BJ_SOURCE="yaml" ;;
        2|layer|layer_file) BJ_SOURCE="layer_file" ;;
        *) echo "ERROR: bjparams_source must be yaml or layer_file, got '$bj_choice'" >&2; exit 2 ;;
    esac
    if [[ "$BJ_SOURCE" == "yaml" ]]; then
        QM_PARAMS_FILE="$(ask_qm_params_file "${CUR_QMFILE:-IrO2}")"
        BJ_LAYER_FILE=""
    else
        BJ_LAYER_FILE="$(ask_qm_layer_file "${CUR_LAYER:-IrO2_2OH_2O_bjparams_layer_avg}")"
        QM_PARAMS_FILE=""
    fi

    echo
    echo "Water model:"
    echo "  1) TIP4P    (recommended default)"
    echo "  2) TIP3P"
    echo "  3) SPCE"
    echo "  4) TIP3PEW"
    wm_choice="$(ask "Choose water_model (number or name)" "$CUR_WMODEL")"
    case "$(printf '%s' "$wm_choice" | tr '[:lower:]' '[:upper:]')" in
        1|TIP4P)   WATER_MODEL="TIP4P" ;;
        2|TIP3P)   WATER_MODEL="TIP3P" ;;
        3|SPCE)    WATER_MODEL="SPCE" ;;
        4|TIP3PEW) WATER_MODEL="TIP3PEW" ;;
        *) echo "ERROR: water_model must be TIP4P, TIP3P, SPCE, or TIP3PEW, got '$wm_choice'" >&2; exit 2 ;;
    esac

    echo
    ADSORB="$(ask "Adsorbate atom count (last N atoms in POSCAR; 0 = none)" "$CUR_ADSORB")"

    # In master mode, pre-relaxation is part of the reusable setup by default.
    DO_PRERELAX=1
fi

# ---- 1. salt preset ----
if [[ "$MODE" == "variant" ]]; then
echo "Salt presets:"
echo "  1) water    (no salt)"
echo "  2) LiOH"
echo "  3) NaOH"
echo "  4) KOH"
echo "  5) CsOH"
echo "  6) LiCl"
echo "  7) NaCl"
echo "  8) KCl"
echo "  9) CsCl"
echo " 10) custom"
SALT_DEFAULT="$CUR_SALT"
salt_choice="$(ask "Choose salt (number or name)" "$SALT_DEFAULT")"

case "$salt_choice" in
    1|water|Water|WATER)         SALT="water" ;;
    2|LiOH)  SALT="LiOH"; CATION="Li_plus"; ANION="OH_minus" ;;
    3|NaOH)  SALT="NaOH"; CATION="Na_plus"; ANION="OH_minus" ;;
    4|KOH)   SALT="KOH";  CATION="K_plus";  ANION="OH_minus" ;;
    5|CsOH)  SALT="CsOH"; CATION="Cs_plus"; ANION="OH_minus" ;;
    6|LiCl)  SALT="LiCl"; CATION="Li_plus"; ANION="Cl_minus" ;;
    7|NaCl)  SALT="NaCl"; CATION="Na_plus"; ANION="Cl_minus" ;;
    8|KCl)   SALT="KCl";  CATION="K_plus";  ANION="Cl_minus" ;;
    9|CsCl)  SALT="CsCl"; CATION="Cs_plus"; ANION="Cl_minus" ;;
    10|custom)
        SALT="$(ask "  custom salt name" "MySalt")"
        CATION="$(ask "  cation species_id (e.g., Li_plus, H3O_plus)" "Li_plus")"
        ANION="$(ask "  anion species_id (e.g., OH_minus, Cl_minus)"  "OH_minus")"
        ;;
    *)
        # treat as preset name passed verbatim
        SALT="$salt_choice"
        CATION="$(ask "  cation species_id" "Li_plus")"
        ANION="$(ask "  anion species_id"  "OH_minus")"
        ;;
esac

# ---- 2. concentration (skip for water) ----
if [[ "$SALT" == "water" ]]; then
    CONC="0"
else
    CONC="$(ask "Concentration [M]" "$CUR_CONC")"
fi

# ---- 3. QM electrode charge ----
echo
echo "QM electrode charge (q_electrode_user_value): -2, -1, 0, +1, +2"
QCHARGE="$(ask "Electrode charge [e]" "$CUR_CHARGE")"

# ---- 4. counterion pool ----
# Always need both a cation and anion in the pool. For salt cases use the salt's
# ions; for pure-water + nonzero charge ask the user which ion to neutralize with.
if [[ "$SALT" == "water" ]]; then
    if (( $(awk "BEGIN{print ($QCHARGE != 0)}") )); then
        echo
        echo "Pure water + nonzero charge: pick a counterion to neutralize the cell."
        if (( $(awk "BEGIN{print ($QCHARGE < 0)}") )); then
            echo "  (negative electrode → cation will be added)"
            CATION="$(ask "  cation species_id" "Li_plus")"
            ANION="$(ask "  anion species_id (kept in pool for completeness)" "OH_minus")"
        else
            echo "  (positive electrode → anion will be added)"
            CATION="$(ask "  cation species_id (kept in pool for completeness)" "Li_plus")"
            ANION="$(ask "  anion species_id" "OH_minus")"
        fi
    else
        # neutral water: still need a default pool so builder is happy
        CATION="${CATION:-Li_plus}"
        ANION="${ANION:-OH_minus}"
    fi
fi

# ---- 5. adsorbate atom count ----
echo
ADSORB="$(ask "Adsorbate atom count (last N atoms in POSCAR; 0 = none)" "$CUR_ADSORB")"

# ---- 6. job name ----
if [[ -n "$SUGGESTED_JOB" ]]; then
    echo
    echo "Suggested job name from pwd: $SUGGESTED_JOB"
    echo "  current YAML job name   : $CUR_JOB"
    JOB_DEFAULT="$SUGGESTED_JOB"
else
    JOB_DEFAULT="$CUR_JOB"
fi
JOBNAME="$(ask "Job name (sets ces2_script.jobname AND ces2_script.pbs.job_name)" "$JOB_DEFAULT")"
fi

# ---- 7. (optional, --box) electrolyte box geometry ----
if [[ "$DO_BOX" == "1" ]]; then
    echo
    echo "Box geometry  (current values shown as defaults — Enter to keep)"
    echo "  thickness/vacuum_z must satisfy:"
    echo "    0.8·(z_top_slab + thickness + vacuum_z + z_buffer_lo) ≥ z_top_slab + thickness + 5"
    echo "  Recommended: thickness=50, vacuum_z=30 (~6 Å margin to QE emaxpos zone)."
    THICKNESS="$(ask "  electrolyte_box.thickness [Å]" "$CUR_THICK")"
    VACUUM_Z="$(ask  "  electrolyte_box.vacuum_z  [Å]" "$CUR_VACZ")"
    if ! [[ "$THICKNESS" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "ERROR: thickness must be a positive number, got '$THICKNESS'" >&2; exit 2
    fi
    if ! [[ "$VACUUM_Z" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "ERROR: vacuum_z must be a positive number, got '$VACUUM_Z'" >&2; exit 2
    fi

    # --- water.count helper: always fires under --box -------------------------
    # Show the current count and the auto count that *would* be computed for
    # the (possibly new) thickness, then let the user accept, switch, or
    # override.  Default answer is "auto" when an explicit integer is stale
    # (thickness actually changed), otherwise "keep" so a no-op --box run
    # doesn't silently rewrite the count.
    thickness_changed=$(awk -v a="$THICKNESS" -v b="$CUR_THICK" \
        'BEGIN{ if (a+0 != b+0) print 1; else print 0 }')
    wcount_lc="$(printf '%s' "$CUR_WCOUNT" | tr '[:upper:]' '[:lower:]')"

    # Compute a suggested auto count if we managed to resolve Lx, Ly.
    # Formula mirrors builder/composition.py:auto_water_count.
    sugg=""
    if [[ -n "$SLAB_LX" && -n "$SLAB_LY" ]]; then
        sugg=$(awk -v lx="$SLAB_LX" -v ly="$SLAB_LY" -v t="$THICKNESS" \
                  -v rho="${CUR_WRHO:-1.0}" -v uf="${CUR_WUF:-0.97}" \
                  'BEGIN{
                      NA = 6.02214076e23; M = 18.01528;
                      V_cm3 = lx*ly*t * 1.0e-24;
                      n     = V_cm3 * rho * NA / M * uf;
                      printf("%d", int(n + 0.5));
                  }')
    fi

    echo
    echo "  water.count:"
    echo "    current        : $CUR_WCOUNT"
    if [[ -n "$sugg" ]]; then
        echo "    auto would be  : $sugg   (Lx·Ly·t = ${SLAB_LX}·${SLAB_LY}·${THICKNESS} Å³,"
        echo "                              ρ=${CUR_WRHO:-1.0} g/ml, underfill=${CUR_WUF:-0.97})"
    else
        echo "    (POSCAR not resolvable from this directory — no numeric suggestion;"
        echo "     'auto' will still be evaluated correctly at build time.)"
    fi
    if [[ "$thickness_changed" == "1" && "$wcount_lc" != "auto" ]]; then
        echo "    NOTE: thickness changed → explicit count is sized for the old box."
        default_wc="auto"
    else
        default_wc="keep"
    fi
    while :; do
        read -r -p "  set water.count to [auto/keep/<int>] (default: $default_wc): " wc_choice
        wc_choice="${wc_choice:-$default_wc}"
        case "$(printf '%s' "$wc_choice" | tr '[:upper:]' '[:lower:]')" in
            auto)  WATER_COUNT="auto"; break ;;
            keep)  WATER_COUNT=""; break ;;   # leave YAML untouched
            *)
                if [[ "$wc_choice" =~ ^[1-9][0-9]*$ ]]; then
                    WATER_COUNT="$wc_choice"; break
                fi
                echo "  please answer 'auto', 'keep', or a positive integer"
                ;;
        esac
    done
fi

# ---- 8. (optional/master, --qe) QE setup review ----------------------------
if [[ "$DO_QE" == "1" ]]; then
    echo
    if [[ "$MASTER_MODE_REQUESTED" == "1" ]]; then
        review_qe="$(ask "Review QE setup now? (Y/n)" "Y")"
        if [[ "$review_qe" =~ ^[Nn] ]]; then
            DO_QE=0
        fi
    fi
fi
if [[ "$DO_QE" == "1" ]]; then
    echo
    echo "QE setup  (fresh-start review; Enter keeps current values)"
    echo "  k_points format: n1 n2 n3 s1 s2 s3"
    QE_PREFIX="$(ask "  qe.prefix" "$CUR_QE_PREFIX")"
    QE_OUTDIR="$(ask "  qe.outdir" "$CUR_QE_OUTDIR")"
    QE_PSEUDO_DIR="$(ask "  qe.pseudo_dir" "$CUR_QE_PDIR")"
    QE_PSEUDO_SET="$(ask "  qe.pseudo_set" "$CUR_QE_PSET")"
    QE_PSEUDOS="$(ask "  qe.pseudopotentials overrides (El=file,El=file; blank = keep)" "$CUR_QE_PSEUDOS")"
    QE_ECUTWFC="$(ask "  qe.ecutwfc [Ry]" "$CUR_QE_ECUTWFC")"
    QE_ECUTRHO="$(ask "  qe.ecutrho [Ry]" "$CUR_QE_ECUTRHO")"
    QE_KPOINTS="$(ask "  qe.k_points" "$CUR_QE_KPTS")"
    QE_OCCUPATIONS="$(ask "  qe.occupations" "$CUR_QE_OCC")"
    QE_SMEARING="$(ask "  qe.smearing" "$CUR_QE_SMEAR")"
    QE_DEGAUSS="$(ask "  qe.degauss [Ry]" "$CUR_QE_DEGAUSS")"
    QE_CONV_THR="$(ask "  qe.conv_thr" "$CUR_QE_CONV")"
    QE_ELECTRON_MAXSTEP="$(ask "  qe.electron_maxstep" "$CUR_QE_EMAX")"
    QE_MIXING_BETA="$(ask "  qe.mixing_beta" "$CUR_QE_MIX")"
    QE_DIAGONALIZATION="$(ask "  qe.diagonalization" "$CUR_QE_DIAG")"

    for pair in \
        "QE_ECUTWFC:$QE_ECUTWFC" "QE_ECUTRHO:$QE_ECUTRHO" "QE_DEGAUSS:$QE_DEGAUSS" \
        "QE_CONV_THR:$QE_CONV_THR" "QE_MIXING_BETA:$QE_MIXING_BETA"; do
        name="${pair%%:*}"; val="${pair#*:}"
        if ! [[ "$val" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
            echo "ERROR: $name must be numeric, got '$val'" >&2; exit 2
        fi
    done
    if ! [[ "$QE_ELECTRON_MAXSTEP" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: QE_ELECTRON_MAXSTEP must be a positive integer, got '$QE_ELECTRON_MAXSTEP'" >&2; exit 2
    fi
    if [[ "$(wc -w <<<"$QE_KPOINTS" | tr -d ' ')" != "6" ]]; then
        echo "ERROR: qe.k_points must have 6 integers, got '$QE_KPOINTS'" >&2; exit 2
    fi
    for k in $QE_KPOINTS; do
        if ! [[ "$k" =~ ^[0-9]+$ ]]; then
            echo "ERROR: qe.k_points must contain integers only, got '$QE_KPOINTS'" >&2; exit 2
        fi
    done
fi

# ---- 9. (optional/master, --lammps) production LAMMPS setup review ---------
if [[ "$DO_LAMMPS" == "1" ]]; then
    echo
    if [[ "$MASTER_MODE_REQUESTED" == "1" ]]; then
        review_lmp="$(ask "Review LAMMPS/QM-MM production setup now? (Y/n)" "Y")"
        if [[ "$review_lmp" =~ ^[Nn] ]]; then
            DO_LAMMPS=0
        fi
    fi
fi
if [[ "$DO_LAMMPS" == "1" ]]; then
    echo
    echo "LAMMPS/QM-MM production setup  (ces2.md; Enter keeps current values)"
    LMP_TIMESTEP="$(ask "  timestep_fs" "$CUR_LMP_TIMESTEP")"
    LMP_N_STEPS="$(ask "  n_steps (0 = single-point)" "$CUR_LMP_NSTEPS")"
    LMP_THERMO_EVERY="$(ask "  thermo_every" "$CUR_LMP_THERMO")"
    LMP_DUMP_EVERY="$(ask "  dump_every" "$CUR_LMP_DUMP")"
    LMP_TEMPERATURE="$(ask "  temperature [K]" "$CUR_LMP_TEMP")"
    LMP_RESTART_EVERY="$(ask "  restart_every" "$CUR_LMP_RESTART")"

    if ! [[ "$LMP_TIMESTEP" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
        echo "ERROR: timestep_fs must be numeric, got '$LMP_TIMESTEP'" >&2; exit 2
    fi
    if ! [[ "$LMP_TEMPERATURE" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
        echo "ERROR: temperature must be numeric, got '$LMP_TEMPERATURE'" >&2; exit 2
    fi
    for pair in \
        "n_steps:$LMP_N_STEPS" "thermo_every:$LMP_THERMO_EVERY" \
        "dump_every:$LMP_DUMP_EVERY" "restart_every:$LMP_RESTART_EVERY"; do
        name="${pair%%:*}"; val="${pair#*:}"
        if ! [[ "$val" =~ ^(0|[1-9][0-9]*)$ ]]; then
            echo "ERROR: $name must be a non-negative integer, got '$val'" >&2; exit 2
        fi
    done
fi

# ---- 8. (optional, --prerelax) pre-relaxation pipeline toggle ----
if [[ "$DO_PRERELAX" == "1" ]]; then
    echo
    echo "Pre-relaxation pipeline  (md_relax.enabled + ces2.initial_dump)"
    if [[ "$CUR_RELAX" == "True" ]] || [[ "$CUR_RELAX" == "true" ]]; then
        cur_pre="on"
    else
        cur_pre="off"
    fi
    echo "  current: enabled=$CUR_RELAX, initial_dump='$CUR_INITDMP'"
    echo "  on  → MM relax (min→heat→equil) before QM/MM, base.in.lammps reads equilibrated.dump"
    echo "  off → skip relax, QM/MM starts directly from packmol coords"
    while :; do
        PRERELAX="$(ask "  enable pre-relaxation? (on/off)" "$cur_pre")"
        case "$(printf '%s' "$PRERELAX" | tr '[:upper:]' '[:lower:]')" in
            on|off|true|false|1|0|yes|no) break ;;
            *) echo "  please answer on or off" ;;
        esac
    done
fi

# ---- 9. (optional, --parallel) QE parallelization flags ----
if [[ "$DO_PARALLEL" == "1" ]]; then
    echo
    echo "QE parallelization flags  (ces2_script.{npool, ntg, ndiag})"
    echo "  npool : k-point pools     — must divide n_kpoints"
    echo "  ntg   : FFT task groups   — divisor of (np/npool); 2~4 typical"
    echo "  ndiag : ScaLAPACK procs   — perfect square (1,4,9,16,25,...)"
    echo "  Tip: ndiag=1 disables parallel Cholesky; useful when pzpotrf fails."
    NPOOL="$(ask "  npool" "$CUR_NPOOL")"
    NTG="$(ask   "  ntg"   "$CUR_NTG")"
    NDIAG="$(ask "  ndiag" "$CUR_NDIAG")"
    for var in NPOOL NTG NDIAG; do
        val="${!var}"
        if [[ -n "$val" && ! "$val" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: $var must be a positive integer, got '$val'" >&2; exit 2
        fi
    done
    if [[ -n "$NDIAG" ]]; then
        sq=$(awk -v n="$NDIAG" 'BEGIN{r=int(sqrt(n)); print (r*r==n) ? 1 : 0}')
        if [[ "$sq" != "1" ]]; then
            echo "ERROR: ndiag must be a perfect square (1,4,9,16,25,...), got '$NDIAG'" >&2; exit 2
        fi
    fi
fi

# ---- 10. (optional, --qmmm) QMMMFINSTEP ----
if [[ "$DO_QMMM" == "1" ]]; then
    echo
    echo "QM/MM outer-loop length  (QMMMFINSTEP in qmmm_dftces2_charging_pts.sh)"
    echo "  Inclusive 0-based upper bound: total steps = QMMMFINSTEP + 1."
    echo "  YAML's ces2_script.n_qmmm_steps is auto-synced to (QMMMFINSTEP + 1)."
    if [[ -n "$CUR_QMMMFINSTEP" ]]; then
        echo "  current QMMMFINSTEP in $QMMM_SCRIPT_PATH : $CUR_QMMMFINSTEP"
        DEFAULT_FIN="$CUR_QMMMFINSTEP"
    elif [[ -n "$CUR_NSTEPS" ]]; then
        echo "  qmmm script not found in CWD; deriving default from YAML"
        echo "    YAML n_qmmm_steps = $CUR_NSTEPS  →  QMMMFINSTEP = $((CUR_NSTEPS - 1))"
        DEFAULT_FIN="$((CUR_NSTEPS - 1))"
    else
        DEFAULT_FIN=""
    fi
    if [[ -z "$CUR_QMMMFINSTEP" ]]; then
        echo "  (no qmmm script in CWD — only YAML's n_qmmm_steps will be updated)"
    fi
    QMMMFINSTEP="$(ask "  QMMMFINSTEP" "$DEFAULT_FIN")"
    if ! [[ "$QMMMFINSTEP" =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo "ERROR: QMMMFINSTEP must be a non-negative integer, got '$QMMMFINSTEP'" >&2; exit 2
    fi
fi

echo
echo "------------------------------------------"
echo "Summary:"
echo "  mode           : $MODE"
if [[ "$MODE" == "master" ]]; then
    echo "  bjparams_source: $BJ_SOURCE"
    if [[ "$BJ_SOURCE" == "yaml" ]]; then
        echo "  qm_params_file : $QM_PARAMS_FILE"
    else
        echo "  layer file     : $BJ_LAYER_FILE"
    fi
    echo "  water_model    : $WATER_MODEL"
    echo "  adsorbate N    : $ADSORB"
fi
if [[ "$MODE" == "variant" ]]; then
    echo "  salt           : $SALT"
    echo "  concentration  : $CONC M"
    echo "  cation in pool : $CATION"
    echo "  anion  in pool : $ANION"
    echo "  electrode q    : $QCHARGE e"
    echo "  job name       : $JOBNAME"
    echo "  build dir      : $OUTPUT_BUILD_DIR"
    echo "  export dir     : $OUTPUT_EXPORT_DIR"
fi
if [[ "$DO_BOX" == "1" ]]; then
    echo "  thickness      : $THICKNESS Å         (was $CUR_THICK)"
    echo "  vacuum_z       : $VACUUM_Z Å         (was $CUR_VACZ)"
    if [[ -n "$WATER_COUNT" ]]; then
        echo "  water.count    : $WATER_COUNT          (was $CUR_WCOUNT)"
    fi
fi
if [[ "$DO_QE" == "1" ]]; then
    echo "  qe.k_points    : $QE_KPOINTS          (was $CUR_QE_KPTS)"
    echo "  qe.pseudo_dir  : $QE_PSEUDO_DIR"
    echo "  qe.pseudo_set  : $QE_PSEUDO_SET"
    echo "  qe.cutoffs     : ecutwfc=$QE_ECUTWFC, ecutrho=$QE_ECUTRHO"
fi
if [[ "$DO_LAMMPS" == "1" ]]; then
    echo "  lammps timestep: $LMP_TIMESTEP fs     (was $CUR_LMP_TIMESTEP)"
    echo "  lammps run     : n_steps=$LMP_N_STEPS, thermo=$LMP_THERMO_EVERY, dump=$LMP_DUMP_EVERY"
    echo "  lammps temp    : $LMP_TEMPERATURE K"
fi
if [[ "$DO_PRERELAX" == "1" ]]; then
    case "$(printf '%s' "$PRERELAX" | tr '[:upper:]' '[:lower:]')" in
        on|true|1|yes)  PR_VIEW="on  (md_relax.enabled=true,  initial_dump=equilibrated.dump)" ;;
        *)              PR_VIEW="off (md_relax.enabled=false, initial_dump removed)" ;;
    esac
    echo "  pre-relaxation : $PR_VIEW           (was enabled=$CUR_RELAX)"
fi
if [[ "$DO_PARALLEL" == "1" ]]; then
    echo "  npool          : $NPOOL          (was $CUR_NPOOL)"
    echo "  ntg            : $NTG          (was $CUR_NTG)"
    echo "  ndiag          : $NDIAG          (was $CUR_NDIAG)"
fi
if [[ "$DO_QMMM" == "1" ]]; then
    new_nsteps=$((QMMMFINSTEP + 1))
    echo "  QMMMFINSTEP    : $QMMMFINSTEP          (was ${CUR_QMMMFINSTEP:-?})"
    echo "  n_qmmm_steps   : $new_nsteps          (was ${CUR_NSTEPS:-?})  [YAML, auto-synced]"
fi
echo "------------------------------------------"
read -r -p "Apply these changes to $CONFIG? [Y/n] " confirm
confirm="${confirm:-Y}"
if [[ ! "$confirm" =~ ^[Yy] ]]; then
    echo "Aborted."
    exit 0
fi

# ---- backup: .<basename>.bak in same dir ----
DIR="$(dirname -- "$CONFIG")"
BASE="$(basename -- "$CONFIG")"
BAK="${DIR}/.${BASE}.bak"
cp -p -- "$CONFIG" "$BAK"
echo "backup -> $BAK"

# ---- apply edits with ruamel.yaml ----
export CONFIG MODE SALT CONC CATION ANION QCHARGE ADSORB JOBNAME THICKNESS VACUUM_Z WATER_COUNT PRERELAX NPOOL NTG NDIAG QMMMFINSTEP BJ_SOURCE QM_PARAMS_FILE BJ_LAYER_FILE WATER_MODEL OUTPUT_BUILD_DIR OUTPUT_EXPORT_DIR QE_PREFIX QE_OUTDIR QE_PSEUDO_DIR QE_PSEUDO_SET QE_PSEUDOS QE_ECUTWFC QE_ECUTRHO QE_KPOINTS QE_OCCUPATIONS QE_SMEARING QE_DEGAUSS QE_CONV_THR QE_ELECTRON_MAXSTEP QE_MIXING_BETA QE_DIAGONALIZATION LMP_TIMESTEP LMP_N_STEPS LMP_THERMO_EVERY LMP_DUMP_EVERY LMP_TEMPERATURE LMP_RESTART_EVERY
python3 <<'PY'
import os, sys
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)
yaml.width = 4096   # don't wrap long inline flow-style maps/sequences

path = os.environ["CONFIG"]
with open(path) as f:
    d = yaml.load(f)

mode = os.environ.get("MODE", "variant")

recipe = d.setdefault("electrolyte_recipe", CommentedMap())
cs = d.setdefault("ces2_script", CommentedMap())

if mode == "variant":
    salt    = os.environ["SALT"]
    conc    = float(os.environ["CONC"])
    cation  = os.environ["CATION"]
    anion   = os.environ["ANION"]
    qcharge = float(os.environ["QCHARGE"])
    jobname = os.environ["JOBNAME"]

    # --- electrolyte_recipe.salts ---
    if salt == "water":
        recipe["salts"] = []
    else:
        entry = CommentedMap()
        entry["name"] = salt
        entry["concentration_M"] = conc
        stoich = CommentedMap()
        stoich[cation] = 1
        stoich[anion]  = 1
        stoich.fa.set_flow_style()     # keep {a: 1, b: 1} style
        entry["stoich"] = stoich
        seq = CommentedSeq()
        seq.append(entry)
        recipe["salts"] = seq

    # --- counterion_pool ---
    pool = CommentedSeq()
    pool.append(cation)
    pool.append(anion)
    pool.fa.set_flow_style()            # keep ["a", "b"] style
    recipe["counterion_pool"] = pool

    # --- charge_control.q_electrode_user_value ---
    cc = d.setdefault("charge_control", CommentedMap())
    cc["q_electrode_user_value"] = qcharge

    # --- ces2_script.jobname / pbs.job_name ---
    cs["jobname"] = jobname
    if "pbs" in cs and isinstance(cs["pbs"], dict):
        cs["pbs"]["job_name"] = jobname

    out = d.setdefault("output", CommentedMap())
    build_dir = os.environ.get("OUTPUT_BUILD_DIR", "").strip()
    export_dir = os.environ.get("OUTPUT_EXPORT_DIR", "").strip()
    if build_dir:
        out["build_dir"] = build_dir
    if export_dir:
        out["export_dir"] = export_dir

if mode == "master":
    slab = d.setdefault("slab", CommentedMap())
    bj_source = os.environ.get("BJ_SOURCE", "").strip()
    if bj_source:
        slab["bjparams_source"] = bj_source
        if bj_source == "yaml":
            qm_file = os.environ.get("QM_PARAMS_FILE", "").strip()
            if qm_file:
                slab["qm_params_file"] = qm_file
            slab.pop("bjparams_layer_file", None)
        elif bj_source == "layer_file":
            layer_file = os.environ.get("BJ_LAYER_FILE", "").strip()
            if layer_file:
                slab["bjparams_layer_file"] = layer_file
            slab.pop("qm_params_file", None)

    water_model = os.environ.get("WATER_MODEL", "").strip()
    if water_model:
        ces2_blk = d.setdefault("ces2", CommentedMap())
        ces2_blk["water_model"] = water_model
        # Let builder derive the matching species from ces2.water_model unless
        # the user later opts into a custom species_id.
        water_blk = recipe.setdefault("water", CommentedMap())
        water_blk.pop("species_id", None)

    adsorb = os.environ.get("ADSORB", "").strip()
    if adsorb:
        cs["adsorbate"] = str(adsorb)

# --- (optional, --box) electrolyte_box geometry ----------------------------
# Empty env var means the section flag wasn't given; preserve current YAML.
thickness_str = os.environ.get("THICKNESS", "").strip()
vacuum_z_str  = os.environ.get("VACUUM_Z",  "").strip()
if thickness_str or vacuum_z_str:
    ebox = d.setdefault("electrolyte_box", CommentedMap())
    if thickness_str:
        ebox["thickness"] = float(thickness_str)
    if vacuum_z_str:
        ebox["vacuum_z"] = float(vacuum_z_str)

# --- (optional, --box helper) electrolyte_recipe.water.count -----------------
# WATER_COUNT is set by the bash helper only when thickness changed AND the
# user picked a non-"keep" answer.  "auto" -> bare auto string;  positive int
# -> explicit override.  Empty string -> leave YAML untouched.
water_count_str = os.environ.get("WATER_COUNT", "").strip()
if water_count_str:
    water_blk = recipe.setdefault("water", CommentedMap())
    if water_count_str.lower() == "auto":
        water_blk["count"] = "auto"
    else:
        water_blk["count"] = int(water_count_str)

# --- (optional/master, --qe) QE setup review -------------------------------
qe_prefix = os.environ.get("QE_PREFIX", "").strip()
if qe_prefix:
    qe = d.setdefault("qe", CommentedMap())
    qe["prefix"] = qe_prefix
    qe["outdir"] = os.environ.get("QE_OUTDIR", "").strip()
    qe["pseudo_dir"] = os.environ.get("QE_PSEUDO_DIR", "").strip()
    qe["pseudo_set"] = os.environ.get("QE_PSEUDO_SET", "").strip()
    qe["ecutwfc"] = float(os.environ.get("QE_ECUTWFC", "").strip())
    qe["ecutrho"] = float(os.environ.get("QE_ECUTRHO", "").strip())
    kseq = CommentedSeq(int(x) for x in os.environ.get("QE_KPOINTS", "").split())
    kseq.fa.set_flow_style()
    qe["k_points"] = kseq
    qe["occupations"] = os.environ.get("QE_OCCUPATIONS", "").strip()
    qe["smearing"] = os.environ.get("QE_SMEARING", "").strip()
    qe["degauss"] = float(os.environ.get("QE_DEGAUSS", "").strip())
    qe["conv_thr"] = float(os.environ.get("QE_CONV_THR", "").strip())
    qe["electron_maxstep"] = int(os.environ.get("QE_ELECTRON_MAXSTEP", "").strip())
    qe["mixing_beta"] = float(os.environ.get("QE_MIXING_BETA", "").strip())
    qe["diagonalization"] = os.environ.get("QE_DIAGONALIZATION", "").strip()

    pseudos_s = os.environ.get("QE_PSEUDOS", "").strip()
    if pseudos_s:
        overrides = CommentedMap()
        for item in pseudos_s.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise SystemExit(f"Invalid qe.pseudopotentials override '{item}'. Use El=file,El=file")
            el, fname = item.split("=", 1)
            overrides[el.strip()] = fname.strip()
        qe["pseudopotentials"] = overrides
    elif "pseudopotentials" in qe and not pseudos_s:
        # Empty prompt with no current overrides means no explicit overrides.
        qe.pop("pseudopotentials", None)

# --- (optional/master, --lammps) production LAMMPS/QM-MM setup -------------
lmp_timestep = os.environ.get("LMP_TIMESTEP", "").strip()
if lmp_timestep:
    ces2_blk = d.setdefault("ces2", CommentedMap())
    md = ces2_blk.setdefault("md", CommentedMap())
    md["timestep_fs"] = float(lmp_timestep)
    md["n_steps"] = int(os.environ.get("LMP_N_STEPS", "").strip())
    md["thermo_every"] = int(os.environ.get("LMP_THERMO_EVERY", "").strip())
    md["dump_every"] = int(os.environ.get("LMP_DUMP_EVERY", "").strip())
    md["temperature"] = float(os.environ.get("LMP_TEMPERATURE", "").strip())
    md["restart_every"] = int(os.environ.get("LMP_RESTART_EVERY", "").strip())

# --- (optional, --prerelax) pre-relaxation toggle --------------------------
# on  → md_relax.enabled=true,  ces2.initial_dump="equilibrated.dump"
# off → md_relax.enabled=false, drop ces2.initial_dump
prerelax_str = os.environ.get("PRERELAX", "").strip().lower()
if prerelax_str:
    enable_relax = prerelax_str in ("on", "true", "1", "yes", "y")
    relax = d.setdefault("md_relax", CommentedMap())
    relax["enabled"] = enable_relax
    ces2_blk = d.setdefault("ces2", CommentedMap())
    if enable_relax:
        ces2_blk["initial_dump"] = "equilibrated.dump"
    else:
        # Remove the key entirely so base.in.lammps emits no read_dump line.
        # Block-level documentation comments above the (now-removed) key are
        # left in place by ruamel.yaml's comment preservation.
        ces2_blk.pop("initial_dump", None)

# --- (optional, --parallel) QE parallelization flags -----------------------
# Sets ces2_script.{npool, ntg, ndiag}. Empty env var (section flag absent)
# leaves the YAML untouched; otherwise each provided value overwrites.
for env_key, yaml_key in (("NPOOL", "npool"), ("NTG", "ntg"), ("NDIAG", "ndiag")):
    val = os.environ.get(env_key, "").strip()
    if val:
        cs.setdefault(yaml_key, 0)
        cs[yaml_key] = int(val)

# --- (optional, --qmmm) QMMMFINSTEP → YAML n_qmmm_steps sync ---------------
# QMMMFINSTEP is the inclusive 0-based upper bound used by the qmmm script's
# for-loop. Total iterations = QMMMFINSTEP + 1, which is what n_qmmm_steps
# represents in the YAML. Keep them in lockstep so the next rebuild doesn't
# silently revert the script.
qmmmfinstep_str = os.environ.get("QMMMFINSTEP", "").strip()
if qmmmfinstep_str:
    cs["n_qmmm_steps"] = int(qmmmfinstep_str) + 1

with open(path, "w") as f:
    yaml.dump(d, f)

print(f"updated -> {path}")
PY

validate_config

# ---- (optional, --qmmm) patch QMMMFINSTEP in qmmm_dftces2_charging_pts.sh ----
# The YAML side was already synced inside the Python block above. Now also
# patch the in-place qmmm script if it exists in CWD, so a resubmit picks up
# the new bound without having to rebuild.
if [[ "$DO_QMMM" == "1" ]] && [[ -f "$QMMM_SCRIPT_PATH" ]]; then
    QMMM_BAK="${QMMM_SCRIPT_PATH}.bak.$(date +%Y%m%d_%H%M%S)"
    cp -p -- "$QMMM_SCRIPT_PATH" "$QMMM_BAK"
    # Replace ONLY the numeric value, preserve any trailing comment.
    sed -i.tmp -E "s|^([[:space:]]*QMMMFINSTEP=)[0-9]+([[:space:]]*.*)?$|\1${QMMMFINSTEP}\2|" "$QMMM_SCRIPT_PATH"
    rm -f "${QMMM_SCRIPT_PATH}.tmp"
    echo "qmmm script -> $QMMM_SCRIPT_PATH  (backup: $QMMM_BAK)"
    grep -E '^[[:space:]]*QMMMFINSTEP=' "$QMMM_SCRIPT_PATH" | sed 's/^/  /'
fi

echo "Done."
