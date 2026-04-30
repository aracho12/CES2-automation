#!/usr/bin/env bash
# Interactive wizard to configure a CES2 build YAML.
#
# Usage:
#   tools/configure.sh [OPTIONS] [path/to/config.yaml]
#
# Without options, prompts for: salt, concentration, electrode charge,
# adsorbate count, job name (and only edits those keys).
#
# Optional section flags — each opens an extra interactive prompt block that
# shows the current YAML value as the default, so hitting <Enter> keeps it.
# Sections that aren't requested are NOT touched at all:
#
#   --box           Box geometry: electrolyte_box.thickness, vacuum_z [Å]
#   --prerelax      Pre-relaxation pipeline: md_relax.enabled (on/off) and
#                   ces2.initial_dump (auto-set when on, removed when off)
#   --parallel      QE parallelization flags: ces2_script.{npool, ntg, ndiag}
#                   (pw.x mpirun options; see config_example.yaml comments)
#   --qmmm          Edit QMMMFINSTEP in ./qmmm_dftces2_charging_pts.sh (if
#                   present in CWD) and keep YAML's ces2_script.n_qmmm_steps
#                   in sync (n_qmmm_steps = QMMMFINSTEP + 1).
#   -h, --help      Show this help and exit.
#
# Combine freely:
#   tools/configure.sh --box --prerelax --parallel --qmmm myconfig.yaml
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
CONFIG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --box)              DO_BOX=1; shift ;;
        --prerelax)         DO_PRERELAX=1; shift ;;
        --parallel)         DO_PARALLEL=1; shift ;;
        --qmmm)             DO_QMMM=1; shift ;;
        -h|--help)          print_help; exit 0 ;;
        --)                 shift; CONFIG="${1:-}"; break ;;
        -*)                 echo "ERROR: unknown option: $1" >&2; print_help; exit 2 ;;
        *)                  CONFIG="$1"; shift ;;
    esac
done

CONFIG="${CONFIG:-${REPO_ROOT}/config_example.yaml}"

# These get populated by interactive prompts only when their section flag is set.
# Empty value at python-apply time means "do not touch".
THICKNESS=""
VACUUM_Z=""
PRERELAX=""
NPOOL=""
NTG=""
NDIAG=""
QMMMFINSTEP=""

# Path to the qmmm script in the current working directory (if any).
QMMM_SCRIPT_PATH="./qmmm_dftces2_charging_pts.sh"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

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

# ---- read current values for default-prompts ----
read_defaults() {
    CONFIG="$CONFIG" python3 <<'PY'
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

print(f"{salt_name}|{salt_conc}|{qcharge}|{adsorb}|{jobname}|{cur_thick}|{cur_vacz}|{cur_relax}|{cur_initdmp}|{cur_npool}|{cur_ntg}|{cur_ndiag}|{cur_nsteps}")
PY
}

IFS='|' read -r CUR_SALT CUR_CONC CUR_CHARGE CUR_ADSORB CUR_JOB CUR_THICK CUR_VACZ CUR_RELAX CUR_INITDMP CUR_NPOOL CUR_NTG CUR_NDIAG CUR_NSTEPS <<<"$(read_defaults)"

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

echo
echo "=========================================="
echo "  CES2 build config wizard"
echo "  target: $CONFIG"
echo "=========================================="
echo

# ---- 1. salt preset ----
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
JOBNAME="$(ask "Job name (sets ces2_script.jobname AND ces2_script.pbs.job_name)" "$CUR_JOB")"

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
echo "  salt           : $SALT"
echo "  concentration  : $CONC M"
echo "  cation in pool : $CATION"
echo "  anion  in pool : $ANION"
echo "  electrode q    : $QCHARGE e"
echo "  adsorbate N    : $ADSORB"
echo "  job name       : $JOBNAME"
if [[ "$DO_BOX" == "1" ]]; then
    echo "  thickness      : $THICKNESS Å         (was $CUR_THICK)"
    echo "  vacuum_z       : $VACUUM_Z Å         (was $CUR_VACZ)"
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
export CONFIG SALT CONC CATION ANION QCHARGE ADSORB JOBNAME THICKNESS VACUUM_Z PRERELAX NPOOL NTG NDIAG QMMMFINSTEP
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

salt    = os.environ["SALT"]
conc    = float(os.environ["CONC"])
cation  = os.environ["CATION"]
anion   = os.environ["ANION"]
qcharge = float(os.environ["QCHARGE"])
adsorb  = os.environ["ADSORB"]
jobname = os.environ["JOBNAME"]

# --- electrolyte_recipe.salts ---
recipe = d.setdefault("electrolyte_recipe", CommentedMap())
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

# --- ces2_script.adsorbate / jobname / pbs.job_name ---
cs = d.setdefault("ces2_script", CommentedMap())
cs["adsorbate"] = str(adsorb)
cs["jobname"]   = jobname
if "pbs" in cs and isinstance(cs["pbs"], dict):
    cs["pbs"]["job_name"] = jobname

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
