#!/usr/bin/env bash
# Interactive wizard to configure a CES2 build YAML.
# Usage:  tools/configure.sh [path/to/config.yaml]
# Backup: writes .<basename>.bak alongside the original before editing.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${REPO_ROOT}/config_example.yaml}"

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

print(f"{salt_name}|{salt_conc}|{qcharge}|{adsorb}|{jobname}")
PY
}

IFS='|' read -r CUR_SALT CUR_CONC CUR_CHARGE CUR_ADSORB CUR_JOB <<<"$(read_defaults)"

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
export CONFIG SALT CONC CATION ANION QCHARGE ADSORB JOBNAME
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

with open(path, "w") as f:
    yaml.dump(d, f)

print(f"updated -> {path}")
PY

echo "Done."
