#!/usr/bin/env bash
# extract_fiscs.sh — Extract vdW forcefield parameters from VASP OUTCAR
#
# Usage:
#   ./extract_fiscs.sh [OUTCAR]          (default: ./OUTCAR)
#
# Output:
#   fiscs.out          — full per-atom table with column header
#   fiscs-average.out  — per-element averages (C6ts, ALPHAscs, C6 in kcal/mol·Å^6)

OUTCAR="${1:-OUTCAR}"

if [ ! -f "$OUTCAR" ]; then
    echo "ERROR: file not found: $OUTCAR" >&2
    exit 1
fi

# ── fiscs.out : full per-atom table ──────────────────────────────────────────
# Matches any element line: leading spaces, capital letter (optionally followed
# by a lowercase letter), then a number — works for any element symbol.
awk '
/Parameters of vdW forcefield/ { delete a; i=0; f=1 }
f && /^[[:space:]]+[A-Z][a-z]?[[:space:]]+[0-9]/ { a[i++] = $0 }
END {
    printf "%-8s  %16s  %14s  %12s  %12s  %12s  %10s\n",
           "Element", "C6ts(au)", "R0ts(au)", "ALPHAts(au)", "R0scs(au)", "ALPHAscs(au)", "RELVOL"
    print  "--------------------------------------------------------------------------------------------"
    for (j=0; j<i; j++) print a[j]
}
' "$OUTCAR" > fiscs.out

n_atoms=$(awk '/^[[:space:]]+[A-Z]/ && !/Element/' fiscs.out | wc -l)
echo "[extract_fiscs] Written: fiscs.out  (${n_atoms} atoms)"

# ── fiscs-average.out : per-element averages ──────────────────────────────────
awk '
/Parameters of vdW forcefield/ { delete a; i=0; f=1 }
f && /^[[:space:]]+[A-Z][a-z]?[[:space:]]+[0-9]/ { a[i++] = $0 }
END {
    for (j=0; j<i; j++) {
        split(a[j], c)
        el = c[1]
        if (!(el in order_idx)) {
            order[n_el++] = el   # preserve first-appearance order
            order_idx[el] = 1
        }
        sum_c6[el]    += c[2]
        sum_r0ts[el]  += c[3]
        sum_alpha[el] += c[6]
        sum_relvol[el]+= c[7]
        cnt[el]++
    }
    # C6 unit conversion: au → kcal/mol·Å^6
    # 1 Ha·a0^6 ; 1 Ha = 627.5094740631 kcal/mol ; 1 a0 = 0.529177 Å
    # factor = 627.5094740631 * 0.529177^6 = 13.77928721
    conv = 13.77928721

    printf "%-8s  %5s  %14s  %15s  %14s  %18s\n",
           "Element", "N", "C6ts_avg(au)", "ALPHAscs_avg(au)", "RELVOL_avg", "C6ts_avg(kcal/mol*A6)"
    print  "--------------------------------------------------------------------------------------------"
    for (k=0; k<n_el; k++) {
        el = order[k]
        printf "%-8s  %5d  %14.3f  %15.3f  %14.3f  %18.3f\n",
               el, cnt[el],
               sum_c6[el]/cnt[el],
               sum_alpha[el]/cnt[el],
               sum_relvol[el]/cnt[el],
               sum_c6[el]/cnt[el] * conv
    }
}
' "$OUTCAR" > fiscs-average.out

echo "[extract_fiscs] Written: fiscs-average.out"
