"""
ces2_script_writer.py — Generate CES2 QM/MM run scripts.

Based on the DFT-CES2 production script by Taehwan Jang, M-design group @ KAIST.
This writer substitutes system-specific values from config.yaml into the script
template.  The core bash logic (Blur_MM, Blur_QM, post_dipole, main loop) is kept
identical to the production script; only the top-level variable declarations are
auto-generated from config.

Files generated
---------------
  qmmm_dftces2_charging_pts.sh   — main QM/MM outer-loop wrapper script
  submit_ces2.sh                  — SLURM / PBS batch submission script

Config keys consumed (ces2_script:)
------------------------------------
  dft_ces2_path    : root directory of the DFT-CES2 installation
  qe_binary        : full path to pw.x
  pp_binary        : full path to pp.x
  lmp_binary       : full path to lmp_mpi
  chg2pot_binary   : full path to chg2pot utility
  mdipc_binary     : full path to mdipc utility  (default: DFT_CES2_PATH/tools/3-poisson/mdipc)
  chgplate_binary  : full path to make_rho_mino   (default: needs explicit path)
  jobname          : job name used for SBATCH -J and prefix= variable in submit script
  np               : MPI tasks for QE + LAMMPS (single value)
  qm_type          : "scf" or "opt"  (default "scf")
  qm_relax_high    : relax QM atoms below this z value [Ang] (for opt, default 7)
  qmmm_ini_step    : first QM/MM step index (default 0)
  n_qmmm_steps     : total QM/MM steps; QMMMFINSTEP = n_qmmm_steps - 1
  qm_max_step      : max SCF/relax trials per step (default 5)
  md_equil_steps   : LAMMPS equilibration MD steps per QM/MM iteration
  md_average_steps : LAMMPS averaging MD steps per QM/MM iteration
  dipole_corr      : "yes"/"no" — enable slab dipole correction (default "yes")
  tot_chg          : total extra charge on QM system (e.g. "-0.01652...")
  mpc_layer        : MPC layer position [bohr]
  tot_layer        : number of total MPC layers
  mpc_one          : mpcone parameter for mdipc
  adsorbate        : number of adsorbate atoms for mdipc
  plate_pos        : plate position [bohr] between electrode bottom and top
  initial_qm       : 1 = initial QM already done (skip step-0 pw.x); 0 = run it
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Element full name → cube file naming convention
# ---------------------------------------------------------------------------
_ELEMENT_FULLNAME: Dict[str, str] = {
    "H":  "hydrogen",  "He": "helium",
    "Li": "lithium",   "Be": "beryllium",
    "B":  "boron",     "C":  "carbon",
    "N":  "nitrogen",  "O":  "oxygen",
    "F":  "fluorine",  "Ne": "neon",
    "Na": "sodium",    "Mg": "magnesium",
    "Al": "aluminum",  "Si": "silicon",
    "P":  "phosphorus","S":  "sulfur",
    "Cl": "chlorine",  "Ar": "argon",
    "K":  "potassium", "Ca": "calcium",
    "Sc": "scandium",  "Ti": "titanium",
    "Cr": "chromium",  "Mn": "manganese",
    "Fe": "iron",      "Co": "cobalt",
    "Ni": "nickel",    "Cu": "copper",
    "Zn": "zinc",      "Rb": "rubidium",
    "Cs": "cesium",    "Br": "bromine",
    "I":  "iodine",    "Ir": "iridium",
    "Pt": "platinum",  "Au": "gold",
    "Ag": "silver",    "W":  "tungsten",
}


def _element_cube_name(element: str) -> str:
    """Return the cube output filename for a given element. E.g. K → potassium.cube"""
    return _ELEMENT_FULLNAME.get(element, element.lower()) + ".cube"


# ---------------------------------------------------------------------------
# Build MM arrays from species_order + species_db
# ---------------------------------------------------------------------------

def _build_mm_arrays(
    water_sid: str,
    species_order: List[Tuple[str, Any]],
    species_db: Dict[str, Any],
    type_id_by_label: Dict[str, int],
) -> Tuple[List[str], List[str], List[str], bool, int]:
    """
    Build MM species arrays needed by the qmmm script.

    The DFT-CES2 framework uses ONE cube per unique element in the MM system
    (not one per type_label).  All H atoms — whether from water or OH⁻ — share
    hydrogen.cube (cube index 0), and all O atoms share oxygen.cube (cube index 1).
    This matches lammps_input_writer.py: cube_idx.get(el, next_cube) re-uses the
    index already assigned to the same element by water.

    Returns
    -------
    mm_elements    : unique element symbol list (MMrepA bash array), cube-index order
    mm_charges     : representative partial charge per element (MMpartialCharge)
    mm_cube_output : cube output file name list (cube_output_MM bash array)
    is_tip4p       : True when TIP4P water is detected
    tip4p_O_idx    : 0-based index of O in mm_elements (TIP4Pinvolved[1])

    Cube index ordering (matches lammps_input_writer gridforce convention):
      idx 0 : H  (from water Hw — all H atoms share this cube)
      idx 1 : O  (from water Ow — all O atoms, including OH⁻, share this cube)
      idx 2+: new elements from non-water species, first-appearance order
    """
    mm_elements:    List[str] = []
    mm_charges_str: List[str] = []
    mm_cube_output: List[str] = []
    is_tip4p = False
    tip4p_O_idx = 0
    seen_elements: set = set()   # de-dup by element symbol

    # ── Water species: H first (cube idx 0), O second (cube idx 1) ────────
    water_sp = species_db.get(water_sid)
    if water_sp is not None:
        h_atom = next((a for a in water_sp.atoms if a.element == "H"), None)
        o_atom = next((a for a in water_sp.atoms if a.element == "O"), None)
        # TIP4P M-site: element "M", "MW", or "EP" (virtual charge site)
        m_atom = next((a for a in water_sp.atoms
                       if a.element in ("M", "MW", "EP")), None)

        if h_atom is not None:
            mm_elements.append("H")
            mm_charges_str.append(f"{h_atom.charge:.4f}")
            mm_cube_output.append("hydrogen.cube")
            seen_elements.add("H")

        if o_atom is not None:
            tip4p_O_idx = len(mm_elements)
            # TIP4P: Ow charge is 0; actual electrostatic charge is on M-site
            if m_atom is not None and abs(o_atom.charge) < 1e-6:
                is_tip4p = True
                o_charge_val = m_atom.charge
            else:
                o_charge_val = o_atom.charge
            mm_elements.append("O")
            mm_charges_str.append(f"{o_charge_val:.4f}")
            mm_cube_output.append("oxygen.cube")
            seen_elements.add("O")

    # ── Non-water species: one entry per NEW element only ─────────────────
    # Elements already assigned to water H/O are skipped (same cube shared).
    for sid, _ in species_order:
        if sid == water_sid:
            continue
        sp = species_db.get(sid)
        if sp is None:
            continue
        sorted_atoms = sorted(
            sp.atoms,
            key=lambda a: type_id_by_label.get(a.type_label, 9999)
        )
        for atom in sorted_atoms:
            el = atom.element
            if el in seen_elements:
                continue   # H or O from OH⁻ already covered by water cubes
            seen_elements.add(el)
            mm_elements.append(el)
            mm_charges_str.append(f"{atom.charge:.4f}")
            mm_cube_output.append(_element_cube_name(el))

    return mm_elements, mm_charges_str, mm_cube_output, is_tip4p, tip4p_O_idx


# ---------------------------------------------------------------------------
# SLURM / PBS header helpers (unchanged)
# ---------------------------------------------------------------------------

def _slurm_header(slurm_cfg: Dict[str, Any], jobname: str = "ces2_qmmm") -> List[str]:
    h: List[str] = ["#!/bin/bash"]
    h.append(f"#SBATCH -J {jobname}")
    if "account" in slurm_cfg:
        h.append(f"#SBATCH -A {slurm_cfg['account']}")
    if "partition" in slurm_cfg:
        h.append(f"#SBATCH -p {slurm_cfg['partition']}")
    nodes = int(slurm_cfg.get("nodes", 1))
    h.append(f"#SBATCH -N {nodes}")
    if slurm_cfg.get("no_requeue", False):
        h.append("#SBATCH --no-requeue")
    if "time" in slurm_cfg:
        h.append(f"#SBATCH -t {slurm_cfg['time']}")
    # default stderr/stdout to %x.e%j / %x.o%j if not explicitly set
    stderr = slurm_cfg.get("stderr", "%x.e%j")
    stdout = slurm_cfg.get("stdout", "%x.o%j")
    h.append(f"#SBATCH -e {stderr}")
    h.append(f"#SBATCH -o {stdout}")
    if "comment" in slurm_cfg:
        h.append(f"#SBATCH --comment {slurm_cfg['comment']}")
    for extra in slurm_cfg.get("extra_lines", []):
        h.append(extra)
    return h


def _pbs_header(pbs_cfg: Dict[str, Any]) -> List[str]:
    h: List[str] = ["#!/bin/bash"]
    h.append(f"#PBS -N {pbs_cfg.get('job_name', 'ces2_qmmm')}")
    nodes = int(pbs_cfg.get("nodes", 1))
    ppn   = int(pbs_cfg.get("ppn",   32))
    h.append(f"#PBS -l nodes={nodes}:ppn={ppn}")
    if "walltime" in pbs_cfg:
        h.append(f"#PBS -l walltime={pbs_cfg['walltime']}")
    if "queue" in pbs_cfg:
        h.append(f"#PBS -q {pbs_cfg['queue']}")
    return h


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_ces2_scripts(
    *,
    export_dir: Path,
    cfg: Dict[str, Any],
    n_mm: int,
    n_qm: int,
    species_order: Optional[List[Tuple[str, Any]]] = None,
    species_db:    Optional[Dict[str, Any]]         = None,
    water_sid:     Optional[str]                    = None,
    type_id_by_label: Optional[Dict[str, int]]      = None,
    slab_elements: Optional[List[str]]              = None,
) -> Dict[str, Path]:
    """
    Generate qmmm_dftces2_charging_pts.sh and submit_ces2.sh.

    The wrapper script follows the exact structure of the DFT-CES2 production
    script (M-design group @ KAIST).  Only the top-level variable block is
    auto-generated; all bash functions and the main loop are reproduced verbatim.

    Parameters
    ----------
    export_dir        : output directory (same as for data.file, base.in.lammps …)
    cfg               : full config dict
    n_mm              : number of MM atoms (= NSOLVATOM)
    n_qm              : number of QM atoms in supercell
    species_order     : [(species_id, count), …] in LAMMPS ordering
    species_db        : loaded species database
    water_sid         : water species_id (e.g. "water_tip4p")
    type_id_by_label  : type_label → LAMMPS type integer
    slab_elements     : element symbols of QM slab atoms (for QMLMPTYPE)
    """
    sc_cfg   = cfg.get("ces2_script", {})
    ces2_cfg = cfg.get("ces2",        {})
    qe_cfg   = cfg.get("qe",          {})
    cell_cfg = cfg.get("cell",        {})

    # ── binary paths ──────────────────────────────────────────────────────
    dft_ces2_path  = str(sc_cfg.get("dft_ces2_path",   "/path/to/dft-ces2"))
    qe_binary      = str(sc_cfg.get("qe_binary",       "/path/to/pw.x"))
    pp_binary      = str(sc_cfg.get("pp_binary",       "/path/to/pp.x"))
    lmp_binary     = str(sc_cfg.get("lmp_binary",      "/path/to/lmp_mpi"))
    chg2pot_bin    = str(sc_cfg.get("chg2pot_binary",  "/path/to/chg2pot"))
    mdipc_bin      = str(sc_cfg.get("mdipc_binary",
                                     "${DFT_CES2_PATH}/tools/3-poisson/mdipc"))
    chgplate_bin   = str(sc_cfg.get("chgplate_binary", "/path/to/make_rho"))

    # ── run control ───────────────────────────────────────────────────────
    np_procs       = int(sc_cfg.get("np",               24))
    qm_type        = str(sc_cfg.get("qm_type",         "scf"))
    qm_relax_high  = int(sc_cfg.get("qm_relax_high",   7))
    qmmm_ini_step  = int(sc_cfg.get("qmmm_ini_step",   0))
    n_qmmm_steps   = int(sc_cfg.get("n_qmmm_steps",    7))
    qmmm_fin_step  = n_qmmm_steps - 1
    qm_max_step    = int(sc_cfg.get("qm_max_step",     5))
    md_equil       = int(sc_cfg.get("md_equil_steps",  2000000))
    md_average     = int(sc_cfg.get("md_average_steps",2000000))
    dipole_corr    = str(sc_cfg.get("dipole_corr",     "yes"))
    initial_qm     = int(sc_cfg.get("initial_qm",      0))

    # ── charged system parameters ─────────────────────────────────────────
    tot_chg        = str(sc_cfg.get("tot_chg",         "0"))
    mpc_layer      = str(sc_cfg.get("mpc_layer",       "0.0"))   # bohr
    tot_layer      = str(sc_cfg.get("tot_layer",       "4"))
    mpc_one        = str(sc_cfg.get("mpc_one",         "1"))
    adsorbate      = str(sc_cfg.get("adsorbate",       "0"))
    plate_pos      = str(sc_cfg.get("plate_pos",       "0.0"))   # bohr

    # ── dipole correction (from qe: section) ──────────────────────────────
    dipole_dir     = str(qe_cfg.get("edir",   3))
    dipole_pos     = str(qe_cfg.get("emaxpos", 0.80))

    # ── supercell ─────────────────────────────────────────────────────────
    rep       = list(cell_cfg.get("supercell", [1, 1, 1]))
    sc_factor = int(rep[0]) * int(rep[1]) * int(rep[2])

    # ── QE prefix for cube file names ─────────────────────────────────────
    qe_prefix = str(qe_cfg.get("prefix", "solute"))

    # ── MM species arrays (from species_order + species_db) ───────────────
    if (species_order is not None and species_db is not None
            and water_sid is not None and type_id_by_label is not None):
        mm_elements, mm_charges, mm_cube_output, is_tip4p, tip4p_O_idx = \
            _build_mm_arrays(water_sid, species_order, species_db, type_id_by_label)
    else:
        # Fallback: TIP4P water + placeholder
        mm_elements   = ["H", "O"]
        mm_charges    = ["0.5242", "-1.0484"]
        mm_cube_output= ["hydrogen.cube", "oxygen.cube"]
        is_tip4p      = True
        tip4p_O_idx   = 1

    # Derive c_rho hat cube names (one per MM element)
    cube_qm_rho_hat = [
        f"c_rho_{_ELEMENT_FULLNAME.get(el, el.lower())}.cube"
        for el in mm_elements
    ]
    # TIP4P O gets c_rep_ naming via the special Blur_QM path (standard)
    tip4p_flag   = 1 if is_tip4p else 0
    tip4p_second = tip4p_O_idx

    # ── QMLMPTYPE: LAMMPS type IDs for QM (slab) atoms ───────────────────
    if slab_elements is not None and type_id_by_label is not None:
        qm_type_ids = sorted(set(
            type_id_by_label[el]
            for el in set(slab_elements)
            if el in type_id_by_label
        ))
    else:
        qm_type_ids = []   # user must fill in manually

    # ── QM cube names (from QE prefix) ────────────────────────────────────
    cube_coul_qm = f"{qe_prefix}.pot.cube"
    cube_ind_qm  = f"{qe_prefix}.ind.cube"
    cube_rho_qm  = f"{qe_prefix}.rho.cube"

    # ======================================================================
    # Build qmmm_dftces2_charging_pts.sh
    # ======================================================================
    lines: List[str] = []

    def L(s: str = "") -> None:
        lines.append(s)

    # ── Header comment ─────────────────────────────────────────────────────
    L("#!/bin/bash")
    L("# DFT-CES2 wrapper script written by Taehwan Jang, Copyright (c) 2024 M-design group @ KAIST")
    L("# Variable declarations auto-generated by cesbuild (ces2_script_writer.py)")
    L(f"# QM atoms  : {n_qm}  (slab, total supercell)    MM atoms: {n_mm}")
    L(f"# Supercell : {rep[0]}x{rep[1]}x{rep[2]}  (sc_factor={sc_factor})")
    L()

    # ── Global variables (auto-generated from config) ─────────────────────
    L("# Global variables")
    L(f"NP={np_procs}")
    L(f'QMTYPE="{qm_type}" # scf or opt')
    L(f"QMRELAX_HIGH={qm_relax_high} # QM atoms below the specified value are relaxed in unit of Angs")
    L(f"QMMMINISTEP={qmmm_ini_step} # no of initial QMMM step")
    L(f"QMMMFINSTEP={qmmm_fin_step} # no of final QMMM step")
    L(f'QMMAXSTEP="{qm_max_step}" # QM max trial for "opt"')
    L(f"SUPERCELL=({rep[0]} {rep[1]} {rep[2]}) # (x y z)")
    L("SUPERCELLFACTOR=`echo ${SUPERCELL[0]}*${SUPERCELL[1]}*${SUPERCELL[2]} | bc`")
    L(f'DIPOLECORR="{dipole_corr}" # dipole correction for MM charge density')
    L(f'DIPOLEDIR="{dipole_dir}" # dipole correction direction x=1 y=2 z=3')
    L(f'DIPOLEPOS="{dipole_pos}" # dipole correction fractional position along DIPOLEDIR')
    L(f'export DFT_CES2_PATH="{dft_ces2_path}"')
    L(f'LAMMPS="{lmp_binary}"')
    L(f'QEPW="{qe_binary}"')
    L(f'QEPP="{pp_binary}"')
    L('CUBEADD="${DFT_CES2_PATH}/newtools/1-cube/cube_add" # Working for n cubes. Input: n cube1 ... cuben')
    L('CUBEMULTI="${DFT_CES2_PATH}/tools/1-cube/cube_multi"')
    L('CUBESUB="${DFT_CES2_PATH}/tools/1-cube/cube_sub"')
    L('BLUR="${DFT_CES2_PATH}/tools/2-blur/blur"')
    L('MDDIPOLE="${DFT_CES2_PATH}/tools/3-poisson/dipc2"')
    L(f'CHG2POT="{chg2pot_bin}"')
    L('LAMMPSIN="base.in.lammps"')
    L('QMIN="base.pw.in"')
    L('QMIN2="base.pp.in"')
    L('LAMMPSDATA="data.file"')
    L('LAMMPSRESTART="ini.restart"')
    L(f'MDEQUIL="{md_equil}" # no. of equilibration steps')
    L(f'MDAVERAGE="{md_average}" # no. of averaging steps')

    # QMLMPTYPE
    if qm_type_ids:
        L(f"QMLMPTYPE=({' '.join(str(t) for t in qm_type_ids)}) # type atom type index of QM system defined in LAMMPS data")
    else:
        L("QMLMPTYPE=(FILL_IN_QM_TYPE_IDS) # type atom type index of QM system defined in LAMMPS data")

    L(f'NSOLVATOM="{n_mm}" # no. of MM (solvent) atoms')
    L(f"MMpartialCharge=({' '.join(mm_charges)})")
    L(f"MMrepA=({' '.join(mm_elements)})")
    L("unit_LAMMPS2QE=0.001686594 # Ry/a.u. to kcal/mol Angs = 592.9107")
    L("ATOMNAME=(H Li Be B C N O F Na Mg Al Si P S Cl K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Ag I W Re Os Ir Pt Au)")
    L("# Truncate to 1 decimal place")
    L("ATOMMASS=(1.0 6.9 9.0 10.8 12.0 14.0 15.9 18.9 22.9 24.3 26.9 28.0 30.9 32.0 35.4 39.0 40.0 44.9 47.8 50.9 51.9 54.9 55.8 58.9 58.6 63.5 65.3 69.7 72.6 74.9 78.9 79.9 107.8 126.9 183.8 186.2 190.2 192.2 195.0 196.9)")
    L("# Charged system. Point charge scheme. For electrostatic interactions.")
    L(f'MDIPC="{mdipc_bin}"')
    L(f'mpclayer="{mpc_layer}" #bohr unit mpc layer')
    L(f'totlayer="{tot_layer}"')
    L(f'mpcone="{mpc_one}"')
    L(f'adsorbate="{adsorbate}"')
    L("# Charged system. Plate scheme. For induction interactions.")
    L(f'CHGPLATE="{chgplate_bin}"')
    L(f'PLATEPOS="{plate_pos}" # somewhere in between bottom and the top layer of the electrode, bohr unit')
    L(f'TOTCHG="{tot_chg}" # (-): put more electrons in QM system, 0 for PZC')
    L(f"initialqm={initial_qm} #1, when the initial qm has been done.")
    L("firstrun=1 # Should be 1 for all the time")
    L(f"cube_coul_QM=({cube_coul_qm}) # v_q^A. neutralized solute.pot.cube by subtracting point charges")
    L(f"cube_ind_QM=({cube_ind_qm}) # non-neutral solute.pot.cube. For induction. Uniform charge scheme is used.")
    L(f"cube_rho_QM=({cube_rho_qm}) # rho^A")
    L(f"cube_QM_rho_hat=({' '.join(cube_qm_rho_hat)}) # \\hat rho^A sigma_b")
    L(f"cube_output_MM=({' '.join(mm_cube_output)}) # rho_q^B")
    L()
    L("cube_output_PolarDen=(px.cube py.cube pz.cube) # Polarization density")
    L(f"TIP4Pinvolved=({tip4p_flag} {tip4p_second}) # when TIP4P water is involved in the system, first: 1, second: order of output MM cube for TIP4P oxygen ")
    L("if [ ${TIP4Pinvolved[0]} -eq 1 ]; then")
    L("\tcube_output_TIP4P_O=(rep_TIP4P_oxygen.cube)")
    L("fi ")
    L('cubeinum=`echo "${#cube_coul_QM[@]}+${#cube_ind_QM[@]}+${#cube_QM_rho_hat[@]}" | bc`')
    L('cubeonum=`echo "${#cube_output_MM[@]}+${#cube_output_PolarDen[@]}+${#cube_output_TIP4P_O[@]}" | bc`')
    L("cubeio=`echo $cubeinum $cubeonum ${cube_coul_QM[@]} ${cube_ind_QM[@]} ${cube_QM_rho_hat[@]} ${cube_output_MM[@]} ${cube_output_PolarDen[@]} ${cube_output_TIP4P_O[@]}`")
    L("cubeioequil=`echo $cubeinum 0 ${cube_coul_QM[@]} ${cube_ind_QM[@]} ${cube_QM_rho_hat[@]}`")
    L("ATOMrepANAME=(H O N S C Li Na K Rb Cs He Ne Ar Kr Xe F Cl Br I P)")
    L("ATOMrepA=(1.381 15.56 10.51 3.39 5.15 11.30 24.82 23.51 19.26 11.23 23.78 22.01 11.08 7.617 4.532 2.600 2.694 2.175 1.938 4.38)")
    L("ATOMalpha=(1.693 5.20 7.25 19.5 11.7 0.193 0.93 5.05 8.32 15.0 1.38 2.67 11.1 16.8 27.2 15.0 30.3 42.8 61.7 24.8)")
    L("PI=3.141592")
    L("temprep=${cube_output_MM}")
    L("tempalpha=${cube_outpt_MM}")
    L("for ((i=0; i<${#MMrepA[@]}; i++));do")
    L("  for ((j=0; j<${#ATOMrepANAME[@]}; j++));do")
    L("\t\tif [ \"${ATOMrepANAME[$j]}\" == \"${MMrepA[$i]}\" ];then ")
    L("      temprep[$i]=${ATOMrepA[$j]}")
    L("      tempalpha[$i]=${ATOMalpha[$j]}")
    L("    fi")
    L("\tdone")
    L("done")
    L()

    # ── Blur_MM function (verbatim from production) ────────────────────────
    L("Blur_MM (){")
    L("  for ((i=0; i<${#cube_output_MM[@]}; i++));do")
    L("\t\tif [ \"$i\" == \"${TIP4Pinvolved[1]}\" ]; then")
    L("        \t\t$BLUR rep_TIP4P_oxygen.cube 1.114142")
    L("\t\t\tmv blurred.cube c_rep_TIP4P_oxygen.cube")
    L("\t\telse ")
    L("\t\t\tconv=`echo ${MMpartialCharge[$i]} ${temprep[$i]} | awk '{printf(\"%f\",1/$1*$2*2)}'`")
    L("\t\t\techo $conv")
    L("\t\t\tradius=`echo ${PI} ${tempalpha[$i]} | awk '{printf(\"%f\",((2/$1)**0.5*$2/3)**(1/3))}'`")
    L("\t\t\techo $radius")
    L("    \t\t\t$CUBEMULTI ${cube_output_MM[$i]} $conv # 1/q*repA/313.7545 ")
    L("    \t\t\t$BLUR multiplied.cube ${radius} # atomic radius from isotropic polarizability.")
    L()
    L("\t\t\tmv blurred.cube c_rep_${cube_output_MM[$i]}")
    L("\t\tfi")
    L("  done")
    L("  list=$(ls c_rep_*.cube) #convoluted repulsion potential from mm; fixed external potential by mm")
    L("  listnum=$(ls c_rep_*.cube | wc -l) #convoluted repulsion potential from mm; fixed external potential by mm")
    L("  $CUBEADD $listnum ${list[@]}")
    L("  mv add.cube repA.cube")
    L("}")

    # ── Blur_QM function (verbatim from production) ────────────────────────
    L("Blur_QM (){")
    L("  for ((i=0; i<${#cube_output_MM[@]}; i++));do")
    L("\t\t\tradius=`echo ${PI} ${tempalpha[$i]} | awk '{printf(\"%f\",((2/$1)**0.5*$2/3)**(1/3))}'`")
    L("        $BLUR  $1 ${radius}")
    L("        mv  blurred.cube  ${cube_QM_rho_hat[$i]}")
    L("\tdone")
    L("}")

    # ── post_dipole function (verbatim from production) ────────────────────
    L("post_dipole (){")
    L('        dip_parse="${DFT_CES2_PATH}/tools/1-cube/cube_dip_parse" # post-process for polarization density')
    L('        addthree="${DFT_CES2_PATH}/tools/1-cube/cube_addthree"')
    L('\taddfour="${DFT_CES2_PATH}/tools/1-cube/cube_addfour"')
    L('\taddelev="${DFT_CES2_PATH}/tools/1-cube/cube_addelev" ')
    L("        $dip_parse ${cube_output_PolarDen[0]} 1          # compute distribution of bound charge along x-direction")
    L("        mv postdipole.cube postdipx.cube")
    L("        $dip_parse ${cube_output_PolarDen[1]} 2          # compute distribution of bound charge along y-direction")
    L("        mv postdipole.cube postdipy.cube")
    L("        $dip_parse ${cube_output_PolarDen[2]} 3          # compute distribution of bound charge along z-direction")
    L("        mv postdipole.cube postdipz.cube")
    L()
    L("        # time-averaged charge distribution from MM")
    L("\t$CUBEADD ${#cube_output_MM[@]} ${cube_output_MM[@]}")
    L("\tmv add.cube mobile.cube")
    L("        $addfour postdipx.cube postdipy.cube postdipz.cube mobile.cube")
    L("        mv add.cube MOBILE_final.cube")
    L("        rm add*.cube post*.cube dip?.cube")
    L("}")
    L()

    # ── Main QM/MM loop (verbatim from production) ─────────────────────────
    L("# main loop")
    L("for ((qmmmstep=$QMMMINISTEP; qmmmstep<=$QMMMFINSTEP; qmmmstep++));do")
    L('  echo "######### Starting $qmmmstep QMMM step #########"$\'\\n\'')
    L("  # Make saving directory")
    L("  mkdir qm_$qmmmstep mm_$qmmmstep")
    L()
    L("  ## Preparation of QM")
    L('  if [ -e "$LAMMPSRESTART" ]; then')
    L("    mpirun -np 1 $LAMMPS -r $LAMMPSRESTART $LAMMPSDATA")
    L("  else")
    L('    if [ -e "$LAMMPSDATA" ]; then')
    L("      echo \"Using $LAMMPSDATA directly (no restart file)\"  # data.file is used as-is")
    L("    else")
    L('      echo "no data file or restart file for initialization"')
    L("      exit 1")
    L("    fi")
    L("  fi")
    L()
    L("  if [ $firstrun -eq 1 ]; then")
    L("    # Extract to data from restart")
    L("    natoms_qm=`awk '/^[[:space:]]*nat[[:space:]]*=/{gsub(\",\",\"\"); print $3}' $QMIN`")
    L("    nqm_extend=`echo ${SUPERCELLFACTOR}*${natoms_qm} | bc `")
    L("    # Parsing cell box & atom names")
    L('    echo "### Parsing cell box & atom species from data file.."')
    L("    Box=(`awk '{if(NF==4 && ($3==\"xlo\"||$3==\"ylo\"||$3==\"zlo\")) print $0}' $LAMMPSDATA`)")
    L("    cells=( `printf \"%f\" ${Box[1]}` `printf \"%f\" ${Box[5]}` `printf \"%f\" ${Box[9]}` ) # Cell box sizes 0:x 1:y 2:z")
    L('    echo "### Parsed lammps cell box size : ${cells[@]}"$\'\\n\'')
    L("    Masses=(`awk 'BEGIN{tag=0;cnt=0}{if($2==\"atom\"&&$3==\"types\") natoms=$1; if($1==\"Masses\") tag=1; if(tag==1 && NF==2 && cnt<natoms) {print $0; cnt++}}' $LAMMPSDATA`)")
    L("    for ((i=0; i<${#Masses[@]}; i++));do")
    L("      if [ $((i%2)) -eq 1 ]; then")
    L("        temp=`echo \"${Masses[$i]}*10/1\" | bc`  # mass round down to one decimal place")
    L("        temp=`echo \"$temp/10\" | bc -l`")
    L("        for ((j=0; j<${#ATOMMASS[@]}; j++));do")
    L("          temp2=`echo $temp'=='${ATOMMASS[$j]} | bc -l` # match atomname and atommass")
    L("          if [ \"$temp2\" == \"1\" ]; then")
    L("            atoms=\"$atoms ${ATOMNAME[$j]}\"")
    L("          fi")
    L("        done")
    L("      fi")
    L("    done")
    L("    atoms=($atoms) # Atomic species by Type in lammps starting from 0..")
    L("    echo \"### Parsed lammps atomic species : ${atoms[@]}\"$'\\n' # for all atomic species")
    L("    if [ -z \"${atoms[*]}\" ]; then")
    L("      echo \"Parsed lammps atomic species are empty. Please make sure that mass information in data file is NF=2\\n\"")
    L("      exit 1")
    L("    fi ")
    L("  fi # 0th step initial parsing end")
    L()
    L("  # Parsing solute xyz")
    L("  # Parsing dispersion force xyz")
    L("  awk -v nsolvatom=$NSOLVATOM '{if(NF==10 && $1>nsolvatom) print $0}' $LAMMPSDATA > data.solute.temp")
    L("  if [ ! -s \"data.solute.temp\" ]; then")
    L("    echo \"Error: File $filename is empty. Make sure that NF=10 in Atoms section\\n\"")
    L("    exit 1")
    L("  fi ")
    L("  forloopend=$((NSOLVATOM+nqm_extend))")
    L("  for ((i=${NSOLVATOM}; i<${forloopend}; i++));do")
    L("      awk -v var=$i '{if($1==var+1) print $0}' data.solute.temp >> data.solute.temp2")
    L("  done")
    L("  nodispf=`ls dispf.ave`")
    L("  if [ \"$nodispf\" == \"\" ] || [ ! -s dispf.ave ]; then")
    L("        for ((i=0; i<$nqm_extend; i++));do")
    L("                        echo \"0 0 0 0 0 0 0 0\">> dispf.ave")
    L("                done")
    L("  fi")
    L("  tail -n $nqm_extend dispf.ave | awk -v unit=$unit_LAMMPS2QE '{print $6*unit,$7*unit,$8*unit}' > data.dispforce # sorted id ")
    L("  paste data.solute.temp2 data.dispforce > data.solute")
    L("  rm data.solute.temp data.solute.temp2")
    L("  solute=(`awk -v var=\"${QMLMPTYPE[*]}\" 'BEGIN{split(var,list,\" \")} {if(NF==13) for(x in list) if($3==list[x]) print $1,$3,$5,$6,$7,$11,$12,$13}' data.solute`) # format: id type x y z bfx bfy bfz, only solute position, x in list: from 1 to the number of QMLMPTYPE")
    L("  echo \"### Parsed number of atoms in solute : $natoms_qm\"")
    L()
    L("  for ((i=0;i<${#solute[@]};i++));do # i from 0 to 8*natoms; 8 since the element for each atom is id type x y z d_bfx d_bfy d_bfz")
    L("    if [ $((i%8)) -eq 1 ]; then")
    L("      idx=`echo ${solute[$i]}'-'1 | bc` # since atomic species has allocated from 0; the number is where the atomname is stored.")
    L("      solute[$i]=${atoms[$idx]} # swap and allocate the atom type of each solute atom.")
    L("    fi")
    L("    if [ $((i%8)) -eq 2 -o $((i%8)) -eq 3 -o $((i%8)) -eq 4 -o $((i%8)) -eq 5 -o $((i%8)) -eq 6 -o $((i%8)) -eq 7 ]; then")
    L("      solute[$i]=`printf \"%lf\" ${solute[$i]}`")
    L("    fi")
    L("  done")
    L("  ")
    L("  # Solute xyz to qmxyz string")
    L("  qmxyz=\"\"")
    L("  force=\"\"")
    L("  for ((i=0; i<$natoms_qm; i++));do")
    L("    qmxyz=\"$qmxyz${solute[$((8*i+1))]} ${solute[$((8*i+2))]} ${solute[$((8*i+3))]} ${solute[$((8*i+4))]}\"$'\\n'")
    L("    fx=0;   fy=0;   fz=0;")
    L("    for ((j=0; j<${SUPERCELLFACTOR}; j++));do")
    L("     fx=`echo ${fx} ${solute[$((8*(i+j*natoms_qm)+5))]} | awk -v super=${SUPERCELLFACTOR} '{print $1+$2/super}' `")
    L("     fy=`echo ${fy} ${solute[$((8*(i+j*natoms_qm)+6))]} | awk -v super=${SUPERCELLFACTOR} '{print $1+$2/super}' `")
    L("     fz=`echo ${fz} ${solute[$((8*(i+j*natoms_qm)+7))]} | awk -v super=${SUPERCELLFACTOR} '{print $1+$2/super}' `")
    L("    done")
    L("    force=\"$force${solute[$((8*i+1))]} ${fx} ${fy} ${fz}\"$'\\n'")
    L("  done")
    L("  echo \"###qmxyz\"")
    L("  echo \"$qmxyz\"")
    L("  echo \"###force\"")
    L("  echo \"$force\"")
    L()
    L("  firstrun=0")
    L("  ")
    L("  # Make pw.in based on $QMIN")
    L("  awk -v \"geo=$qmxyz\" -v \"dispf=$force\" '{if($0==\"###qmxyz\") {print geo} else if($0==\"###dispf\") {print dispf} else {print $0} }' $QMIN > pw.in")
    L("  if [ $qmmmstep -gt 0 ]; then ")
    L("    sed -i \"s/.*\\&CONTROL.*/&\\ndft_ces = .true./\" pw.in")
    L("    sed -i \"s/.*\\&CONTROL.*/&\\nrho_ces = '.\\/MOBILE_final.cube'/\" pw.in")
    L("    sed -i \"s/.*\\&CONTROL.*/&\\npauli_rep_ces = '.\\/repA.cube'/\" pw.in")
    L("    sed -i \"s/.*\\&ELECTRONS.*/&\\nstartingwfc = 'file'/\" pw.in")
    L("    sed -i \"s/.*\\&ELECTRONS.*/&\\nstartingpot = 'file'/\" pw.in")
    L("    sed -i \"s/.*\\&SYSTEM.*/&\\ntot_charge = $TOTCHG/\" pw.in")
    L("  fi")
    L("  if [ $qmmmstep -gt 0 -a \"$QMTYPE\" == \"opt\" ]; then # relax geometry")
    L("\tsed -i \"s/.*\\&CONTROL.*/&\\nforc_conv_thr = 1.0D-3/\" pw.in")
    L("\tsed -i \"s/.*\\&CONTROL.*/&\\nnstep = 150/\" pw.in")
    L("\tsed -i \"s/.*calculation.*/calculation = 'relax'/\" pw.in")
    L("\tsed -i \"s/.*ATOMIC_SPECIES.*/\\&IONS\\n&/\" pw.in")
    L("\tsed -i \"s/.*ATOMIC_SPECIES.*/\\/\\n&/\" pw.in")
    L("\tsed -i \"s/.*\\&IONS.*/&\\ntrust_radius_max = 0.05D0/\" pw.in")
    L("\tnr=$(grep -n \"ATOMIC_POSITIONS\" pw.in | cut -d: -f1)")
    L("\tnrf=$(grep -n \"ATOMIC_FORCES\" pw.in | cut -d: -f1)")
    L("\tawk -v line=$nr '{if(NR<=line) {print $0}}' pw.in >tmp")
    L("\tawk -v line=$nr -v zval=$QMRELAX_HIGH -v nrf=$nrf '{if(NR>line) {if(NF==4 && $4<zval && NR<nrf) {print $0, 0,\"\", 0, \"\", 0 } else {print $0}} }' pw.in >>tmp")
    L("\tmv tmp pw.in")
    L("  fi")
    L(" ")
    L("  if [ $qmmmstep -gt 0 ]; then")
    L("    cp mm_$((qmmmstep-1))/MOBILE_final.cube ./")
    L("    cp mm_$((qmmmstep-1))/empty.cube ./")
    L("    cp mm_$((qmmmstep-1))/repA.cube ./")
    L("  fi")
    L("        ## end of Preparation of QM")
    L()
    L("  # Run QM calculation")
    L("  # 1. Orthogonalized QM by MM -> Repulsion of MM is considered")
    L("  # For final QM rho or potential calculation")
    L("  # For qm_0, neutral calculation is performed ")
    L("  finished=\"\"")
    L("  for ((i=0; i<$QMMAXSTEP; i++));do")
    L("    if [ $initialqm -eq 0 ]; then")
    L("      mpirun -np $NP $QEPW < pw.in > pw.out")
    L("      if [ \"$QMTYPE\" == \"scf\" ]; then")
    L("        finished=`grep \"JOB DONE\" pw.out`")
    L("        if [ \"$finished\" != \"\" ]; then ")
    L('          echo "### QM calculation done"$\'\\n\'')
    L("          cp v_saw.cube v_saw_pw_ortho.cube")
    L("\t  break")
    L("        else")
    L('          echo "### QM calculation aborted, check output file"$\'\\n\'')
    L("          exit")
    L("        fi")
    L("      elif [ $qmmmstep -eq 0 ]; then")
    L("        finished=`grep \"JOB DONE\" pw.out`")
    L("        if [ \"$finished\" != \"\" ]; then ")
    L('          echo "### QM calculation done"$\'\\n\'')
    L("\t  cp v_saw.cube v_saw_pw_ortho.cube")
    L("          break")
    L("        else")
    L('          echo "### QM calculation aborted, check output file"$\'\\n\'')
    L("          exit")
    L("        fi")
    L("      else")
    L("        finished=`grep \"JOB DONE\" pw.out`")
    L("        if [ \"$finished\" != \"\" ]; then")
    L('          echo "### QM calculation done"$\'\\n\'')
    L("\t  cp v_saw.cube v_saw_pw_ortho.cube")
    L("          break")
    L("        else")
    L('          echo "### QM calculation aborted, check output file"$\'\\n\'')
    L("          exit")
    L("        fi")
    L("      fi")
    L("    fi")
    L("  done")
    L("  ## Postprocess of QM")
    L("  # Generating QM solute potential")
    L('  echo "### QM potential post processing.. "$\'\\n\'')
    L("  cp $QMIN2 pp.pot_ortho.in")
    L("  cp $QMIN2 pp.rho_ortho.in")
    L("  sed -i \"s/.*fileout.*/fileout = 'solute.pot_ortho.cube'/\" pp.pot_ortho.in")
    L("  sed -i \"s/.*plot_num.*/plot_num = 0/\" pp.rho_ortho.in")
    L("  sed -i \"s/.*fileout.*/fileout = 'solute.rho_ortho.cube'/\" pp.rho_ortho.in")
    L(" ")
    L("  if [ $initialqm -eq 0 ]; then")
    L("        mpirun -np $NP $QEPP < pp.pot_ortho.in > pp.pot_ortho.out")
    L("        ppdone1=`ls solute.pot_ortho.cube`")
    L("        mpirun -np $NP $QEPP < pp.rho_ortho.in > pp.rho_ortho.out")
    L("        ppdone2=`ls solute.rho_ortho.cube`")
    L("        if [ \"$ppdone1\" == \"\" ] || [ \"$ppdone2\" == \"\" ] ; then")
    L('            echo "### QM post processing failed, aborting whole qmmm loop"$\'\\n\'')
    L("        else")
    L("\tcp v_saw.cube v_saw_ortho.cube")
    L("\tif [ $qmmmstep -gt 0 ]; then")
    L("\t    $CHG2POT MOBILE_final.cube 3")
    L("\t    $CUBEMULTI pot.cube -2")
    L("\t    $CUBEADD 2 multiplied.cube v_saw_pw_ortho.cube ")
    L("\t    $CUBEADD 2 add.cube solute.pot_ortho.cube")
    L("            mv add.cube total_pot_ortho.cube")
    L("\t    $CUBEADD 2 v_saw_ortho.cube solute.pot_ortho.cube")
    L("            $CUBEMULTI solute.pot_ortho.cube 0")
    L("            mv multiplied.cube empty.cube")
    L("            mv add.cube solute.pot_ortho.cube")
    L("            cp pw.in pw.out pp.pot_ortho.out pp.rho_ortho.out solute.rho_ortho.cube solute.pot_ortho.cube total_pot_ortho.cube v_saw_pw_ortho.cube v_saw_ortho.cube qm_$qmmmstep")
    L("            #cp -r solute qm_$qmmmstep")
    L("\telse # qm_0")
    L("\t    $CUBEADD 2 v_saw_pw_ortho.cube solute.pot_ortho.cube")
    L("\t    $CUBEMULTI solute.pot_ortho.cube 0")
    L("\t    mv multiplied.cube empty.cube")
    L("\t    mv add.cube ref_solute.pot.cube")
    L("\t    mv pp.pot_ortho.in pp.pot.in")
    L("\t    mv pp.pot_ortho.out pp.pot.out")
    L("\t    mv pp.rho_ortho.in pp.rho.in")
    L("\t    mv pp.rho_ortho.out pp.rho.out")
    L("\t    mv solute.rho_ortho.cube ref_solute.rho.cube")
    L("\t    mv v_saw_pw_ortho.cube v_saw.cube")
    L("\t    cp pw.in pw.out pp.pot.out pp.rho.out ref_solute.rho.cube ref_solute.pot.cube v_saw.cube qm_$qmmmstep")
    L("\tfi")
    L("        fi")
    L("  else")
    L("        emptyexst=`ls empty.cube`")
    L("        if [ \"$emptyexst\" == \"\" ] ; then")
    L("\t    if [ $qmmmstep -gt 0 ]; then")
    L("\t\tcp qm_$qmmmstep/solute.rho_ortho.cube .")
    L("                $CUBEMULTI solute.rho_ortho.cube 0")
    L("                mv multiplied.cube empty.cube")
    L("\t    else # qm_0")
    L("\t\tcp qm_$qmmmstep/ref_solute.rho.cube .")
    L("\t\t$CUBEMULTI ref_solute.rho.cube 0")
    L("\t\tmv multiplied.cube empty.cube")
    L("\t    fi")
    L("        fi")
    L("  fi")
    L()
    L("  # 2. Non-orthogonalized QM by MM -> Repulsion of MM is not considered")
    L("  # For QM/MM iteration")
    L("  # For qm_0, skip")
    L("  if [ $initialqm -eq 0 -a \"$QMTYPE\" == \"scf\" -a $qmmmstep -gt 0 ]; then")
    L("\tawk -v \"geo=$qmxyz\" -v \"dispf=$force\" '{if($0==\"###qmxyz\") {print geo} else if($0==\"###dispf\") {print dispf} else {print $0} }' $QMIN > pw.nonortho.in")
    L("\tsed -i \"s/.*\\&CONTROL.*/&\\ndft_ces = .true./\" pw.nonortho.in")
    L("\tsed -i \"s/.*\\&CONTROL.*/&\\nrho_ces = '.\\/MOBILE_final.cube'/\" pw.nonortho.in")
    L("\tsed -i \"s/.*\\&CONTROL.*/&\\npauli_rep_ces = '.\\/empty.cube'/\" pw.nonortho.in")
    L("        sed -i \"s/.*\\&ELECTRONS.*/&\\nstartingwfc = 'file'/\" pw.nonortho.in")
    L("        sed -i \"s/.*\\&ELECTRONS.*/&\\nstartingpot = 'file'/\" pw.nonortho.in")
    L("        sed -i \"s/.*\\&SYSTEM.*/&\\ntot_charge = $TOTCHG/\" pw.nonortho.in")
    L("\tmpirun -np $NP $QEPW < pw.nonortho.in > pw.nonortho.out")
    L("\tcp v_saw.cube v_saw_pw_nonortho.cube")
    L("  elif [ $initialqm -eq 0 -a \"$QMTYPE\" == \"opt\" -a $qmmmstep -gt 0 ]; then")
    L("\tsolute_nonortho=(`grep -A ${natoms_qm} \"ATOMIC_POSITIONS\" pw.out | tail -n ${natoms_qm} | awk '{print $1,$2,$3,$4 }'`)")
    L("\tqmxyz_nonortho=\"\"")
    L("\tforce_nonortho=\"\"")
    L("\tfor ((i=0; i<$natoms_qm; i++));do")
    L("\t\tqmxyz_nonortho=\"$qmxyz_nonortho${solute_nonortho[$((4*i))]} ${solute_nonortho[$((4*i+1))]} ${solute_nonortho[$((4*i+2))]} ${solute_nonortho[$((4*i+3))]}\"$'\\n'")
    L("\t\tforce_nonortho=\"$force_nonortho${solute_nonortho[$((4*i))]} 0 0 0\"$'\\n'")
    L("\tdone")
    L("\tawk -v \"geo=$qmxyz_nonortho\" -v \"dispf=$force_nonortho\" '{if($0==\"###qmxyz\") {print geo} else if($0==\"###dispf\") {print dispf} else {print $0} }' $QMIN > pw.nonortho.in")
    L("\tsed -i \"s/.*\\&CONTROL.*/&\\ndft_ces = .true./\" pw.nonortho.in")
    L("\tsed -i \"s/.*\\&CONTROL.*/&\\nrho_ces = '.\\/MOBILE_final.cube'/\" pw.nonortho.in")
    L("\tsed -i \"s/.*\\&CONTROL.*/&\\npauli_rep_ces = '.\\/empty.cube'/\" pw.nonortho.in")
    L("\tmpirun -np $NP $QEPW < pw.nonortho.in > pw.nonortho.out")
    L("\tcp v_saw.cube v_saw_pw_nonortho.cube")
    L("  fi")
    L()
    L("  ## Postprocess of QM")
    L("  # Generating QM solute potential")
    L('  echo "### QM potential post processing.. "$\'\\n\'')
    L("  cp $QMIN2 pp.pot.in")
    L("  cp $QMIN2 pp.rho.in")
    L("  sed -i \"s/.*plot_num.*/plot_num = 0/\" pp.rho.in")
    L("  sed -i \"s/.*fileout.*/fileout = 'solute.rho.cube'/\" pp.rho.in")
    L()
    L("  if [ $qmmmstep -gt 0 ]; then ")
    L("    if [ $initialqm -eq 0 ]; then")
    L("        mpirun -np $NP $QEPP < pp.pot.in > pp.pot.out")
    L("        ppdone1=`ls solute.pot.cube`")
    L("        mpirun -np $NP $QEPP < pp.rho.in > pp.rho.out")
    L("        ppdone2=`ls solute.rho.cube`")
    L("        if [ \"$ppdone1\" == \"\" ] || [ \"$ppdone2\" == \"\" ] ; then")
    L('            echo "### QM post processing failed, aborting whole qmmm loop"$\'\\n\'')
    L("        else")
    L("\tcp v_saw.cube v_saw_nonortho.cube")
    L()
    L("\t# Make total_pot_nonortho.cube")
    L("\t$CHG2POT MOBILE_final.cube 3")
    L("\t$CUBEMULTI pot.cube -2")
    L("\t$CUBEADD 2 multiplied.cube v_saw_pw_nonortho.cube ")
    L("\t$CUBEADD 2 add.cube solute.pot.cube")
    L("        mv add.cube total_pot_nonortho.cube")
    L()
    L("        Blur_QM solute.rho.cube")
    L()
    L("\t# Make solute.pot.cube")
    L("        $CUBEADD 2 v_saw_nonortho.cube solute.pot.cube")
    L("        $CUBEMULTI solute.pot.cube 0")
    L("        mv multiplied.cube empty.cube")
    L("        mv add.cube solute.pot.cube")
    L()
    L("\t# Make solute.ind.cube, Same procedure for both neutral and non-neutral case")
    L("\t$CUBESUB solute.rho.cube qm_0/ref_solute.rho.cube")
    L("\tdipolegrid=`awk -v dir=$DIPOLEDIR -v pos=$DIPOLEPOS '{if(NR==(dir+3)) printf \"%d\", $1*pos}' solute.rho.cube`")
    L("\t$CHGPLATE subtracted.cube 1 $dipolegrid 0 0 $PLATEPOS")
    L("\t$CUBESUB subtracted.cube q.cube")
    L("\t$CHG2POT subtracted.cube 3")
    L("\t$CUBEMULTI pot.cube 2")
    L("\tcp multiplied.cube solute.pot_uni.cube")
    L("\t$CUBEMULTI V.cube 2")
    L("\t$CUBEADD 2 multiplied.cube solute.pot_uni.cube")
    L("\t$CUBEADD 2 add.cube qm_0/ref_solute.pot.cube")
    L("\tcp add.cube solute.ind.cube")
    L()
    L("        cp pw.nonortho.out pp.pot.out pp.rho.out solute.rho.cube solute.pot.cube solute.pot_uni.cube solute.ind.cube total_pot_nonortho.cube v_saw_pw_nonortho.cube v_saw_nonortho.cube qm_$qmmmstep")
    L("        fi")
    L("    else")
    L("        rhoexst=`ls solute.rho.cube`")
    L("        emptyexst=`ls empty.cube`")
    L("        if [ \"$rhoexst\" != \"\" ] ; then")
    L("            Blur_QM solute.rho.cube")
    L("        else")
    L("            cp qm_$qmmmstep/*.cube .")
    L("            Blur_QM solute.rho.cube")
    L("        fi")
    L("        if [ \"$emptyexst\" == \"\" ] ; then")
    L("            $CUBEMULTI solute.rho.cube 0")
    L("            mv multiplied.cube empty.cube")
    L("        fi")
    L("    fi")
    L("  else ")
    L("    rhoexst=`ls ref_solute.rho.cube`")
    L("    if [ \"$rhoexst\" != \"\" ] ; then")
    L("        Blur_QM ref_solute.rho.cube")
    L("    else")
    L("        cp qm_$qmmmstep/*.cube .")
    L("        Blur_QM ref_solute.rho.cube")
    L("    fi")
    L("    cp qm_$qmmmstep/ref_solute.pot.cube .")
    L("    cp ref_solute.pot.cube solute.pot.cube")
    L("    cp solute.pot.cube solute.ind.cube")
    L("    cp solute.pot.cube solute.ind.cube qm_$qmmmstep")
    L("  fi")
    L("    # Charged system neutralize")
    L("    # Point charge in QM")
    L("    # slab dipole correction")
    L("    # Using non-ortho calculation results")
    L("    if [ $initialqm -eq 0 -a $qmmmstep -gt 0 ] && [ \"$(echo \"$TOTCHG != 0\" | bc)\" -eq 1 ]; then")
    L("      $CUBESUB solute.rho.cube qm_0/ref_solute.rho.cube")
    L("      dipolegrid=`awk -v dir=$DIPOLEDIR -v pos=$DIPOLEPOS '{if(NR==(dir+3)) printf \"%d\", $1*pos}' solute.pot.cube`")
    L("      echo \"### dipole correction has been applied: dir=$DIPOLEDIR, grid=$dipolegrid\"")
    L("      $MDIPC subtracted.cube $DIPOLEDIR $dipolegrid $mpclayer $totlayer $mpcone $adsorbate")
    L("      $CHG2POT mdipc.cube $DIPOLEDIR")
    L("      $CUBEMULTI pot.cube 2  #to set Ha to ryd")
    L("      $CUBEADD 2 multiplied.cube qm_0/ref_solute.pot.cube")
    L("      cp add.cube solute.pot.cube")
    L("      cp solute.pot.cube qm_$qmmmstep")
    L("    fi")
    L()
    L("    initialqm=0")
    L("    echo \"qmmstep is $qmmmstep\"")
    L("    # Updating QM optimized geometry")
    L("    cp qm_$qmmmstep/pw.out .")
    L("    if [ $qmmmstep -gt 0 -a \"$QMTYPE\" == \"opt\" ]; then")
    L("        solutefinal=(`grep -A ${natoms_qm} \"ATOMIC_POSITIONS\" pw.out | tail -n ${natoms_qm} | awk '{print $1,$2,$3,$4 }'`)")
    L("        cnt=0")
    L("        for ((i=0; i<${#solute[@]}; i++));do")
    L("            if [ $((i%8)) != 0 -a $((i%8)) != 1 -a $((i%8)) != 5 -a $((i%8)) != 6 ]; then")
    L("                solute[$i]=${solutefinal[$((i-4*cnt-1))]}")
    L("            fi")
    L("            if [ $((i%8)) == 7 ]; then let cnt=cnt+1; fi")
    L("        done")
    L("        echo \"### Updating QM optimized geometry.. \"$'\\n'")
    L("    fi")
    L("    ## end of Postprocess of QM")
    L()
    L("    ## Preparation of MM")
    L("    # Modifying LAMMPS INPUT : restart, geometry")
    L("    cp $LAMMPSIN in.lammps")
    L("    if [ $qmmmstep -gt 0 -a \"$QMTYPE\" == \"opt\" ]; then")
    L("\t\t\tfor ((i=0; i<$natoms_qm; i++));do")
    L("\t\t\t\tcnt=0")
    L("\t\t\t\tfor ((j=1; j<=${SUPERCELL[0]}; j++));do")
    L("\t\t\t\t\tfor ((k=1; k<=${SUPERCELL[1]}; k++));do")
    L("\t\t\t\t\t\tfor ((l=1; l<=${SUPERCELL[2]}; l++));do")
    L("\t\t\t\t\t\t\tindex=`echo \"${solute[$((i*8))]} + $cnt*$natoms_qm\" | bc`")
    L("\t\t\t\t\t\t\txpos=`echo ${solute[$((i*8+2))]} ${cells[0]} $((j-1)) ${SUPERCELL[0]} | awk '{print $1+$2*$3/$4}'`")
    L("\t\t\t\t\t\t\typos=`echo ${solute[$((i*8+3))]} ${cells[1]} $((k-1)) ${SUPERCELL[1]} | awk '{print $1+$2*$3/$4}'`")
    L("\t\t\t\t\t\t\tzpos=`echo ${solute[$((i*8+4))]} ${cells[2]} $((l-1)) ${SUPERCELL[2]} | awk '{print $1+$2*$3/$4}'`")
    L("\t\t\t\t\t\t\tcnt=$((cnt+1))")
    L("\t\t\t\t\t\t\tsed -i \"s/.*xyz.*/&\\nset atom ${index} x ${xpos} y ${ypos} z ${zpos}/\" in.lammps")
    L("\t\t\t\t\t\tdone")
    L("\t\t\t\t\tdone")
    L("\t\t\t\tdone")
    L("\t\t\tdone")
    L("    fi")
    L("    ## end of Preparation of MM")
    L()
    L("    # RUN LAMMPS equilibration step")
    L("    ## Preparation of QM")
    L('    if [ -e "$LAMMPSRESTART" ]; then')
    L("      mpirun -np 1 $LAMMPS -r $LAMMPSRESTART data.md")
    L("    else")
    L('      if [ -e "$LAMMPSDATA" ]; then')
    L("        cp $LAMMPSDATA data.md # If there is no restart file")
    L("      else")
    L('        echo "no data file or restart file for lammps equilibration"')
    L("        exit 1")
    L("      fi")
    L("    fi")
    L()
    L("    sed -i \"s/.*read_data.*/read_data data.md/\" in.lammps")
    L("    echo \"### Running LAMMPS(emxext) for equilibration $qmmmstep QMMM iterations\"$'\\n'")
    L("    sed -i \"s/.*run.*/run\\t\\t$MDEQUIL/\" in.lammps")
    L("    if [ $qmmmstep -eq 0 ]; then")
    L("        LAMMPSRESTARTtime=0")
    L("    else")
    L("        LAMMPSRESTARTtime=`ls -lrt *.restart | tail -n 1 | awk '{print $9}' | cut -f2 -d '.'`")
    L("    fi")
    L("  sed -i \"s/.*reset_timestep.*/reset_timestep ${LAMMPSRESTARTtime}/\" in.lammps")
    L("  cp in.lammps in.lammps.equil ")
    L("  finaltime=`echo ${LAMMPSRESTARTtime} ${MDEQUIL} | awk '{print $1+$2}'`")
    L("  sed -i \"s/.*c_fdisp.*/fix\\t\\tshowf all ave\\/atom 1 $MDEQUIL $finaltime c_fdisp[*]/\" in.lammps.equil # MDEQUIL is a divisor of MDAVERAGE.")
    L("  sed -i \"s/.*dispf.ave.*/dump\\t\\t11 SOLUTE custom $finaltime dispf.ave id type xu yu zu f_showf[*]/\" in.lammps.equil")
    L("  sed -i \"s/.*#CUBEPOSITION.*/&\\ngrid\\t\\t ${cubeioequil} /\" in.lammps.equil")
    L("  mpirun -np $NP $LAMMPS -in in.lammps.equil > lammps.equil.out")
    L()
    L("  # RUN LAMMPS averaging step")
    L("  LAMMPSRESTART=`ls -lrt *.restart | tail -n 1 | awk '{print $9}'`")
    L("  LAMMPSRESTARTtime=`ls -lrt *.restart | tail -n 1 | awk '{print $9}' | cut -f2 -d '.'`")
    L("  sed -i \"s/.*reset_timestep.*/reset_timestep ${LAMMPSRESTARTtime}/\" in.lammps")
    L("sed -i \"s/.*#CUBEPOSITION.*/&\\ngrid\\t\\t ${cubeio} /\" in.lammps")
    L("  mpirun -np 1 $LAMMPS -r $LAMMPSRESTART data.md")
    L("  finaltime=`echo ${LAMMPSRESTARTtime} ${MDAVERAGE} | awk '{print $1+$2}'`")
    L("  sed -i \"s/.*read_data.*/read_data data.md/\" in.lammps")
    L("  sed -i \"s/.*c_fdisp.*/fix\\t\\tshowf all ave\\/atom 1 $MDAVERAGE $finaltime c_fdisp[*]/\" in.lammps # MDEQUIL is a divisor of MDAVERAGE.")
    L("  sed -i \"s/.*dispf.ave.*/dump\\t\\t11 SOLUTE custom $finaltime dispf.ave id type xu yu zu f_showf[*]/\" in.lammps # MDEQUIL is a divisor of MDAVERAGE.")
    L("  sed -i \"s/.*run.*/run\\t\\t$MDAVERAGE/\" in.lammps")
    L("  echo \"### Running LAMMPS(emdext) for averaging solvent charge density $qmmmstep QMMM iterations\"$'\\n'")
    L("  mpirun -np $NP $LAMMPS -in in.lammps > lammps.average.out")
    L("  LAMMPSRESTART=`ls -lrt *.restart | tail -n 1 | awk '{print $9}'`")
    L("  rm -f log.lammps")
    L("  post_dipole")
    L("\tBlur_MM")
    L('  if [ "$DIPOLECORR" == "yes" ]; then')
    L("    $MDDIPOLE MOBILE_final.cube $DIPOLEDIR $DIPOLEPOS")
    L("  fi")
    L("  cp dispf.ave repA.cube in.lammps* empty.cube MOBILE_final.cube *.lammpstrj $LAMMPSRESTART lammps.*.out mm_$qmmmstep")
    L()
    L("done # qmmm loop")
    L()
    L()

    qmmm_path = export_dir / "qmmm_dftces2_charging_pts.sh"
    qmmm_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    qmmm_path.chmod(0o755)
    print(f"[ces2_script_writer] Written: {qmmm_path}")

    # ======================================================================
    # submit_ces2.sh  — SLURM / PBS batch submission script
    # ======================================================================
    slurm_cfg = sc_cfg.get("slurm", {})
    pbs_cfg   = sc_cfg.get("pbs",   {})
    jobname   = str(sc_cfg.get("jobname", "ces2_qmmm"))

    sub_lines: List[str] = []

    def SL(s: str = "") -> None:
        sub_lines.append(s)

    use_pbs = (not slurm_cfg and pbs_cfg)
    if use_pbs:
        for h in _pbs_header(pbs_cfg):
            SL(h)
    else:
        for h in _slurm_header(slurm_cfg if slurm_cfg else {}, jobname=jobname):
            SL(h)

    SL()
    SL("# submit_ces2.sh — SLURM/PBS batch submission for CES2 QM/MM run")
    SL("# Auto-generated by cesbuild  (ces2_script_writer.py)")
    SL()

    module_lines_sub = str(sc_cfg.get("module_lines",
                                       slurm_cfg.get("module_lines", "")))
    if module_lines_sub.strip():
        for ml in module_lines_sub.strip().splitlines():
            SL(ml.strip())
        SL()

    SL(f'prefix="{jobname}"')
    if use_pbs:
        SL("curr_dir=${PBS_O_WORKDIR}")
    else:
        SL("curr_dir=${SLURM_SUBMIT_DIR}")
    SL("cd $curr_dir")
    SL()
    SL("./qmmm_dftces2_charging_pts.sh > $curr_dir/qmmm.out")

    sub_path = export_dir / "submit_ces2.sh"
    sub_path.write_text("\n".join(sub_lines) + "\n", encoding="utf-8")
    sub_path.chmod(0o755)
    print(f"[ces2_script_writer] Written: {sub_path}")

    return {"qmmm_sh": qmmm_path, "slurm_sh": sub_path}
