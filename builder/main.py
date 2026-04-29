from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import time
from ase.io import read, write

from .config import load_config
from .species import load_species_db
from .vasp_io import read_vasp, write_vasp, write_xyz, make_supercell, is_orthogonal_cell
from .box import compute_box_meta
from .composition import water_volume_L, auto_water_count, counts_from_salts, compute_total_charge, adjust_with_counterions
from .packmol import PackmolJob, StructureReq, write_packmol_input, run_packmol
from .builder import (
    write_species_xyz, make_type_registry, assign_mm_types_charges_by_order,
    build_mm_connectivity, apply_slab_charging, masses_by_type_from_labels,
    detect_top_layer_z, compute_charged_system_params,
)
from .lammps_writer import write_data_file_reference_style, DataFileFormat
from .md_workflow import generate_md_bundle
from .lammps_input_writer import generate_lammps_input, collect_relax_ff_params
from .qe_writer import generate_qe_input
from .ces2_script_writer import generate_ces2_scripts  # type_id_by_label, species info passed at call
from .bjdisp_db import (
    parse_layer_file, assign_layer_labels, layer_label_to_element, BjdispParams,
)

def run(config_path: str | Path, vasp_file: str | Path | None = None) -> Dict:
    start_time = time.perf_counter()
    timings: Dict[str, float] = {}

    cfg = load_config(config_path).raw
    workdir = Path(cfg.get("project", {}).get("workdir","./")).resolve()
    seed = int(cfg.get("project", {}).get("seed", 1234))
    np.random.seed(seed)

    build_dir = workdir / cfg.get("output", {}).get("build_dir","build")
    export_dir = workdir / cfg.get("output", {}).get("export_dir","export")
    build_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    timings["config_and_dirs"] = time.perf_counter() - start_time
    print(f"[TIMING] config_and_dirs: {timings['config_and_dirs']:.3f} s")

    # load species DB
    # Priority: 1) absolute path in config, 2) relative to workdir, 3) bundled with package
    t0 = time.perf_counter()
    _package_dir = Path(__file__).parent.parent  # CES2-automation/
    species_db_dir = cfg.get("species_db", None)
    if species_db_dir is None:
        species_db_path = _package_dir / "species_db"
    else:
        species_db_path = Path(species_db_dir)
        if not species_db_path.is_absolute():
            candidate = workdir / species_db_path
            if candidate.exists():
                species_db_path = candidate
            else:
                species_db_path = _package_dir / species_db_path
    species_db = load_species_db(species_db_path)
    timings["species_db_load"] = time.perf_counter() - t0
    print(f"[TIMING] species_db_load: {timings['species_db_load']:.3f} s")

    # read slab + supercell
    # Priority: --input/-i argument > config input.vasp_file
    t0 = time.perf_counter()
    if vasp_file is not None:
        _vasp_path = Path(vasp_file).resolve()
    else:
        _vasp_path = (workdir / cfg["input"]["vasp_file"]).resolve()
    slab = read_vasp(_vasp_path.as_posix())
    if cfg.get("cell", {}).get("require_orthogonal", True) and not is_orthogonal_cell(slab.cell.array):
        raise ValueError("Cell is not orthogonal. v0.2 still assumes orthogonal (matches your current writer).")

    rep = tuple(int(x) for x in cfg["cell"]["supercell"])
    slab_sc = make_supercell(slab, rep)
    write_vasp((build_dir/"slab_supercell.vasp").as_posix(), slab_sc)
    write_xyz((build_dir/"slab_supercell.xyz").as_posix(), slab_sc)
    timings["slab_supercell"] = time.perf_counter() - t0
    print(f"[TIMING] slab_supercell: {timings['slab_supercell']:.3f} s")

    # box meta
    t0 = time.perf_counter()
    _ebox_cfg = cfg["electrolyte_box"]
    box = compute_box_meta(slab_sc,
                           z_gap=float(_ebox_cfg["z_gap"]),
                           thickness=float(_ebox_cfg["thickness"]),
                           z_margin_top=float(_ebox_cfg["z_margin_top"]),
                           vacuum_z=float(_ebox_cfg.get("vacuum_z", 20.0)),
                           z_buffer_lo=float(_ebox_cfg.get("z_buffer_lo", 1.0)))
    (build_dir/"box_meta.json").write_text(json.dumps(box.__dict__, indent=2), encoding="utf-8")
    timings["box_meta"] = time.perf_counter() - t0
    print(f"[TIMING] box_meta: {timings['box_meta']:.3f} s")

    # electrolyte recipe
    t0 = time.perf_counter()
    recipe = cfg["electrolyte_recipe"]

    # Auto-derive water species_id from ces2.water_model when not explicitly set.
    # Mapping: TIP4P → water_tip4p,  TIP3P → water_tip3p
    _WATER_MODEL_TO_SID = {"TIP4P": "water_tip4p", "TIP3P": "water_tip3p"}
    _water_model = str(cfg.get("ces2", {}).get("water_model", "TIP4P")).upper()
    _default_sid = _WATER_MODEL_TO_SID.get(_water_model, "water_tip4p")
    water_sid = recipe["water"].get("species_id", _default_sid)

    rho = float(recipe["water"].get("density_g_per_ml", cfg.get("composition", {}).get("density_g_per_ml", 1.0)))
    if water_sid not in species_db:
        raise KeyError(f"Water species_id '{water_sid}' not found in species_db")

    # Water count: explicit integer or auto-derive from electrolyte-box geometry.
    # Accepts:  count: <int>   (explicit)
    #           count: auto    (default — derived from Lx·Ly·thickness·ρ·underfill)
    #           count omitted  (treated as 'auto')
    _w_count = recipe["water"].get("count", "auto")
    if isinstance(_w_count, str) and _w_count.strip().lower() == "auto":
        underfill = float(recipe["water"].get("packmol_underfill", 0.97))
        thickness_A = box.z_el_hi - box.z_el_lo
        n_water = auto_water_count(box.Lx, box.Ly, thickness_A,
                                   rho_g_ml=rho, underfill=underfill)
        print(f"[main] water.count: auto → {n_water}  "
              f"(Lx·Ly·t = {box.Lx:.2f}·{box.Ly:.2f}·{thickness_A:.2f} = "
              f"{box.Lx*box.Ly*thickness_A:.0f} Å³, ρ={rho:g} g/ml, "
              f"underfill={underfill:g})")
    else:
        n_water = int(_w_count)

    V_L = water_volume_L(n_water, rho)

    counts = {water_sid: n_water}
    # salts -> counts
    salts = recipe.get("salts", [])
    counts_salts = counts_from_salts(V_L, salts)
    for sid, n in counts_salts.items():
        counts[sid] = counts.get(sid, 0) + int(n)

    # extras (explicit counts)
    for ex in recipe.get("extras", []):
        sid = ex["species_id"]
        n = int(ex["count"])
        counts[sid] = counts.get(sid, 0) + n

    # charge control
    q_electrode = float(cfg.get("charge_control", {}).get("q_electrode_user_value", 0.0))
    q_target = float(cfg.get("charge_control", {}).get("q_target_total", 0.0))

    net_charge_by_species = {sid: float(species_db[sid].net_charge) for sid in counts.keys()}
    q_total = compute_total_charge({k:v for k,v in counts.items() if k!=water_sid}, net_charge_by_species, q_electrode)
    # include water net charge too (usually 0)
    q_total = compute_total_charge(counts, {sid: float(species_db[sid].net_charge) for sid in counts.keys()}, q_electrode)

    counter_pool = recipe.get("counterion_pool", [])
    if counter_pool:
        # ensure net_charge map has entries
        for sid in counter_pool:
            if sid not in species_db:
                raise KeyError(f"counterion_pool species_id '{sid}' not found in species_db")
        net_charge_all = {sid: float(species_db[sid].net_charge) for sid in species_db.keys()}
        counts_adj, q_final = adjust_with_counterions(
            counts=counts,
            net_charge_by_species=net_charge_all,
            q_electrode=q_electrode,
            q_target_total=q_target,
            counterion_pool=counter_pool,
            max_pair_search=int(recipe.get("counterion_pair_search", 2000))
        )
        counts = counts_adj
        q_total = q_final

    (build_dir/"composition_final.json").write_text(json.dumps({
        "V_L_approx": V_L,
        "counts": counts,
        "Q_electrode": q_electrode,
        "Q_target_total": q_target,
        "Q_total_final": q_total
    }, indent=2), encoding="utf-8")
    timings["composition_and_charge"] = time.perf_counter() - t0
    print(f"[TIMING] composition_and_charge: {timings['composition_and_charge']:.3f} s")

    # write per-species xyz templates for packmol
    t0 = time.perf_counter()
    structures: List[StructureReq] = []
    z_excl = recipe.get("z_exclusion", {})
    water_excl = float(z_excl.get("water_A", 2.0))
    ion_excl = float(z_excl.get("ions_A", 4.0))
    neutral_excl = float(z_excl.get("neutral_A", 2.0))

    species_order: List[Tuple[str,int]] = []
    # packmol order: water first, then everything else sorted for reproducibility
    # (you can override by giving explicit packmol_order in recipe)
    order = recipe.get("packmol_order", None)
    if order is None:
        order = [water_sid] + sorted([sid for sid in counts.keys() if sid != water_sid])

    for sid in order:
        n = int(counts.get(sid, 0))
        if n <= 0:
            continue
        sp = species_db[sid]
        xyz_path = build_dir / f"{sid}.xyz"
        write_species_xyz(xyz_path, sp)

        # zmin by charge class
        if abs(sp.net_charge) < 1e-12:
            zmin = box.z_el_lo + (water_excl if sid == water_sid else neutral_excl)
        else:
            zmin = box.z_el_lo + ion_excl

        structures.append(StructureReq(species_id=sid, xyz_file=xyz_path.name, count=n, zmin=zmin))
        species_order.append((sid, n))

    (build_dir/"species_order.json").write_text(json.dumps(species_order, indent=2), encoding="utf-8")
    timings["species_xyz_and_order"] = time.perf_counter() - t0
    print(f"[TIMING] species_xyz_and_order: {timings['species_xyz_and_order']:.3f} s")

    # run packmol
    t0 = time.perf_counter()
    pack = cfg["packmol"]
    job = PackmolJob(
        binary=str(pack.get("binary","packmol")),
        tolerance=float(pack.get("tolerance",2.0)),
        maxit=int(pack.get("maxit",200)),
        seed=seed,
        Lx=box.Lx, Ly=box.Ly,
        z_lo=box.z_el_lo, z_hi=box.z_el_hi,
        output_xyz="mm_packmol.xyz",
        structures=structures
    )
    inp = build_dir/"packmol.inp"
    write_packmol_input(inp, job)
    pmol = run_packmol(job.binary, inp, build_dir)
    timings["packmol"] = time.perf_counter() - t0
    print(f"[TIMING] packmol: {timings['packmol']:.3f} s")
    # ── Packmol convergence report ──────────────────────────────────────────
    _status = "SOLUTION FOUND" if pmol.converged else "NOT CONVERGED (overlaps may remain!)"
    print(f"[PACKMOL] status      : {_status}")
    print(f"[PACKMOL] obj_final   : {pmol.obj_final:.6g}  (0 = perfect, >0 = overlaps remain)")
    print(f"[PACKMOL] iterations  : {pmol.n_iter}  (maxit={job.maxit})")
    print(f"[PACKMOL] full log    : {build_dir/'packmol.log'}")
    if not pmol.converged:
        print(f"[PACKMOL] WARNING: increase packmol.maxit in config.yaml "
              f"(current={job.maxit}) or check packmol.log")

    mm_xyz = build_dir/"mm_packmol.xyz"
    if not mm_xyz.exists():
        raise RuntimeError("PACKMOL did not produce mm_packmol.xyz")

    t0 = time.perf_counter()
    mm = read(mm_xyz.as_posix())
    n_mm = len(mm)

    combined = mm + slab_sc
    cell = slab_sc.cell.copy()
    cell[2,2] = box.z_el_hi
    combined.set_cell(cell)
    combined.set_pbc([True,True,True])
    write((build_dir/"combined.xyz").as_posix(), combined)
    write((export_dir/"combined.xyz").as_posix(), combined)
    timings["ase_read_and_combine"] = time.perf_counter() - t0
    print(f"[TIMING] ase_read_and_combine: {timings['ase_read_and_combine']:.3f} s")

    # type registry
    t0 = time.perf_counter()

    # ── slab.bjparams_source + slab.bjparams_layer_file ─────────────────────
    # bjparams_source controls which database is used for QM slab BJ parameters:
    #   "yaml"       (default) — load from species_db/qm_params/*.yaml
    #   "layer_file"           — load from slab.bjparams_layer_file (.dat);
    #                            requires bjparams_layer_file to be set
    _slab_cfg = cfg.get("slab", {}) or {}
    _bjparams_source = str(_slab_cfg.get("bjparams_source", "yaml")).lower()
    if _bjparams_source not in ("yaml", "layer_file"):
        raise ValueError(
            f"slab.bjparams_source must be 'yaml' or 'layer_file', got '{_bjparams_source}'"
        )

    _layer_file = _slab_cfg.get("bjparams_layer_file")
    _qm_params_file: Optional[str] = None
    _layer_db: Dict[str, BjdispParams] = {}
    _layer_label_by_sc_idx: Dict[int, str] = {}
    _layer_label_to_element: Dict[str, str] = {}
    if _bjparams_source == "layer_file":
        if not _layer_file:
            raise ValueError(
                "slab.bjparams_source is 'layer_file' but slab.bjparams_layer_file is not set"
            )
        _layer_path = Path(_layer_file)
        if not _layer_path.is_absolute():
            _layer_path = (workdir / _layer_path).resolve()
        if not _layer_path.exists():
            raise FileNotFoundError(
                f"slab.bjparams_layer_file: {_layer_path} not found"
            )
        if int(rep[2]) != 1:
            raise ValueError(
                f"slab.bjparams_layer_file requires supercell rep[2]==1 "
                f"(got rep={rep}); z-tiled slabs aren't supported because "
                f"the layer file's z values are primitive-cell relative."
            )
        _z_tol, _layer_entries, _layer_db = parse_layer_file(_layer_path)
        _layer_label_to_element = layer_label_to_element(_layer_entries)
        _sc_syms_tmp = list(slab_sc.get_chemical_symbols())
        _sc_z_tmp = [float(z) for z in slab_sc.positions[:, 2]]
        _layer_labels_sc = assign_layer_labels(
            _sc_syms_tmp, _sc_z_tmp, _layer_entries, _z_tol,
        )
        _layer_label_by_sc_idx = {i: lbl for i, lbl in enumerate(_layer_labels_sc)}
        _n_unique_layers = len(set(_layer_labels_sc))
        print(f"[layer file] {_layer_path.name}: z_tol={_z_tol} Å, "
              f"{len(_layer_entries)} layer(s) → {len(_layer_labels_sc)} slab atoms "
              f"labelled into {_n_unique_layers} unique type_label(s)")
    else:
        _qm_params_file = _slab_cfg.get("qm_params_file")
        if _qm_params_file:
            print(f"[bjparams] source=yaml — loading from species_db/qm_params/{Path(_qm_params_file).stem}.yaml")
        else:
            print(f"[bjparams] source=yaml — loading all files from species_db/qm_params/")

    # ── slab.type_label_overrides ────────────────────────────────────────────
    # config: slab.type_label_overrides: {<1-based primitive index>: <type_label>}
    # Expands primitive-cell indices to the full supercell automatically.
    # ASE repeat([na,nb,nc]) ordering: for primitive atom p (0-based),
    #   supercell index = m0*nb*nc*N + m1*nc*N + m2*N + p
    #   for m0∈[0,na), m1∈[0,nb), m2∈[0,nc)
    _raw_overrides = cfg.get("slab", {}).get("type_label_overrides", {})
    _slab_sc_overrides: Dict[int, str] = {}   # supercell 0-based index → type_label
    if _raw_overrides:
        _n_prim = len(slab)
        _na, _nb, _nc = rep
        for _pkey, _plabel in _raw_overrides.items():
            _p = int(_pkey) - 1   # 1-based → 0-based
            if _p < 0 or _p >= _n_prim:
                raise ValueError(
                    f"slab.type_label_overrides: primitive index {int(_pkey)} out of range "
                    f"(primitive cell has {_n_prim} atoms, valid: 1–{_n_prim})"
                )
            for _m0 in range(_na):
                for _m1 in range(_nb):
                    for _m2 in range(_nc):
                        _sc_idx = _m0*_nb*_nc*_n_prim + _m1*_nc*_n_prim + _m2*_n_prim + _p
                        _slab_sc_overrides[_sc_idx] = str(_plabel)
        _n_unique = len(set(_slab_sc_overrides.values()))
        print(f"[slab overrides] {len(_raw_overrides)} primitive override(s) → "
              f"{len(_slab_sc_overrides)} supercell atoms relabelled "
              f"({_n_unique} custom type_label(s))")

    # Per-atom type_label list for slab supercell. Priority:
    #   type_label_overrides  >  layer-file label  >  element symbol
    _slab_syms = list(slab_sc.get_chemical_symbols())
    slab_type_labels = [
        _slab_sc_overrides.get(i, _layer_label_by_sc_idx.get(i, el))
        for i, el in enumerate(_slab_syms)
    ]

    # Map custom QM labels → base element (for mass lookup)
    _extra_label_to_element: Dict[str, str] = dict(_layer_label_to_element)
    for _sc_idx, _lbl in _slab_sc_overrides.items():
        _extra_label_to_element[_lbl] = _slab_syms[_sc_idx]

    species_ids_in_order = [sid for sid,_ in species_order]
    type_id_by_label, label_by_type_id = make_type_registry(species_ids_in_order, species_db, slab_type_labels)

    # assign MM types/charges by order
    mm_types, mm_charges, mm_slices = assign_mm_types_charges_by_order(mm, species_order, species_db, type_id_by_label)

    # connectivity
    bonds, angles, bond_coeffs, angle_coeffs = build_mm_connectivity(mm_slices, species_db)

    # slab types/charges
    slab_types = [type_id_by_label[lbl] for lbl in slab_type_labels]

    # ── charged system: top-layer detection & charge distribution ──────────
    chg_cfg = cfg.get("charge_control", {})
    _top_layer_tol = float(chg_cfg.get("top_layer_tolerance", 0.5))  # Å
    _exclude_labels = list(chg_cfg.get("exclude_labels", []))  # e.g. ["O_ads","H_ads"]

    if q_electrode != 0.0:
        z_cutoff = detect_top_layer_z(slab_sc, tolerance=_top_layer_tol)
        slab_charges = apply_slab_charging(
            slab_sc, q_electrode,
            z_cutoff=z_cutoff,
            exclude_labels=_exclude_labels if _exclude_labels else None,
            slab_type_labels=slab_type_labels if _exclude_labels else None,
        )
        print(f"[charged] top-layer z_cutoff={z_cutoff:.3f} Å, "
              f"q_electrode={q_electrode}, "
              f"n_top_atoms={sum(1 for c in slab_charges if c != 0.0)}")
    else:
        z_cutoff = None
        slab_charges = apply_slab_charging(slab_sc, q_electrode)

    # ── auto-calculate CES2 charged-system parameters ──────────────────────
    sc_factor = int(rep[0]) * int(rep[1]) * int(rep[2])
    import numpy as _np2
    _max_z_all = float(_np2.max(combined.positions[:, 2]))
    _z_hi_data = max(box.z_el_hi, _max_z_all + 1.0) + box.vacuum_z
    _z_lo_data = -box.z_buffer_lo
    _box_z_data = _z_hi_data - _z_lo_data

    charged_params = compute_charged_system_params(
        slab_sc, q_electrode, sc_factor, _box_z_data,
        z_cutoff=z_cutoff,
        top_layer_tol=_top_layer_tol,
    )
    (export_dir / "charged_system_params.json").write_text(
        json.dumps(charged_params, indent=2), encoding="utf-8")
    print(f"[charged] auto-calculated: tot_chg={charged_params['tot_chg']:.8f}, "
          f"mpc_layer={charged_params['mpc_layer']:.4f} bohr, "
          f"plate_pos={charged_params['plate_pos']:.6f}")

    atom_types = mm_types + slab_types
    charges = mm_charges + slab_charges

    # masses by type
    masses_by_type = masses_by_type_from_labels(label_by_type_id, type_id_by_label, species_db,
                                                extra_label_to_element=_extra_label_to_element)
    timings["types_and_connectivity"] = time.perf_counter() - t0
    print(f"[TIMING] types_and_connectivity: {timings['types_and_connectivity']:.3f} s")

    # write data.file in reference style
    t0 = time.perf_counter()
    out_data = export_dir / "data.file"
    write_data_file_reference_style(
        out_data, combined,
        atom_types=atom_types,
        charges=charges,
        masses_by_type=masses_by_type,
        bond_coeffs=bond_coeffs if bond_coeffs else None,
        angle_coeffs=angle_coeffs if angle_coeffs else None,
        bonds=bonds,
        angles=angles,
        fmt=DataFileFormat(title="data"),
        vacuum_z=box.vacuum_z,
        z_buffer_lo=box.z_buffer_lo,
    )
    timings["write_data_file"] = time.perf_counter() - t0
    print(f"[TIMING] write_data_file: {timings['write_data_file']:.3f} s")

    # QM/MM lists (QM=slab)
    t0 = time.perf_counter()
    qm_ids = list(range(n_mm+1, len(combined)+1))
    mm_ids = list(range(1, n_mm+1))
    (export_dir/"qm_atoms.txt").write_text("\n".join(map(str,qm_ids))+"\n", encoding="utf-8")
    (export_dir/"mm_atoms.txt").write_text("\n".join(map(str,mm_ids))+"\n", encoding="utf-8")
    timings["qm_mm_lists"] = time.perf_counter() - t0
    print(f"[TIMING] qm_mm_lists: {timings['qm_mm_lists']:.3f} s")

    # Compute full simulation-box z span (must match data.file exactly).
    # Uses the same formula as write_data_file_reference_style:
    #   z_hi = max(z_el_hi, max_atom_z + 1.0) + vacuum_z
    #   z_lo = -z_buffer_lo
    # Computed here (before generate_lammps_input) so the writer can clamp the
    # SOLVENT upper wall safely below QE's emaxpos zone.
    import numpy as _np
    _max_atom_z = float(_np.max(combined.positions[:, 2]))
    _z_hi = max(box.z_el_hi, _max_atom_z + 1.0) + box.vacuum_z
    _z_lo = -box.z_buffer_lo
    box_z_total = _z_hi - _z_lo
    box.box_z_total = box_z_total
    box.box_zlo     = _z_lo
    box.box_zhi     = _z_hi

    # Generate base.in.lammps (CES2 QM/MM LAMMPS input)
    t0 = time.perf_counter()
    qm_params_dir = species_db_path / "qm_params" if _bjparams_source == "yaml" else None
    generate_lammps_input(
        export_dir=export_dir,
        type_id_by_label=type_id_by_label,
        label_by_type_id=label_by_type_id,
        species_order=species_order,
        species_db=species_db,
        box=box,
        bond_coeffs=bond_coeffs,
        angle_coeffs=angle_coeffs,
        qm_params_dir=qm_params_dir,
        qm_params_file=_qm_params_file if _bjparams_source == "yaml" else None,
        cfg=cfg,
        n_mm=n_mm,
        charged_params=charged_params,
        extra_qm_bjdisp=_layer_db if _layer_db else None,
    )
    timings["lammps_input"] = time.perf_counter() - t0
    print(f"[TIMING] lammps_input: {timings['lammps_input']:.3f} s")

    # Generate base.pw.in + base.pp.in (QE input)
    t0 = time.perf_counter()
    qm_elements_ordered = sorted(set(slab_sc.get_chemical_symbols()))
    generate_qe_input(
        export_dir=export_dir,
        slab_cell=slab_sc.cell.array,
        n_qm_total=len(slab_sc),
        qm_elements=qm_elements_ordered,
        box_z_total=box_z_total,
        cfg=cfg,
    )
    timings["qe_input"] = time.perf_counter() - t0
    print(f"[TIMING] qe_input: {timings['qe_input']:.3f} s")

    # Build type_id → element mapping for direct atoms=(...) generation in shell script.
    # This avoids the fragile mass-matching approach (which breaks when an element
    # like Cs is absent from the hardcoded ATOMMASS table).
    _label_to_element: Dict[str, str] = {}
    for sid in species_ids_in_order:
        for a in species_db[sid].atoms:
            _label_to_element[a.type_label] = a.element
    for lbl in set(slab_type_labels):
        if lbl in _extra_label_to_element:
            _label_to_element[lbl] = _extra_label_to_element[lbl]
        elif lbl not in _label_to_element:
            _label_to_element[lbl] = lbl   # slab label IS the element symbol (e.g. "Ir", "O")
    type_id_to_element: Dict[int, str] = {
        tid: _label_to_element[lbl]
        for tid, lbl in label_by_type_id.items()
    }

    # Generate qmmm wrapper + SLURM submit scripts
    t0 = time.perf_counter()
    n_qm = len(slab_sc)
    generate_ces2_scripts(
        export_dir=export_dir,
        cfg=cfg,
        n_mm=n_mm,
        n_qm=n_qm,
        species_order=species_order,
        species_db=species_db,
        water_sid=water_sid,
        type_id_by_label=type_id_by_label,
        slab_elements=slab_type_labels,
        type_id_to_element=type_id_to_element,
        charged_params=charged_params,
    )
    timings["ces2_scripts"] = time.perf_counter() - t0
    print(f"[TIMING] ces2_scripts: {timings['ces2_scripts']:.3f} s")

    summary = {
        "box": box.__dict__,
        "counts": counts,
        "species_order": species_order,
        "q_electrode": q_electrode,
        "q_total_final": q_total,
        "n_total": len(combined),
        "n_mm": n_mm,
        "n_qm": len(combined)-n_mm,
        "n_bonds": len(bonds),
        "n_angles": len(angles),
        "n_atom_types": max(atom_types) if atom_types else 0,
        "packmol": {
            "converged":  pmol.converged,
            "obj_final":  pmol.obj_final,
            "iterations": pmol.n_iter,
            "maxit":      job.maxit,
            "tolerance":  job.tolerance,
        },
    }
    (export_dir/"build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
# ---- Optional MD pre-relax bundle ----
    md_cfg = cfg.get("md_relax", None)
    if md_cfg and bool(md_cfg.get("enabled", False)):
        t0 = time.perf_counter()
        relax_cutoff = float(md_cfg.get("lj_cutoff", 10.0))
        relax_ff = collect_relax_ff_params(
            type_id_by_label=type_id_by_label,
            label_by_type_id=label_by_type_id,
            species_order=species_order,
            species_db=species_db,
            bond_coeffs=bond_coeffs,
            angle_coeffs=angle_coeffs,
            cfg=cfg,
            relax_cutoff=relax_cutoff,
        )
        md_out = generate_md_bundle(export_dir, md_cfg, relax_ff=relax_ff,
                                    qe_cfg=cfg.get("qe", {}))
        timings["md_bundle"] = time.perf_counter() - t0
        print(f"[TIMING] md_bundle: {timings['md_bundle']:.3f} s")
        summary["md_relax_bundle"] = md_out.__dict__

    total_elapsed = time.perf_counter() - start_time
    timings["total"] = total_elapsed
    print(f"[TIMING] total: {timings['total']:.3f} s")
    (export_dir/"timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    summary["timings_sec"] = timings

    summary["export_dir"] = str(export_dir.resolve())
    return summary
