# Changelog

All changes relative to [`aracho12/CES2-automation@main`](https://github.com/aracho12/CES2-automation/tree/main) are documented here.
Each entry includes the modified file, the exact diff, and the reason.

---

## [Unreleased]

### `run_builder.py`

**What:** Added an explicit error when multiple `config*.yaml` files are found in cwd.

```diff
+    if len(candidates) > 1:
+        names = ", ".join(str(c) for c in candidates)
+        raise SystemExit(f"Multiple config*.yaml files found ({names}). Please specify one with --config.")
```

**Why:** Previously, if both `config.yaml` and `config_example.yaml` existed (the typical state after cloning), auto-detection silently returned `None` and failed with a cryptic message. The new error names the conflicting files and tells the user what to do.

---

### `builder/main.py`

**What:** Replaced the unconditional `bjparams_layer_file` block with an explicit `bjparams_source` selector.

```diff
-    # ── slab.bjparams_layer_file (optional) ──────────────────────────────────
-    # When provided, every slab atom is auto-labelled from a per-layer table
-    # (bjparams_layer_avg.dat): one type_label per (element, z-layer).
-    # This replaces hand-written type_label_overrides for systems whose QM
-    # atoms have z-dependent bjdisp parameters.
-    # Overrides still apply on top, so a layer-assigned label can be changed
-    # for specific primitive indices.
+    # ── slab.bjparams_source + slab.bjparams_layer_file ─────────────────────
+    # bjparams_source controls which database is used for QM slab BJ parameters:
+    #   "yaml"       (default) — load from species_db/qm_params/*.yaml
+    #   "layer_file"           — load from slab.bjparams_layer_file (.dat);
+    #                            requires bjparams_layer_file to be set
     _slab_cfg = cfg.get("slab", {}) or {}
+    _bjparams_source = str(_slab_cfg.get("bjparams_source", "yaml")).lower()
+    if _bjparams_source not in ("yaml", "layer_file"):
+        raise ValueError(
+            f"slab.bjparams_source must be 'yaml' or 'layer_file', got '{_bjparams_source}'"
+        )
+
     _layer_file = _slab_cfg.get("bjparams_layer_file")
     ...
-    if _layer_file:
+    if _bjparams_source == "layer_file":
+        if not _layer_file:
+            raise ValueError(
+                "slab.bjparams_source is 'layer_file' but slab.bjparams_layer_file is not set"
+            )
         ...
+    else:
+        print(f"[bjparams] source=yaml — loading from species_db/qm_params/")

-    qm_params_dir = species_db_path / "qm_params"
+    qm_params_dir = species_db_path / "qm_params" if _bjparams_source == "yaml" else None
```

**Why:** With `bjparams_layer_file` set in `config.yaml` but the `.dat` file not yet generated, the builder crashed unconditionally. There was no way to use YAML-based BJ parameters while keeping `bjparams_layer_file` in the config for future use.

**Usage in `config.yaml`:**
```yaml
slab:
  bjparams_source: yaml        # "yaml" (default) or "layer_file"
  bjparams_layer_file: bjparams_layer_avg.dat   # required only when source=layer_file
```

---

### `builder/bjdisp_db.py` — `load_all()`

**What:** Made `qm_params_dir` accept `Optional[Path]`; skips YAML loading when `None`.

```diff
 def load_all(
     species_db: Dict[str, Any],
-    qm_params_dir: Path,
+    qm_params_dir: Optional[Path],
     config_bjdisp: Optional[Dict] = None,
 ) -> ...:
+    """
+    ...
+    Pass qm_params_dir=None to skip loading the qm_params YAML database
+    (e.g. when slab.bjparams_source is "layer_file").
+    """
     mm_db = extract_mm_bjdisp_from_species(species_db)
-    qm_db = load_qm_params_db(qm_params_dir)
+    qm_db = load_qm_params_db(qm_params_dir) if qm_params_dir is not None else {}
```

**Why:** Required to support `bjparams_source: "layer_file"` — when the layer file is the sole BJ source, the YAML database should not be loaded.

---

### `builder/lammps_input_writer.py` — `generate_lammps_input()`

**What:** Changed the type annotation of `qm_params_dir` from `Path` to `Optional[Path]`.

```diff
-    qm_params_dir: Path,
+    qm_params_dir: Optional[Path],
```

**Why:** Consistent with the `load_all()` change. Allows `main.py` to pass `None` when `bjparams_source == "layer_file"`.

---

### `README.md`

Two separate sets of changes are present:

#### Pre-existing local changes (not tracked in upstream)

```diff
-### 1. Clone and set up Python environment
+### 1-1. Clone and set up Python environment
 ...
+### 1-2. Clone and set up Python environment using conda
+...
```

```diff
-For running simulations: LAMMPS with `gridforce/net` and `bjdisp` pair style, Quantum ESPRESSO ≥ 7.0
+For running simulations: LAMMPS with `gridforce/net` and `bjdisp` pair style, modified Quantum ESPRESSO ≥ 7.0 for DFT-CES2 simulation. (Refer https://github.com/dft-ces/dft-ces2)
```

#### Changes made in this session

```diff
-PACKMOL must be installed separately and on your PATH.
+PACKMOL must be installed separately.
 ...
-which packmol                          # verify
+which packmol                          # get the full path
+```
+
+Set the full binary path in `config.yaml`:
+
+```yaml
+packmol:
+  binary: "/path/to/your/packmol"   # e.g. /home/user/anaconda3/envs/ces2/bin/packmol
```

**Why:** Relying on `PATH` is fragile in conda environments. Requiring an explicit path in `config.yaml` makes the setup reproducible and consistent with how other binaries (`qe_binary`, `lmp_binary`, etc.) are handled.

---

### `config.yaml` (local only — not tracked in git)

```yaml
# Added:
packmol:
  binary: "/home/mino/anaconda3/envs/ces2/bin/packmol"

slab:
  bjparams_source: yaml        # use species_db/qm_params/ YAML; ignore missing .dat file
  bjparams_layer_file: bjparams_layer_avg.dat
```

---

### `run_builder.py` + `builder/main.py` — build completion message

**What:** Print the export directory path and all generated files at the very end of output (after the JSON summary).

`builder/main.py`:
```diff
+    summary["export_dir"] = str(export_dir.resolve())
     return summary
```

`run_builder.py`:
```diff
     print(json.dumps(summary, indent=2))
+
+    export_dir = Path(summary["export_dir"])
+    print(f"\nBuild complete. Output files in: {export_dir}")
+    for f in sorted(export_dir.iterdir()):
+        print(f"  {f.name}")
```

**Why:** The builder previously had no indication of where output files were written. The message is placed after the JSON so it appears last in the terminal output.

---

### `slab.qm_params_file` — specify which QM params YAML to load

**What:** Added `slab.qm_params_file` config key to load only a specific file from `species_db/qm_params/` instead of all files.

**Changes:**

`builder/bjdisp_db.py` — `load_qm_params_db()`:
```diff
-def load_qm_params_db(qm_params_dir: Path) -> Dict[str, BjdispParams]:
+def load_qm_params_db(qm_params_dir: Path, filename: Optional[str] = None) -> Dict[str, BjdispParams]:
+    if filename is not None:
+        stem = Path(filename).stem
+        target = qm_params_dir / f"{stem}.yaml"
+        if not target.exists():
+            available = [f.name for f in sorted(qm_params_dir.glob("*.yaml"))]
+            raise FileNotFoundError(
+                f"slab.qm_params_file: '{target.name}' not found in {qm_params_dir}. "
+                f"Available: {available}"
+            )
+        yaml_files = [target]
+    else:
+        yaml_files = sorted(qm_params_dir.glob("*.yaml"))
```

`builder/bjdisp_db.py` — `load_all()`:
```diff
+    qm_params_file: Optional[str] = None,
     ...
-    qm_db = load_qm_params_db(qm_params_dir) if qm_params_dir is not None else {}
+    qm_db = load_qm_params_db(qm_params_dir, filename=qm_params_file) if qm_params_dir is not None else {}
```

`builder/lammps_input_writer.py` — `generate_lammps_input()` and `builder/main.py` — `run()`: propagate `qm_params_file` through the call chain.

**Usage in `config.yaml`:**
```yaml
slab:
  bjparams_source: yaml
  qm_params_file: "IrO2"   # load only species_db/qm_params/IrO2.yaml
                             # omit to load all *.yaml files (original behavior)
```

**Why:** When multiple electrode systems exist in `species_db/qm_params/`, loading all files risks silent type_label collisions (e.g., two systems both defining `O` with different parameters). Specifying the file explicitly makes the parameter source unambiguous.

---

### `config_example.yaml` — `slab` section updated

**What:** Replaced the bare `bjparams_layer_file` line with a fully documented `slab` block showing both usage options.

- **Option 1 (yaml, uncommented):** `bjparams_source: yaml` + `qm_params_file: "IrO2"` as the active default.
- **Option 2 (layer_file, commented out):** shows how to switch to `.dat`-based parameters, including how to generate the file first with `tools/extract_bjparams.py`.
- `type_label_overrides` example preserved as a comment.

**Why:** The original example only showed `bjparams_layer_file` with no explanation of alternatives or how to switch between them.

---

### `builder/builder.py` — `PLATEPOS` unit fix

**What:** Fixed `plate_pos` calculation from dimensionless fraction to bohr.

```diff
-    # Plate position: midpoint of slab as fraction of box z
-    slab_mid_z = 0.5 * (z_min + z_max)
-    plate_pos = slab_mid_z / box_z_total if box_z_total > 0 else 0.0
+    # Plate position: midpoint of slab in bohr
+    slab_mid_z = 0.5 * (z_min + z_max)
+    plate_pos = slab_mid_z * ANG_TO_BOHR
```

**Why:** `CHGPLATE` expects `PLATEPOS` in bohr (as documented in the generated script comment), but the code was passing a dimensionless fraction of box height. `mpc_layer` was already converted to bohr correctly.

---

### `builder/builder.py` — `ATOMIC_MASS` table + error message

**What 1:** Added `Au` to the hardcoded atomic mass table.

```diff
     "Ir": 192.217,
+    "Au": 196.967,
 }
```

**What 2:** Improved error message when an element is missing from `ATOMIC_MASS` to tell the user exactly what to add and where.

```diff
-        el = lbl if lbl in ATOMIC_MASS else label_to_element.get(lbl, None)
-        if el is None:
-            raise ValueError(f"Cannot infer atomic mass for type_label '{lbl}'. ...")
+        if lbl in ATOMIC_MASS:
+            el = lbl
+        else:
+            el = label_to_element.get(lbl)
+            if el is None:
+                raise ValueError(
+                    f"... If '{lbl}' is an element symbol missing from the table, add it to "
+                    f"ATOMIC_MASS in builder/builder.py:\n"
+                    f'    "{lbl}": <mass>,    # g/mol'
+                )
+            if el not in ATOMIC_MASS:
+                raise ValueError(
+                    f"Element '{el}' (from type_label '{lbl}') is not in ATOMIC_MASS. "
+                    f"Add it to builder/builder.py:\n"
+                    f'    "{el}": <mass>,    # g/mol'
+                )
```

**Why:** Previously the error didn't hint at the fix. Two cases covered: (1) plain element symbol not in table (e.g. `Pt`), (2) custom type_label (e.g. `Pt_surf`) mapped to an element not in the table.

---

### `CHANGELOG.md` (new file)

This file.

---

## How to generate `bjparams_layer_avg.dat`

To switch to `bjparams_source: layer_file`, first generate the `.dat` file:

```bash
python tools/extract_bjparams.py <OUTCAR>
```

Requires a VASP OUTCAR with FISCS vdW data. Produces:
- `bjparams_layer_avg.dat` — layer-averaged BJ parameters (use as `bjparams_layer_file`)
- `bjparams_zsorted.dat`, `bjparams_elem_zsorted.dat` — intermediate tables

Then update `config.yaml`:
```yaml
slab:
  bjparams_source: layer_file
  bjparams_layer_file: bjparams_layer_avg.dat
```
