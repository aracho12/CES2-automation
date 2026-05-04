# Changelog

All changes relative to [`aracho12/CES2-automation@main`](https://github.com/aracho12/CES2-automation/tree/main) are documented here.
Each entry includes the modified file, the exact diff, and the reason.

---

## [Unreleased]

### `builder/lammps_input_writer.py` — selectable TIP3P pair style

**What:** Made the TIP3P pair_style configurable via `ces2.pair_style_tip3p`.
Default is now `lj/charmm/coul/long/opt`; the previous hard-coded `lj/cut/long`
has been removed.

```diff
     else:
-        L(f"pair_style      hybrid/overlay lj/cut/long {lj_cut:.1f} bjdisp {bjd_cutoff:.0f}")
-        L(f"kspace_style    pppm {kspace_acc:.1e}")
+        # TIP3P: user-selectable pair_style (default: lj/charmm/coul/long/opt)
+        tip3p_substyle = str(ces2_cfg.get("pair_style_tip3p", "lj/charmm/coul/long/opt"))
+        _valid_tip3p = {
+            "lj/charmm/coul/long/opt",
+            "lj/charmm/coul/long",
+            "lj/cut/coul/long",
+        }
+        if tip3p_substyle not in _valid_tip3p:
+            raise ValueError(
+                f"ces2.pair_style_tip3p must be one of {sorted(_valid_tip3p)}, "
+                f"got '{tip3p_substyle}'"
+            )
+        if tip3p_substyle.startswith("lj/charmm"):
+            inner = float(ces2_cfg.get("charmm_inner_cutoff", lj_cut - 2.0))
+            L(f"pair_style      hybrid/overlay {tip3p_substyle} {inner:.1f} {lj_cut:.1f}"
+              f" bjdisp {bjd_cutoff:.0f}")
+        else:
+            L(f"pair_style      hybrid/overlay {tip3p_substyle} {lj_cut:.1f}"
+              f" bjdisp {bjd_cutoff:.0f}")
+        L(f"kspace_style    pppm {kspace_acc:.1e}")
```

The substyle name used for `pair_coeff` lines also follows the user choice:

```diff
-    lj_substyle = "lj/cut/tip4p/long/opt" if use_tip4p else "lj/cut/long"
+    lj_substyle = "lj/cut/tip4p/long/opt" if use_tip4p else tip3p_substyle
```

**Why:** `lj/cut/long` is not a Coulomb-aware pair style — TIP3P needs explicit
long-range Coulomb. `lj/charmm/coul/long/opt` is a sensible default for TIP3P
(CHARMM-style switching + long-range Coulomb, OMP-accelerated). Allowing
`lj/charmm/coul/long` and `lj/cut/coul/long` covers builds without the OMP
package or users who don't want CHARMM switching.

**Usage in `config.yaml`:**
```yaml
ces2:
  water_model: TIP3P
  # pair_style_tip3p: "lj/charmm/coul/long/opt"   # default
  # charmm_inner_cutoff: 13.0                      # default: lj_cutoff - 2
```

---

### `config_example.yaml` — document TIP3P pair_style options

**What:** Added a commented-out usage block under the `ces2` section
explaining the three valid `pair_style_tip3p` values and the
`charmm_inner_cutoff` option.

```diff
 ces2:
   # Water model: TIP4P (default, TIP4P-EW recommended for CES2) or TIP3P
   water_model: TIP4P

+  # ---- Pair style for TIP3P (ignored when water_model: TIP4P) ----
+  # TIP4P always uses lj/cut/tip4p/long/opt.
+  # For TIP3P, choose one of:
+  #   "lj/charmm/coul/long/opt" (default) — CHARMM-style switching, OMP accelerated
+  #   "lj/charmm/coul/long"               — CHARMM-style switching, no /opt
+  #   "lj/cut/coul/long"                  — plain cutoff LJ + long-range coul
+  # pair_style_tip3p: "lj/charmm/coul/long/opt"
+  # charmm_inner_cutoff: 13.0   # used only for lj/charmm/* styles (default: lj_cutoff - 2)
+
   # Pair-style cutoffs
   lj_cutoff:        15.0   # Angstrom - real-space LJ cutoff
```

**Why:** Mirrors the new `pair_style_tip3p` config key in code so users can
discover the options from the example without reading the source.

---

### `builder/vasp_io.py` + `builder/main.py` — read QE input files

**What:** The slab structure file no longer has to be a VASP `CONTCAR`. ASE-
supported formats are accepted, with the format auto-detected from the
filename. Quantum ESPRESSO `pw.x` input/output (e.g. `pw.in`, `*.pwi`,
`*.pwo`) is handled explicitly; VASP files are auto-detected.

`builder/vasp_io.py`:
```diff
+def _detect_format(path: str) -> Optional[str]:
+    name = Path(path).name.lower()
+    if name.endswith((".in", ".pwi")) or name.startswith("pw."):
+        return "espresso-in"
+    if name.endswith((".out", ".pwo")) or name.endswith(".pw.out"):
+        return "espresso-out"
+    return None
+
+def read_structure(path: str, fmt: Optional[str] = None) -> Atoms:
+    if fmt is None:
+        fmt = _detect_format(path)
+    if fmt is None:
+        return read(path)
+    return read(path, format=fmt)
+
 def read_vasp(path: str) -> Atoms:
-    return read(path)
+    """Backwards-compatible alias for read_structure."""
+    return read_structure(path)
```

`builder/main.py`:
```diff
-from .vasp_io import read_vasp, ...
+from .vasp_io import read_structure, ...
 ...
-    if vasp_file is not None:
-        _vasp_path = Path(vasp_file).resolve()
-    else:
-        _vasp_path = (workdir / cfg["input"]["vasp_file"]).resolve()
-    slab = read_vasp(_vasp_path.as_posix())
+    _input_cfg = cfg.get("input", {})
+    if vasp_file is not None:
+        _vasp_path = Path(vasp_file).resolve()
+    else:
+        _file = _input_cfg.get("structure_file") or _input_cfg.get("vasp_file")
+        if _file is None:
+            raise ValueError("Config must set input.structure_file (or input.vasp_file)")
+        _vasp_path = (workdir / _file).resolve()
+    _input_fmt = _input_cfg.get("format")
+    slab = read_structure(_vasp_path.as_posix(), fmt=_input_fmt)
+    print(f"[input] read {_vasp_path.name} ({len(slab)} atoms) ...")
```

**Why:** Some users only have QE `pw.x` inputs/outputs and shouldn't have to
convert them to VASP format. Format auto-detection covers the common cases;
the explicit `input.format` override handles ambiguous filenames.

**Usage in `config.yaml`:**
```yaml
input:
  structure_file: "scf.pw.in"   # or "CONTCAR", "POSCAR", "*.pwo", ...
  # vasp_file: "CONTCAR"         # legacy key, still supported
  # format: "espresso-in"        # optional override
```

The `--input` / `-i` CLI flag also works with non-VASP files.

---

### `run_builder.py` + `builder/main.py` — fix `--input` default overriding config

**What:** `--input` no longer defaults to `"CONTCAR"`; the default is now
`None` so that `input.structure_file` / `input.vasp_file` from the config
takes effect when the user doesn't pass `--input`.

`run_builder.py`:
```diff
-    ap.add_argument("--input", "-i", default="CONTCAR",
-                    help="Path to CONTCAR (overrides input.vasp_file in config)")
+    ap.add_argument("--input", "-i", default=None,
+                    help="Path to slab structure file (overrides input.structure_file / "
+                         "input.vasp_file in config). If neither is set, defaults to CONTCAR.")
```

`builder/main.py`:
```diff
-        _file = _input_cfg.get("structure_file") or _input_cfg.get("vasp_file")
-        if _file is None:
-            raise ValueError("Config must set input.structure_file (or input.vasp_file)")
-        _vasp_path = (workdir / _file).resolve()
+        # Priority: structure_file > vasp_file > "CONTCAR" (default)
+        _file = _input_cfg.get("structure_file") or _input_cfg.get("vasp_file") or "CONTCAR"
+        _vasp_path = (workdir / _file).resolve()
```

**Why:** Argparse's `default="CONTCAR"` always supplied a value, so the
config's `structure_file` was silently ignored. Bug surfaced when running
`python run_builder.py --config config_Pt111.yaml` with `structure_file:
"base.pw.in"` — it kept reading `CONTCAR` instead.

---

### `species_db/water_tip3p.yaml` — fix charges to standard TIP3P values

**What:** Replaced SPC/E charges (`-0.8476` / `+0.4238`) that were
incorrectly placed in the TIP3P file with the standard TIP3P values from
Jorgensen et al. (1983). Also added a source/parameter header comment and
renamed `(TIP3P-like)` → `(TIP3P)`.

```diff
-name: Water (TIP3P-like)
+name: Water (TIP3P)
 ...
+# Source: Jorgensen et al., J. Chem. Phys. 79, 926 (1983)
+# Geometry: d(OH)=0.9572 Å, angle(HOH)=104.52°
+# Charges:  q(O)=-0.834, q(H)=+0.417
+# LJ (O):   ε=0.1521 kcal/mol, σ=3.1507 Å (defined in lammps_input_writer.py)
 ...
-    charge: -0.8476
+    charge: -0.834
 ...
-    charge: 0.4238
+    charge: 0.417
```

**Why:** `-0.8476` is SPC/E, not TIP3P. The mismatch silently produced
non-standard TIP3P runs. Now that there is a dedicated `water_spce.yaml`,
each file holds the charges that match its model.

---

### `species_db/water_spce.yaml` (new) — SPC/E water model

**What:** New water species file for SPC/E (Berendsen, Grigera & Straatsma 1987).

```yaml
id: water_spce
atoms:
  - element: O
    type_label: Ow_spce
    charge: -0.8476
    ...
  - element: H
    type_label: Hw_spce
    charge: 0.4238
    ...
coeffs:
  bond_coeffs:  { 1: [450.0, 1.0000] }    # SPC/E rigid bond
  angle_coeffs: { 1: [55.0, 109.47] }     # SPC/E rigid angle
```

Distinct `Ow_spce` / `Hw_spce` type_labels prevent LJ collisions with TIP3P.

---

### `species_db/water_tip3p_ew.yaml` (new) — TIP3P-Ew water model

**What:** New water species file for TIP3P-Ew (Price & Brooks 2004).

```yaml
id: water_tip3p_ew
atoms:
  - element: O
    type_label: Ow_tip3pew
    charge: -0.830
    ...
  - element: H
    type_label: Hw_tip3pew
    charge: 0.415
    ...
coeffs:
  bond_coeffs:  { 1: [450.0, 0.9572] }    # same geometry as TIP3P
  angle_coeffs: { 1: [55.0, 104.52] }
```

---

### `builder/lammps_input_writer.py` + `builder/main.py` — wire SPC/E and TIP3P-Ew into water_model

**What:**
1. Added new water_model values `SPCE` and `TIP3PEW`.
2. Added LJ defaults for `Ow_spce` / `Ow_tip3pew` (water-O LJ; H has no LJ).
3. Updated three `_WATER_MODEL_TO_SID` lookups (one in `main.py`, two in
   `lammps_input_writer.py`) so water species selection follows water_model.

`builder/lammps_input_writer.py`:
```diff
 _DEFAULT_LJ_TIP3P: Dict[str, Tuple[float, float]] = {
     "Ow":    (0.1521,      3.1507),
     "Hw":    (0.0000,      1.0000),
     ...
+    # SPC/E water (water_spce.yaml)
+    "Ow_spce":    (0.1553,  3.166),
+    "Hw_spce":    (0.0000,  1.0000),
+    # TIP3P-Ew water (water_tip3p_ew.yaml)
+    "Ow_tip3pew": (0.102,   3.188),
+    "Hw_tip3pew": (0.0000,  1.0000),
     ...
 }
-    _WATER_MODEL_TO_SID = {"TIP4P": "water_tip4p", "TIP3P": "water_tip3p"}
+    _WATER_MODEL_TO_SID = {
+        "TIP4P":   "water_tip4p",
+        "TIP3P":   "water_tip3p",
+        "SPCE":    "water_spce",
+        "TIP3PEW": "water_tip3p_ew",
+    }
```

`builder/main.py`: identical update to its own `_WATER_MODEL_TO_SID`.

**Why:** The previous `_WATER_MODEL_TO_SID` only knew TIP4P/TIP3P, so
`water_model: SPCE` silently fell through to the `water_tip4p` default.
Three separate copies of the dict had to be updated to fully wire up the
new models.

**Usage in `config.yaml`:**
```yaml
ces2:
  water_model: SPCE      # or TIP3PEW
```

The TIP3P pair_style logic (CHARMM / cut + long-range coul) applies to all
3-site models, so `pair_style_tip3p` and `charmm_inner_cutoff` work the
same way under SPCE / TIP3PEW.

**Caveat:** Joung-Cheatham ion LJ defaults are taken from the TIP3P column
of JC 2008. SPC/E and TIP3P-Ew formally have their own ion sets — override
via `ces2.lj_params` in `config.yaml` if strict consistency is required.

---

### `config_example.yaml` — document new water_model values

**What:** Extended the `ces2.water_model` and `electrolyte_recipe.water`
comments to list the four supported models and their auto-derived species_ids.

```diff
-  # Water model: TIP4P (default, TIP4P-EW recommended for CES2) or TIP3P
+  # Water model — selects default species_id, charges, LJ defaults, and pair_style:
+  #   TIP4P    (default) — 4-site TIP4P-Ew (lj/cut/tip4p/long/opt)
+  #   TIP3P              — 3-site TIP3P
+  #   SPCE               — 3-site SPC/E
+  #   TIP3PEW            — 3-site TIP3P-Ew (Price & Brooks 2004)
   water_model: TIP4P
```

---

### `builder/lammps_input_writer.py` — refactor LJ defaults, auto-pick JC ion column per water model

**What:** Split the monolithic `_DEFAULT_LJ_TIP4P` / `_DEFAULT_LJ_TIP3P`
dicts into separate water-LJ and ion-LJ tables and added a builder function
that selects the matching Joung-Cheatham 2008 ion column based on
`water_model`.

```diff
-_DEFAULT_LJ_TIP4P = { ... water + JC TIP4P-Ew ions ... }
-_DEFAULT_LJ_TIP3P = { ... water (TIP3P + SPCE + TIP3PEW) + JC TIP3P ions ... }
+# Water LJ (one dict per model)
+_LJ_WATER_TIP4P_EW = {"Ow": (0.16275, 3.16435), "Hw": (0.0, 1.0)}
+_LJ_WATER_TIP3P    = {"Ow": (0.1521, 3.1507),  "Hw": (0.0, 1.0)}
+_LJ_WATER_SPCE     = {"Ow_spce":    (0.1553, 3.166), "Hw_spce":    (0.0, 1.0)}
+_LJ_WATER_TIP3P_EW = {"Ow_tip3pew": (0.102,  3.188), "Hw_tip3pew": (0.0, 1.0)}
+
+# Joung-Cheatham 2008 ion LJ (one column per matching water model)
+_LJ_IONS_JC_TIP4P_EW = { Li, Na, K, Rb, Cs, F, Cl, Br, I }   # existing
+_LJ_IONS_JC_TIP3P    = { ... }                                 # existing
+_LJ_IONS_JC_SPCE     = { ... }                                 # NEW
+
+def _build_lj_defaults(water_model: str) -> Dict:
+    if   wm == "TIP4P":   return {**_LJ_WATER_TIP4P_EW, ..., **_LJ_IONS_JC_TIP4P_EW, ...}
+    elif wm == "SPCE":    return {**_LJ_WATER_SPCE,     ..., **_LJ_IONS_JC_SPCE,     ...}
+    elif wm == "TIP3PEW": return {**_LJ_WATER_TIP3P_EW, ..., **_LJ_IONS_JC_TIP3P,    ...}
+    else:                 return {**_LJ_WATER_TIP3P,    ..., **_LJ_IONS_JC_TIP3P,    ...}
```

JC SPC/E values added (from Joung-Cheatham 2008 SPC/E column):

```python
_LJ_IONS_JC_SPCE = {
    "Li":  (0.3367050, 1.40880),
    "Na":  (0.3526418, 2.15952),
    "K":   (0.4297054, 2.83840),
    "Rb":  (0.4451081, 3.04509),
    "Cs":  (0.0898565, 3.83159),
    "F":   (0.0074005, 4.10219),
    "Cl":  (0.0127850, 4.83045),
    "Br":  (0.0269586, 4.90412),
    "I":   (0.0427845, 5.20892),
}
```

The two `lj_db` build sites (`generate_lammps_input` and the MD relax helper)
plus `print_lj_db` now all call `_build_lj_defaults(water_model)` instead
of inlining the model-specific dict choice.

**Why:** Previously `water_model: SPCE` produced SPC/E water LJ but JC
**TIP3P** ion LJ — silently inconsistent. Now the JC column is locked to
the water model. TIP3P-Ew has no published JC set, so JC TIP3P is used as
the closest fallback (documented in code).

**Verified per-model behaviour:**
| `water_model` | water-O LJ | Na+ LJ | Cl- LJ |
|---|---|---|---|
| TIP4P    | (0.16275, 3.16435) | JC TIP4P-Ew (0.168, 2.184) | JC TIP4P-Ew (0.012, 4.918) |
| TIP3P    | (0.15210, 3.15070) | JC TIP3P    (0.353, 2.160) | JC TIP3P    (0.720, 4.417) |
| SPCE     | (0.15530, 3.16600) | JC SPC/E    (0.353, 2.160) | JC SPC/E    (0.013, 4.830) |
| TIP3PEW  | (0.10200, 3.18800) | JC TIP3P fallback          | JC TIP3P fallback          |

---

### `builder/builder.py` — add Pt to `ATOMIC_MASS`

```diff
     "Ir": 192.217,
+    "Pt": 195.078,
     "Au": 196.967,
```

**Why:** Required for the new `species_db/qm_params/Pt111.yaml` slab; without
it the builder fails with `Cannot infer atomic mass for type_label 'Pt'`.

---

### `species_db/qm_params/Pt111.yaml` — add Pt(111) QM parameters with H placeholder

**What:** New file. Defines `Pt`, `Pt_surf` for layer-dependent BJ parameters
on a Pt(111) slab, plus an `H` entry with placeholder values for adsorbed H.

```yaml
system: Pt111
types:
  Pt:        # inner layers
    alpha_iso: 14.5533125
    C6: 4642.241861
    s: 1.40
  Pt_surf:   # top layer (assigned via type_label_overrides)
    alpha_iso: 42.27259375
    C6: 4642.241861
    s: 1.40
  H:         # placeholder — fill in before production runs
    alpha_iso: 0.0
    C6: 0.0
    s: 0.53
```

**Why:** Companion to the `config_Pt111.yaml` setup. The `H` placeholder
allows the build to complete so the rest of the pipeline can be tested
even before final dispersion parameters are available.

---

### `config_example.yaml` — document structure file format options

**What:** Replaced the bare `vasp_file: "CONTCAR"` line with a documented
block that lists the auto-detected formats, both supported config keys
(`structure_file` preferred, `vasp_file` legacy alias), and the optional
`format` override.

```diff
-input:
-  vasp_file: "CONTCAR"
+input:
+  # Slab structure file. ASE-supported formats are auto-detected from filename:
+  #   CONTCAR, POSCAR, *.vasp        → VASP
+  #   *.in, *.pwi, pw.*              → Quantum ESPRESSO pw.x input
+  #   *.out, *.pwo, *.pw.out         → Quantum ESPRESSO pw.x output
+  #
+  # Use either key — `structure_file` (preferred) or `vasp_file` (legacy alias):
+  vasp_file: "CONTCAR"
+  # structure_file: "scf.pw.in"      # equivalent
+  #
+  # Optional: explicit format override when auto-detection fails
+  # (e.g. unusual filename). Any ASE format string is accepted.
+  # format: "espresso-in"
```

**Why:** Surfaces the new format-detection behaviour and the
`structure_file` / `format` keys in the example so users discover them
without reading the source.
