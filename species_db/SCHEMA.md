# species_db YAML schema (v0.2)

Required
- id: unique species id (string)
- net_charge: net molecular charge (float)
- atoms: list of atoms in a *fixed order*
  - element: atomic symbol (string)
  - type_label: label used for LAMMPS atom types (string)
  - charge: per-atom charge (float) or null
  - xyz: [x,y,z] in Angstrom (used by PACKMOL as rigid template)

Optional
- charge_scheme:
  - explicit_or_zero (default): null charges -> 0.0
  - uniform_from_net_charge: distribute remaining (net_charge - sum(explicit)) uniformly over null entries

- connectivity:
  - bonds: list of [bond_type, i, j] in 1-based indices within the molecule
  - angles: list of [angle_type, i, j, k] 1-based
- coeffs:
  - bond_coeffs: { bond_type: [k, r0] }
  - angle_coeffs: { angle_type: [k, theta0] }

Notes
- For monoatomic ions, omit connectivity.
- For polyatomic ions, fill connectivity even if you postpone full force-field.
- If you don't want Bond/Angle Coeffs blocks in output, just keep coeffs empty; the writer omits them when absent.

---

# LJ force-field database — `forcefields/lj_forcefield.yaml`

Single source of truth for Lennard-Jones `(epsilon, sigma)` of every MM
`type_label`.  Lives in the `forcefields/` subdirectory so it is **not** picked
up by the species loader (which globs `species_db/*.yaml`).  Loaded by
`builder/lammps_input_writer.py::_build_lj_defaults(water_model, path)`.

Resolution for a given `ces2.water_model` (later sets win on label collisions,
but labels are normally disjoint):

```
always_sets  +  water_set  +  ion_set  +  hydroxide_set  +  ces2.lj_params (config override)
```

## Top-level keys

- `schema_version`: int.
- `units`: informational (`epsilon: kcal/mol`, `sigma: Angstrom`). Values in the
  file MUST already be in these units.
- `mixing_rule`: informational only — the writer always applies Lorentz-Berthelot
  (`eps_ij = sqrt(eps_i*eps_j)`, `sigma_ij = (sigma_i+sigma_j)/2`).
- `fallback`: `{epsilon, sigma}` used for any `type_label` absent from every
  applied set (writer also flags it `[!NO DEFAULT]` in the output).
- `always_sets`: list of set names applied for every water model (e.g. `common`
  generic atoms C/N/P/S used by polyatomic ions).
- `water_models`: map `WATER_MODEL → { water_set, ion_set, hydroxide_set, note? }`.
  Keys are upper-case (`TIP4P`, `TIP3P`, `SPCE`, `TIP3PEW`). A `water_model` with
  no entry falls back to the `TIP3P` entry (with a warning).
- `sets`: map `set_name → { source?, note?, params: { type_label: {epsilon, sigma} } }`.

## Example

```yaml
schema_version: 1
units: { epsilon: kcal/mol, sigma: Angstrom }
mixing_rule: lorentz_berthelot
fallback: { epsilon: 0.0, sigma: 1.0 }
always_sets: [common]

water_models:
  TIP4P:
    water_set:     water_tip4p_ew
    ion_set:       jc2008_tip4p_ew
    hydroxide_set: hydroxide_tip4p2005

sets:
  jc2008_tip4p_ew:
    source: "Joung & Cheatham, JPCB 112, 9020 (2008), TIP4P-Ew column"
    params:
      Li: { epsilon: 0.10398840, sigma: 1.43969 }
      # ...
  hydroxide_tip4p2005:
    source: "de Lucas et al., JPCL 15, 9411 (2024)"
    params:
      O_oh: { epsilon: 0.05996, sigma: 3.4000 }
      H_oh: { epsilon: 0.04396, sigma: 1.4430 }
```

## How to extend

- **New ion / generic atom**: add its `type_label` to the relevant `ion_set`
  (per water-model column) or to `common`.
- **Swap a force-field family per category**: point a `water_models[...]` entry's
  `*_set` at a different set name (e.g. a `hydroxide_bonthuis2016` set), or add a
  new set and reference it. Sets are reusable across water models.
- **One-off per-build tweak**: prefer `ces2.lj_params` in config — it overrides
  this DB by `type_label` without editing the shared file.

Notes
- `type_label`s here MUST match the labels used in the species YAML files
  (e.g. `Ow`/`Hw`, `O_oh`/`H_oh`, ion element labels). Water sets use
  model-specific labels where the species files do (`Ow_spce`, `Ow_tip3pew`).
- Hydroxide is a first-class set (previously the writer borrowed the active
  water-oxygen LJ for `O_oh`, which over-bound cations such as Li⁺).
