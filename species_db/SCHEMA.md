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
