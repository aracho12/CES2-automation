from pathlib import Path

import pytest

from builder.lammps_input_writer import _build_lj_defaults


FF_PATH = (
    Path(__file__).resolve().parents[1]
    / "species_db"
    / "forcefields"
    / "lj_forcefield.yaml"
)


@pytest.mark.parametrize(
    ("water_model", "label", "expected"),
    [
        ("TIP4P", "Na", (0.1684375, 2.18448)),
        ("TIP3P", "Na", (0.0874393, 2.43928)),
        ("TIP3P", "Cl", (0.0355910, 4.47766)),
        ("SPCE", "Na", (0.3526418, 2.15954)),
        ("SPCE", "Rb", (0.4451036, 3.09498)),
        ("SPCE", "Cs", (0.0898565, 3.60101)),
        ("TIP3PEW", "Na", (0.0874393, 2.43928)),
    ],
)
def test_representative_joung_cheatham_values(water_model, label, expected):
    db = _build_lj_defaults(water_model, FF_PATH)
    assert db[label] == pytest.approx(expected)
