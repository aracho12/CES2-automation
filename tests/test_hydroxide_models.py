from pathlib import Path

import pytest

from builder.lammps_input_writer import _build_lj_defaults
from builder.species import (
    load_species_db,
    validate_hydroxide_species_models,
)
from builder.forcefield_db import load_forcefield_db, resolve_hydroxide_set_name


ROOT = Path(__file__).resolve().parents[1]
SPECIES_DIR = ROOT / "species_db"
FF_PATH = SPECIES_DIR / "forcefields" / "lj_forcefield.yaml"


@pytest.mark.parametrize(
    ("model", "expected_length", "expected_oxygen_lj", "expected_charges"),
    [
        ("bonthuis2016", 1.00, (0.01195, 3.81), (-1.0, 0.0)),
        ("tip4p2005", 0.98, (0.05996, 3.40), (-1.0727, 0.2227)),
    ],
)
def test_hydroxide_model_species_owns_geometry_charge_and_matching_lj(
    model, expected_length, expected_oxygen_lj, expected_charges
):
    species = load_species_db(SPECIES_DIR)
    ff_data = load_forcefield_db(FF_PATH)
    set_name = resolve_hydroxide_set_name(
        ff_data, "TIP4P", model, source=FF_PATH
    )
    lj = _build_lj_defaults("TIP4P", FF_PATH, model)

    assert set_name == f"hydroxide_{model}"
    assert lj["O_oh"] == pytest.approx(expected_oxygen_lj)
    expected_species = {
        "bonthuis2016": (
            "OH_minus",
            "OH_minus_frac",
        ),
        "tip4p2005": ("OH_minus_tip4p2005",),
    }[model]
    for sid in expected_species:
        sp = species[sid]
        assert sp.atoms[1].xyz[2] - sp.atoms[0].xyz[2] == pytest.approx(expected_length)
        assert sp.bond_coeffs[2][1] == pytest.approx(expected_length)
        assert tuple(atom.charge for atom in sp.atoms) == pytest.approx(expected_charges)
        assert sp.net_charge == pytest.approx(sum(expected_charges))

    counts = {sid: 1 for sid in expected_species}
    assert validate_hydroxide_species_models(
        counts, species, set_name
    ) == pytest.approx(expected_length)


def test_tip4p2005_has_no_frac_species_file():
    assert not (SPECIES_DIR / "OH_minus_tip4p2005_frac.yaml").exists()


def test_no_redundant_model_named_bonthuis_species_files():
    assert not (SPECIES_DIR / "OH_minus_bonthuis2016.yaml").exists()
    assert not (SPECIES_DIR / "OH_minus_bonthuis2016_frac.yaml").exists()


def test_active_hydroxide_species_must_match_selected_model():
    species = load_species_db(SPECIES_DIR)
    validate_hydroxide_species_models(
        {"OH_minus_tip4p2005": 1}, species, "hydroxide_tip4p2005"
    )
    with pytest.raises(ValueError, match="different model"):
        validate_hydroxide_species_models(
            {"OH_minus": 1}, species, "hydroxide_tip4p2005"
        )


def test_hydroxide_auto_follows_water_model_mapping():
    ff_data = load_forcefield_db(FF_PATH)
    set_name = resolve_hydroxide_set_name(
        ff_data, "TIP3P", "auto", source=FF_PATH
    )
    assert set_name == "hydroxide_bonthuis2016"


def test_unknown_hydroxide_model_is_rejected():
    ff_data = load_forcefield_db(FF_PATH)
    with pytest.raises(KeyError, match="Unknown hydroxide model/set"):
        resolve_hydroxide_set_name(
            ff_data, "TIP4P", "missing", source=FF_PATH
        )
