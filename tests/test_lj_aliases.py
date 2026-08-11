import unittest

from builder.lammps_input_writer import _apply_lj_type_aliases
from builder.species import AtomDef, Species


def _species(species_id, type_label, lj_type_label=None):
    return Species(
        id=species_id,
        name=species_id,
        net_charge=0.0,
        atoms=[AtomDef(
            element="Na",
            type_label=type_label,
            charge=0.0,
            xyz=(0.0, 0.0, 0.0),
            lj_type_label=lj_type_label,
        )],
        bonds=[],
        angles=[],
        bond_coeffs={},
        angle_coeffs={},
    )


class LjTypeAliasTests(unittest.TestCase):
    def test_distinct_type_inherits_source_parameters(self):
        lj_db = {"Na": (0.3526418, 2.16)}
        species_db = {"frac": _species("frac", "Na_frac", "Na")}

        _apply_lj_type_aliases(lj_db, [("frac", 1)], species_db)

        self.assertEqual(lj_db["Na_frac"], lj_db["Na"])

    def test_unknown_source_is_rejected(self):
        lj_db = {"Na": (0.3526418, 2.16)}
        species_db = {"frac": _species("frac", "Na_frac", "Missing")}

        with self.assertRaisesRegex(KeyError, "unknown lj_type_label 'Missing'"):
            _apply_lj_type_aliases(lj_db, [("frac", 1)], species_db)

    def test_conflicting_aliases_are_rejected(self):
        lj_db = {"Na": (0.3526418, 2.16), "K": (0.4, 3.3)}
        species_db = {
            "a": _species("a", "ion_frac", "Na"),
            "b": _species("b", "ion_frac", "K"),
        }

        with self.assertRaisesRegex(ValueError, "Conflicting lj_type_label aliases"):
            _apply_lj_type_aliases(lj_db, [("a", 1), ("b", 1)], species_db)


if __name__ == "__main__":
    unittest.main()
