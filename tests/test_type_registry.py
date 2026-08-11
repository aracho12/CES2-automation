import unittest

from builder.builder import make_type_registry
from builder.species import AtomDef, Species


def _species(species_id, atoms):
    return Species(
        id=species_id,
        name=species_id,
        net_charge=0.0,
        atoms=[
            AtomDef(element=element, type_label=label, charge=0.0, xyz=(0.0, 0.0, 0.0))
            for element, label in atoms
        ],
        bonds=[],
        angles=[],
        bond_coeffs={},
        angle_coeffs={},
    )


class TypeRegistryTests(unittest.TestCase):
    def setUp(self):
        self.species_db = {
            "water": _species("water", [("O", "Ow"), ("H", "Hw"), ("H", "Hw")]),
            "salt": _species("salt", [("Na", "Na")]),
        }
        self.species_order = ["water", "salt"]
        self.electrode_labels = ["Pt_surf", "Pt", "Pt_surf"]

    def test_electrode_types_last_by_default(self):
        by_label, by_id = make_type_registry(
            self.species_order, self.species_db, self.electrode_labels
        )

        self.assertEqual(by_label, {"Hw": 1, "Ow": 2, "Na": 3, "Pt": 4, "Pt_surf": 5})
        self.assertEqual(by_id[1], "Hw")

    def test_electrode_types_first(self):
        by_label, by_id = make_type_registry(
            self.species_order,
            self.species_db,
            self.electrode_labels,
            electrode_type_position="first",
        )

        self.assertEqual(by_label, {"Pt": 1, "Pt_surf": 2, "Hw": 3, "Ow": 4, "Na": 5})
        self.assertEqual(by_id[5], "Na")

    def test_position_is_case_insensitive(self):
        by_label, _ = make_type_registry(
            self.species_order,
            self.species_db,
            self.electrode_labels,
            electrode_type_position=" FIRST ",
        )

        self.assertEqual(by_label["Pt"], 1)

    def test_invalid_position_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be 'first' or 'last'"):
            make_type_registry(
                self.species_order,
                self.species_db,
                self.electrode_labels,
                electrode_type_position="middle",
            )


if __name__ == "__main__":
    unittest.main()
