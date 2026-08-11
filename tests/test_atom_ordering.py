import tempfile
import unittest
from pathlib import Path

from builder.builder import build_mm_connectivity
from builder.ces2_script_writer import generate_ces2_scripts
from builder.md_workflow import _qm_atom_range_from_summary
from builder.species import AtomDef, Species


def _bonded_species():
    return Species(
        id="OH",
        name="hydroxide",
        net_charge=-1.0,
        atoms=[
            AtomDef(element="O", type_label="O_OH", charge=-1.0, xyz=(0.0, 0.0, 0.0)),
            AtomDef(element="H", type_label="H_OH", charge=0.0, xyz=(0.0, 0.0, 1.0)),
        ],
        bonds=[(1, 1, 2)],
        angles=[],
        bond_coeffs={1: (450.0, 1.0)},
        angle_coeffs={},
    )


class AtomOrderingTests(unittest.TestCase):
    def test_mm_connectivity_can_follow_leading_qm_atoms(self):
        species_db = {"OH": _bonded_species()}
        bonds, angles, _, _ = build_mm_connectivity(
            [("OH", 0, 2), ("OH", 2, 4)], species_db, atom_id_offset=6
        )

        self.assertEqual(bonds, [(1, 7, 8), (1, 9, 10)])
        self.assertEqual(angles, [])

    def test_md_range_prefers_explicit_summary_range(self):
        summary = {"n_mm": 100, "n_qm": 6, "qm_atom_id_range": [1, 6]}
        self.assertEqual(_qm_atom_range_from_summary(summary), (1, 6))

    def test_md_range_supports_legacy_mm_first_summary(self):
        self.assertEqual(
            _qm_atom_range_from_summary({"n_mm": 100, "n_qm": 6}),
            (101, 106),
        )

    def test_ces2_script_uses_explicit_qm_id_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_ces2_scripts(
                export_dir=Path(tmp),
                cfg={},
                n_mm=4,
                n_qm=2,
                electrode_atom_position="first",
            )
            script = paths["qmmm_sh"].read_text(encoding="utf-8")

        self.assertIn('QMATOMLO="1"', script)
        self.assertIn('QMATOMHI="2"', script)
        self.assertIn("$1>=lo && $1<=hi", script)
        self.assertNotIn("$1>nsolvatom", script)


if __name__ == "__main__":
    unittest.main()
