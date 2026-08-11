import unittest

from builder.composition import resolve_packmol_order


class ResolvePackmolOrderTests(unittest.TestCase):
    def test_default_without_extras_preserves_water_then_sorted_order(self):
        counts = {"water": 100, "Na": 2, "Cl": 2}

        self.assertEqual(
            resolve_packmol_order(counts, "water"),
            ["water", "Cl", "Na"],
        )

    def test_extras_follow_declaration_order_before_remaining_species(self):
        counts = {"water": 100, "Na": 2, "OH_frac": 3, "OH": 4, "Cl": 2}
        extras = [
            {"species_id": "Na", "count": 2},
            {"species_id": "OH_frac", "count": 3},
            {"species_id": "OH", "count": 4},
        ]

        self.assertEqual(
            resolve_packmol_order(counts, "water", extras=extras),
            ["water", "Na", "OH_frac", "OH", "Cl"],
        )

    def test_duplicate_extras_are_emitted_once(self):
        counts = {"water": 100, "Na": 5}
        extras = [
            {"species_id": "Na", "count": 2},
            {"species_id": "Na", "count": 3},
        ]

        self.assertEqual(
            resolve_packmol_order(counts, "water", extras=extras),
            ["water", "Na"],
        )

    def test_inactive_extra_is_skipped(self):
        counts = {"water": 100, "Na": 0, "Cl": 2}
        extras = [{"species_id": "Na", "count": 0}]

        self.assertEqual(
            resolve_packmol_order(counts, "water", extras=extras),
            ["water", "Cl"],
        )

    def test_explicit_order_remains_authoritative(self):
        counts = {"water": 100, "Na": 2, "Cl": 2}
        extras = [{"species_id": "Na", "count": 2}]

        self.assertEqual(
            resolve_packmol_order(
                counts,
                "water",
                extras=extras,
                explicit_order=["water", "Cl", "Na"],
            ),
            ["water", "Cl", "Na"],
        )


if __name__ == "__main__":
    unittest.main()
