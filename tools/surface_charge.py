#!/usr/bin/env python3
"""
surface_charge.py -- Surface charge density calculator for CES2 exports.

Reads the xy box area from a CES2 export directory, build_summary.json, or a
LAMMPS data.file, then computes

    sigma [uC/cm^2] = charge_e * e_C / (area_A2 * surfaces * 1e-16) * 1e6

Usage
-----
  # Use build_summary.json in an export directory
  python tools/surface_charge.py export --charge-e -4

  # Read directly from a CES2 build summary
  python tools/surface_charge.py export/build_summary.json --charge-e -4

  # Fall back to LAMMPS data.file box bounds
  python tools/surface_charge.py export/data.file --charge-e -4

  # If a total charge is shared by two equivalent surfaces
  python tools/surface_charge.py export --charge-e -4 --surfaces 2

  # Machine-readable output
  python tools/surface_charge.py export --charge-e -4 --json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


ELEMENTARY_CHARGE_C = 1.602176634e-19
ANG2_TO_CM2 = 1.0e-16


@dataclass(frozen=True)
class SurfaceGeometry:
    source: Path
    source_type: str
    lx_a: float
    ly_a: float
    xlo_a: Optional[float] = None
    xhi_a: Optional[float] = None
    ylo_a: Optional[float] = None
    yhi_a: Optional[float] = None

    @property
    def area_a2(self) -> float:
        return self.lx_a * self.ly_a


def _as_float(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric: {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} is not finite: {value!r}")
    return out


def _positive(value: float, name: str) -> float:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value:g}")
    return value


def parse_build_summary(path: Path) -> SurfaceGeometry:
    """Read Lx and Ly from CES2 build_summary.json."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}") from exc

    box = payload.get("box")
    if not isinstance(box, dict):
        raise ValueError(f"{path} has no top-level 'box' object")

    if "Lx" in box and "Ly" in box:
        lx = _positive(_as_float(box["Lx"], "box.Lx"), "box.Lx")
        ly = _positive(_as_float(box["Ly"], "box.Ly"), "box.Ly")
        return SurfaceGeometry(path, "build_summary", lx, ly)

    packmol = payload.get("packmol")
    pbc_box = packmol.get("pbc_box") if isinstance(packmol, dict) else None
    if isinstance(pbc_box, list) and len(pbc_box) >= 6:
        xlo = _as_float(pbc_box[0], "packmol.pbc_box[0]")
        ylo = _as_float(pbc_box[1], "packmol.pbc_box[1]")
        xhi = _as_float(pbc_box[3], "packmol.pbc_box[3]")
        yhi = _as_float(pbc_box[4], "packmol.pbc_box[4]")
        lx = _positive(xhi - xlo, "packmol.pbc_box Lx")
        ly = _positive(yhi - ylo, "packmol.pbc_box Ly")
        return SurfaceGeometry(path, "build_summary", lx, ly, xlo, xhi, ylo, yhi)

    raise ValueError(f"{path} does not contain box.Lx/box.Ly or packmol.pbc_box")


_BOX_LINE_RE = re.compile(
    r"^\s*"
    r"(?P<lo>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
    r"\s+"
    r"(?P<hi>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
    r"\s+"
    r"(?P<axis>[xyz])lo\s+"
    r"(?P=axis)hi\b",
    re.IGNORECASE,
)


def parse_lammps_data(path: Path) -> SurfaceGeometry:
    """Read xy box bounds from a LAMMPS data file."""
    bounds: dict[str, tuple[float, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = _BOX_LINE_RE.match(line)
            if not match:
                if bounds and line.strip().lower() == "masses":
                    break
                continue
            axis = match.group("axis").lower()
            lo = _as_float(match.group("lo"), f"{axis}lo")
            hi = _as_float(match.group("hi"), f"{axis}hi")
            bounds[axis] = (lo, hi)
            if "x" in bounds and "y" in bounds:
                break

    missing = [axis for axis in ("x", "y") if axis not in bounds]
    if missing:
        raise ValueError(f"{path} is missing LAMMPS {'/'.join(missing)} box bounds")

    xlo, xhi = bounds["x"]
    ylo, yhi = bounds["y"]
    lx = _positive(xhi - xlo, "xhi - xlo")
    ly = _positive(yhi - ylo, "yhi - ylo")
    return SurfaceGeometry(path, "lammps_data", lx, ly, xlo, xhi, ylo, yhi)


def resolve_geometry(input_path: Path) -> SurfaceGeometry:
    """Resolve an export directory, build_summary.json, or data.file."""
    path = input_path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"input not found: {path}")

    if path.is_dir():
        summary = path / "build_summary.json"
        data_file = path / "data.file"
        errors: list[str] = []
        if summary.exists():
            try:
                return parse_build_summary(summary)
            except ValueError as exc:
                errors.append(str(exc))
        if data_file.exists():
            try:
                return parse_lammps_data(data_file)
            except ValueError as exc:
                errors.append(str(exc))
        detail = "; ".join(errors) if errors else "no build_summary.json or data.file found"
        raise ValueError(f"could not read CES2 xy area from {path}: {detail}")

    if path.suffix.lower() == ".json" or path.name == "build_summary.json":
        return parse_build_summary(path)
    return parse_lammps_data(path)


def compute_surface_charge(charge_e: float, area_a2: float, surfaces: int) -> dict[str, float]:
    """Compute charge and surface charge density in common units."""
    if not math.isfinite(charge_e):
        raise ValueError(f"charge_e is not finite: {charge_e!r}")
    if surfaces < 1:
        raise ValueError(f"surfaces must be >= 1, got {surfaces}")

    effective_area_a2 = area_a2 * surfaces
    area_cm2 = effective_area_a2 * ANG2_TO_CM2
    charge_c = charge_e * ELEMENTARY_CHARGE_C
    sigma_uc_cm2 = charge_c / area_cm2 * 1.0e6
    sigma_c_m2 = sigma_uc_cm2 * 1.0e-2
    return {
        "charge_C": charge_c,
        "effective_area_A2": effective_area_a2,
        "area_cm2": area_cm2,
        "sigma_uC_cm2": sigma_uc_cm2,
        "sigma_C_m2": sigma_c_m2,
    }


def format_result(
    geometry: SurfaceGeometry,
    charge_e: float,
    surfaces: int,
    result: dict[str, float],
    precision: int,
) -> str:
    """Create a compact human-readable report."""
    p = max(1, precision)
    lines = [
        "Surface charge density",
        f"  source       : {geometry.source} ({geometry.source_type})",
        f"  Lx, Ly       : {geometry.lx_a:.{p}g} A, {geometry.ly_a:.{p}g} A",
        f"  xy area      : {geometry.area_a2:.{p}g} A^2",
    ]
    if surfaces != 1:
        lines.append(
            f"  surfaces     : {surfaces} "
            f"(effective area {result['effective_area_A2']:.{p}g} A^2)"
        )
    lines.extend(
        [
            f"  charge       : {charge_e:.{p}g} e = {result['charge_C']:.{p}e} C",
            f"  sigma        : {result['sigma_uC_cm2']:.{p}g} uC/cm^2",
            f"  sigma        : {result['sigma_C_m2']:.{p}g} C/m^2",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute surface charge density from CES2 export geometry "
            "(build_summary.json or LAMMPS data.file)."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="CES2 export directory, build_summary.json, or LAMMPS data.file",
    )
    parser.add_argument(
        "-q",
        "--charge-e",
        "--ne",
        "--n-electrons",
        dest="charge_e",
        type=float,
        required=True,
        help="Net charge in elementary-charge units, e.g. -4 for -4e",
    )
    parser.add_argument(
        "--surfaces",
        type=int,
        default=1,
        help="Number of equivalent surfaces sharing the provided charge (default: 1)",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Significant digits for text output (default: 6)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print machine-readable JSON",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        geometry = resolve_geometry(args.input)
        result = compute_surface_charge(args.charge_e, geometry.area_a2, args.surfaces)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json_output:
        payload = {
            "source": str(geometry.source),
            "source_type": geometry.source_type,
            "Lx_A": geometry.lx_a,
            "Ly_A": geometry.ly_a,
            "area_A2": geometry.area_a2,
            "surfaces": args.surfaces,
            "charge_e": args.charge_e,
            **result,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(format_result(geometry, args.charge_e, args.surfaces, result, args.precision))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
