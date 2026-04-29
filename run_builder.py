#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from builder import run

def _find_config_yaml() -> Path | None:
    """Auto-detect a config*.yaml file in cwd if exactly one exists."""
    candidates = sorted(Path(".").glob("config*.yaml"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(str(c) for c in candidates)
        raise SystemExit(f"Multiple config*.yaml files found ({names}). Please specify one with --config.")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="Path to config YAML (default: auto-detect config*.yaml in cwd)")
    ap.add_argument("--input", "-i", default="CONTCAR",
                    help="Path to CONTCAR (overrides input.vasp_file in config)")
    args = ap.parse_args()

    config_path = Path(args.config) if args.config else _find_config_yaml()
    if config_path is None:
        ap.error("No --config given and could not auto-detect a single config*.yaml in cwd")

    summary = run(config_path, vasp_file=args.input)
    print(json.dumps(summary, indent=2))

    export_dir = Path(summary["export_dir"])
    print(f"\nBuild complete. Output files in: {export_dir}")
    for f in sorted(export_dir.iterdir()):
        print(f"  {f.name}")

if __name__ == "__main__":
    main()
