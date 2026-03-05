#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from ces2_builder import run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    summary = run(Path(args.config))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
