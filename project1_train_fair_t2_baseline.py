#!/usr/bin/env python3
"""Train fair T2-only 3D CNN holdout baseline (128×128×96, canonical 80/20 split)."""
from __future__ import annotations

import argparse
import json

from run_project1_methodological_upgrade import train_final_t2_baseline_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="Match mask-guided final training epochs")
    args = parser.parse_args()
    out = train_final_t2_baseline_model(epochs=args.epochs)
    print(json.dumps({k: out[k] for k in out if k != "protocol"}, indent=2))


if __name__ == "__main__":
    main()
