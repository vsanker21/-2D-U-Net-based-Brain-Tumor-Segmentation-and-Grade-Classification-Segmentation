#!/usr/bin/env python3
"""Train random-mask classification ablation (negative control for mask-guided 3D CNN)."""
from __future__ import annotations

import argparse
import json

from run_project1_methodological_upgrade import train_final_random_mask_ablation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    out = train_final_random_mask_ablation(epochs=args.epochs)
    print(json.dumps({k: out[k] for k in out if k not in ("protocol", "history")}, indent=2))


if __name__ == "__main__":
    main()
