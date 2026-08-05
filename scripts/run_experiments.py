from __future__ import annotations

import argparse
from pathlib import Path

import torch

from twostage_transfer.scenarios import SCENARIOS, get_scenario
from twostage_transfer.experiment import run_many, save_results


def parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run two-stage transfer learning simulations.")
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    parser.add_argument("--dataset-sizes", default=None, help="Comma-separated sizes. Defaults to the scenario settings.")
    parser.add_argument("--seeds", default="42,43,44,45,46", help="Comma-separated seeds.")
    parser.add_argument("--output-dir", default="outputs", type=Path)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Override torch device.")
    args = parser.parse_args()

    config = get_scenario(args.scenario)
    dataset_sizes = parse_ints(args.dataset_sizes) if args.dataset_sizes else list(config.dataset_sizes)
    seeds = parse_ints(args.seeds)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    run_output = run_many(config, dataset_sizes=dataset_sizes, seeds=seeds, device=device)
    json_path, csv_path = save_results(config, run_output, dataset_sizes, seeds, args.output_dir)
    print(f"\nSaved full results: {json_path}")
    print(f"Saved summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
