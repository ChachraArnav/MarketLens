"""Command line: python -m marketlens [download|build|evaluate|crosscheck|all]"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from . import analytics, config, crosscheck, data, evaluate, features
from .splits import chronological_split


def cmd_download(_args) -> None:
    print("Downloading prices from Yahoo Finance...")
    data.download()


def cmd_build(_args) -> pd.DataFrame:
    prices = data.load_raw()
    problems = data.validate(prices)
    if problems:
        raise SystemExit("Data validation failed: " + "; ".join(problems))
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_csv(config.PRICES_FILE, index=False)

    dataset = features.build_dataset(prices)
    dataset.to_csv(config.DATASET_FILE, index=False)

    analytics.summary_table(prices).to_csv(config.RESULTS_DIR / "etf_statistics.csv", index=False)
    risk = dataset.groupby("Ticker")[config.TARGETS].mean().mul(100).round(2)
    risk.to_csv(config.RESULTS_DIR / "risk_definition_comparison.csv")

    print(
        f"Prices: {len(prices):,} rows, {prices['Ticker'].nunique()} ETFs, "
        f"{prices['Date'].min():%Y-%m-%d} to {prices['Date'].max():%Y-%m-%d}"
    )
    print(f"Dataset: {len(dataset):,} labelled observations")
    for target in config.TARGETS:
        print(f"  {target}: {dataset[target].mean():.2%} positive")
    return dataset


def cmd_evaluate(_args) -> None:
    dataset = pd.read_csv(config.DATASET_FILE, parse_dates=["Date"])
    splits = chronological_split(dataset)
    print(
        f"Train {len(splits.train):,} | validation {len(splits.validation):,} "
        f"| test {len(splits.test):,} observations"
    )
    summary = evaluate.run(splits)
    comparison = pd.read_csv(config.RESULTS_DIR / "final_model_comparison.csv")
    print("\n2025 validation (threshold 0.5):")
    print(comparison.round(4).to_string(index=False))
    print("\n2026 untouched test (model and threshold chosen on 2025):")
    print(json.dumps(summary["results"], indent=2))


def cmd_crosscheck(_args) -> None:
    result = crosscheck.run()
    if not result:
        print("No TMX files in data/external/tmx; skipping the cross-check.")
        return
    print(f"\nCross-check against {result['source']}:")
    for w in result["windows"]:
        print(
            f"  {w['ticker']} {w['start']} to {w['end']}: {w['tmx_days']} TMX days, "
            f"{len(w['dates_only_in_tmx']) + len(w['dates_only_in_yahoo'])} date mismatches, "
            f"max close gap {w['max_close_difference_cents']} cents, "
            f"{w['share_within_one_cent']:.0%} within 1 cent"
        )
        for step in w["distribution_steps"]:
            print(
                f"    ex-date {step['Ex_Date']}: implied ${step['Implied_Cash_Per_Unit']:.3f}, "
                f"announced {step['Announced_Cash_Per_Unit']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(prog="marketlens", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="fetch fresh prices from Yahoo Finance")
    sub.add_parser("build", help="validate prices and build features + labels")
    sub.add_parser("evaluate", help="compare models on 2025, test once on 2026")
    sub.add_parser("crosscheck", help="compare Yahoo prices with TMX Money downloads")
    sub.add_parser("all", help="build + evaluate + crosscheck using the saved prices")
    args = parser.parse_args()
    if args.command == "download":
        cmd_download(args)
    elif args.command == "build":
        cmd_build(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "crosscheck":
        cmd_crosscheck(args)
    else:
        cmd_build(args)
        cmd_evaluate(args)
        cmd_crosscheck(args)


if __name__ == "__main__":
    main()
