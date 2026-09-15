"""Batch scoring CLI pipeline for insurance default prediction.

Usage:
    python pipeline.py --data data/premium_dataset.csv
    python pipeline.py --data path/to/new_customers.csv --output outputs/scored_customers.csv
"""

import argparse
from pathlib import Path
import pandas as pd
from src.inference import InferencePipeline


def run_batch_scoring(data_path: str, output_path: str = None, model_path: str = None):
    project_root = Path(__file__).resolve().parent
    input_file = Path(data_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input customer dataset not found at: {input_file}")

    print("\n" + "=" * 80)
    print("INSURANCE PREMIUM DEFAULT RISK PROFILER - BATCH INFERENCE")
    print("=" * 80)
    print(f"Loading customer records from: {input_file}")

    df_raw = pd.read_csv(input_file)
    print(f"  Total records ingested: {len(df_raw):,}")

    pipeline = InferencePipeline(model_path=model_path)
    print(f"  Model loaded: {pipeline.model_path.name}")
    print(f"  Optimal decision threshold: {pipeline.optimal_threshold:.3f}")

    print("\nExecuting feature transformations and risk scoring...")
    predictions = pipeline.predict(df_raw)

    if output_path is None:
        output_file = project_root / "outputs" / "predictions.csv"
    else:
        output_file = Path(output_path)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_file, index=False)
    print(f"\n[SUCCESS] Scored predictions persisted to:\n  -> {output_file.resolve()}")

    # Summary Statistics
    print("\n" + "-" * 40)
    print("OPERATIONAL RISK TIER BREAKDOWN")
    print("-" * 40)
    tier_summary = predictions.groupby('risk_tier').agg(
        Count=('customer_id', 'count'),
        Avg_Default_Risk=('default_probability', lambda x: x.mean() * 100),
        Total_Cost=('intervention_cost', 'sum')
    ).reindex(['High Risk', 'Medium Risk', 'Low-Medium Risk', 'Low Risk']).fillna(0)

    tier_summary['Pct_Total'] = tier_summary['Count'] / len(predictions) * 100

    for tier, row in tier_summary.iterrows():
        print(f"  • {tier:15s}: {int(row['Count']):6,d} ({row['Pct_Total']:5.1f}%) | "
              f"Avg Default Risk: {row['Avg_Default_Risk']:4.1f}% | Budget: ${int(row['Total_Cost']):,d}")

    total_cost = predictions['intervention_cost'].sum()
    print("-" * 40)
    print(f"Total Projected Intervention Budget: ${total_cost:,.0f}")
    print("=" * 80 + "\n")

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Insurance Premium Default Risk Profiler - Batch Scoring CLI"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/premium_dataset.csv",
        help="Path to input customer CSV file (default: data/premium_dataset.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output predictions CSV file (default: outputs/predictions.csv)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional custom path to champion model .pkl",
    )

    args = parser.parse_args()
    run_batch_scoring(data_path=args.data, output_path=args.output, model_path=args.model)
