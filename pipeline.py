import pandas as pd
import argparse
from pathlib import Path
import sys

from src.inference import InferencePipeline


def run_inference(data_path, output_path=None, model_path=None):
    project_root = Path(__file__).resolve().parent
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Input dataset not found at {data_path}")

    print("\n" + "="*75)
    print("INSURANCE DEFAULT PREDICTION - INFERENCE PIPELINE")
    print("="*75)
    print(f"Loading customer data from: {data_path}")

    df_raw = pd.read_csv(data_path)
    print(f"   Loaded {len(df_raw)} records")

    pipeline = InferencePipeline(model_path=model_path)
    print(f"Loaded trained model artifact: {pipeline.model_path}")

    print("\nRunning feature engineering and risk scoring...")
    predictions = pipeline.predict(df_raw)

    if output_path is None:
        output_path = project_root / "outputs" / "predictions.csv"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(exist_ok=True, parents=True)
    predictions.to_csv(output_path, index=False)

    print(f"\n[SUCCESS] Predictions successfully generated and saved to:")

    print(f"   {output_path}")

    print("\nRISK TIER SUMMARY:")
    tier_counts = predictions['risk_tier'].value_counts()
    for tier, count in tier_counts.items():
        pct = count / len(predictions) * 100
        print(f"   • {tier}: {count} customers ({pct:.1f}%)")

    total_cost = predictions['intervention_cost'].sum()
    print(f"\nTotal Intervention Cost: ${total_cost:,.0f}")
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insurance Premium Payment Default - Inference & Batch Scoring Pipeline")
    parser.add_argument("--data", type=str, default=None, help="Path to input dataset CSV for inference")
    parser.add_argument("--output", type=str, default=None, help="Path to save output predictions CSV")
    parser.add_argument("--model", type=str, default=None, help="Custom path to trained model .pkl file")
    parser.add_argument("--train", action="store_true", help="Trigger model re-training workflow (invokes train.py)")
    parser.add_argument("--fast", action="store_true", help="Fast training mode (used with --train)")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent

    if args.train:
        print("Redirecting to model training pipeline (train.py)...")
        from train import run_training
        data_path = Path(args.data) if args.data else project_root / "data" / "premium_dataset.csv"
        run_training(filepath=data_path, run_tuning=not args.fast, fast_mode=args.fast)
    else:
        data_path = Path(args.data) if args.data else project_root / "data" / "premium_dataset.csv"
        run_inference(data_path=data_path, output_path=args.output, model_path=args.model)
