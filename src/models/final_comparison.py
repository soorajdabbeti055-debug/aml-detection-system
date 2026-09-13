from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "reports"
    / "final_model_comparison.csv"
)


def main():

    print("=" * 70)
    print("PHASE 3.8 - FINAL MODEL COMPARISON")
    print("=" * 70)

    results = [
        {
            "model": "Baseline XGBoost",
            "feature_strategy": "Full-dataset account statistics",
            "test_pr_auc": 0.7321,
            "test_roc_auc": 0.9917,
            "test_precision": 0.6849,
            "test_recall": 0.7042,
            "test_f1": 0.6944,
            "test_false_positives": 46,
            "test_false_negatives": 42,
        },
        {
            "model": "Leakage-Safe PIT XGBoost",
            "feature_strategy": "Historical information only",
            "test_pr_auc": 0.0686,
            "test_roc_auc": 0.9332,
            "test_precision": 0.0544,
            "test_recall": 0.2887,
            "test_f1": 0.0916,
            "test_false_positives": 712,
            "test_false_negatives": 101,
        },
        {
            "model": "PIT XGBoost - Threshold 0.70",
            "feature_strategy": "Historical information + optimized threshold",
            "test_pr_auc": 0.0686,
            "test_roc_auc": 0.9332,
            "test_precision": 0.1027,
            "test_recall": 0.2113,
            "test_f1": 0.1382,
            "test_false_positives": 262,
            "test_false_negatives": 112,
        },
        {
            "model": "PIT XGBoost - Threshold 0.75",
            "feature_strategy": "Historical information + best validation F1 threshold",
            "test_pr_auc": 0.0686,
            "test_roc_auc": 0.9332,
            "test_precision": 0.1050,
            "test_recall": 0.1620,
            "test_f1": 0.1274,
            "test_false_positives": 196,
            "test_false_negatives": 119,
        },
    ]

    df = pd.DataFrame(results)

    print("\nFINAL TEST COMPARISON")
    print("=" * 70)

    print(
        df[
            [
                "model",
                "test_pr_auc",
                "test_roc_auc",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_false_positives",
                "test_false_negatives",
            ]
        ].to_string(index=False)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved comparison to:\n{OUTPUT_FILE}"
    )

    print("\n" + "=" * 70)
    print("PHASE 3.8 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()