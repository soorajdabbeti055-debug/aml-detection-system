from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


BASE_DIR = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "transactions_features.parquet"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "xgboost_aml_model.joblib"
)


FEATURES = [
    "Amount",
    "log_amount",
    "currency_changed",
    "cross_border",
    "same_country",
    "sender_transaction_count",
    "sender_total_amount",
    "sender_avg_amount",
    "sender_max_amount",
    "sender_unique_receivers",
    "receiver_transaction_count",
    "receiver_total_amount",
    "receiver_avg_amount",
    "receiver_max_amount",
    "receiver_unique_senders",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "day_of_week",
    "seconds_since_previous_sender_transaction",
]

TARGET = "Is_laundering"


def main():

    print("=" * 70)
    print("PHASE 3.3 - AML THRESHOLD OPTIMIZATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    print("\nLoading feature dataset...")

    df = pd.read_parquet(FEATURE_FILE)

    print(f"Rows loaded: {len(df):,}")

    # ---------------------------------------------------------
    # Prepare data
    # ---------------------------------------------------------

    df = df[
        FEATURES + [TARGET, "datetime"]
    ].copy()

    df[FEATURES] = (
        df[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
    )

    df = df.dropna(
        subset=FEATURES + [TARGET, "datetime"]
    )

    df = df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Time-based split
    # ---------------------------------------------------------

    n = len(df)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    validation_df = df.iloc[
        train_end:validation_end
    ]

    # ---------------------------------------------------------
    # Validation data
    # ---------------------------------------------------------

    X_validation = validation_df[FEATURES]
    y_validation = validation_df[TARGET]

    print("\nValidation rows:")
    print(f"{len(validation_df):,}")

    print("\nValidation target distribution:")
    print(y_validation.value_counts())

    # ---------------------------------------------------------
    # Load trained model
    # ---------------------------------------------------------

    print("\nLoading trained model...")

    model = joblib.load(MODEL_FILE)

    print("✓ Model loaded.")

    # ---------------------------------------------------------
    # Generate probabilities
    # ---------------------------------------------------------

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    # ---------------------------------------------------------
    # Test thresholds
    # ---------------------------------------------------------

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ]

    results = []

    print("\n" + "=" * 70)
    print("THRESHOLD RESULTS")
    print("=" * 70)

    print(
        f"{'Threshold':>10} "
        f"{'Precision':>10} "
        f"{'Recall':>10} "
        f"{'F1':>10} "
        f"{'FP':>8} "
        f"{'FN':>8} "
        f"{'Flagged':>10}"
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1]
        ).ravel()

        flagged = int(predictions.sum())

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "flagged_transactions": flagged,
        })

        print(
            f"{threshold:10.2f} "
            f"{precision:10.4f} "
            f"{recall:10.4f} "
            f"{f1:10.4f} "
            f"{fp:8d} "
            f"{fn:8d} "
            f"{flagged:10d}"
        )

    # ---------------------------------------------------------
    # Find best F1 threshold
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    best_row = results_df.loc[
        results_df["f1"].idxmax()
    ]

    print("\n" + "=" * 70)
    print("BEST THRESHOLD BY F1")
    print("=" * 70)

    print(
        f"Threshold:          {best_row['threshold']:.2f}"
    )
    print(
        f"Precision:          {best_row['precision']:.4f}"
    )
    print(
        f"Recall:             {best_row['recall']:.4f}"
    )
    print(
        f"F1:                 {best_row['f1']:.4f}"
    )
    print(
        f"False positives:    {int(best_row['false_positives'])}"
    )
    print(
        f"False negatives:    {int(best_row['false_negatives'])}"
    )
    print(
        f"Flagged transactions:{int(best_row['flagged_transactions'])}"
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    output_file = (
        BASE_DIR
        / "data"
        / "reports"
        / "threshold_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\nResults saved to:")
    print(output_file)

    print("\n" + "=" * 70)
    print("PHASE 3.3 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()