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

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "transactions_pit_features.parquet"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "xgboost_aml_pit_model.joblib"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "reports"
    / "pit_operational_thresholds.csv"
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


def evaluate_threshold(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
    ).ravel()

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "alerts": int(predictions.sum()),
    }


def main():

    print("=" * 70)
    print("PHASE 3.7 - OPERATIONAL THRESHOLD ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading PIT dataset...")

    df = pd.read_parquet(INPUT_FILE)

    df = (
        df.sort_values("datetime")
        .reset_index(drop=True)
    )

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    # --------------------------------------------------------
    # Same chronological split
    # --------------------------------------------------------

    n = len(df)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    X_validation = X.iloc[
        train_end:validation_end
    ]

    y_validation = y.iloc[
        train_end:validation_end
    ]

    X_test = X.iloc[
        validation_end:
    ]

    y_test = y.iloc[
        validation_end:
    ]

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = joblib.load(MODEL_FILE)

    print("✓ PIT model loaded.")

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------

    thresholds = [
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

    validation_results = []
    test_results = []

    # --------------------------------------------------------
    # Evaluate thresholds
    # --------------------------------------------------------

    for threshold in thresholds:

        validation_results.append(
            evaluate_threshold(
                y_validation,
                validation_probabilities,
                threshold,
            )
        )

        test_results.append(
            evaluate_threshold(
                y_test,
                test_probabilities,
                threshold,
            )
        )

    validation_df = pd.DataFrame(
        validation_results
    )

    test_df = pd.DataFrame(
        test_results
    )

    # --------------------------------------------------------
    # Display validation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION THRESHOLD ANALYSIS")
    print("=" * 70)

    print(
        validation_df[
            [
                "threshold",
                "precision",
                "recall",
                "f1",
                "false_positives",
                "false_negatives",
                "alerts",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Display test
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST THRESHOLD ANALYSIS")
    print("=" * 70)

    print(
        test_df[
            [
                "threshold",
                "precision",
                "recall",
                "f1",
                "false_positives",
                "false_negatives",
                "alerts",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Best F1 from validation
    # --------------------------------------------------------

    best_index = validation_df["f1"].idxmax()

    best_threshold = float(
        validation_df.loc[
            best_index,
            "threshold",
        ]
    )

    # --------------------------------------------------------
    # Recommended high-recall threshold
    #
    # Select the highest threshold that still provides
    # at least 20% recall on validation.
    # --------------------------------------------------------

    recall_candidates = validation_df[
        validation_df["recall"] >= 0.20
    ]

    if not recall_candidates.empty:

        high_recall_threshold = float(
            recall_candidates["threshold"].max()
        )

    else:

        high_recall_threshold = None

    # --------------------------------------------------------
    # Print recommendations
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("THRESHOLD RECOMMENDATIONS")
    print("=" * 70)

    print(
        f"\nBest validation F1 threshold: "
        f"{best_threshold:.2f}"
    )

    if high_recall_threshold is not None:

        print(
            f"Highest threshold with >=20% "
            f"validation recall: "
            f"{high_recall_threshold:.2f}"
        )

    # --------------------------------------------------------
    # Save combined results
    # --------------------------------------------------------

    validation_df["dataset"] = "validation"
    test_df["dataset"] = "test"

    combined = pd.concat(
        [
            validation_df,
            test_df,
        ],
        ignore_index=True,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved results to:\n{OUTPUT_FILE}"
    )

    print("\n" + "=" * 70)
    print("PHASE 3.7 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()