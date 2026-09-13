from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)


# ============================================================
# PATHS
# ============================================================

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
    / "pit_threshold_results.csv"
)


# ============================================================
# FEATURES
# ============================================================

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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 3.6 - PIT THRESHOLD OPTIMIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading PIT features...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Rows loaded: {len(df):,}")

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    # --------------------------------------------------------
    # Same chronological split as training
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

    print("\nValidation rows:", len(X_validation))
    print("Test rows:", len(X_test))

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading PIT model...")

    model = joblib.load(MODEL_FILE)

    print("✓ Model loaded.")

    # --------------------------------------------------------
    # Validation probabilities
    # --------------------------------------------------------

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    # --------------------------------------------------------
    # Threshold search
    # --------------------------------------------------------

    thresholds = np.arange(
        0.05,
        0.96,
        0.05
    )

    results = []

    print("\n" + "=" * 70)
    print("VALIDATION THRESHOLD RESULTS")
    print("=" * 70)

    print(
        f"{'Threshold':<10}"
        f"{'Precision':<12}"
        f"{'Recall':<10}"
        f"{'F1':<10}"
        f"{'FP':<10}"
        f"{'FN':<10}"
        f"{'Flagged':<10}"
    )

    for threshold in thresholds:

        predictions = (
            validation_probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        cm = confusion_matrix(
            y_validation,
            predictions,
        )

        tn, fp, fn, tp = cm.ravel()

        flagged = int(
            predictions.sum()
        )

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fp": fp,
            "fn": fn,
            "flagged": flagged,
        })

        print(
            f"{threshold:<10.2f}"
            f"{precision:<12.4f}"
            f"{recall:<10.4f}"
            f"{f1:<10.4f}"
            f"{fp:<10}"
            f"{fn:<10}"
            f"{flagged:<10}"
        )

    results_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Select best validation F1
    # --------------------------------------------------------

    best_row = results_df.loc[
        results_df["f1"].idxmax()
    ]

    best_threshold = float(
        best_row["threshold"]
    )

    print("\n" + "=" * 70)
    print("BEST VALIDATION THRESHOLD")
    print("=" * 70)

    print(
        f"\nThreshold: {best_threshold:.2f}"
    )

    print(
        f"Precision: {best_row['precision']:.4f}"
    )

    print(
        f"Recall:    {best_row['recall']:.4f}"
    )

    print(
        f"F1:        {best_row['f1']:.4f}"
    )

    print(
        f"False positives: {int(best_row['fp'])}"
    )

    print(
        f"False negatives: {int(best_row['fn'])}"
    )

    print(
        f"Flagged:         {int(best_row['flagged'])}"
    )

    # --------------------------------------------------------
    # Save threshold results
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved threshold results to:\n"
        f"{OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Apply selected threshold to TEST
    # --------------------------------------------------------

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    test_precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    test_recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    test_f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    test_cm = confusion_matrix(
        y_test,
        test_predictions,
    )

    test_pr_auc = average_precision_score(
        y_test,
        test_probabilities,
    )

    tn, fp, fn, tp = test_cm.ravel()

    # --------------------------------------------------------
    # Final test results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST RESULTS USING VALIDATION-SELECTED THRESHOLD")
    print("=" * 70)

    print(
        f"\nThreshold used: {best_threshold:.2f}"
    )

    print(
        f"Precision: {test_precision:.4f}"
    )

    print(
        f"Recall:    {test_recall:.4f}"
    )

    print(
        f"F1:        {test_f1:.4f}"
    )

    print(
        f"PR-AUC:    {test_pr_auc:.4f}"
    )

    print("\nConfusion Matrix:")
    print(test_cm)

    print(
        f"\nTrue Negatives:  {tn}"
    )

    print(
        f"False Positives: {fp}"
    )

    print(
        f"False Negatives: {fn}"
    )

    print(
        f"True Positives:  {tp}"
    )

    print("\n" + "=" * 70)
    print("PHASE 3.6 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()