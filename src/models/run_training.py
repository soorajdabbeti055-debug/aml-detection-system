from pathlib import Path

import numpy as np
import pandas as pd

from src.models.classifier import train_xgboost
from src.models.evaluation import evaluate_model


BASE_DIR = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "transactions_features.parquet"
)

TARGET = "Is_laundering"


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


def main():

    print("=" * 70)
    print("PHASE 3.2 - AML XGBOOST TRAINING")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    print("\nLoading feature dataset...")

    df = pd.read_parquet(FEATURE_FILE)

    print(f"Rows loaded: {len(df):,}")

    # ---------------------------------------------------------
    # Check features
    # ---------------------------------------------------------

    print("\nChecking features...")

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        print("ERROR: Missing features:")
        for feature in missing_features:
            print(f"  - {feature}")
        return

    print(f"✓ {len(FEATURES)} features available.")

    # ---------------------------------------------------------
    # Check target
    # ---------------------------------------------------------

    if TARGET not in df.columns:
        print(f"ERROR: Target column '{TARGET}' not found.")
        return

    # ---------------------------------------------------------
    # Check missing values BEFORE cleaning
    # ---------------------------------------------------------

    print("\nChecking missing values...")

    feature_missing = df[FEATURES].isna().sum()

    missing_features_with_values = (
        feature_missing[feature_missing > 0]
        .sort_values(ascending=False)
    )

    if len(missing_features_with_values) > 0:
        print(missing_features_with_values)
    else:
        print("✓ No missing feature values.")

    # ---------------------------------------------------------
    # Check infinite values
    # ---------------------------------------------------------

    print("\nChecking infinite values...")

    numeric_features = df[FEATURES].select_dtypes(
        include="number"
    ).columns

    infinity_counts = {}

    for column in numeric_features:
        count = np.isinf(df[column]).sum()

        if count > 0:
            infinity_counts[column] = int(count)

    if infinity_counts:
        print(infinity_counts)
    else:
        print("✓ No infinite values.")

    # ---------------------------------------------------------
    # Prepare modeling data
    # ---------------------------------------------------------

    model_df = df[
        FEATURES + [TARGET, "datetime"]
    ].copy()

    # Replace infinities only
    model_df[FEATURES] = (
        model_df[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
    )

    rows_before = len(model_df)

    model_df = model_df.dropna(
        subset=FEATURES + [TARGET, "datetime"]
    )

    rows_after = len(model_df)

    print("\nCleaning:")
    print(f"Rows before cleaning: {rows_before:,}")
    print(f"Rows after cleaning:  {rows_after:,}")
    print(f"Rows removed:         {rows_before - rows_after:,}")

    if rows_after == 0:
        print("ERROR: No rows remain after cleaning.")
        return

    # ---------------------------------------------------------
    # Sort chronologically
    # ---------------------------------------------------------

    model_df = model_df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Time-based split
    # ---------------------------------------------------------

    n = len(model_df)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train_df = model_df.iloc[:train_end]
    validation_df = model_df.iloc[
        train_end:validation_end
    ]
    test_df = model_df.iloc[
        validation_end:
    ]

    print("\n" + "=" * 70)
    print("TIME-BASED SPLIT")
    print("=" * 70)

    print(f"Training:    {len(train_df):,}")
    print(f"Validation:  {len(validation_df):,}")
    print(f"Testing:     {len(test_df):,}")

    # ---------------------------------------------------------
    # Split X / y
    # ---------------------------------------------------------

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_validation = validation_df[FEATURES]
    y_validation = validation_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    # ---------------------------------------------------------
    # Target distributions
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("TARGET DISTRIBUTION")
    print("=" * 70)

    print("\nTraining:")
    print(y_train.value_counts())

    print("\nValidation:")
    print(y_validation.value_counts())

    print("\nTesting:")
    print(y_test.value_counts())

    # ---------------------------------------------------------
    # Train model
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING XGBOOST")
    print("=" * 70)

    model = train_xgboost(
        X_train,
        y_train
    )

    # ---------------------------------------------------------
    # Validation evaluation
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    validation_results = evaluate_model(
        model,
        X_validation,
        y_validation
    )

    # ---------------------------------------------------------
    # Test evaluation
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    test_results = evaluate_model(
        model,
        X_test,
        y_test
    )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 3.2 COMPLETE")
    print("=" * 70)

    print("\nTest metrics:")
    print(
        f"PR-AUC:  {test_results['pr_auc']:.4f}"
    )
    print(
        f"ROC-AUC: {test_results['roc_auc']:.4f}"
    )

    print("\n✓ Model training complete.")


if __name__ == "__main__":
    main()