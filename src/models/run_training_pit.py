from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
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


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Amount",
    "log_amount",
    "currency_changed",
    "cross_border",
    "same_country",

    # Leakage-safe sender features
    "sender_transaction_count",
    "sender_total_amount",
    "sender_avg_amount",
    "sender_max_amount",
    "sender_unique_receivers",

    # Leakage-safe receiver features
    "receiver_transaction_count",
    "receiver_total_amount",
    "receiver_avg_amount",
    "receiver_max_amount",
    "receiver_unique_senders",

    # Temporal features
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
    print("PHASE 3.5 - LEAKAGE-SAFE XGBOOST TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading leakage-safe feature dataset...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Rows loaded: {len(df):,}")

    # --------------------------------------------------------
    # Validate features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    print(f"✓ {len(FEATURES)} features available.")

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    if "datetime" not in df.columns:
        raise ValueError(
            "datetime column is required for time-based split."
        )

    df = df.sort_values("datetime").reset_index(drop=True)

    # --------------------------------------------------------
    # Check missing values
    # --------------------------------------------------------

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    missing = X.isna().sum()

    if missing.sum() > 0:
        print("\nMissing values detected:")
        print(missing[missing > 0])
        raise ValueError("Missing feature values detected.")

    print("✓ No missing feature values.")

    # --------------------------------------------------------
    # Check infinite values
    # --------------------------------------------------------

    numeric_values = X.select_dtypes(
        include=[np.number]
    )

    infinite_count = np.isinf(
        numeric_values.to_numpy()
    ).sum()

    if infinite_count > 0:
        raise ValueError(
            f"Found {infinite_count:,} infinite values."
        )

    print("✓ No infinite values.")

    # --------------------------------------------------------
    # Time-based split
    #
    # 70% training
    # 15% validation
    # 15% testing
    # --------------------------------------------------------

    n = len(df)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_validation = X.iloc[
        train_end:validation_end
    ]
    y_validation = y.iloc[
        train_end:validation_end
    ]

    X_test = X.iloc[validation_end:]
    y_test = y.iloc[validation_end:]

    print("\n" + "=" * 70)
    print("TIME-BASED SPLIT")
    print("=" * 70)

    print(f"Training:    {len(X_train):,}")
    print(f"Validation:  {len(X_validation):,}")
    print(f"Testing:     {len(X_test):,}")

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TARGET DISTRIBUTION")
    print("=" * 70)

    print("\nTraining:")
    print(y_train.value_counts().sort_index())

    print("\nValidation:")
    print(y_validation.value_counts().sort_index())

    print("\nTesting:")
    print(y_test.value_counts().sort_index())

    # --------------------------------------------------------
    # Class imbalance
    # --------------------------------------------------------

    normal_count = (y_train == 0).sum()
    laundering_count = (y_train == 1).sum()

    scale_pos_weight = (
        normal_count / laundering_count
    )

    print("\n" + "=" * 70)
    print("CLASS IMBALANCE")
    print("=" * 70)

    print(
        f"Normal transactions:     {normal_count:,}"
    )

    print(
        f"Laundering transactions:  {laundering_count:,}"
    )

    print(
        f"scale_pos_weight:         {scale_pos_weight:.2f}"
    )

    # --------------------------------------------------------
    # Train XGBoost
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING XGBOOST")
    print("=" * 70)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_validation, y_validation)
        ],
        verbose=False,
    )

    print("✓ XGBoost training complete.")

    # --------------------------------------------------------
    # Validation evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    validation_predictions = (
        validation_probabilities >= 0.50
    ).astype(int)

    print(
        classification_report(
            y_validation,
            validation_predictions,
            digits=4,
            zero_division=0,
        )
    )

    validation_cm = confusion_matrix(
        y_validation,
        validation_predictions,
    )

    validation_pr_auc = average_precision_score(
        y_validation,
        validation_probabilities,
    )

    validation_roc_auc = roc_auc_score(
        y_validation,
        validation_probabilities,
    )

    print("Confusion Matrix:")
    print(validation_cm)

    print(
        f"\nValidation PR-AUC:  "
        f"{validation_pr_auc:.4f}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{validation_roc_auc:.4f}"
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_predictions = (
        test_probabilities >= 0.50
    ).astype(int)

    print(
        classification_report(
            y_test,
            test_predictions,
            digits=4,
            zero_division=0,
        )
    )

    test_cm = confusion_matrix(
        y_test,
        test_predictions,
    )

    test_pr_auc = average_precision_score(
        y_test,
        test_probabilities,
    )

    test_roc_auc = roc_auc_score(
        y_test,
        test_probabilities,
    )

    print("Confusion Matrix:")
    print(test_cm)

    print(
        f"\nTest PR-AUC:  "
        f"{test_pr_auc:.4f}"
    )

    print(
        f"Test ROC-AUC: "
        f"{test_roc_auc:.4f}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print("\n" + "=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        f"\nSaved to:\n{MODEL_FILE}"
    )

    print("\n" + "=" * 70)
    print("PHASE 3.5 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()