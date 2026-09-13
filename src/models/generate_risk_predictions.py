import joblib
import numpy as np
import pandas as pd


INPUT_FILE = (
    "data/processed/transactions_graph_features.parquet"
)

MODEL_FILE = (
    "models/xgboost_aml_pit_model.joblib"
)

OUTPUT_FILE = (
    "data/processed/transactions_risk_predictions.parquet"
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


def main():

    print("=" * 70)
    print("PHASE 5.1 - TRANSACTION RISK PREDICTIONS")
    print("=" * 70)

    print("\nLoading transaction data...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Transactions loaded: {len(df):,}")

    print("\nChecking ML features...")

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing ML features: {missing_features}"
        )

    X = df[FEATURES].copy()

    if X.isna().any().any():
        raise ValueError(
            "Missing values detected in ML features."
        )

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "Infinite values detected in ML features."
        )

    print(
        f"✓ {len(FEATURES)} ML features available."
    )

    print("\nLoading leakage-safe XGBoost model...")

    model = joblib.load(MODEL_FILE)

    print("✓ Model loaded.")

    print("\nGenerating laundering probabilities...")

    probabilities = model.predict_proba(X)[:, 1]

    df["ml_risk_probability"] = probabilities

    print("\nRisk probability summary:")

    print(
        df["ml_risk_probability"]
        .describe()
        .round(6)
    )

    print("\nProbability distribution:")

    print(
        pd.cut(
            df["ml_risk_probability"],
            bins=[
                -np.inf,
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                np.inf,
            ],
        )
        .value_counts()
        .sort_index()
    )

    print("\nValidating predictions...")

    if len(df) != 1_048_575:
        raise ValueError(
            "Transaction count changed."
        )

    if df["ml_risk_probability"].isna().any():
        raise ValueError(
            "Missing ML risk probabilities."
        )

    if not np.isfinite(
        df["ml_risk_probability"].to_numpy()
    ).all():
        raise ValueError(
            "Invalid ML risk probabilities."
        )

    print("✓ All transactions have ML risk probabilities.")
    print("✓ No transactions lost.")

    print("\nSaving predictions...")

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nSaved to:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("PHASE 5.1 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()