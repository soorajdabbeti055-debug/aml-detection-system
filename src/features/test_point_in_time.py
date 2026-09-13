from pathlib import Path

import pandas as pd

from src.features.point_in_time_features import (
    create_point_in_time_account_features
)


BASE_DIR = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "transactions_features.parquet"
)


def main():

    print("=" * 70)
    print("TESTING POINT-IN-TIME ACCOUNT FEATURES")
    print("=" * 70)

    df = pd.read_parquet(
        FEATURE_FILE
    )

    # Use a small sample for initial testing
    df = (
        df.sort_values("datetime")
        .head(10000)
        .copy()
    )

    print(
        f"\nTesting rows: {len(df):,}"
    )

    result = create_point_in_time_account_features(
        df
    )

    columns = [
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
    ]

    print("\nGenerated features:")

    print(
        result[columns].head(10)
    )

    print("\nMissing values:")

    print(
        result[columns]
        .isna()
        .sum()
    )

    print("\n✓ Point-in-time feature test complete.")


if __name__ == "__main__":
    main()