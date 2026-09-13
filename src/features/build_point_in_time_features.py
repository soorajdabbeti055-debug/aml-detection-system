from pathlib import Path

import pandas as pd

from src.features.point_in_time_features import (
    create_point_in_time_account_features
)


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "transactions_features.parquet"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "transactions_pit_features.parquet"
)


def main():

    print("=" * 70)
    print("PHASE 3.4 - LEAKAGE-SAFE FEATURE ENGINEERING")
    print("=" * 70)

    print("\nLoading existing features...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Rows loaded: {len(df):,}")

    # ---------------------------------------------------------
    # Validate datetime
    # ---------------------------------------------------------

    print("\nChecking datetime...")

    invalid_datetime = df["datetime"].isna().sum()

    print(
        f"Invalid datetime rows: "
        f"{invalid_datetime:,}"
    )

    if invalid_datetime > 0:
        raise ValueError(
            "Invalid datetime values detected."
        )

    # ---------------------------------------------------------
    # Create point-in-time features
    # ---------------------------------------------------------

    print("\nCreating leakage-safe account features...")

    df = create_point_in_time_account_features(df)

    print("✓ Point-in-time features created.")

    # ---------------------------------------------------------
    # Check feature values
    # ---------------------------------------------------------

    pit_features = [
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

    print("\nChecking missing values...")

    missing = df[pit_features].isna().sum()

    print(missing[missing > 0])

    if missing.sum() == 0:
        print("✓ No missing point-in-time features.")

    # ---------------------------------------------------------
    # Check row count
    # ---------------------------------------------------------

    print("\nChecking row count...")

    print(
        f"Input rows:  {1_048_575:,}"
    )

    print(
        f"Output rows: {len(df):,}"
    )

    if len(df) != 1_048_575:
        raise ValueError(
            "Row count changed during feature engineering."
        )

    print("✓ No rows lost.")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    print("\nSaving leakage-safe feature dataset...")

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("PHASE 3.4 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()