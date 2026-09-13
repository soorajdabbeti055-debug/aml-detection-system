from pathlib import Path

import pandas as pd

from src.features.transaction_features import create_transaction_features
from src.features.account_features import create_account_features
from src.features.temporal_features import  create_temporal_features

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "SAML-D.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():

    print("=" * 70)
    print("PHASE 2 - FEATURE ENGINEERING")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows loaded: {len(df):,}")

    # =========================================================
    # Transaction features
    # =========================================================

    print("\nCreating transaction features...")

    df = create_transaction_features(df)

    # =========================================================
    # Account features
    # =========================================================

    print("Creating account behavior features...")

    df = create_account_features(df)

    # =========================================================
    # Temporal features
    # =========================================================

    print("Creating temporal features...")

    df = create_temporal_features(df)

    # =========================================================
    # Remove helper columns
    # =========================================================

    columns_to_drop = [
        "previous_sender_datetime"
    ]

    df = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    # =========================================================
    # Save CSV
    # =========================================================

    csv_output = (
        OUTPUT_DIR
        / "transactions_features.csv"
    )

    print("\nSaving CSV...")

    df.to_csv(
        csv_output,
        index=False
    )

    # =========================================================
    # Save Parquet
    # =========================================================

    parquet_output = (
        OUTPUT_DIR
        / "transactions_features.parquet"
    )

    print("Saving Parquet...")

    df.to_parquet(
        parquet_output,
        index=False
    )

    # =========================================================
    # Summary
    # =========================================================

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\nNew features:")

    original_columns = [
        "Time",
        "Date",
        "Sender_account",
        "Receiver_account",
        "Amount",
        "Payment_currency",
        "Received_currency",
        "Sender_bank_location",
        "Receiver_bank_location",
        "Payment_type",
        "Is_laundering",
        "Laundering_type",
    ]

    new_features = [
        col
        for col in df.columns
        if col not in original_columns
    ]

    for feature in new_features:
        print(f"  ✓ {feature}")

    print("\nFiles created:")

    print(csv_output)
    print(parquet_output)


if __name__ == "__main__":
    main()