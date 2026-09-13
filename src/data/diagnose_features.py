from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

RAW_FILE = BASE_DIR / "data" / "raw" / "SAML-D.csv"
FEATURE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "transactions_features.parquet"
)


def main():

    print("=" * 70)
    print("PHASE 3.1 - DATA LOSS DIAGNOSTICS")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load original data
    # ---------------------------------------------------------

    print("\nLoading original dataset...")

    raw = pd.read_csv(RAW_FILE)

    print(
        f"Original rows: {len(raw):,}"
    )

    # ---------------------------------------------------------
    # Check original missing values
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("1. ORIGINAL MISSING VALUES")
    print("=" * 70)

    missing = raw.isna().sum()

    print(missing[missing > 0])

    if missing.sum() == 0:
        print("✓ No missing values in original dataset.")

    # ---------------------------------------------------------
    # Load feature dataset
    # ---------------------------------------------------------

    print("\nLoading feature dataset...")

    features = pd.read_parquet(FEATURE_FILE)

    print(
        f"Feature rows: {len(features):,}"
    )

    print(
        f"Rows apparently lost: "
        f"{len(raw) - len(features):,}"
    )

    # ---------------------------------------------------------
    # Check datetime
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("2. DATETIME CHECK")
    print("=" * 70)

    datetime_value = pd.to_datetime(
    raw["Date"].astype(str).str.strip()
    + " "
    + raw["Time"].astype(str).str.strip(),
    format="%d-%m-%Y %H:%M:%S",
    errors="coerce"
)

    invalid_datetime = datetime_value.isna().sum()

    print(
        f"Invalid datetime rows: "
        f"{invalid_datetime:,}"
    )

    print(
        f"Valid datetime rows: "
        f"{datetime_value.notna().sum():,}"
    )

    # ---------------------------------------------------------
    # Check individual Date
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("3. DATE CHECK")
    print("=" * 70)

    parsed_date = pd.to_datetime(
    raw["Date"].astype(str).str.strip(),
    format="%d-%m-%Y",
    errors="coerce"
)

    print(
        f"Invalid dates: "
        f"{parsed_date.isna().sum():,}"
    )

    # ---------------------------------------------------------
    # Check individual Time
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("4. TIME CHECK")
    print("=" * 70)

    parsed_time = pd.to_datetime(
        raw["Time"],
        errors="coerce"
    )

    print(
        f"Invalid times: "
        f"{parsed_time.isna().sum():,}"
    )

    # ---------------------------------------------------------
    # Feature missing values
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("5. FEATURE DATASET MISSING VALUES")
    print("=" * 70)

    feature_missing = features.isna().sum()

    print(
        feature_missing[
            feature_missing > 0
        ]
    )

    # ---------------------------------------------------------
    # Feature infinities
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("6. INFINITE VALUES")
    print("=" * 70)

    numeric_columns = features.select_dtypes(
        include="number"
    ).columns

    infinity_counts = {}

    for column in numeric_columns:

        count = (
            ~pd.Series(
                pd.isna(features[column])
            )
            &
            ~pd.Series(
                np.isfinite(features[column])
            )
        ).sum()

        if count > 0:
            infinity_counts[column] = count

    print(infinity_counts)

    # ---------------------------------------------------------
    # Target values
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("7. TARGET CHECK")
    print("=" * 70)

    print(
        raw["Is_laundering"]
        .value_counts(dropna=False)
    )

    # ---------------------------------------------------------
    # Date range
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("8. DATE RANGE")
    print("=" * 70)

    print(
        "Earliest:",
        datetime_value.min()
    )

    print(
        "Latest:",
        datetime_value.max()
    )

    # ---------------------------------------------------------
    # Result
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()