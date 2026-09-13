import pandas as pd


INPUT_FILE = "data/processed/account_graph_features.parquet"
OUTPUT_FILE = "data/processed/account_aml_patterns.parquet"


def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fan-in: account receives from multiple counterparties
    df["is_fan_in"] = (
        df["in_degree"] >= 5
    ).astype("int8")

    # Fan-out: account sends to multiple counterparties
    df["is_fan_out"] = (
        df["out_degree"] >= 5
    ).astype("int8")

    # Potential layering:
    # meaningful activity in both directions
    df["is_layering_candidate"] = (
        (df["in_degree"] >= 2)
        & (df["out_degree"] >= 2)
    ).astype("int8")

    # Highly connected account
    df["is_high_connectivity"] = (
        df["total_degree"] >= 10
    ).astype("int8")

    # Incoming/outgoing amount ratio
    df["amount_flow_ratio"] = (
        df["outgoing_amount"]
        / (df["incoming_amount"] + 1e-9)
    )

    # Accounts that receive substantially more than they send
    df["is_accumulation_candidate"] = (
        (df["incoming_amount"] > 0)
        & (df["outgoing_amount"] == 0)
    ).astype("int8")

    # Accounts that both receive and send significant amounts
    df["is_flow_through_candidate"] = (
        (df["incoming_amount"] > 0)
        & (df["outgoing_amount"] > 0)
    ).astype("int8")

    return df


def main():

    print("=" * 70)
    print("PHASE 4.3 - AML NETWORK PATTERN DETECTION")
    print("=" * 70)

    print("\nLoading graph features...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Accounts loaded: {len(df):,}")

    print("\nDetecting network patterns...")

    df = detect_patterns(df)

    pattern_columns = [
        "is_fan_in",
        "is_fan_out",
        "is_layering_candidate",
        "is_high_connectivity",
        "is_accumulation_candidate",
        "is_flow_through_candidate",
    ]

    print("\nPattern counts:")

    for column in pattern_columns:
        count = int(df[column].sum())
        percentage = count / len(df) * 100

        print(
            f"{column:30s}: "
            f"{count:8,} accounts "
            f"({percentage:.2f}%)"
        )

    print("\nTop accounts by connectivity:")

    top_accounts = (
        df.sort_values(
            "total_degree",
            ascending=False
        )
        .head(10)
        [
            [
                "account",
                "in_degree",
                "out_degree",
                "total_degree",
                "incoming_amount",
                "outgoing_amount",
            ]
        ]
    )

    print(top_accounts.to_string(index=False))

    print("\nSaving AML network patterns...")

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nSaved to:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("PHASE 4.3 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()