import pandas as pd


TRANSACTION_FILE = (
    "data/processed/transactions_pit_features.parquet"
)

ACCOUNT_FILE = (
    "data/processed/account_aml_profile.parquet"
)

OUTPUT_FILE = (
    "data/processed/transactions_graph_features.parquet"
)


GRAPH_FEATURES = [
    "in_degree",
    "out_degree",
    "total_degree",
    "incoming_amount",
    "outgoing_amount",
    "unique_counterparties",
    "fan_in_ratio",
    "fan_out_ratio",
    "is_fan_in",
    "is_fan_out",
    "is_layering_candidate",
    "is_high_connectivity",
    "is_accumulation_candidate",
    "is_flow_through_candidate",
    "cycle_component_size",
    "is_cycle_candidate",
]


def main():

    print("=" * 70)
    print("PHASE 4.6 - ATTACH GRAPH FEATURES TO TRANSACTIONS")
    print("=" * 70)

    print("\nLoading transactions...")

    transactions = pd.read_parquet(
        TRANSACTION_FILE
    )

    print(
        f"Transactions loaded: "
        f"{len(transactions):,}"
    )

    print("\nLoading account AML profile...")

    accounts = pd.read_parquet(
        ACCOUNT_FILE
        )

    print(
        f"Accounts loaded: "
        f"{len(accounts):,}"
    )

# Ensure account identifiers use the same datatype
# in both datasets.
    transactions["Sender_account"] = (
        transactions["Sender_account"]
        .astype(str)
        )

    transactions["Receiver_account"] = (
        transactions["Receiver_account"]
        .astype(str)
        )
    accounts["account"] = (
        accounts["account"]
        .astype(str)
        )

    # ---------------------------------------------------------
    # Sender features
    # ---------------------------------------------------------

    sender_features = accounts[
        ["account"] + GRAPH_FEATURES
    ].copy()

    sender_features = sender_features.rename(
        columns={
            column: f"sender_{column}"
            for column in GRAPH_FEATURES
        }
    )

    sender_features = sender_features.rename(
        columns={
            "account": "Sender_account"
        }
    )

    print("\nAttaching sender network features...")

    transactions = transactions.merge(
        sender_features,
        on="Sender_account",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------
    # Receiver features
    # ---------------------------------------------------------

    receiver_features = accounts[
        ["account"] + GRAPH_FEATURES
    ].copy()

    receiver_features = receiver_features.rename(
        columns={
            column: f"receiver_{column}"
            for column in GRAPH_FEATURES
        }
    )

    receiver_features = receiver_features.rename(
        columns={
            "account": "Receiver_account"
        }
    )

    print("Attaching receiver network features...")

    transactions = transactions.merge(
        receiver_features,
        on="Receiver_account",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    print("\nValidating result...")

    print(
        f"Original transactions: "
        f"{len(transactions):,}"
    )

    expected_rows = 1_048_575

    if len(transactions) != expected_rows:
        raise ValueError(
            "Transaction count changed after graph merge."
        )

    graph_columns = [
        column
        for column in transactions.columns
        if column.startswith("sender_")
        or column.startswith("receiver_")
    ]

    missing = transactions[
        graph_columns
    ].isna().sum()

    missing = missing[
        missing > 0
    ]

    if len(missing) > 0:

        print("\nMissing graph features:")
        print(missing)

        raise ValueError(
            "Missing graph features detected."
        )

    print(
        f"Graph features attached: "
        f"{len(graph_columns)}"
    )

    print("✓ No missing graph features.")
    print("✓ No transactions lost.")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    print("\nSaving transaction graph features...")

    transactions.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )

    print("\n" + "=" * 70)
    print("PHASE 4.6 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()