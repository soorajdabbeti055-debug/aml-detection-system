import pandas as pd


GRAPH_FILE = "data/processed/account_graph_features.parquet"
PATTERN_FILE = "data/processed/account_aml_patterns.parquet"
CYCLE_FILE = "data/processed/account_cycle_features.parquet"

OUTPUT_FILE = "data/processed/account_aml_profile.parquet"


def main():

    print("=" * 70)
    print("PHASE 4.5 - COMBINE AML NETWORK FEATURES")
    print("=" * 70)

    print("\nLoading graph features...")

    graph = pd.read_parquet(GRAPH_FILE)

    print(f"Graph accounts: {len(graph):,}")

    print("\nLoading AML pattern features...")

    patterns = pd.read_parquet(PATTERN_FILE)

    print(f"Pattern accounts: {len(patterns):,}")

    print("\nLoading cycle features...")

    cycles = pd.read_parquet(CYCLE_FILE)

    print(f"Cycle accounts: {len(cycles):,}")

    # Remove duplicate base columns before merging.
    pattern_columns = [
        column
        for column in patterns.columns
        if column != "account"
        and column not in graph.columns
    ]

    patterns = patterns[
        ["account"] + pattern_columns
    ]

    cycle_columns = [
        column
        for column in cycles.columns
        if column != "account"
        and column not in graph.columns
        and column not in patterns.columns
    ]

    cycles = cycles[
        ["account"] + cycle_columns
    ]

    print("\nMerging features...")

    result = graph.merge(
        patterns,
        on="account",
        how="left"
    )

    result = result.merge(
        cycles,
        on="account",
        how="left"
    )

    # Accounts that do not participate in a cycle
    # receive zero for cycle features.
    cycle_fill_columns = [
        "cycle_component_size",
        "is_cycle_candidate",
    ]

    for column in cycle_fill_columns:

        if column in result.columns:

            result[column] = (
                result[column]
                .fillna(0)
            )

    # Ensure pattern flags are integer values.
    pattern_flags = [
        "is_fan_in",
        "is_fan_out",
        "is_layering_candidate",
        "is_high_connectivity",
        "is_accumulation_candidate",
        "is_flow_through_candidate",
        "is_cycle_candidate",
    ]

    for column in pattern_flags:

        if column in result.columns:

            result[column] = (
                result[column]
                .fillna(0)
                .astype("int8")
            )

    print("\nFinal account profile:")

    print(
        f"Accounts: {len(result):,}"
    )

    print(
        f"Features: {len(result.columns) - 1}"
    )

    print("\nMissing values:")

    missing = result.isna().sum()

    print(
        missing[
            missing > 0
        ]
    )

    print("\nNetwork risk indicators:")

    indicators = [
        "is_fan_in",
        "is_fan_out",
        "is_layering_candidate",
        "is_high_connectivity",
        "is_accumulation_candidate",
        "is_flow_through_candidate",
        "is_cycle_candidate",
    ]

    for column in indicators:

        if column in result.columns:

            count = int(
                result[column].sum()
            )

            percentage = (
                count / len(result) * 100
            )

            print(
                f"{column:30s}: "
                f"{count:8,} "
                f"({percentage:.2f}%)"
            )

    print("\nSaving combined account profile...")

    result.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )

    print("\n" + "=" * 70)
    print("PHASE 4.5 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()