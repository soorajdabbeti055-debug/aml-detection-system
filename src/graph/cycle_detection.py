import pandas as pd
import networkx as nx


GRAPH_FILE = "data/processed/aml_transaction_graph.graphml"
OUTPUT_FILE = "data/processed/account_cycle_features.parquet"


def detect_cycles(graph: nx.DiGraph) -> pd.DataFrame:
    """
    Detect accounts participating in directed transaction cycles.

    To keep computation practical on a large graph, we use
    strongly connected components rather than enumerating every
    individual cycle.
    """

    print("\nFinding strongly connected components...")

    components = nx.strongly_connected_components(graph)

    rows = []

    cycle_accounts = 0
    cycle_components = 0

    for component in components:

        component_size = len(component)

        # A component with >1 node contains a directed cycle.
        if component_size > 1:

            cycle_components += 1
            cycle_accounts += component_size

            for account in component:

                rows.append(
                    {
                        "account": account,
                        "cycle_component_size": component_size,
                        "is_cycle_candidate": 1,
                    }
                )

    result = pd.DataFrame(rows)

    if result.empty:
        result = pd.DataFrame(
            columns=[
                "account",
                "cycle_component_size",
                "is_cycle_candidate",
            ]
        )

    print(f"Cycle components found: {cycle_components:,}")
    print(f"Accounts participating in cycles: {cycle_accounts:,}")

    return result


def main():

    print("=" * 70)
    print("PHASE 4.4 - AML CYCLE DETECTION")
    print("=" * 70)

    print("\nLoading transaction graph...")

    graph = nx.read_graphml(GRAPH_FILE)

    print(f"Nodes: {graph.number_of_nodes():,}")
    print(f"Edges: {graph.number_of_edges():,}")

    result = detect_cycles(graph)

    print("\nCycle feature summary:")

    if not result.empty:

        print(
            result[
                [
                    "cycle_component_size",
                    "is_cycle_candidate",
                ]
            ]
            .describe()
            .round(2)
        )

        print("\nLargest cycle components:")

        print(
            result.sort_values(
                "cycle_component_size",
                ascending=False
            )
            .head(20)
            .to_string(index=False)
        )

    else:

        print("No directed cycles detected.")

    print("\nSaving cycle features...")

    result.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nSaved to:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("PHASE 4.4 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()