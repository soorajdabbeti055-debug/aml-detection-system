import pandas as pd
import networkx as nx


GRAPH_FILE = "data/processed/aml_transaction_graph.graphml"
OUTPUT_FILE = "data/processed/account_graph_features.parquet"


def calculate_graph_features(graph: nx.DiGraph) -> pd.DataFrame:
    """Calculate AML-relevant network features for every account."""

    rows = []

    for node in graph.nodes():

        in_edges = graph.in_edges(node, data=True)
        out_edges = graph.out_edges(node, data=True)

        in_degree = graph.in_degree(node)
        out_degree = graph.out_degree(node)

        incoming_amount = sum(
            float(data.get("total_amount", 0))
            for _, _, data in in_edges
        )

        outgoing_amount = sum(
            float(data.get("total_amount", 0))
            for _, _, data in out_edges
        )

        unique_counterparties = len(
            set(graph.predecessors(node))
            | set(graph.successors(node))
        )

        total_degree = in_degree + out_degree

        fan_in_ratio = (
            in_degree / total_degree
            if total_degree > 0
            else 0
        )

        fan_out_ratio = (
            out_degree / total_degree
            if total_degree > 0
            else 0
        )

        rows.append(
            {
                "account": node,
                "in_degree": in_degree,
                "out_degree": out_degree,
                "total_degree": total_degree,
                "incoming_amount": incoming_amount,
                "outgoing_amount": outgoing_amount,
                "unique_counterparties": unique_counterparties,
                "fan_in_ratio": fan_in_ratio,
                "fan_out_ratio": fan_out_ratio,
            }
        )

    return pd.DataFrame(rows)


def main():

    print("=" * 70)
    print("PHASE 4.2 - GRAPH-BASED AML FEATURES")
    print("=" * 70)

    print("\nLoading transaction graph...")

    graph = nx.read_graphml(GRAPH_FILE)

    print(f"Nodes loaded: {graph.number_of_nodes():,}")
    print(f"Edges loaded: {graph.number_of_edges():,}")

    print("\nCalculating account-level graph features...")

    features = calculate_graph_features(graph)

    print(f"Accounts processed: {len(features):,}")
    print(f"Features created: {len(features.columns) - 1}")

    print("\nFeature summary:")
    print(features.describe().round(2))

    print("\nSaving features...")

    features.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nSaved to:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("PHASE 4.2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()