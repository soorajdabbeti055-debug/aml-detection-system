import pandas as pd
import networkx as nx


INPUT_FILE = "data/processed/transactions_pit_features.parquet"
GRAPH_FILE = "data/processed/aml_transaction_graph.graphml"


def build_transaction_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed transaction graph.

    Nodes:
        Bank accounts

    Edges:
        Sender -> Receiver

    Edge attributes:
        transaction_count
        total_amount
        average_amount
    """

    graph = nx.DiGraph()

    grouped = (
        df.groupby(
            ["Sender_account", "Receiver_account"],
            as_index=False
        )
        .agg(
            transaction_count=("Amount", "count"),
            total_amount=("Amount", "sum"),
            average_amount=("Amount", "mean"),
        )
    )

    for row in grouped.itertuples(index=False):

        sender = row.Sender_account
        receiver = row.Receiver_account

        graph.add_edge(
            sender,
            receiver,
            transaction_count=int(row.transaction_count),
            total_amount=float(row.total_amount),
            average_amount=float(row.average_amount),
        )

    return graph


def main():

    print("=" * 70)
    print("PHASE 4.1 - AML TRANSACTION GRAPH")
    print("=" * 70)

    print("\nLoading transaction data...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Rows loaded: {len(df):,}")

    print("\nBuilding transaction graph...")

    graph = build_transaction_graph(df)

    print(f"Nodes: {graph.number_of_nodes():,}")
    print(f"Edges: {graph.number_of_edges():,}")

    print("\nSaving graph...")

    nx.write_graphml(graph, GRAPH_FILE)

    print(f"Saved to:")
    print(GRAPH_FILE)

    print("\n" + "=" * 70)
    print("PHASE 4.1 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()