import pandas as pd
import plotly.graph_objects as go


def get_network_edges(
    transactions_df,
    sender_account,
    receiver_account,
    max_neighbors=8,
):
    sender_account = str(sender_account)
    receiver_account = str(receiver_account)

    df = transactions_df.copy()

    df["Sender_account"] = df["Sender_account"].astype(str)
    df["Receiver_account"] = df["Receiver_account"].astype(str)

    # Transactions directly involving the selected sender
    sender_edges = df[
        (df["Sender_account"] == sender_account)
        | (df["Receiver_account"] == sender_account)
    ].copy()

    # Transactions directly involving the selected receiver
    receiver_edges = df[
        (df["Sender_account"] == receiver_account)
        | (df["Receiver_account"] == receiver_account)
    ].copy()

    edges = pd.concat(
        [sender_edges, receiver_edges],
        ignore_index=True,
    ).drop_duplicates()

    # Keep strongest connections by transaction amount
    edges = (
        edges.sort_values(
            "Amount",
            ascending=False
        )
        .head(max_neighbors * 2)
    )

    return edges


def create_network_graph(
    edges,
    sender_account,
    receiver_account,
):
    sender_account = str(sender_account)
    receiver_account = str(receiver_account)

    nodes = set()

    for _, row in edges.iterrows():
        nodes.add(str(row["Sender_account"]))
        nodes.add(str(row["Receiver_account"]))

    nodes.add(sender_account)
    nodes.add(receiver_account)

    nodes = list(nodes)

    # Simple circular layout
    positions = {}

    import math

    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / max(len(nodes), 1)

        positions[node] = (
            math.cos(angle),
            math.sin(angle),
        )

    fig = go.Figure()

    # Draw edges
    for _, row in edges.iterrows():

        source = str(row["Sender_account"])
        target = str(row["Receiver_account"])

        x0, y0 = positions[source]
        x1, y1 = positions[target]

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=1),
                hoverinfo="text",
                text=(
                    f"{source} → {target}"
                    f"<br>Amount: {row['Amount']:,.2f}"
                ),
                showlegend=False,
            )
        )

    # Draw nodes
    x_values = []
    y_values = []
    labels = []
    hover_text = []

    for node in nodes:

        x, y = positions[node]

        x_values.append(x)
        y_values.append(y)

        if node == sender_account:
            label = "SENDER"
        elif node == receiver_account:
            label = "RECEIVER"
        else:
            label = node

        labels.append(label)

        hover_text.append(
            f"Account: {node}"
        )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers+text",
            text=labels,
            textposition="top center",
            hovertext=hover_text,
            hoverinfo="text",
            marker=dict(
                size=18,
            ),
            showlegend=False,
        )
    )

    fig.update_layout(
        title="AML Transaction Network",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        height=600,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return fig