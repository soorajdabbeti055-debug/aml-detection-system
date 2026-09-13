import pandas as pd


def create_point_in_time_account_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    # Preserve original transaction order.
    # This provides a deterministic tie-breaker when
    # multiple transactions have the same timestamp.
    df["_original_row_id"] = range(len(df))

    # Sort chronologically.
    # For identical timestamps, preserve original dataset order.
    df = (
        df.sort_values(
            ["datetime", "_original_row_id"]
        )
        .reset_index(drop=True)
    )

    # =========================================================
    # SENDER HISTORICAL FEATURES
    # =========================================================

    # Number of previous transactions from this sender.
    df["sender_transaction_count"] = (
        df.groupby("Sender_account")
        .cumcount()
        .astype("int32")
    )

    # Cumulative amount including current transaction.
    sender_cumulative_amount = (
        df.groupby("Sender_account")["Amount"]
        .cumsum()
    )

    # Remove current transaction.
    df["sender_total_amount"] = (
        sender_cumulative_amount - df["Amount"]
    )

    # Historical average amount.
    df["sender_avg_amount"] = (
        df["sender_total_amount"]
        / df["sender_transaction_count"].replace(0, 1)
    )

    # Historical maximum transaction amount.
    sender_cumulative_max = (
        df.groupby("Sender_account")["Amount"]
        .cummax()
    )

    df["sender_max_amount"] = (
        sender_cumulative_max
        .groupby(df["Sender_account"])
        .shift(1)
        .fillna(0)
    )

    # =========================================================
    # RECEIVER HISTORICAL FEATURES
    # =========================================================

    # Number of previous transactions received.
    df["receiver_transaction_count"] = (
        df.groupby("Receiver_account")
        .cumcount()
        .astype("int32")
    )

    # Cumulative received amount including current transaction.
    receiver_cumulative_amount = (
        df.groupby("Receiver_account")["Amount"]
        .cumsum()
    )

    # Remove current transaction.
    df["receiver_total_amount"] = (
        receiver_cumulative_amount - df["Amount"]
    )

    # Historical average received amount.
    df["receiver_avg_amount"] = (
        df["receiver_total_amount"]
        / df["receiver_transaction_count"].replace(0, 1)
    )

    # Historical maximum received amount.
    receiver_cumulative_max = (
        df.groupby("Receiver_account")["Amount"]
        .cummax()
    )

    df["receiver_max_amount"] = (
        receiver_cumulative_max
        .groupby(df["Receiver_account"])
        .shift(1)
        .fillna(0)
    )

    # =========================================================
    # HISTORICAL UNIQUE COUNTERPARTIES
    # =========================================================

    # A sender-receiver pair is "new" the first time it appears.
    sender_pair_first = ~df.duplicated(
        ["Sender_account", "Receiver_account"]
    )

    # Number of unique receivers seen BEFORE this transaction.
    sender_unique_cumulative = (
        sender_pair_first
        .groupby(df["Sender_account"])
        .cumsum()
    )

    df["sender_unique_receivers"] = (
        sender_unique_cumulative
        - sender_pair_first.astype("int32")
    ).astype("int32")

    # A receiver-sender pair is "new" the first time it appears.
    receiver_pair_first = ~df.duplicated(
        ["Receiver_account", "Sender_account"]
    )

    # Number of unique senders seen BEFORE this transaction.
    receiver_unique_cumulative = (
        receiver_pair_first
        .groupby(df["Receiver_account"])
        .cumsum()
    )

    df["receiver_unique_senders"] = (
        receiver_unique_cumulative
        - receiver_pair_first.astype("int32")
    ).astype("int32")

    # =========================================================
    # CLEANUP
    # =========================================================

    # Internal helper column is no longer needed.
    df = df.drop(columns=["_original_row_id"])

    return df