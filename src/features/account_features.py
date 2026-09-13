import pandas as pd


def create_account_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create behavioral features for sender and receiver accounts.
    """

    df = df.copy()

    # =========================================================
    # SENDER FEATURES
    # =========================================================

    sender_stats = (
        df.groupby("Sender_account")
        .agg(
            sender_transaction_count=(
                "Receiver_account",
                "count"
            ),
            sender_total_amount=(
                "Amount",
                "sum"
            ),
            sender_avg_amount=(
                "Amount",
                "mean"
            ),
            sender_max_amount=(
                "Amount",
                "max"
            ),
            sender_unique_receivers=(
                "Receiver_account",
                "nunique"
            ),
        )
        .reset_index()
    )

    df = df.merge(
        sender_stats,
        on="Sender_account",
        how="left"
    )

    # =========================================================
    # RECEIVER FEATURES
    # =========================================================

    receiver_stats = (
        df.groupby("Receiver_account")
        .agg(
            receiver_transaction_count=(
                "Sender_account",
                "count"
            ),
            receiver_total_amount=(
                "Amount",
                "sum"
            ),
            receiver_avg_amount=(
                "Amount",
                "mean"
            ),
            receiver_max_amount=(
                "Amount",
                "max"
            ),
            receiver_unique_senders=(
                "Sender_account",
                "nunique"
            ),
        )
        .reset_index()
    )

    df = df.merge(
        receiver_stats,
        on="Receiver_account",
        how="left"
    )

    return df