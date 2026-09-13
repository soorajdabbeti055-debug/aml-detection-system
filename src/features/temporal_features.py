import pandas as pd


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Explicit format: DD-MM-YYYY HH:MM:SS
    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str).str.strip()
        + " "
        + df["Time"].astype(str).str.strip(),
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce"
    )

    # Calendar features
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["second"] = df["datetime"].dt.second
    df["day_of_week"] = df["datetime"].dt.dayofweek

    # Time since previous transaction from the same sender
    df = df.sort_values(
        ["Sender_account", "datetime"]
    ).reset_index(drop=True)

    df["previous_sender_datetime"] = (
        df.groupby("Sender_account")["datetime"].shift(1)
    )

    df["seconds_since_previous_sender_transaction"] = (
        df["datetime"] - df["previous_sender_datetime"]
    ).dt.total_seconds()

    # First transaction of each sender
    df["seconds_since_previous_sender_transaction"] = (
        df["seconds_since_previous_sender_transaction"]
        .fillna(-1)
    )

    # Restore chronological order
    df = df.sort_values("datetime").reset_index(drop=True)

    return df