import numpy as np
import pandas as pd


def create_transaction_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["log_amount"] = np.log1p(df["Amount"])

    df["currency_changed"] = (
        df["Payment_currency"] != df["Received_currency"]
    ).astype("int8")

    df["cross_border"] = (
        df["Sender_bank_location"]
        != df["Receiver_bank_location"]
    ).astype("int8")

    df["same_country"] = (
        df["Sender_bank_location"]
        == df["Receiver_bank_location"]
    ).astype("int8")

    return df