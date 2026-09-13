import pandas as pd


ACCOUNT_PROFILE_FILE = "data/processed/account_aml_profile.parquet"


def load_account_profiles():
    return pd.read_parquet(ACCOUNT_PROFILE_FILE)


def get_account_profile(account_id, account_profiles):
    account_id = str(account_id)

    profile = account_profiles[
        account_profiles["account"] == account_id
    ]

    if profile.empty:
        return None

    return profile.iloc[0]


def get_network_summary(account_id, account_profiles):
    profile = get_account_profile(
        account_id,
        account_profiles
    )

    if profile is None:
        return None

    return {
        "in_degree": int(profile["in_degree"]),
        "out_degree": int(profile["out_degree"]),
        "total_degree": int(profile["total_degree"]),
        "incoming_amount": float(profile["incoming_amount"]),
        "outgoing_amount": float(profile["outgoing_amount"]),
        "unique_counterparties": int(
            profile["unique_counterparties"]
        ),
        "fan_in": bool(profile["is_fan_in"]),
        "fan_out": bool(profile["is_fan_out"]),
        "layering_candidate": bool(
            profile["is_layering_candidate"]
        ),
        "high_connectivity": bool(
            profile["is_high_connectivity"]
        ),
        "cycle_candidate": bool(
            profile["is_cycle_candidate"]
        ),
    }