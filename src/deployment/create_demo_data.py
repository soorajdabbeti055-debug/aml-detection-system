from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED = PROJECT_ROOT / "data" / "processed"
DEMO = PROJECT_ROOT / "data" / "demo"

DEMO.mkdir(parents=True, exist_ok=True)

FINAL_RISK_FILE = PROCESSED / "transactions_final_risk.parquet"
ALERT_FILE = PROCESSED / "aml_alerts.parquet"
INVESTIGATION_FILE = PROCESSED / "aml_investigation_report.parquet"
ACCOUNT_PROFILE_FILE = PROCESSED / "account_aml_profile.parquet"


def main():
    print("Loading processed AML data...")

    risk = pd.read_parquet(FINAL_RISK_FILE)
    alerts = pd.read_parquet(ALERT_FILE)
    investigations = pd.read_parquet(INVESTIGATION_FILE)
    accounts = pd.read_parquet(ACCOUNT_PROFILE_FILE)

    print(f"Risk transactions: {len(risk):,}")
    print(f"Alerts: {len(alerts):,}")
    print(f"Investigations: {len(investigations):,}")
    print(f"Accounts: {len(accounts):,}")

    # ---------------------------------------------------------
    # Create deterministic transaction IDs
    # ---------------------------------------------------------
    risk = risk.reset_index(drop=True)
    risk["transaction_id"] = [
        f"TXN-{i:09d}"
        for i in range(1, len(risk) + 1)
    ]

    # ---------------------------------------------------------
    # Identify transactions represented by alerts.
    #
    # We match using transaction-level fields rather than
    # assuming that an alert ID is a transaction ID.
    # ---------------------------------------------------------
    possible_keys = [
        "Sender_account",
        "Receiver_account",
        "Amount",
        "Date",
        "Time",
    ]

    available_keys = [
        col for col in possible_keys
        if col in risk.columns and col in alerts.columns
    ]

    if not available_keys:
        raise ValueError(
            "Could not find common transaction columns between "
            "transactions_final_risk.parquet and aml_alerts.parquet."
        )

    print("Matching alert transactions using:")
    print(available_keys)

    risk["_match_key"] = (
        risk[available_keys]
        .astype(str)
        .agg("|".join, axis=1)
    )

    alerts["_match_key"] = (
        alerts[available_keys]
        .astype(str)
        .agg("|".join, axis=1)
    )

    alert_keys = set(alerts["_match_key"])

    alert_transactions = risk[
        risk["_match_key"].isin(alert_keys)
    ].copy()

    print(
        f"Alert transactions matched: "
        f"{len(alert_transactions):,}"
    )

    # ---------------------------------------------------------
    # Select a manageable demo dataset.
    #
    # Keep up to 500 suspicious transactions plus normal
    # transactions for realistic dashboard behavior.
    # ---------------------------------------------------------
    suspicious = alert_transactions.sort_values(
        "risk_score",
        ascending=False
    ).head(500)

    suspicious_ids = set(suspicious["transaction_id"])

    normal = risk[
        ~risk["transaction_id"].isin(suspicious_ids)
    ].sample(
        n=min(4500, len(risk) - len(suspicious)),
        random_state=42
    )

    demo_transactions = pd.concat(
        [suspicious, normal],
        ignore_index=True
    )

    demo_transactions = demo_transactions.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    demo_transactions = demo_transactions.drop(
        columns=["_match_key"],
        errors="ignore"
    )

    # ---------------------------------------------------------
    # Keep only investigations associated with demo alerts.
    # ---------------------------------------------------------
    demo_alert_keys = set(
        suspicious["_match_key"]
    )

    demo_investigations = investigations.copy()

    if all(col in demo_investigations.columns for col in available_keys):
        demo_investigations["_match_key"] = (
            demo_investigations[available_keys]
            .astype(str)
            .agg("|".join, axis=1)
        )

        demo_investigations = demo_investigations[
            demo_investigations["_match_key"].isin(demo_alert_keys)
        ].drop(columns=["_match_key"])

    # ---------------------------------------------------------
    # Keep only accounts involved in demo transactions.
    # ---------------------------------------------------------
    demo_accounts_used = set(
        demo_transactions["Sender_account"].astype(str)
    ) | set(
        demo_transactions["Receiver_account"].astype(str)
    )

    accounts = accounts.copy()
    accounts["account"] = accounts["account"].astype(str)

    demo_accounts = accounts[
        accounts["account"].isin(demo_accounts_used)
    ].copy()

    # ---------------------------------------------------------
    # Save deployment data.
    # ---------------------------------------------------------
    demo_transactions.to_parquet(
        DEMO / "transactions_demo.parquet",
        index=False
    )

    demo_investigations.to_parquet(
        DEMO / "investigation_demo.parquet",
        index=False
    )

    demo_accounts.to_parquet(
        DEMO / "account_profiles_demo.parquet",
        index=False
    )

    print()
    print("=" * 60)
    print("DEMO DATA CREATED")
    print("=" * 60)
    print(
        f"Transactions : {len(demo_transactions):,}"
    )
    print(
        f"Investigations: {len(demo_investigations):,}"
    )
    print(
        f"Accounts     : {len(demo_accounts):,}"
    )
    print()
    print("Files:")
    print(DEMO / "transactions_demo.parquet")
    print(DEMO / "investigation_demo.parquet")
    print(DEMO / "account_profiles_demo.parquet")
    print()

    print("Example transaction IDs:")
    print(
        demo_transactions[
            ["transaction_id", "Sender_account",
             "Receiver_account", "Amount"]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()