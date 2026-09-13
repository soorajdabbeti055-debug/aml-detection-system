from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "raw_dataset": ROOT / "data/raw/SAML-D.csv",
    "features": ROOT / "data/processed/transactions_features.parquet",
    "pit_features": ROOT / "data/processed/transactions_pit_features.parquet",
    "graph_features": ROOT / "data/processed/account_aml_profile.parquet",
    "risk_predictions": ROOT / "data/processed/transactions_risk_predictions.parquet",
    "final_risk": ROOT / "data/processed/transactions_final_risk.parquet",
    "alerts": ROOT / "data/processed/aml_alerts.parquet",
    "investigation": ROOT / "data/processed/aml_investigation_report.parquet",
}

EXPECTED_TRANSACTIONS = 1_048_575


def check_file(name, path):
    if not path.exists():
        print(f"❌ {name}: missing")
        return False

    print(f"✓ {name}: exists")
    return True


def main():

    print("=" * 70)
    print("AML DETECTION SYSTEM - FINAL VALIDATION")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Check required files
    # --------------------------------------------------

    print("\n1. CHECKING REQUIRED FILES")
    print("-" * 70)

    all_files_exist = True

    for name, path in FILES.items():
        if not check_file(name, path):
            all_files_exist = False

    if not all_files_exist:
        print("\n❌ Validation failed: one or more required files are missing.")
        return

    # --------------------------------------------------
    # 2. Transaction counts
    # --------------------------------------------------

    print("\n2. CHECKING TRANSACTION COUNTS")
    print("-" * 70)

    transaction_files = [
        "features",
        "pit_features",
        "risk_predictions",
        "final_risk",
    ]

    counts_ok = True

    for name in transaction_files:
        df = pd.read_parquet(FILES[name])

        if len(df) != EXPECTED_TRANSACTIONS:
            print(
                f"❌ {name}: {len(df):,} rows "
                f"(expected {EXPECTED_TRANSACTIONS:,})"
            )
            counts_ok = False
        else:
            print(f"✓ {name}: {len(df):,} rows")

    # --------------------------------------------------
    # 3. Raw dataset
    # --------------------------------------------------

    print("\n3. CHECKING RAW DATASET")
    print("-" * 70)

    raw = pd.read_csv(FILES["raw_dataset"])

    print(f"Rows: {len(raw):,}")
    print(f"Columns: {len(raw.columns)}")

    if len(raw) == EXPECTED_TRANSACTIONS:
        print("✓ Raw transaction count correct")
    else:
        print("❌ Raw transaction count incorrect")

    # --------------------------------------------------
    # 4. Target distribution
    # --------------------------------------------------

    print("\n4. CHECKING TARGET DISTRIBUTION")
    print("-" * 70)

    features = pd.read_parquet(
        FILES["features"],
        columns=["Is_laundering"]
    )

    normal = int((features["Is_laundering"] == 0).sum())
    laundering = int((features["Is_laundering"] == 1).sum())

    print(f"Normal transactions:     {normal:,}")
    print(f"Laundering transactions: {laundering:,}")

    if normal == 1_047_619 and laundering == 956:
        print("✓ Target distribution correct")
    else:
        print("⚠ Target distribution differs from expected values")

    # --------------------------------------------------
    # 5. PIT feature validation
    # --------------------------------------------------

    print("\n5. CHECKING POINT-IN-TIME FEATURES")
    print("-" * 70)

    pit = pd.read_parquet(FILES["pit_features"])

    pit_columns = [
        "sender_tx_count_before",
        "sender_total_amount_before",
        "sender_avg_amount_before",
        "sender_max_amount_before",
        "sender_unique_receivers_before",
        "receiver_tx_count_before",
        "receiver_total_amount_before",
        "receiver_avg_amount_before",
        "receiver_max_amount_before",
        "receiver_unique_senders_before",
    ]

    missing_pit = [
        col for col in pit_columns
        if col not in pit.columns
    ]

    if missing_pit:
        print(f"❌ Missing PIT features: {missing_pit}")
    else:
        print(f"✓ All {len(pit_columns)} PIT features present")

    # --------------------------------------------------
    # 6. Graph validation
    # --------------------------------------------------

    print("\n6. CHECKING GRAPH FEATURES")
    print("-" * 70)

    graph = pd.read_parquet(FILES["graph_features"])

    print(f"Accounts: {len(graph):,}")
    print(f"Graph feature columns: {len(graph.columns)}")

    required_graph_columns = [
        "account",
        "in_degree",
        "out_degree",
        "total_degree",
        "incoming_amount",
        "outgoing_amount",
        "is_fan_in",
        "is_fan_out",
        "is_layering_candidate",
        "is_high_connectivity",
        "is_cycle_candidate",
    ]

    missing_graph = [
        col for col in required_graph_columns
        if col not in graph.columns
    ]

    if missing_graph:
        print(f"❌ Missing graph columns: {missing_graph}")
    else:
        print("✓ Required graph features present")

    # --------------------------------------------------
    # 7. Risk validation
    # --------------------------------------------------

    print("\n7. CHECKING RISK SCORES")
    print("-" * 70)

    risk = pd.read_parquet(FILES["final_risk"])

    print(f"Transactions: {len(risk):,}")

    if "risk_score" in risk.columns:
        print(
            f"Risk score range: "
            f"{risk['risk_score'].min():.2f} - "
            f"{risk['risk_score'].max():.2f}"
        )
        print("✓ Risk scores present")
    else:
        print("❌ risk_score column missing")

    if "risk_category" in risk.columns:

        print("\nRisk categories:")
        print(
            risk["risk_category"]
            .value_counts()
            .to_string()
        )

        print("✓ Risk categories present")

    else:
        print("❌ risk_category column missing")

    # --------------------------------------------------
    # 8. Alert validation
    # --------------------------------------------------

    print("\n8. CHECKING AML ALERTS")
    print("-" * 70)

    alerts = pd.read_parquet(FILES["alerts"])

    print(f"Total alerts: {len(alerts):,}")

    if len(alerts) > 0:

        if "alert_id" in alerts.columns:

            unique_ids = alerts["alert_id"].nunique()

            print(f"Unique alert IDs: {unique_ids:,}")

            if unique_ids == len(alerts):
                print("✓ Alert IDs are unique")
            else:
                print("❌ Duplicate alert IDs found")

        if "risk_score" in alerts.columns:

            if alerts["risk_score"].notna().all():
                print("✓ Alert risk scores present")
            else:
                print("❌ Missing alert risk scores")

        if "alert_severity" in alerts.columns:

            print("\nAlert severity:")
            print(
                alerts["alert_severity"]
                .value_counts()
                .to_string()
            )

    # --------------------------------------------------
    # 9. Investigation report
    # --------------------------------------------------

    print("\n9. CHECKING INVESTIGATION REPORT")
    print("-" * 70)

    investigation = pd.read_parquet(
        FILES["investigation"]
    )

    print(
        f"Investigation records: "
        f"{len(investigation):,}"
    )

    if len(investigation) == len(alerts):
        print("✓ All alerts have investigation records")
    else:
        print("❌ Alert/investigation count mismatch")

    required_investigation_columns = [
        "alert_id",
        "risk_score",
        "recommended_action",
        "investigation_status",
        "risk_reasons",
    ]

    missing_investigation = [
        col
        for col in required_investigation_columns
        if col not in investigation.columns
    ]

    if missing_investigation:
        print(
            f"❌ Missing investigation columns: "
            f"{missing_investigation}"
        )
    else:
        print("✓ Required investigation fields present")

    # --------------------------------------------------
    # 10. Final result
    # --------------------------------------------------

    print("\n" + "=" * 70)

    if all_files_exist and counts_ok:
        print("✅ FINAL VALIDATION PASSED")
        print("=" * 70)
        print("\nAML Detection System is ready for final presentation.")
    else:
        print("❌ FINAL VALIDATION FAILED")
        print("=" * 70)


if __name__ == "__main__":
    main()