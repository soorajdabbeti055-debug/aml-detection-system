import pandas as pd
import numpy as np


INPUT_FILE = "data/processed/transactions_final_risk.parquet"
ALERT_FILE = "data/processed/aml_alerts.parquet"
SUMMARY_FILE = "data/reports/aml_alert_summary.csv"


def assign_priority(row):
    if row["risk_category"] == "CRITICAL":
        return "P1 - Immediate Review"

    if row["risk_category"] == "HIGH":
        return "P2 - High Priority"

    if row["risk_category"] == "MEDIUM":
        return "P3 - Review"

    return "P4 - Monitor"


def main():

    print("=" * 70)
    print("PHASE 5.3 - AML ALERT GENERATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load final risk dataset
    # ---------------------------------------------------------

    print("\nLoading final risk dataset...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Transactions loaded: {len(df):,}")

    # ---------------------------------------------------------
    # 2. Validate required columns
    # ---------------------------------------------------------

    required_columns = [
        "Sender_account",
        "Receiver_account",
        "Amount",
        "Payment_currency",
        "Received_currency",
        "Sender_bank_location",
        "Receiver_bank_location",
        "Payment_type",
        "Is_laundering",
        "ml_risk_probability",
        "risk_score",
        "risk_category",
        "risk_reasons",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("✓ Required columns available.")

    # ---------------------------------------------------------
    # 3. Assign investigation priority
    # ---------------------------------------------------------

    print("\nAssigning investigation priorities...")

    df["investigation_priority"] = df.apply(
        assign_priority,
        axis=1
    )

    print("✓ Priorities assigned.")

    # ---------------------------------------------------------
    # 4. Generate alert IDs
    # ---------------------------------------------------------

    # Only transactions with risk >= 50 become investigation alerts.
    #
    # LOW-risk transactions remain available in the complete
    # scored dataset but are not sent to the investigation queue.

    alerts = df[
        df["risk_score"] >= 50
    ].copy()

    alerts = alerts.sort_values(
        ["risk_score", "ml_risk_probability"],
        ascending=False
    ).reset_index(drop=True)

    alerts.insert(
        0,
        "alert_id",
        [
            f"AML-{i:07d}"
            for i in range(1, len(alerts) + 1)
        ]
    )

    print(
        f"Investigation alerts generated: {len(alerts):,}"
    )

    # ---------------------------------------------------------
    # 5. Add alert severity
    # ---------------------------------------------------------

    alerts["alert_severity"] = alerts["risk_category"]

    # ---------------------------------------------------------
    # 6. Add useful investigation fields
    # ---------------------------------------------------------

    alerts["amount_log"] = np.log1p(
        alerts["Amount"]
    )

    alerts["is_high_value"] = (
        alerts["Amount"]
        >= df["Amount"].quantile(0.95)
    ).astype("int8")

    alerts["is_very_high_value"] = (
        alerts["Amount"]
        >= df["Amount"].quantile(0.99)
    ).astype("int8")

    # ---------------------------------------------------------
    # 7. Select investigator-friendly columns
    # ---------------------------------------------------------

    alert_columns = [
        "alert_id",
        "investigation_priority",
        "alert_severity",
        "risk_score",
        "ml_risk_probability",

        "Sender_account",
        "Receiver_account",
        "Amount",

        "Payment_currency",
        "Received_currency",

        "Sender_bank_location",
        "Receiver_bank_location",

        "Payment_type",

        "cross_border",
        "currency_changed",

        "sender_transaction_count",
        "receiver_transaction_count",

        "seconds_since_previous_sender_transaction",

        "risk_reasons",

        "Is_laundering",
    ]

    alerts = alerts[alert_columns]

    # ---------------------------------------------------------
    # 8. Summary
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("ALERT SUMMARY")
    print("-" * 70)

    print(
        f"\nTotal transactions: {len(df):,}"
    )

    print(
        f"Total alerts:        {len(alerts):,}"
    )

    alert_rate = (
        len(alerts) / len(df) * 100
    )

    print(
        f"Alert rate:          {alert_rate:.4f}%"
    )

    print("\nAlerts by severity:")

    severity_counts = (
        alerts["alert_severity"]
        .value_counts()
        .reindex(
            ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            fill_value=0
        )
    )

    print(severity_counts)

    print("\nAlerts by investigation priority:")

    priority_counts = (
        alerts["investigation_priority"]
        .value_counts()
    )

    print(priority_counts)

    # ---------------------------------------------------------
    # 9. Calculate summary report
    # ---------------------------------------------------------

    summary_rows = []

    total_transactions = len(df)

    for severity in [
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW",
    ]:

        count = int(
            (
                alerts["alert_severity"]
                == severity
            ).sum()
        )

        percentage = (
            count / total_transactions * 100
        )

        summary_rows.append(
            {
                "severity": severity,
                "alert_count": count,
                "percentage_of_all_transactions":
                    round(percentage, 6),
            }
        )

    summary = pd.DataFrame(summary_rows)

    # Add overall totals

    summary = pd.concat(
        [
            summary,
            pd.DataFrame(
                [
                    {
                        "severity": "TOTAL_ALERTS",
                        "alert_count": len(alerts),
                        "percentage_of_all_transactions":
                            round(alert_rate, 6),
                    }
                ]
            ),
        ],
        ignore_index=True
    )

    # ---------------------------------------------------------
    # 10. Validation
    # ---------------------------------------------------------

    print("\nValidating alert dataset...")

    if alerts["alert_id"].duplicated().any():
        raise ValueError(
            "Duplicate alert IDs detected."
        )

    if alerts["risk_score"].isna().any():
        raise ValueError(
            "Missing risk scores detected."
        )

    if alerts["risk_reasons"].isna().any():
        raise ValueError(
            "Missing risk reasons detected."
        )

    if not (
        alerts["risk_score"] >= 50
    ).all():
        raise ValueError(
            "Invalid alert threshold detected."
        )

    if not alerts["risk_score"].is_monotonic_decreasing:
        raise ValueError(
            "Alerts are not sorted by risk score."
        )

    print("✓ Alert IDs are unique.")
    print("✓ Risk scores are valid.")
    print("✓ Risk explanations are present.")
    print("✓ All alerts meet the risk threshold.")
    print("✓ Alerts sorted by risk score.")

    # ---------------------------------------------------------
    # 11. Save outputs
    # ---------------------------------------------------------

    print("\nSaving alert dataset...")

    alerts.to_parquet(
        ALERT_FILE,
        index=False
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    print(
        f"\nAlerts saved to:\n{ALERT_FILE}"
    )

    print(
        f"\nSummary saved to:\n{SUMMARY_FILE}"
    )

    # ---------------------------------------------------------
    # 12. Show top alerts
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("TOP 10 AML ALERTS")
    print("-" * 70)

    if len(alerts) > 0:

        display_columns = [
            "alert_id",
            "alert_severity",
            "risk_score",
            "ml_risk_probability",
            "Amount",
            "risk_reasons",
        ]

        print(
            alerts.head(10)[display_columns]
            .to_string(index=False)
        )

    else:
        print("No alerts generated.")

    print("\n" + "=" * 70)
    print("PHASE 5.3 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()