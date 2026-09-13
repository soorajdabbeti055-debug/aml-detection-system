import pandas as pd
import numpy as np


INPUT_FILE = "data/processed/aml_alerts.parquet"
OUTPUT_FILE = "data/processed/aml_investigation_report.parquet"
SUMMARY_FILE = "data/reports/aml_investigation_summary.csv"


def recommended_action(row):
    severity = row["alert_severity"]

    if severity == "CRITICAL":
        return (
            "Immediate manual investigation; verify transaction "
            "purpose, source of funds, beneficiary relationship, "
            "and supporting documentation."
        )

    if severity == "HIGH":
        return (
            "Prioritized AML investigation; review transaction "
            "history, counterparties, geographic exposure, and "
            "account behaviour."
        )

    if severity == "MEDIUM":
        return (
            "Secondary review; examine transaction context and "
            "monitor for repeated suspicious activity."
        )

    return "Continue monitoring."


def investigation_outcome(row):
    if row["alert_severity"] == "CRITICAL":
        return "OPEN - URGENT"

    if row["alert_severity"] == "HIGH":
        return "OPEN - PRIORITY"

    return "OPEN - REVIEW"


def main():

    print("=" * 70)
    print("PHASE 5.4 - AML INVESTIGATION REPORT")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load alerts
    # ---------------------------------------------------------

    print("\nLoading AML alerts...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Alerts loaded: {len(df):,}")

    # ---------------------------------------------------------
    # 2. Validate required fields
    # ---------------------------------------------------------

    required_columns = [
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

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("✓ Required investigation fields available.")

    # ---------------------------------------------------------
    # 3. Create investigation fields
    # ---------------------------------------------------------

    print("\nBuilding investigation fields...")

    df["recommended_action"] = df.apply(
        recommended_action,
        axis=1
    )

    df["investigation_status"] = df.apply(
        investigation_outcome,
        axis=1
    )

    # Convert probability to percentage for investigators
    df["ml_probability_percent"] = (
        df["ml_risk_probability"] * 100
    ).round(2)

    # ---------------------------------------------------------
    # 4. Transaction context
    # ---------------------------------------------------------

    df["transaction_context"] = np.select(
        [
            (
                (df["cross_border"] == 1)
                & (df["currency_changed"] == 1)
            ),

            df["cross_border"] == 1,

            df["currency_changed"] == 1,
        ],
        [
            "Cross-border + currency change",
            "Cross-border transaction",
            "Currency change",
        ],
        default="Domestic / same-currency transaction",
    )

    # ---------------------------------------------------------
    # 5. Behavioural context
    # ---------------------------------------------------------

    df["sender_activity_level"] = np.select(
        [
            df["sender_transaction_count"] >= 100,
            df["sender_transaction_count"] >= 25,
            df["sender_transaction_count"] >= 10,
        ],
        [
            "Very high",
            "High",
            "Elevated",
        ],
        default="Normal",
    )

    df["receiver_activity_level"] = np.select(
        [
            df["receiver_transaction_count"] >= 100,
            df["receiver_transaction_count"] >= 25,
            df["receiver_transaction_count"] >= 10,
        ],
        [
            "Very high",
            "High",
            "Elevated",
        ],
        default="Normal",
    )

    # ---------------------------------------------------------
    # 6. Temporal context
    # ---------------------------------------------------------

    df["temporal_risk_indicator"] = np.select(
        [
            (
                df["seconds_since_previous_sender_transaction"]
                .between(0, 60)
            ),

            (
                df["seconds_since_previous_sender_transaction"]
                .between(61, 300)
            ),
        ],
        [
            "Very rapid repeat transaction",
            "Rapid repeat transaction",
        ],
        default="No rapid repeat indicator",
    )

    # ---------------------------------------------------------
    # 7. Investigation label
    # ---------------------------------------------------------

    df["investigation_label"] = np.select(
        [
            df["risk_score"] >= 75,
            df["risk_score"] >= 50,
        ],
        [
            "CRITICAL AML ALERT",
            "HIGH-CONFIDENCE AML SCREENING ALERT",
        ],
        default="AML SCREENING ALERT",
    )

    # ---------------------------------------------------------
    # 8. Final investigator report columns
    # ---------------------------------------------------------

    report_columns = [
        "alert_id",

        "investigation_status",
        "investigation_priority",
        "investigation_label",

        "alert_severity",
        "risk_score",
        "ml_risk_probability",
        "ml_probability_percent",

        "Sender_account",
        "Receiver_account",
        "Amount",

        "Payment_currency",
        "Received_currency",

        "Sender_bank_location",
        "Receiver_bank_location",

        "Payment_type",

        "transaction_context",

        "cross_border",
        "currency_changed",

        "sender_transaction_count",
        "receiver_transaction_count",

        "sender_activity_level",
        "receiver_activity_level",

        "seconds_since_previous_sender_transaction",
        "temporal_risk_indicator",

        "risk_reasons",

        "recommended_action",

        "Is_laundering",
    ]

    report = df[report_columns].copy()

    # ---------------------------------------------------------
    # 9. Sort by severity and score
    # ---------------------------------------------------------

    severity_order = {
        "CRITICAL": 0,
        "HIGH": 1,
        "MEDIUM": 2,
        "LOW": 3,
    }

    report["_severity_order"] = (
        report["alert_severity"]
        .map(severity_order)
    )

    report = (
        report
        .sort_values(
            ["_severity_order", "risk_score"],
            ascending=[True, False]
        )
        .drop(columns="_severity_order")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # 10. Create summary
    # ---------------------------------------------------------

    print("\nCreating investigation summary...")

    summary = (
        report
        .groupby(
            [
                "alert_severity",
                "investigation_priority",
            ],
            dropna=False
        )
        .agg(
            alert_count=("alert_id", "count"),
            average_risk_score=("risk_score", "mean"),
            maximum_risk_score=("risk_score", "max"),
            average_amount=("Amount", "mean"),
            total_amount=("Amount", "sum"),
            laundering_cases=("Is_laundering", "sum"),
        )
        .reset_index()
    )

    summary["average_risk_score"] = (
        summary["average_risk_score"].round(2)
    )

    summary["maximum_risk_score"] = (
        summary["maximum_risk_score"].round(2)
    )

    summary["average_amount"] = (
        summary["average_amount"].round(2)
    )

    summary["total_amount"] = (
        summary["total_amount"].round(2)
    )

    # ---------------------------------------------------------
    # 11. Validation
    # ---------------------------------------------------------

    print("\nValidating investigation report...")

    if len(report) != len(df):
        raise ValueError(
            "Investigation report row count changed."
        )

    if report["alert_id"].duplicated().any():
        raise ValueError(
            "Duplicate alert IDs detected."
        )

    if report["risk_score"].isna().any():
        raise ValueError(
            "Missing risk scores detected."
        )

    if report["risk_reasons"].isna().any():
        raise ValueError(
            "Missing risk reasons detected."
        )

    if report["recommended_action"].isna().any():
        raise ValueError(
            "Missing investigation actions detected."
        )

    print("✓ All alerts retained.")
    print("✓ Alert IDs are unique.")
    print("✓ Risk scores are present.")
    print("✓ Investigation explanations are present.")
    print("✓ Recommended actions are present.")

    # ---------------------------------------------------------
    # 12. Display summary
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("INVESTIGATION SUMMARY")
    print("-" * 70)

    print(
        summary.to_string(index=False)
    )

    # ---------------------------------------------------------
    # 13. Display top critical cases
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("TOP CRITICAL INVESTIGATION CASES")
    print("-" * 70)

    critical = report[
        report["alert_severity"] == "CRITICAL"
    ]

    if len(critical) > 0:

        columns = [
            "alert_id",
            "risk_score",
            "ml_probability_percent",
            "Amount",
            "transaction_context",
            "risk_reasons",
        ]

        print(
            critical.head(10)[columns]
            .to_string(index=False)
        )

    # ---------------------------------------------------------
    # 14. Save outputs
    # ---------------------------------------------------------

    print("\nSaving investigation report...")

    report.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    print(
        f"\nInvestigation report saved to:\n{OUTPUT_FILE}"
    )

    print(
        f"\nInvestigation summary saved to:\n{SUMMARY_FILE}"
    )

    print("\n" + "=" * 70)
    print("PHASE 5.4 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()