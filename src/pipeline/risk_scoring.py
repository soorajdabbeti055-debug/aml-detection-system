import numpy as np
import pandas as pd


INPUT_FILE = "data/processed/transactions_risk_predictions.parquet"
OUTPUT_FILE = "data/processed/transactions_final_risk.parquet"


def normalize(value, minimum, maximum):
    if maximum == minimum:
        return 0.0
    return (value - minimum) / (maximum - minimum)


def calculate_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---------------------------------------------------------
    # 1. ML risk component
    # ---------------------------------------------------------
    df["ml_risk_score"] = (
        df["ml_risk_probability"].clip(0, 1) * 100
    )

    # ---------------------------------------------------------
    # 2. Transaction amount risk
    # ---------------------------------------------------------
    amount_95 = df["Amount"].quantile(0.95)
    amount_99 = df["Amount"].quantile(0.99)

    df["amount_risk"] = np.where(
        df["Amount"] >= amount_99,
        100,
        np.where(
            df["Amount"] >= amount_95,
            70,
            normalize(
                df["Amount"],
                df["Amount"].min(),
                amount_95
            ) * 50
        )
    )

    # ---------------------------------------------------------
    # 3. Cross-border risk
    # ---------------------------------------------------------
    df["cross_border_risk"] = (
        df["cross_border"] * 100
    )

    # ---------------------------------------------------------
    # 4. Currency-change risk
    # ---------------------------------------------------------
    df["currency_change_risk"] = (
        df["currency_changed"] * 100
    )

    # ---------------------------------------------------------
    # 5. Account activity risk
    # ---------------------------------------------------------
    sender_activity_95 = (
        df["sender_transaction_count"].quantile(0.95)
    )

    receiver_activity_95 = (
        df["receiver_transaction_count"].quantile(0.95)
    )

    sender_activity_score = (
        df["sender_transaction_count"]
        .clip(upper=sender_activity_95)
        / max(sender_activity_95, 1)
        * 100
    )

    receiver_activity_score = (
        df["receiver_transaction_count"]
        .clip(upper=receiver_activity_95)
        / max(receiver_activity_95, 1)
        * 100
    )

    df["account_activity_risk"] = (
        0.6 * sender_activity_score
        + 0.4 * receiver_activity_score
    )

    # ---------------------------------------------------------
    # 6. Temporal risk
    # ---------------------------------------------------------
    # Transactions occurring very soon after another sender
    # transaction can indicate rapid movement of funds.
    df["rapid_transaction_risk"] = np.select(
        [
            df["seconds_since_previous_sender_transaction"]
            .between(0, 60),

            df["seconds_since_previous_sender_transaction"]
            .between(61, 300),

            df["seconds_since_previous_sender_transaction"]
            .between(301, 900),
        ],
        [
            100,
            70,
            40,
        ],
        default=0,
    )

    # ---------------------------------------------------------
    # 7. Graph/network risk
    # ---------------------------------------------------------
    graph_components = []

    if "sender_total_degree" in df.columns:
        sender_degree_95 = df["sender_total_degree"].quantile(0.95)

        df["sender_network_risk"] = (
            df["sender_total_degree"]
            .clip(upper=sender_degree_95)
            / max(sender_degree_95, 1)
            * 100
        )

        graph_components.append("sender_network_risk")

    else:
        df["sender_network_risk"] = 0

    if "receiver_total_degree" in df.columns:
        receiver_degree_95 = df["receiver_total_degree"].quantile(0.95)

        df["receiver_network_risk"] = (
            df["receiver_total_degree"]
            .clip(upper=receiver_degree_95)
            / max(receiver_degree_95, 1)
            * 100
        )

        graph_components.append("receiver_network_risk")

    else:
        df["receiver_network_risk"] = 0

    df["network_risk"] = (
        df["sender_network_risk"]
        + df["receiver_network_risk"]
    ) / 2

    # ---------------------------------------------------------
    # 8. Composite AML risk score
    # ---------------------------------------------------------
    #
    # ML = primary predictive component
    # Transaction/context = supporting evidence
    # Network = supporting structural evidence
    #
    df["risk_score"] = (
        0.50 * df["ml_risk_score"]
        + 0.15 * df["amount_risk"]
        + 0.10 * df["cross_border_risk"]
        + 0.05 * df["currency_change_risk"]
        + 0.05 * df["account_activity_risk"]
        + 0.05 * df["rapid_transaction_risk"]
        + 0.10 * df["network_risk"]
    )

    df["risk_score"] = (
        df["risk_score"]
        .clip(0, 100)
        .round(2)
    )

    # ---------------------------------------------------------
    # 9. Risk category
    # ---------------------------------------------------------
    df["risk_category"] = pd.cut(
        df["risk_score"],
        bins=[-np.inf, 25, 50, 75, np.inf],
        labels=[
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL",
        ]
    )

    # ---------------------------------------------------------
    # 10. Explainable risk reasons
    # ---------------------------------------------------------
    def generate_reasons(row):

        reasons = []

        if row["ml_risk_probability"] >= 0.75:
            reasons.append("Very high ML laundering probability")
        elif row["ml_risk_probability"] >= 0.50:
            reasons.append("High ML laundering probability")
        elif row["ml_risk_probability"] >= 0.25:
            reasons.append("Elevated ML laundering probability")

        if row["Amount"] >= amount_99:
            reasons.append("Extremely high transaction amount")
        elif row["Amount"] >= amount_95:
            reasons.append("Unusually high transaction amount")

        if row["cross_border"] == 1:
            reasons.append("Cross-border transaction")

        if row["currency_changed"] == 1:
            reasons.append("Currency conversion/change")

        if row["seconds_since_previous_sender_transaction"] >= 0:
            if row["seconds_since_previous_sender_transaction"] <= 60:
                reasons.append("Very rapid repeat transaction")
            elif row["seconds_since_previous_sender_transaction"] <= 300:
                reasons.append("Rapid repeat transaction")

        if row["sender_transaction_count"] >= sender_activity_95:
            reasons.append("Highly active sender account")

        if row["receiver_transaction_count"] >= receiver_activity_95:
            reasons.append("Highly active receiver account")

        if row["network_risk"] >= 75:
            reasons.append("Highly connected account/network")

        if not reasons:
            reasons.append("No dominant individual risk indicator")

        return "; ".join(reasons)

    df["risk_reasons"] = df.apply(
        generate_reasons,
        axis=1
    )

    return df


def main():

    print("=" * 70)
    print("PHASE 5.2 - EXPLAINABLE AML RISK SCORING")
    print("=" * 70)

    print("\nLoading risk predictions...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Transactions loaded: {len(df):,}")

    required_columns = [
        "ml_risk_probability",
        "Amount",
        "cross_border",
        "currency_changed",
        "sender_transaction_count",
        "receiver_transaction_count",
        "seconds_since_previous_sender_transaction",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("\nCalculating explainable risk score...")

    df = calculate_risk_score(df)

    print("✓ Risk scores generated.")

    print("\nRisk score statistics:")

    print(
        df["risk_score"]
        .describe()
        .round(2)
    )

    print("\nRisk category distribution:")

    print(
        df["risk_category"]
        .value_counts()
        .sort_index()
    )

    print("\nTop 10 highest-risk transactions:")

    top_columns = [
        "Sender_account",
        "Receiver_account",
        "Amount",
        "ml_risk_probability",
        "risk_score",
        "risk_category",
        "risk_reasons",
    ]

    print(
        df.nlargest(10, "risk_score")[top_columns]
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    print("\nValidating output...")

    if len(df) != 1_048_575:
        raise ValueError(
            "Transaction count changed."
        )

    if df["risk_score"].isna().any():
        raise ValueError(
            "Missing risk scores detected."
        )

    if not np.isfinite(
        df["risk_score"].to_numpy()
    ).all():
        raise ValueError(
            "Invalid risk scores detected."
        )

    if df["risk_category"].isna().any():
        raise ValueError(
            "Missing risk categories detected."
        )

    print("✓ All transactions have risk scores.")
    print("✓ All transactions have risk categories.")
    print("✓ No transactions lost.")

    print("\nSaving final risk dataset...")

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )

    print("\n" + "=" * 70)
    print("PHASE 5.2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()