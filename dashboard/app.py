import streamlit as st
import pandas as pd
import plotly.express as px

from network_analysis import (
    load_account_profiles,
    get_network_summary,
)

from graph_visualization import (
    get_network_edges,
    create_network_graph,
)


# ============================================================
# CONFIGURATION
# ============================================================

RISK_FILE = "data/demo/transactions_demo.parquet"
INVESTIGATION_FILE = "data/demo/investigation_demo.parquet"
ACCOUNT_PROFILE_FILE = "data/demo/account_profiles_demo.parquet"


st.set_page_config(
    page_title="AML Transaction Investigation",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    transactions = pd.read_parquet(RISK_FILE)
    investigations = pd.read_parquet(INVESTIGATION_FILE)
    accounts = pd.read_parquet(ACCOUNT_PROFILE_FILE)

    transactions["transaction_id"] = (
        transactions["transaction_id"].astype(str)
    )

    return transactions, investigations, accounts


transactions_df, investigation_df, account_profiles = load_data()


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ AML Transaction Investigation System")

st.markdown(
    """
    **Intelligent Money Laundering Detection using Machine Learning,
    Temporal Analysis and Network Intelligence**
    
    Enter a transaction ID to automatically generate a complete
    AML investigation.
    """
)

st.divider()


# ============================================================
# TRANSACTION SEARCH
# ============================================================

st.subheader("🔎 Investigate a Transaction")

st.markdown(
    "Enter a **Transaction ID** to analyze the transaction, "
    "its risk, associated accounts and transaction network."
)

transaction_ids = transactions_df["transaction_id"].tolist()

search_col, button_col = st.columns([5, 1])

with search_col:
    transaction_id = st.text_input(
        "Transaction ID",
        placeholder="Example: TXN-000650700",
        label_visibility="collapsed",
    )

with button_col:
    investigate = st.button(
        "🔍 Investigate",
        use_container_width=True,
        type="primary",
    )


# ============================================================
# INVESTIGATION
# ============================================================

if investigate:

    transaction_id = transaction_id.strip()

    if not transaction_id:
        st.warning("Please enter a Transaction ID.")

    elif transaction_id not in set(transaction_ids):
        st.error(
            f"Transaction ID **{transaction_id}** was not found "
            "in the demonstration dataset."
        )

        st.info(
            "Try one of the example Transaction IDs shown below."
        )

    else:

        # ----------------------------------------------------
        # Get transaction
        # ----------------------------------------------------

        transaction = transactions_df[
            transactions_df["transaction_id"] == transaction_id
        ].iloc[0]


        # ----------------------------------------------------
        # Find related investigation
        # ----------------------------------------------------

        investigation = None

        match_columns = [
            "Sender_account",
            "Receiver_account",
            "Amount",
        ]

        available_columns = [
            col
            for col in match_columns
            if col in investigation_df.columns
            and col in transactions_df.columns
        ]

        if available_columns:

            matches = investigation_df.copy()

            for column in available_columns:
                matches = matches[
                    matches[column].astype(str)
                    == str(transaction[column])
                ]

            if not matches.empty:
                investigation = matches.iloc[0]


        # ====================================================
        # CASE HEADER
        # ====================================================

        st.divider()

        st.subheader(
            f"🕵️ Investigation: {transaction_id}"
        )

        if investigation is not None:

            risk_score = float(
                investigation.get(
                    "risk_score",
                    transaction.get("risk_score", 0),
                )
            )

            severity = investigation.get(
                "alert_severity",
                transaction.get("risk_category", "UNKNOWN"),
            )

            ml_probability = float(
                investigation.get(
                    "ml_probability_percent",
                    transaction.get(
                        "ml_probability",
                        0,
                    ) * 100,
                )
            )

            amount = float(
                transaction["Amount"]
            )

        else:

            risk_score = float(
                transaction.get("risk_score", 0)
            )

            severity = transaction.get(
                "risk_category",
                "UNKNOWN",
            )

            ml_probability = float(
                transaction.get(
                    "ml_probability",
                    0,
                )
                * 100
            )

            amount = float(
                transaction["Amount"]
            )


        # ====================================================
        # RISK SUMMARY
        # ====================================================

        st.markdown("### 🚨 Risk Assessment")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Risk Score",
            f"{risk_score:.2f} / 100",
        )

        c2.metric(
            "Severity",
            severity,
        )

        c3.metric(
            "ML Probability",
            f"{ml_probability:.2f}%",
        )

        c4.metric(
            "Transaction Amount",
            f"{amount:,.2f}",
        )


        # ====================================================
        # TRANSACTION DETAILS
        # ====================================================

        st.markdown("### 💳 Transaction Details")

        details = pd.DataFrame(
            {
                "Field": [
                    "Transaction ID",
                    "Sender Account",
                    "Receiver Account",
                    "Amount",
                    "Payment Currency",
                    "Received Currency",
                    "Sender Bank Location",
                    "Receiver Bank Location",
                    "Payment Type",
                    "Date",
                    "Time",
                ],
                "Value": [
                    transaction_id,
                    transaction["Sender_account"],
                    transaction["Receiver_account"],
                    f"{amount:,.2f}",
                    transaction.get(
                        "Payment_currency",
                        "N/A",
                    ),
                    transaction.get(
                        "Received_currency",
                        "N/A",
                    ),
                    transaction.get(
                        "Sender_bank_location",
                        "N/A",
                    ),
                    transaction.get(
                        "Receiver_bank_location",
                        "N/A",
                    ),
                    transaction.get(
                        "Payment_type",
                        "N/A",
                    ),
                    transaction.get(
                        "Date",
                        "N/A",
                    ),
                    transaction.get(
                        "Time",
                        "N/A",
                    ),
                ],
            }
        )

        st.dataframe(
            details,
            use_container_width=True,
            hide_index=True,
        )


        # ====================================================
        # ML ANALYSIS
        # ====================================================

        st.markdown("### 🤖 Machine Learning Analysis")

        ml_col1, ml_col2 = st.columns(2)

        with ml_col1:

            st.metric(
                "Laundering Risk Probability",
                f"{ml_probability:.2f}%",
            )

            st.progress(
                min(
                    max(
                        ml_probability / 100,
                        0,
                    ),
                    1,
                )
            )

        with ml_col2:

            st.info(
                """
                The ML model evaluates transaction and historical
                behavioral features to estimate the likelihood that
                the transaction belongs to the laundering class.

                This probability is a **model score**, not proof that
                money laundering occurred.
                """
            )


        # ====================================================
        # RISK DRIVERS
        # ====================================================

        st.markdown("### ⚠️ Risk Drivers")

        risk_reasons = None

        if investigation is not None:

            risk_reasons = investigation.get(
                "risk_reasons",
                None,
            )

        if (
            risk_reasons is not None
            and str(risk_reasons) != "nan"
        ):

            st.warning(
                str(risk_reasons)
            )

        else:

            drivers = []

            if ml_probability >= 70:
                drivers.append(
                    "High ML laundering probability"
                )

            if amount > transactions_df["Amount"].quantile(0.95):
                drivers.append(
                    "High transaction amount"
                )

            if (
                transaction.get(
                    "Sender_bank_location"
                )
                != transaction.get(
                    "Receiver_bank_location"
                )
            ):
                drivers.append(
                    "Cross-border transaction"
                )

            if (
                transaction.get(
                    "Payment_currency"
                )
                != transaction.get(
                    "Received_currency"
                )
            ):
                drivers.append(
                    "Currency conversion/change"
                )

            if drivers:

                for driver in drivers:
                    st.write(f"• {driver}")

            else:

                st.success(
                    "No major individual risk drivers detected."
                )


        # ====================================================
        # ACCOUNT INTELLIGENCE
        # ====================================================

        st.markdown("### 🏦 Account Intelligence")

        sender_id = str(
            transaction["Sender_account"]
        )

        receiver_id = str(
            transaction["Receiver_account"]
        )

        sender_network = get_network_summary(
            sender_id,
            account_profiles,
        )

        receiver_network = get_network_summary(
            receiver_id,
            account_profiles,
        )


        def display_account_profile(
            title,
            account_id,
            profile,
        ):

            st.markdown(f"#### {title}")

            st.caption(
                f"Account: {account_id}"
            )

            if profile is None:

                st.info(
                    "No account network profile available."
                )

                return

            a1, a2, a3, a4 = st.columns(4)

            a1.metric(
                "Incoming Connections",
                profile["in_degree"],
            )

            a2.metric(
                "Outgoing Connections",
                profile["out_degree"],
            )

            a3.metric(
                "Total Connections",
                profile["total_degree"],
            )

            a4.metric(
                "Counterparties",
                profile["unique_counterparties"],
            )

            flags = []

            if profile["fan_in"]:
                flags.append("Fan-In")

            if profile["fan_out"]:
                flags.append("Fan-Out")

            if profile["layering_candidate"]:
                flags.append(
                    "Layering Candidate"
                )

            if profile["high_connectivity"]:
                flags.append(
                    "High Connectivity"
                )

            if profile["cycle_candidate"]:
                flags.append(
                    "Cycle Candidate"
                )

            if flags:

                st.warning(
                    "Network indicators: "
                    + ", ".join(flags)
                )

            else:

                st.success(
                    "No major network pattern indicators."
                )


        sender_col, receiver_col = st.columns(2)

        with sender_col:

            display_account_profile(
                "Sender Account",
                sender_id,
                sender_network,
            )

        with receiver_col:

            display_account_profile(
                "Receiver Account",
                receiver_id,
                receiver_network,
            )


        # ====================================================
        # NETWORK GRAPH
        # ====================================================

        st.markdown("### 🕸️ Transaction Network")

        network_edges = get_network_edges(
            transactions_df,
            sender_id,
            receiver_id,
            max_neighbors=8,
        )

        if not network_edges.empty:

            network_fig = create_network_graph(
                network_edges,
                sender_id,
                receiver_id,
            )

            st.plotly_chart(
                network_fig,
                use_container_width=True,
            )

            st.caption(
                "The network shows transactions involving the "
                "selected sender or receiver. Network indicators "
                "are screening signals and are not proof of "
                "money laundering."
            )

        else:

            st.info(
                "No surrounding transaction network was found."
            )


        # ====================================================
        # INVESTIGATION RECOMMENDATION
        # ====================================================

        st.markdown(
            "### 📋 Investigation Recommendation"
        )

        if investigation is not None:

            recommended_action = investigation.get(
                "recommended_action",
                None,
            )

            if (
                recommended_action is not None
                and str(recommended_action) != "nan"
            ):

                st.info(
                    str(recommended_action)
                )

            else:

                if risk_score > 75:

                    st.error(
                        "P1 — Immediate Review: "
                        "Perform detailed investigation of the "
                        "transaction and associated accounts."
                    )

                elif risk_score > 50:

                    st.warning(
                        "P2 — High Priority: "
                        "Review the transaction and associated "
                        "account/network activity."
                    )

                else:

                    st.info(
                        "P3 — Review: "
                        "Monitor the transaction and investigate "
                        "if additional suspicious activity appears."
                    )

        else:

            if risk_score > 75:

                st.error(
                    "P1 — Immediate Review"
                )

            elif risk_score > 50:

                st.warning(
                    "P2 — High Priority"
                )

            else:

                st.info(
                    "P3 — Review / Monitor"
                )


        # ====================================================
        # CASE CONCLUSION
        # ====================================================

        st.divider()

        st.markdown("### 📝 Investigation Summary")

        st.write(
            f"""
            Transaction **{transaction_id}** has been analyzed
            using machine learning, transaction characteristics,
            account behavior and network intelligence.

            The current composite risk score is
            **{risk_score:.2f}/100**, with a severity classification
            of **{severity}**.

            This assessment is intended to **prioritize investigation**.
            It does not independently establish that money laundering
            has occurred.
            """
        )


else:

    # ========================================================
    # INITIAL SCREEN
    # ========================================================

    st.info(
        "👆 Enter a Transaction ID above and click "
        "**Investigate** to begin."
    )

    st.markdown("### 💡 Example Transaction IDs")

    examples = transactions_df[
        [
            "transaction_id",
            "Sender_account",
            "Receiver_account",
            "Amount",
        ]
    ].head(10)

    st.dataframe(
        examples,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ANALYTICS
# ============================================================

with st.expander(
    "📊 Dataset Analytics",
    expanded=False,
):

    st.subheader(
        "Executive Dataset Overview"
    )

    total_transactions = len(
        transactions_df
    )

    risk_category_counts = (
        transactions_df[
            "risk_category"
        ]
        .value_counts()
        .reindex(
            [
                "LOW",
                "MEDIUM",
                "HIGH",
                "CRITICAL",
            ],
            fill_value=0,
        )
        .reset_index()
    )

    risk_category_counts.columns = [
        "Risk Category",
        "Transactions",
    ]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Transactions",
        f"{total_transactions:,}",
    )

    c2.metric(
        "Investigations",
        f"{len(investigation_df):,}",
    )

    c3.metric(
        "High/Critical Cases",
        f"{(
            transactions_df["risk_category"]
            .isin(["HIGH", "CRITICAL"])
        ).sum():,}",
    )

    chart = px.bar(
        risk_category_counts,
        x="Risk Category",
        y="Transactions",
        title="Risk Category Distribution",
    )

    st.plotly_chart(
        chart,
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AML Detection & Investigation System | "
    "SAML-D Synthetic Dataset | "
    "Machine Learning + Temporal Analysis + "
    "Network Intelligence"
)