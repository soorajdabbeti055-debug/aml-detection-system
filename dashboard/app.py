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

RISK_FILE = "data/processed/transactions_final_risk.parquet"
ALERT_FILE = "data/processed/aml_investigation_report.parquet"


st.set_page_config(
    page_title="AML Detection & Investigation System",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data
def load_data():
    risk_df = pd.read_parquet(RISK_FILE)
    alert_df = pd.read_parquet(ALERT_FILE)

    return risk_df, alert_df


risk_df, alert_df = load_data()
account_profiles = load_account_profiles()


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ AML Detection & Investigation System")

st.markdown(
    """
    **Machine Learning + Behavioral Analytics + Network Intelligence**

    This dashboard identifies transactions requiring AML investigation
    using explainable risk scoring and multiple transaction indicators.
    """
)

st.divider()


# ============================================================
# EXECUTIVE METRICS
# ============================================================

total_transactions = len(risk_df)
total_alerts = len(alert_df)

critical_alerts = (
    alert_df["alert_severity"] == "CRITICAL"
).sum()

high_alerts = (
    alert_df["alert_severity"] == "HIGH"
).sum()

medium_alerts = (
    alert_df["alert_severity"] == "MEDIUM"
).sum()

laundering_transactions = risk_df["Is_laundering"].sum()

alert_rate = (
    total_alerts / total_transactions * 100
)


col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Total Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "AML Alerts",
    f"{total_alerts:,}"
)

col3.metric(
    "Critical Alerts",
    f"{critical_alerts:,}"
)

col4.metric(
    "High Alerts",
    f"{high_alerts:,}"
)

col5.metric(
    "Alert Rate",
    f"{alert_rate:.3f}%"
)


st.divider()


# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.subheader("📊 Risk Analytics")

left, right = st.columns(2)


with left:

    category_counts = (
        risk_df["risk_category"]
        .value_counts()
        .reindex(
            ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            fill_value=0,
        )
        .reset_index()
    )

    category_counts.columns = [
        "Risk Category",
        "Transactions",
    ]

    fig = px.bar(
        category_counts,
        x="Risk Category",
        y="Transactions",
        title="Transaction Risk Categories",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


with right:

    fig = px.histogram(
        risk_df,
        x="risk_score",
        nbins=50,
        title="Risk Score Distribution",
    )

    fig.update_layout(
        xaxis_title="Risk Score",
        yaxis_title="Transactions",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# ALERT SEVERITY
# ============================================================

st.subheader("🚨 AML Alert Severity")

severity_counts = (
    alert_df["alert_severity"]
    .value_counts()
    .reindex(
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        fill_value=0,
    )
    .reset_index()
)

severity_counts.columns = [
    "Severity",
    "Alerts",
]

fig = px.pie(
    severity_counts,
    names="Severity",
    values="Alerts",
    title="Investigation Alert Distribution",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# TOP ALERTS
# ============================================================

st.subheader("🔎 Top AML Investigation Alerts")

top_alerts = alert_df[
    [
        "alert_id",
        "alert_severity",
        "risk_score",
        "ml_probability_percent",
        "Sender_account",
        "Receiver_account",
        "Amount",
        "transaction_context",
        "risk_reasons",
        "recommended_action",
    ]
].head(25)

st.dataframe(
    top_alerts,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# ALERT FILTER
# ============================================================

st.subheader("🔍 Investigation Search")

severity_filter = st.selectbox(
    "Select severity",
    [
        "ALL",
        "CRITICAL",
        "HIGH",
        "MEDIUM",
    ],
)

filtered_alerts = alert_df.copy()

if severity_filter != "ALL":

    filtered_alerts = filtered_alerts[
        filtered_alerts["alert_severity"]
        == severity_filter
    ]


st.write(
    f"Showing {len(filtered_alerts):,} alerts"
)


st.dataframe(
    filtered_alerts[
        [
            "alert_id",
            "alert_severity",
            "risk_score",
            "ml_probability_percent",
            "Sender_account",
            "Receiver_account",
            "Amount",
            "transaction_context",
            "risk_reasons",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# INDIVIDUAL INVESTIGATION
# ============================================================

st.subheader("🕵️ Transaction Investigation")

if len(alert_df) > 0:

    selected_alert = st.selectbox(
        "Select Alert ID",
        alert_df["alert_id"].tolist(),
    )

    case = alert_df[
        alert_df["alert_id"] == selected_alert
    ].iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Risk Score",
        f"{case['risk_score']:.2f}",
    )

    c2.metric(
        "Severity",
        case["alert_severity"],
    )

    c3.metric(
        "ML Probability",
        f"{case['ml_probability_percent']:.2f}%",
    )

    c4.metric(
        "Amount",
        f"{case['Amount']:,.2f}",
    )

    st.markdown("### Transaction Details")

    details = pd.DataFrame(
        {
            "Field": [
                "Alert ID",
                "Sender Account",
                "Receiver Account",
                "Payment Currency",
                "Received Currency",
                "Sender Bank Location",
                "Receiver Bank Location",
                "Payment Type",
                "Transaction Context",
                "Sender Activity",
                "Receiver Activity",
                "Temporal Indicator",
            ],
            "Value": [
                case["alert_id"],
                case["Sender_account"],
                case["Receiver_account"],
                case["Payment_currency"],
                case["Received_currency"],
                case["Sender_bank_location"],
                case["Receiver_bank_location"],
                case["Payment_type"],
                case["transaction_context"],
                case["sender_activity_level"],
                case["receiver_activity_level"],
                case["temporal_risk_indicator"],
            ],
        }
    )

    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### ⚠️ Why Was This Transaction Flagged?")

    st.warning(
        case["risk_reasons"]
    )

    st.markdown("### 📋 Recommended Investigation")

    st.info(
        case["recommended_action"]
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AML Detection & Investigation System | "
    "Synthetic SAML-D Dataset | "
    "Machine Learning + Explainable Risk Scoring"
)
# ============================================================
# NETWORK INTELLIGENCE
# ============================================================

st.subheader("🕸️ Network Intelligence")

if len(alert_df) > 0:

    selected_network_alert = st.selectbox(
        "Select an alert for network analysis",
        alert_df["alert_id"].tolist(),
        key="network_alert_selector",
    )

    network_case = alert_df[
        alert_df["alert_id"] == selected_network_alert
    ].iloc[0]

    sender_id = str(network_case["Sender_account"])
    receiver_id = str(network_case["Receiver_account"])

    sender_network = get_network_summary(
        sender_id,
        account_profiles,
    )

    receiver_network = get_network_summary(
        receiver_id,
        account_profiles,
    )

    st.markdown("### Sender Network Profile")

    if sender_network is not None:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Incoming Connections",
            sender_network["in_degree"],
        )

        c2.metric(
            "Outgoing Connections",
            sender_network["out_degree"],
        )

        c3.metric(
            "Total Connections",
            sender_network["total_degree"],
        )

        c4.metric(
            "Counterparties",
            sender_network["unique_counterparties"],
        )

        sender_flags = []

        if sender_network["fan_in"]:
            sender_flags.append("Fan-In")

        if sender_network["fan_out"]:
            sender_flags.append("Fan-Out")

        if sender_network["layering_candidate"]:
            sender_flags.append("Layering Candidate")

        if sender_network["high_connectivity"]:
            sender_flags.append("High Connectivity")

        if sender_network["cycle_candidate"]:
            sender_flags.append("Cycle Candidate")

        if sender_flags:
            st.warning(
                "Sender network indicators: "
                + ", ".join(sender_flags)
            )
        else:
            st.success(
                "No major network pattern indicators "
                "were detected for the sender."
            )

    else:
        st.info("No sender network profile found.")

    st.markdown("### Receiver Network Profile")

    if receiver_network is not None:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Incoming Connections",
            receiver_network["in_degree"],
        )

        c2.metric(
            "Outgoing Connections",
            receiver_network["out_degree"],
        )

        c3.metric(
            "Total Connections",
            receiver_network["total_degree"],
        )

        c4.metric(
            "Counterparties",
            receiver_network["unique_counterparties"],
        )

        receiver_flags = []

        if receiver_network["fan_in"]:
            receiver_flags.append("Fan-In")

        if receiver_network["fan_out"]:
            receiver_flags.append("Fan-Out")

        if receiver_network["layering_candidate"]:
            receiver_flags.append("Layering Candidate")

        if receiver_network["high_connectivity"]:
            receiver_flags.append("High Connectivity")

        if receiver_network["cycle_candidate"]:
            receiver_flags.append("Cycle Candidate")

        if receiver_flags:
            st.warning(
                "Receiver network indicators: "
                + ", ".join(receiver_flags)
            )
        else:
            st.success(
                "No major network pattern indicators "
                "were detected for the receiver."
            )

    else:
        st.info("No receiver network profile found.")

        st.markdown("### 🔗 Transaction Network")

network_edges = get_network_edges(
    risk_df,
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
        "The graph shows transactions directly involving "
        "the selected sender or receiver. Network patterns "
        "are screening indicators and are not proof of laundering."
    )

else:
    st.info(
        "No surrounding transaction network was found "
        "for this alert."
    )

    # ============================================================
# AML NETWORK PATTERN ANALYTICS
# ============================================================

st.subheader("📈 AML Network Pattern Analytics")

pattern_data = pd.DataFrame(
    {
        "AML Pattern": [
            "Fan-In",
            "Fan-Out",
            "Layering Candidate",
            "High Connectivity",
            "Cycle Candidate",
        ],
        "Accounts": [
            int(account_profiles["is_fan_in"].sum()),
            int(account_profiles["is_fan_out"].sum()),
            int(
                account_profiles[
                    "is_layering_candidate"
                ].sum()
            ),
            int(
                account_profiles[
                    "is_high_connectivity"
                ].sum()
            ),
            int(
                account_profiles[
                    "is_cycle_candidate"
                ].sum()
            ),
        ],
    }
)

# Pattern metrics
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Fan-In Accounts",
    f"{pattern_data.iloc[0]['Accounts']:,}",
)

c2.metric(
    "Fan-Out Accounts",
    f"{pattern_data.iloc[1]['Accounts']:,}",
)

c3.metric(
    "Layering Candidates",
    f"{pattern_data.iloc[2]['Accounts']:,}",
)

c4.metric(
    "High Connectivity",
    f"{pattern_data.iloc[3]['Accounts']:,}",
)

c5.metric(
    "Cycle Candidates",
    f"{pattern_data.iloc[4]['Accounts']:,}",
)

# Pattern chart
fig = px.bar(
    pattern_data,
    x="AML Pattern",
    y="Accounts",
    title="Network AML Pattern Distribution",
)

fig.update_layout(
    xaxis_title="Network Pattern",
    yaxis_title="Number of Accounts",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

st.markdown("### 🧠 What These Patterns Mean")

pattern_explanations = pd.DataFrame(
    {
        "Pattern": [
            "Fan-In",
            "Fan-Out",
            "Layering Candidate",
            "High Connectivity",
            "Cycle Candidate",
        ],
        "Meaning": [
            "An account receives funds from multiple accounts.",
            "An account sends funds to multiple accounts.",
            "An account has multiple incoming and outgoing connections.",
            "An account has unusually high network connectivity.",
            "An account belongs to a directed strongly connected component.",
        ],
        "AML Interpretation": [
            "May indicate aggregation of funds and should be investigated with transaction context.",
            "May indicate distribution or dispersal of funds across multiple accounts.",
            "May indicate movement of funds through multiple intermediaries.",
            "May represent an important hub in the transaction network.",
            "May indicate circular movement of funds and deserves contextual investigation.",
        ],
    }
)

st.dataframe(
    pattern_explanations,
    use_container_width=True,
    hide_index=True,
)

st.info(
    "Network patterns are screening indicators. "
    "They do not by themselves establish that money laundering occurred."
)