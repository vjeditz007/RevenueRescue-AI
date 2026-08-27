import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RevenueRescue AI",
    page_icon="💰",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("RevenueRescue AI")
st.caption("AI-powered autonomous revenue recovery decision system")

st.info(
    "Detect → Predict → Recommend → Recover → Measure Revenue"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("data.csv")


# ============================================================
# TRAIN MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def train_model(df):

    failed_df = df[df["failed"] == 1].copy()

    features = [
        "amount",
        "previous_successes",
        "previous_failures",
        "customer_age_days",
        "is_subscription",
        "is_active",
        "failure_reason"
    ]

    X = failed_df[features]
    y = failed_df["recovered"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    numeric_features = [
        "amount",
        "previous_successes",
        "previous_failures",
        "customer_age_days",
        "is_subscription",
        "is_active"
    ]

    categorical_features = [
        "failure_reason"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight="balanced"
                )
            )
        ]
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        )
    }

    return model, metrics


# ============================================================
# ACTION RECOMMENDATION ENGINE
# ============================================================

def recommend_action(row):

    reason = row["failure_reason"]
    probability = row["recovery_probability"]

    if probability < 0.35:
        return "Manual Review"

    if reason == "expired_method":
        return "Payment Method Update"

    if reason == "checkout_abandoned":
        return "Checkout Reminder"

    if reason in [
        "temporary_failure",
        "technical_error"
    ]:
        return "Smart Retry"

    if reason == "insufficient_funds":
        return "Scheduled Retry"

    if reason == "authentication_failure":
        return "Re-authentication"

    return "Manual Review"


# ============================================================
# LOAD + PREDICT
# ============================================================

df = load_data()

model, metrics = train_model(df)

failed = df[
    df["failed"] == 1
].copy()


features = [
    "amount",
    "previous_successes",
    "previous_failures",
    "customer_age_days",
    "is_subscription",
    "is_active",
    "failure_reason"
]


failed["recovery_probability"] = model.predict_proba(
    failed[features]
)[:, 1]


failed["recovery_probability_pct"] = (
    failed["recovery_probability"] * 100
).round(1)


failed["ai_action"] = failed.apply(
    recommend_action,
    axis=1
)


# ============================================================
# BUSINESS METRICS
# ============================================================

total_failed_revenue = failed[
    "amount"
].sum()


potentially_recoverable = failed.loc[
    failed["recovery_probability"] >= 0.50,
    "amount"
].sum()


expected_recovery = (
    failed["amount"]
    * failed["recovery_probability"]
).sum()


expected_recovery_rate = (
    expected_recovery / total_failed_revenue
    if total_failed_revenue > 0
    else 0
)


# ============================================================
# TOP KPI DASHBOARD
# ============================================================

st.header("Revenue Recovery Overview")


c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "Total Payments",
    f"{len(df):,}"
)


c2.metric(
    "Failed Payments",
    f"{len(failed):,}"
)


c3.metric(
    "Revenue at Risk",
    f"₹{total_failed_revenue:,.0f}"
)


c4.metric(
    "Expected Recovery",
    f"₹{expected_recovery:,.0f}"
)


c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "Recoverable Revenue",
    f"₹{potentially_recoverable:,.0f}"
)


c2.metric(
    "Expected Recovery Rate",
    f"{expected_recovery_rate * 100:.1f}%"
)


c3.metric(
    "Precision",
    f"{metrics['precision'] * 100:.1f}%"
)


c4.metric(
    "Recall",
    f"{metrics['recall'] * 100:.1f}%"
)


st.divider()


# ============================================================
# AI RECOVERY CAMPAIGN
# ============================================================

st.header("AI Recovery Campaign")

st.write(
    "The AI identifies high-confidence failed payments and "
    "simulates the recovery strategy most suitable for each payment."
)


campaign_col1, campaign_col2 = st.columns(
    [2, 1]
)


with campaign_col1:

    campaign_threshold = st.slider(
        "Minimum recovery confidence",
        min_value=0.50,
        max_value=0.90,
        value=0.70,
        step=0.05
    )


with campaign_col2:

    st.write("")

    run_campaign = st.button(
        "RUN AI RECOVERY",
        type="primary",
        use_container_width=True
    )


# Select eligible payments

eligible_payments = failed[
    failed["recovery_probability"]
    >= campaign_threshold
].copy()


eligible_payments[
    "expected_recovery"
] = (
    eligible_payments["amount"]
    * eligible_payments["recovery_probability"]
)


st.write(
    f"Eligible failed payments: "
    f"**{len(eligible_payments):,}**"
)


st.write(
    f"Expected recoverable revenue from this campaign: "
    f"**₹{eligible_payments['expected_recovery'].sum():,.0f}**"
)


# ============================================================
# RUN CAMPAIGN
# ============================================================

if run_campaign:

    # Fixed seed makes the demonstration reproducible.
    np.random.seed(42)

    eligible_payments[
        "recovered"
    ] = (
        np.random.random(
            len(eligible_payments)
        )
        < eligible_payments[
            "recovery_probability"
        ]
    ).astype(int)


    eligible_payments[
        "recovered_amount"
    ] = (
        eligible_payments["amount"]
        * eligible_payments["recovered"]
    )


    payments_processed = len(
        eligible_payments
    )


    payments_recovered = int(
        eligible_payments[
            "recovered"
        ].sum()
    )


    revenue_recovered = (
        eligible_payments[
            "recovered_amount"
        ].sum()
    )


    campaign_recovery_rate = (
        payments_recovered
        / payments_processed
        if payments_processed > 0
        else 0
    )


    # Save results for display

    st.session_state[
        "campaign_results"
    ] = eligible_payments.copy()


# ============================================================
# DISPLAY CAMPAIGN RESULTS
# ============================================================

if "campaign_results" in st.session_state:

    results = st.session_state[
        "campaign_results"
    ]


    payments_processed = len(
        results
    )


    payments_recovered = int(
        results[
            "recovered"
        ].sum()
    )


    revenue_recovered = (
        results[
            "recovered_amount"
        ].sum()
    )


    campaign_recovery_rate = (
        payments_recovered
        / payments_processed
        if payments_processed > 0
        else 0
    )


    st.success(
        "AI Recovery Campaign simulation completed successfully."
    )


    r1, r2, r3, r4 = st.columns(4)


    r1.metric(
        "Payments Processed",
        f"{payments_processed:,}"
    )


    r2.metric(
        "Payments Recovered",
        f"{payments_recovered:,}"
    )


    r3.metric(
        "Simulated Revenue Recovered",
        f"₹{revenue_recovered:,.0f}"
    )


    r4.metric(
        "Campaign Recovery Rate",
        f"{campaign_recovery_rate * 100:.1f}%"
    )


    st.subheader(
        "Recovery Campaign Results"
    )


    results_display = results[
        [
            "transaction_id",
            "customer_id",
            "amount",
            "failure_reason",
            "recovery_probability",
            "ai_action",
            "recovered_amount"
        ]
    ].copy()


    results_display[
        "status"
    ] = np.where(
        results_display[
            "recovered_amount"
        ] > 0,
        "RECOVERED",
        "NOT RECOVERED"
    )


    results_display[
        "recovery_probability"
    ] = (
        results_display[
            "recovery_probability"
        ]
        * 100
    ).round(1).astype(str) + "%"


    st.dataframe(
        results_display,
        use_container_width=True,
        hide_index=True
    )


    # Download report

    report_csv = (
        results_display
        .to_csv(index=False)
        .encode("utf-8")
    )


    st.download_button(
        "Download Recovery Report",
        report_csv,
        "revenue_recovery_report.csv",
        "text/csv"
    )


else:

    st.info(
        "Select the confidence threshold and click "
        "'RUN AI RECOVERY' to simulate a recovery campaign."
    )


st.divider()


# ============================================================
# FAILURE ANALYSIS
# ============================================================

st.header("Recovery Intelligence")


left, right = st.columns(2)


with left:

    reason_counts = (
        failed[
            "failure_reason"
        ]
        .value_counts()
        .reset_index()
    )


    reason_counts.columns = [
        "failure_reason",
        "count"
    ]


    fig = px.bar(
        reason_counts,
        x="failure_reason",
        y="count",
        title="Failed Payments by Reason"
    )


    fig.update_layout(
        xaxis_tickangle=-30
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


with right:

    action_counts = (
        failed[
            "ai_action"
        ]
        .value_counts()
        .reset_index()
    )


    action_counts.columns = [
        "action",
        "count"
    ]


    fig = px.pie(
        action_counts,
        names="action",
        values="count",
        title="AI Recommended Recovery Actions"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# AI RECOVERY ACTION CENTER
# ============================================================

st.header(
    "AI Recovery Action Center"
)


min_probability = st.slider(
    "Show payments with recovery probability above",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05
)


view = failed[
    failed["recovery_probability"]
    >= min_probability
].copy()


view[
    "recovery_probability"
] = (
    view[
        "recovery_probability"
    ]
    * 100
).round(1).astype(str) + "%"


view[
    "amount"
] = view[
    "amount"
].round(2)


st.dataframe(
    view[
        [
            "transaction_id",
            "customer_id",
            "amount",
            "failure_reason",
            "recovery_probability",
            "ai_action"
        ]
    ]
    .sort_values(
        "recovery_probability",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)
# ============================================================
# AI RECOVERY STRATEGY ENGINE
# ============================================================

st.header("AI Recovery Strategy")

st.write(
    "The strategy engine converts the ML prediction into an "
    "actionable recovery plan for the merchant."
)

strategy_transaction = st.selectbox(
    "Choose a payment for AI strategy analysis",
    failed["transaction_id"].head(500).tolist(),
    key="strategy_transaction"
)

strategy_row = failed[
    failed["transaction_id"] == strategy_transaction
].iloc[0]

strategy_probability = strategy_row["recovery_probability"]
strategy_reason = strategy_row["failure_reason"]
strategy_amount = strategy_row["amount"]
strategy_action = strategy_row["ai_action"]


# Determine recommended timing
if strategy_reason == "temporary_failure":
    strategy_timing = "Retry after 2 hours"
elif strategy_reason == "technical_error":
    strategy_timing = "Retry after 1 hour"
elif strategy_reason == "expired_method":
    strategy_timing = "Request update immediately"
elif strategy_reason == "checkout_abandoned":
    strategy_timing = "Send reminder within 30 minutes"
elif strategy_reason == "insufficient_funds":
    strategy_timing = "Schedule retry for the next day"
elif strategy_reason == "authentication_failure":
    strategy_timing = "Request re-authentication immediately"
else:
    strategy_timing = "Manual review"


# Determine recommended channel
if strategy_reason == "checkout_abandoned":
    strategy_channel = "Email / In-app notification"
elif strategy_reason == "expired_method":
    strategy_channel = "Payment update notification"
elif strategy_reason == "authentication_failure":
    strategy_channel = "Authentication prompt"
elif strategy_reason in [
    "temporary_failure",
    "technical_error",
    "insufficient_funds"
]:
    strategy_channel = "Payment retry workflow"
else:
    strategy_channel = "Merchant operations"


# Estimate financial value
strategy_expected_value = (
    strategy_amount * strategy_probability
)


# Determine confidence level
if strategy_probability >= 0.80:
    strategy_confidence = "HIGH"
elif strategy_probability >= 0.60:
    strategy_confidence = "MEDIUM"
else:
    strategy_confidence = "LOW"


s1, s2, s3, s4 = st.columns(4)

s1.metric(
    "Recovery Probability",
    f"{strategy_probability * 100:.1f}%"
)

s2.metric(
    "Expected Value",
    f"₹{strategy_expected_value:,.2f}"
)

s3.metric(
    "Confidence",
    strategy_confidence
)

s4.metric(
    "Recommended Action",
    strategy_action
)


st.subheader("Recommended Recovery Plan")

st.write(
    f"**Why:** The payment failed because of "
    f"**{strategy_reason.replace('_', ' ')}**."
)

st.write(
    f"**Action:** {strategy_action}"
)

st.write(
    f"**When:** {strategy_timing}"
)

st.write(
    f"**Channel:** {strategy_channel}"
)

st.write(
    f"**Expected financial impact:** "
    f"₹{strategy_expected_value:,.2f}"
)


# Recovery message suggestion
if strategy_reason == "checkout_abandoned":

    message = (
        "Reminder: Your payment was not completed. "
        "Please return to checkout to complete your purchase."
    )

elif strategy_reason == "expired_method":

    message = (
        "Your payment method needs to be updated. "
        "Please update your payment details to continue."
    )

elif strategy_reason == "temporary_failure":

    message = (
        "We experienced a temporary payment issue. "
        "Your payment will be retried automatically."
    )

elif strategy_reason == "insufficient_funds":

    message = (
        "Your recent payment could not be completed. "
        "A retry will be scheduled for a later time."
    )

elif strategy_reason == "authentication_failure":

    message = (
        "Additional authentication is required to complete "
        "your payment securely."
    )

else:

    message = (
        "Your payment requires additional attention. "
        "Please review the payment details."
    )


st.subheader("Suggested Customer Communication")

st.info(message)


st.success(
    f"AI strategy selected **{strategy_action}** with "
    f"{strategy_probability * 100:.1f}% recovery probability "
    f"and an estimated recovery value of "
    f"₹{strategy_expected_value:,.2f}."
)

# ============================================================
# AI PAYMENT INVESTIGATOR
# ============================================================

st.header(
    "AI Payment Investigator"
)


selected_transaction = st.selectbox(
    "Select a failed transaction",
    failed[
        "transaction_id"
    ].head(500).tolist()
)


selected_row = failed[
    failed[
        "transaction_id"
    ] == selected_transaction
].iloc[0]


probability = selected_row[
    "recovery_probability"
]


reason = selected_row[
    "failure_reason"
]


a, b, c = st.columns(3)


a.metric(
    "Amount",
    f"₹{selected_row['amount']:,.2f}"
)


b.metric(
    "Recovery Probability",
    f"{probability * 100:.1f}%"
)


c.metric(
    "AI Action",
    selected_row["ai_action"]
)


st.subheader(
    "Why did the AI recommend this action?"
)


explanations = []


if selected_row[
    "previous_successes"
] >= 5:

    explanations.append(
        f"The customer has "
        f"{int(selected_row['previous_successes'])} "
        f"previous successful payments."
    )


if selected_row[
    "previous_failures"
] <= 2:

    explanations.append(
        "The customer's historical failure "
        "count is relatively low."
    )


if reason in [
    "temporary_failure",
    "technical_error"
]:

    explanations.append(
        f"The failure type "
        f"({reason.replace('_', ' ')}) "
        f"is suitable for a retry strategy."
    )


elif reason == "expired_method":

    explanations.append(
        "The payment method appears to require "
        "an update before another payment attempt."
    )


elif reason == "checkout_abandoned":

    explanations.append(
        "The customer started the payment journey "
        "but did not complete checkout."
    )


elif reason == "insufficient_funds":

    explanations.append(
        "The failure is consistent with insufficient "
        "funds, so a scheduled retry is safer."
    )


elif reason == "authentication_failure":

    explanations.append(
        "The payment may require another "
        "authentication attempt."
    )


if probability >= 0.75:

    explanations.append(
        "The ML model predicts a high probability "
        "of successful recovery."
    )


elif probability < 0.35:

    explanations.append(
        "The ML model predicts a low probability "
        "of recovery, so manual review is recommended."
    )


for explanation in explanations:

    st.write(
        "• " + explanation
    )


estimated_value = (
    selected_row["amount"]
    * probability
)


st.success(
    f"Recommended next step: "
    f"**{selected_row['ai_action']}**. "
    f"Estimated recoverable value: "
    f"**₹{estimated_value:,.2f}**."
)


# ============================================================
# MODEL EVALUATION
# ============================================================

st.header(
    "Model Evaluation"
)


m1, m2, m3, m4 = st.columns(4)


m1.metric(
    "Precision",
    f"{metrics['precision'] * 100:.1f}%"
)


m2.metric(
    "Recall",
    f"{metrics['recall'] * 100:.1f}%"
)


m3.metric(
    "F1 Score",
    f"{metrics['f1'] * 100:.1f}%"
)


m4.metric(
    "ROC-AUC",
    f"{metrics['roc_auc'] * 100:.1f}%"
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Prototype uses synthetic payment-recovery data for demonstration. "
    "No real customer or payment data is used."
)

