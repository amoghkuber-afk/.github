
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Telecom Churn Dashboard",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0b1220;
    color: white;
}

h1, h2, h3, h4 {
    color: white;
}

[data-testid="stMetric"] {
    background-color: #111827;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #1f2937;
}

.sidebar .sidebar-content {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# TITLE
# ------------------------------------------------

st.title("📊 TELECOM CHURN PREDICTION DASHBOARD")

st.markdown("### Customer Churn Analysis and Prediction")

# ------------------------------------------------
# SAMPLE DATA
# ------------------------------------------------

np.random.seed(42)

df = pd.DataFrame({
    "tenure": np.random.randint(1, 72, 500),
    "MonthlyCharges": np.random.randint(20, 120, 500),
    "Contract": np.random.choice(
        ["Month-to-month", "One year", "Two year"], 500),
    "InternetService": np.random.choice(
        ["Fiber optic", "DSL", "No internet"], 500),
    "PaymentMethod": np.random.choice(
        ["Electronic check", "Mailed check",
         "Bank transfer", "Credit card"], 500),
    "Churn": np.random.choice(["Yes", "No"], 500)
})

# ------------------------------------------------
# METRICS
# ------------------------------------------------

total_customers = len(df)
churned = len(df[df["Churn"] == "Yes"])
retained = len(df[df["Churn"] == "No"])
churn_rate = round((churned / total_customers) * 100, 2)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", total_customers)
col2.metric("Churned Customers", churned)
col3.metric("Retained Customers", retained)
col4.metric("Churn Rate", f"{churn_rate}%")

st.markdown("---")

# ------------------------------------------------
# CHARTS
# ------------------------------------------------

c1, c2 = st.columns(2)

# Churn Distribution Pie Chart
with c1:
    churn_counts = df["Churn"].value_counts()

    fig = px.pie(
        values=churn_counts.values,
        names=churn_counts.index,
        title="Churn Distribution",
        hole=0.5
    )

    st.plotly_chart(fig, use_container_width=True)

# Contract Type Chart
with c2:
    contract_fig = px.histogram(
        df,
        x="Contract",
        color="Churn",
        barmode="group",
        title="Churn by Contract Type"
    )

    st.plotly_chart(contract_fig, use_container_width=True)

# ------------------------------------------------
# SECOND ROW
# ------------------------------------------------

c3, c4 = st.columns(2)

# Internet Service
with c3:
    internet_fig = px.histogram(
        df,
        x="InternetService",
        color="Churn",
        barmode="group",
        title="Churn by Internet Service"
    )

    st.plotly_chart(internet_fig, use_container_width=True)

# Monthly Charges
with c4:
    charge_fig = px.scatter(
        df,
        x="tenure",
        y="MonthlyCharges",
        color="Churn",
        title="Monthly Charges vs Tenure"
    )

    st.plotly_chart(charge_fig, use_container_width=True)

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.title("Customer Details")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior = st.sidebar.selectbox(
    "Senior Citizen",
    ["Yes", "No"]
)

tenure = st.sidebar.slider(
    "Tenure",
    1, 72, 24
)

monthly = st.sidebar.slider(
    "Monthly Charges",
    20, 120, 70
)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check",
     "Mailed check",
     "Bank transfer",
     "Credit card"]
)

# ------------------------------------------------
# PREDICTION BUTTON
# ------------------------------------------------

if st.sidebar.button("Predict Churn"):

    if monthly > 80 and contract == "Month-to-month":
        prediction = "YES"
        color = "red"
    else:
        prediction = "NO"
        color = "green"

    st.sidebar.markdown(f"""
    <h2 style='color:{color};'>
    Churn Prediction: {prediction}
    </h2>
    """, unsafe_allow_html=True)

# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.markdown("---")

st.markdown(
    "### Telecom Churn Prediction System | Machine Learning Project"
)

