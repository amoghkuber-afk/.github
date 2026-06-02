```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Telecom Churn Prediction",
    layout="wide"
)

# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("📊 Telecom Churn Prediction Dashboard")
st.markdown("### Machine Learning Based Customer Churn Analysis")

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv("telecom_churn.csv")
    return data

df = load_data()

# -------------------------------------------------
# DISPLAY DATASET
# -------------------------------------------------

st.subheader("📁 Dataset Preview")
st.dataframe(df.head())

# -------------------------------------------------
# DATA CLEANING
# -------------------------------------------------

st.subheader("🧹 Data Preprocessing")

# Remove customerID if present
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Handle missing values
df.dropna(inplace=True)

# Encode categorical columns
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

st.success("Data preprocessing completed successfully.")

# -------------------------------------------------
# EDA SECTION
# -------------------------------------------------

st.subheader("📈 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(
        df,
        x='tenure',
        title='Customer Tenure Distribution'
    )
    st.plotly_chart(fig1)

with col2:
    fig2 = px.histogram(
        df,
        x='MonthlyCharges',
        title='Monthly Charges Distribution'
    )
    st.plotly_chart(fig2)

# Churn Distribution
fig3 = px.pie(
    df,
    names='Churn',
    title='Churn Distribution'
)

st.plotly_chart(fig3)

# -------------------------------------------------
# FEATURE SELECTION
# -------------------------------------------------

X = df.drop('Churn', axis=1)
y = df['Churn']

# -------------------------------------------------
# TRAIN TEST SPLIT
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------

model = RandomForestClassifier()

model.fit(X_train, y_train)

# -------------------------------------------------
# MODEL PREDICTION
# -------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# -------------------------------------------------
# MODEL PERFORMANCE
# -------------------------------------------------

st.subheader("🤖 Model Performance")

st.write(f"### Accuracy Score: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)

st.write("### Confusion Matrix")
st.write(cm)

st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))

# -------------------------------------------------
# CUSTOMER CHURN PREDICTION
# -------------------------------------------------

st.subheader("🔍 Predict Customer Churn")

st.markdown("Enter customer details below:")

input_data = {}

for column in X.columns:
    value = st.number_input(
        f"Enter {column}",
        value=float(X[column].mean())
    )
    input_data[column] = value

# Prediction Button
if st.button("Predict Churn"):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to Churn.")
    else:
        st.success("✅ Customer is likely to Stay.")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("---")
st.markdown("Developed using Streamlit and Machine Learning")
```
