```python id="q7rm4a"
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Telecom Churn Prediction",
    layout="wide"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("📊 Telecom Churn Prediction Dashboard")
st.write("Machine Learning Based Customer Churn Prediction System")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("telecom_churn.csv")
    return df

df = load_data()

# ---------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Remove customerID column if present
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric if present
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'],
        errors='coerce'
    )

# Handle missing values
df.dropna(inplace=True)

# ---------------------------------------------------
# ENCODING
# ---------------------------------------------------

label_encoders = {}

for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# ---------------------------------------------------
# EDA
# ---------------------------------------------------

st.subheader("📈 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    if 'tenure' in df.columns:
        fig1 = px.histogram(
            df,
            x='tenure',
            title='Customer Tenure Distribution'
        )
        st.plotly_chart(fig1)

with col2:
    if 'MonthlyCharges' in df.columns:
        fig2 = px.histogram(
            df,
            x='MonthlyCharges',
            title='Monthly Charges Distribution'
        )
        st.plotly_chart(fig2)

if 'Churn' in df.columns:
    fig3 = px.pie(
        df,
        names='Churn',
        title='Churn Distribution'
    )
    st.plotly_chart(fig3)

# ---------------------------------------------------
# FEATURE SELECTION
# ---------------------------------------------------

X = df.drop('Churn', axis=1)
y = df['Churn']

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("🤖 Model Performance")

st.success(f"Model Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
st.write("### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.write(cm)

# Classification Report
st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))

# ---------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------

st.subheader("🔍 Predict Customer Churn")

input_data = {}

for column in X.columns:

    # Numerical columns
    if X[column].dtype in ['int64', 'float64']:

        value = st.number_input(
            f"Enter {column}",
            value=float(X[column].mean())
        )

        input_data[column] = value

    # Categorical columns
    else:

        options = label_encoders[column].classes_

        value = st.selectbox(
            f"Select {column}",
            options
        )

        encoded_value = label_encoders[column].transform([value])[0]

        input_data[column] = encoded_value

# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------

if st.button("Predict Churn"):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)

    prediction_proba = model.predict_proba(input_df)

    if prediction[0] == 1:

        st.error("⚠️ Customer is likely to Churn")

    else:

        st.success("✅ Customer is likely to Stay")

    st.write(
        f"Prediction Confidence: "
        f"{np.max(prediction_proba) * 100:.2f}%"
    )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")
st.markdown(
    "Developed using Streamlit, Scikit-learn, and Machine Learning"
)
```
