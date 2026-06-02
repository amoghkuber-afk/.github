
import streamlit as st
import pandas as pd
import nu```python id="m8s1ke"
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
# LOAD DATASET
# ---------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("telecom_churn.csv")
    return df

df = load_data()

# ---------------------------------------------------
# SHOW DATASET
# ---------------------------------------------------

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Remove customerID column if exists
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges column to numeric
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

# Remove missing values
df.dropna(inplace=True)

# ---------------------------------------------------
# ENCODE CATEGORICAL COLUMNS
# ---------------------------------------------------

label_encoders = {}

for column in df.columns:

    if df[column].dtype == "object":

        le = LabelEncoder()

        df[column] = le.fit_transform(df[column])

        label_encoders[column] = le

# ---------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]

# ---------------------------------------------------
# SPLIT DATA
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------------------------
# MODEL ACCURACY
# ---------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")

st.success(f"Accuracy: {accuracy * 100:.2f}%")

# ---------------------------------------------------
# SIMPLE CHARTS
# ---------------------------------------------------

st.subheader("Dataset Information")

st.write("Shape of Dataset:", df.shape)

st.write("Columns in Dataset:")
st.write(df.columns.tolist())

# ---------------------------------------------------
# CHURN DISTRIBUTION
# ---------------------------------------------------

st.subheader("Churn Distribution")

churn_count = df["Churn"].value_counts()

st.bar_chart(churn_count)

# ---------------------------------------------------
# CUSTOMER PREDICTION SECTION
# ---------------------------------------------------

st.subheader("Predict Customer Churn")

input_data = {}

for column in X.columns:

    value = st.number_input(
        f"Enter {column}",
        value=float(X[column].mean())
    )

    input_data[column] = value

# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------

if st.button("Predict"):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)

    if prediction[0] == 1:

        st.error("⚠️ Customer is likely to Churn")

    else:

        st.success("✅ Customer is likely to Stay")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")
st.markdown("Developed using Streamlit and Machine Learning")
```
mpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# ---------------------------------------------------
# PAGE CONFIGURATION
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
# LOAD DATASET
# ---------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv("telecom_churn.csv")
    return data

df = load_data()

# ---------------------------------------------------
# SHOW DATASET
# ---------------------------------------------------

st.subheader("Dataset Preview")

st.dataframe(df.head())

# ---------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Remove customerID column if available
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

# Remove missing values
df.dropna(inplace=True)

# ---------------------------------------------------
# LABEL ENCODING
# ---------------------------------------------------

label_encoders = {}

for column in df.columns:

    if df[column].dtype == "object":

        le = LabelEncoder()

        df[column] = le.fit_transform(df[column])

        label_encoders[column] = le

# ---------------------------------------------------
# EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------

st.subheader("📈 Exploratory Data Analysis")

col1, col2 = st.columns(2)

# Tenure Distribution
with col1:

    if "tenure" in df.columns:

        fig1 = px.histogram(
            df,
            x="tenure",
            title="Customer Tenure Distribution"
        )

        st.plotly_chart(fig1)

# Monthly Charges Distribution
with col2:

    if "MonthlyCharges" in df.columns:

        fig2 = px.histogram(
            df,
            x="MonthlyCharges",
            title="Monthly Charges Distribution"
        )

        st.plotly_chart(fig2)

# Churn Distribution
if "Churn" in df.columns:

    fig3 = px.pie(
        df,
        names="Churn",
        title="Churn Distribution"
    )

    st.plotly_chart(fig3)

# ---------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------

X = df.drop("Churn", axis=1)

y = df["Churn"]

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
# MODEL PREDICTION
# ---------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# ---------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------

st.subheader("🤖 Model Performance")

st.success(f"Model Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
st.write("### Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

st.write(cm)

# Classification Report
st.write("### Classification Report")

report = classification_report(y_test, y_pred)

st.text(report)

# ---------------------------------------------------
# CUSTOMER CHURN PREDICTION
# ---------------------------------------------------

st.subheader("🔍 Predict Customer Churn")

input_data = {}

for column in X.columns:

    # Numerical Columns
    if X[column].dtype == "int64" or X[column].dtype == "float64":

        value = st.number_input(
            f"Enter {column}",
            value=float(X[column].mean())
        )

        input_data[column] = value

    # Categorical Columns
    else:

        options = label_encoders[column].classes_

        selected_value = st.selectbox(
            f"Select {column}",
            options
        )

        encoded_value = label_encoders[column].transform(
            [selected_value]
        )[0]

        input_data[column] = encoded_value

# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------

if st.button("Predict Churn"):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)

    prediction_probability = model.predict_proba(input_df)

    if prediction[0] == 1:

        st.error("⚠️ Customer is likely to Churn")

    else:

        st.success("✅ Customer is likely to Stay")

    confidence = np.max(prediction_probability) * 100

    st.info(f"Prediction Confidence: {confidence:.2f}%")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.markdown(
    "Developed using Streamlit and Machine Learning"
)

