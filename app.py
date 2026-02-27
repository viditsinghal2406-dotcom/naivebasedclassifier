import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score
)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression

st.title("ML Model App")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# Initialize session state
if "mode" not in st.session_state:
    st.session_state.mode = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ===============================
    # EXPLORATORY DATA ANALYSIS
    # ===============================
    st.markdown("## Exploratory Data Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write("Dataset Shape:", df.shape)

    st.subheader("Column Data Types")
    st.dataframe(df.dtypes)

    st.subheader("Missing Values")
    missing = df.isnull().sum()
    st.dataframe(missing[missing > 0])

    st.subheader("Summary Statistics (Numeric Columns)")
    st.dataframe(df.describe())

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if not numeric_df.empty:
        st.subheader("Correlation Matrix")
        st.dataframe(numeric_df.corr())

        st.subheader("Numeric Feature Distributions")
        for col in numeric_df.columns:
            fig, ax = plt.subplots()
            numeric_df[col].hist(ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

    # ===============================
    # CLEAN DATA (DROP UNNECESSARY)
    # ===============================
    drop_keywords = ["id", "name", "date"]
    filtered_columns = [
        col for col in df.columns
        if not any(keyword in col.lower() for keyword in drop_keywords)
    ]
    clean_df = df[filtered_columns]

    # ===============================
    # MODEL SELECTION
    # ===============================
    st.markdown("## Model Selection")

    col1, col2 = st.columns(2)

    if col1.button("Classification"):
        st.session_state.mode = "classification"

    if col2.button("Regression"):
        st.session_state.mode = "regression"

    if st.session_state.mode is not None:

        mode = st.session_state.mode
        st.subheader(f"{mode.capitalize()} Mode Selected")

        # Filter targets
        if mode == "classification":
            possible_targets = [
                col for col in clean_df.columns
                if clean_df[col].dtype == "object"
            ]
        else:
            possible_targets = [
                col for col in clean_df.columns
                if clean_df[col].dtype in ["int64", "float64"]
            ]

        if not possible_targets:
            st.error("No suitable target columns found.")
        else:
            target_column = st.selectbox("Select Target Column", possible_targets)

            feature_columns = [
                col for col in clean_df.columns
                if col != target_column and clean_df[col].dtype in ["int64", "float64"]
            ]

            selected_features = st.multiselect("Select Feature Columns", feature_columns)

            test_size_percent = st.slider("Test Size (%)", 10, 50, 20)
            test_size = test_size_percent / 100

            if st.button("Run Model"):

                if len(selected_features) == 0:
                    st.error("Please select at least one feature.")
                else:
                    X = clean_df[selected_features].copy()
                    y = clean_df[target_column].copy()

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )

                    # ==========================
                    # CLASSIFICATION
                    # ==========================
                    if mode == "classification":

                        le = LabelEncoder()
                        y_train = le.fit_transform(y_train)
                        y_test = le.transform(y_test)

                        model = GaussianNB()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        st.subheader("Results")
                        st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

                        cm = confusion_matrix(y_test, y_pred)
                        st.write("Confusion Matrix:")
                        st.dataframe(pd.DataFrame(cm))

                        st.write("Classification Report:")
                        st.text(classification_report(y_test, y_pred))

                    # ==========================
                    # REGRESSION
                    # ==========================
                    else:

                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        st.subheader("Results")
                        st.write("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 4))
                        st.write("R² Score:", round(r2_score(y_test, y_pred), 4))stre
