import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

# Streamlit App
st.title("Custom Housing Price Predictor")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    
    # Display dataset
    st.subheader("Dataset Overview")
    st.write(data.head())

    feature_columns = ["bath", "balcony", "bhk", "price_per_sqft", "new_total_sqft"]
    target_column = "price"
    location_column = "site_location"

    # Select features and target
    # st.sidebar.header("Dataset Configuration")
    # target_column = st.sidebar.selectbox("Select Target Column (Price)", data.columns)
    # feature_columns = st.sidebar.multiselect(
    #     "Select Feature Columns (Predictors)", [col for col in data.columns if col != target_column]
    # )
    # location_column = st.sidebar.selectbox(
    #     "Select Location Column (Optional)", [None] + list(data.columns), index=0
    # )

    if target_column and feature_columns:
        X = data[feature_columns]
        y = data[target_column]

        # Add location if selected
        if location_column:
            X["Location"] = data[location_column]
            locations = X["Location"].unique()
        else:
            locations = None

        # One-hot encode the location column
        # encoder = OneHotEncoder(sparse_output=False)
        # location_encoded = encoder.fit_transform(X[[location_column]])
        # 
        # # Replace the location column with encoded values
        # location_columns = encoder.get_feature_names_out([location_column])
        # X_encoded = pd.DataFrame(location_encoded, columns=location_columns, index=X.index)
        # X = pd.concat([X.drop(columns=[location_column]), X_encoded], axis=1)


        # Test size slider
        test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.5, step=0.05, value=0.2)

        # Random state slider
        random_state = st.sidebar.number_input("Random State", min_value=0, step=1, value=42)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Drop location column for training (if applicable)
        if location_column:
            X_train_model = X_train.drop(columns=["Location"])
            X_test_model = X_test.drop(columns=["Location"])
        else:
            X_train_model, X_test_model = X_train, X_test

        # Train model
        model = LinearRegression()
        model.fit(X_train_model, y_train)
        y_pred = model.predict(X_test_model)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Display metrics
        st.subheader("Model Performance")
        st.write("**Mean Absolute Error (MAE):**", mae)
        st.write("**R² Score:**", r2)

        # Visualization
        st.subheader("Actual vs Predicted Prices")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.6, label="Predictions")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Prediction Line")
        ax.set_title("Actual vs Predicted House Prices")
        ax.set_xlabel("Actual Prices")
        ax.set_ylabel("Predicted Prices")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        # Prediction by location (if applicable)
        if location_column:
            st.subheader("Predict Prices for a Location")
            location_selected = st.selectbox("Select a Location", locations)
            location_data = X[X["Location"] == location_selected].drop(columns=["Location"])
            if not location_data.empty:
                location_prediction = model.predict(location_data)
                actual_prices = y[X["Location"] == location_selected]
                st.write(f"**Predicted Price for {location_selected}:** {location_prediction[0]:.2f}")
                st.write(f"**Actual Price for {location_selected}:** {actual_prices.values[0]:.2f}")
            else:
                st.write("No data available for the selected location.")