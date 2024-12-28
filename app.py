import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import streamlit as st


# Function to remove outliers
def remove_outliers(col):
    Q1, Q3 = col.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return lower, upper


# Caching the trained model
@st.cache_resource
def train_model(X_train, y_train, random_state):
    model = RandomForestRegressor(random_state=random_state, n_estimators=200)
    model.fit(X_train, y_train)
    return model


# Caching predictions
@st.cache_data
def get_prediction(_model, input_data):
    return model.predict(input_data)


# Streamlit App
st.title("Custom Housing Price Predictor")

# Upload dataset
uploaded_file = "no_outliers.csv"  # st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)

    # Remove outliers from the price column
    low, high = remove_outliers(data['price'])
    data['price'] = np.where(data['price'] > high, high, data['price'])

    # Predefined columns
    feature_columns = ["bath", "balcony", "bhk", "price_per_sqft", "new_total_sqft"]
    target_column = "price"
    location_column = "site_location"

    # Feature and target selection
    X = data[feature_columns]
    y = data[target_column]

    # Add location if applicable
    if location_column in data.columns:
        locations = data[location_column].unique()
    else:
        locations = None

    # Train-test split
    test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.5, step=0.05, value=0.2)
    random_state = st.sidebar.number_input("Random State", min_value=0, step=1, value=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train model
    model = train_model(X_train, y_train, random_state)

    # Calculate metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display metrics
    st.subheader("Model Performance")
    st.write("*Mean Absolute Error (MAE):*", mae)
    st.write("*R² Score:*", r2)

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

    # User input section for predictions
    st.subheader("Make a Prediction")

    # User inputs
    bath = st.number_input("Number of Bathrooms", min_value=0, step=1, value=1)
    balcony = st.number_input("Number of Balconies", min_value=0, step=1, value=1)
    bhk = st.number_input("Number of BHKs", min_value=1, step=1, value=2)
    price_per_sqft = st.number_input("Price per Square Foot", min_value=0.0, step=1.0, value=5000.0)
    new_total_sqft = st.number_input("Total Square Foot Area", min_value=0.0, step=1.0, value=1000.0)

    if locations is not None:
        location = st.selectbox("Select a Location", options=locations)
    else:
        location = st.text_input("Enter the Location")

    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        "bath": [bath],
        "balcony": [balcony],
        "bhk": [bhk],
        "price_per_sqft": [price_per_sqft],
        "new_total_sqft": [new_total_sqft],
    })

    # Make predictions and provide approximate actual prices
    if st.button("Predict"):
        prediction = get_prediction(model, input_data)[0]  # Cached prediction
        st.success(f"Predicted Price: {prediction:,.2f}")

        # Provide approximate actual prices
        if not data.empty:
            approx_prices = data[
                (data['site_location'] == location) &  # Same location
                (data['new_total_sqft'] <= new_total_sqft * 1.1) &  # +/- 10% sqft range
                (data['new_total_sqft'] >= new_total_sqft * 0.9) &  # +/- 10% sqft range
                (data['price_per_sqft'] <= price_per_sqft * 1.1) &  # +/- 10% price per sqft
                (data['price_per_sqft'] >= price_per_sqft * 0.9)    # +/- 10% price per sqft
            ]['price']

            if not approx_prices.empty:
                min_price = approx_prices.min()
                max_price = approx_prices.max()
                mean_price = approx_prices.mean()
                st.info(
                    f"Approximate Actual Price Range for Similar Properties: {min_price:,.2f} - {max_price:,.2f} "
                    f"(Average: {mean_price:,.2f})"
                )
            else:
                st.warning("No approximate price range available for similar properties in the dataset.")
