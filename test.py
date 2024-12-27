# # # # # # # import numpy as np
# # # # # # # import pandas as pd
# # # # # # # import matplotlib.pyplot as plt
# # # # # # # from sklearn.datasets import fetch_california_housing
# # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # from sklearn.linear_model import LinearRegression
# # # # # # # from sklearn.metrics import mean_absolute_error, r2_score
# # # # # # # import streamlit as st
# # # # # # # 
# # # # # # # # Load California housing data
# # # # # # # housing_data = fetch_california_housing()
# # # # # # # X = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
# # # # # # # y = pd.Series(housing_data.target)
# # # # # # # 
# # # # # # # # Streamlit App
# # # # # # # st.title("California Housing Price Predictor")
# # # # # # # st.sidebar.header("Model Configuration")
# # # # # # # 
# # # # # # # # Test size slider
# # # # # # # test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.5, step=0.05, value=0.2)
# # # # # # # 
# # # # # # # # Random state slider
# # # # # # # random_state = st.sidebar.number_input("Random State", min_value=0, step=1, value=42)
# # # # # # # 
# # # # # # # # Splitting data
# # # # # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
# # # # # # # 
# # # # # # # # Train the Linear Regression model
# # # # # # # model = LinearRegression()
# # # # # # # model.fit(X_train, y_train)
# # # # # # # y_pred = model.predict(X_test)
# # # # # # # 
# # # # # # # # Calculate metrics
# # # # # # # mae = mean_absolute_error(y_test, y_pred)
# # # # # # # r2 = r2_score(y_test, y_pred)
# # # # # # # 
# # # # # # # # Display metrics
# # # # # # # st.subheader("Model Performance")
# # # # # # # st.write("**Mean Absolute Error (MAE):**", mae)
# # # # # # # st.write("**R² Score:**", r2)
# # # # # # # 
# # # # # # # # Visualization
# # # # # # # st.subheader("Actual vs Predicted Prices")
# # # # # # # fig, ax = plt.subplots(figsize=(10, 6))
# # # # # # # ax.scatter(y_test, y_pred, alpha=0.6, label="Predictions")
# # # # # # # ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Prediction Line")
# # # # # # # ax.set_title("Actual vs Predicted House Prices")
# # # # # # # ax.set_xlabel("Actual Prices")
# # # # # # # ax.set_ylabel("Predicted Prices")
# # # # # # # ax.set_xlim([0, 5])
# # # # # # # ax.set_ylim([0, 5])
# # # # # # # ax.legend()
# # # # # # # ax.grid()
# # # # # # # st.pyplot(fig)
# # # # # # # 
# # # # # # # # Allow users to input data for prediction
# # # # # # # st.subheader("Make Predictions")
# # # # # # # input_data = {}
# # # # # # # for feature in housing_data.feature_names:
# # # # # # #     input_data[feature] = st.number_input(f"{feature}", value=float(X[feature].mean()))
# # # # # # # input_df = pd.DataFrame([input_data])
# # # # # # # 
# # # # # # # # Predict new data
# # # # # # # if st.button("Predict"):
# # # # # # #     prediction = model.predict(input_df)
# # # # # # #     st.write("**Predicted House Price:**", prediction[0])
# # # # # # 
# # # # # # 
# # # # # # 
# # # # # # import numpy as np
# # # # # # import pandas as pd
# # # # # # import matplotlib.pyplot as plt
# # # # # # from sklearn.datasets import fetch_california_housing
# # # # # # from sklearn.model_selection import train_test_split
# # # # # # from sklearn.linear_model import LinearRegression
# # # # # # from sklearn.metrics import mean_absolute_error, r2_score
# # # # # # import streamlit as st
# # # # # # 
# # # # # # # Load California housing data
# # # # # # housing_data = fetch_california_housing()
# # # # # # X = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
# # # # # # y = pd.Series(housing_data.target)
# # # # # # 
# # # # # # # Simulate locations (hypothetical mapping of ZIP codes to data rows)
# # # # # # np.random.seed(42)
# # # # # # locations = ["Location_" + str(i) for i in range(1, len(X) + 1)]
# # # # # # X["Location"] = locations
# # # # # # 
# # # # # # # Streamlit App
# # # # # # st.title("California Housing Price Predictor by Location")
# # # # # # st.sidebar.header("Model Configuration")
# # # # # # 
# # # # # # # Test size slider
# # # # # # test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.5, step=0.05, value=0.2)
# # # # # # 
# # # # # # # Random state slider
# # # # # # random_state = st.sidebar.number_input("Random State", min_value=0, step=1, value=42)
# # # # # # 
# # # # # # # Splitting data
# # # # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
# # # # # # 
# # # # # # # Train the Linear Regression model
# # # # # # model = LinearRegression()
# # # # # # model.fit(X_train.drop("Location", axis=1), y_train)
# # # # # # y_pred = model.predict(X_test.drop("Location", axis=1))
# # # # # # 
# # # # # # # Calculate metrics
# # # # # # mae = mean_absolute_error(y_test, y_pred)
# # # # # # r2 = r2_score(y_test, y_pred)
# # # # # # 
# # # # # # # Display metrics
# # # # # # st.subheader("Model Performance")
# # # # # # st.write("**Mean Absolute Error (MAE):**", mae)
# # # # # # st.write("**R² Score:**", r2)
# # # # # # 
# # # # # # # Visualization
# # # # # # st.subheader("Actual vs Predicted Prices")
# # # # # # fig, ax = plt.subplots(figsize=(10, 6))
# # # # # # ax.scatter(y_test, y_pred, alpha=0.6, label="Predictions")
# # # # # # ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Prediction Line")
# # # # # # ax.set_title("Actual vs Predicted House Prices")
# # # # # # ax.set_xlabel("Actual Prices")
# # # # # # ax.set_ylabel("Predicted Prices")
# # # # # # ax.set_xlim([0, 5])
# # # # # # ax.set_ylim([0, 5])
# # # # # # ax.legend()
# # # # # # ax.grid()
# # # # # # st.pyplot(fig)
# # # # # # 
# # # # # # # Allow user to input location for prediction
# # # # # # st.subheader("Predict Prices for a Location")
# # # # # # location_selected = st.selectbox("Select a Location", X["Location"].unique())
# # # # # # 
# # # # # # # Filter data for the selected location
# # # # # # location_data = X[X["Location"] == location_selected].drop("Location", axis=1)
# # # # # # 
# # # # # # if not location_data.empty:
# # # # # #     # Predict the price for the selected location
# # # # # #     location_prediction = model.predict(location_data)
# # # # # #     st.write(f"**Predicted House Price for {location_selected}:**", location_prediction[0])
# # # # # # else:
# # # # # #     st.write("No data available for the selected location.")
# # # # # 
# # # # # 
# # # # # 
# # # # # 
# # # # # import numpy as np
# # # # # import pandas as pd
# # # # # import matplotlib.pyplot as plt
# # # # # from sklearn.datasets import fetch_california_housing
# # # # # from sklearn.model_selection import train_test_split
# # # # # from sklearn.linear_model import LinearRegression
# # # # # from sklearn.metrics import mean_absolute_error, r2_score
# # # # # import streamlit as st
# # # # # 
# # # # # # Load California housing data
# # # # # housing_data = fetch_california_housing()
# # # # # X = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
# # # # # y = pd.Series(housing_data.target, name="Current Price")
# # # # # 
# # # # # # Simulate locations (hypothetical mapping of ZIP codes to data rows)
# # # # # np.random.seed(42)
# # # # # locations = ["Location_" + str(i) for i in range(1, len(X) + 1)]
# # # # # X["Location"] = locations
# # # # # 
# # # # # # Combine X and y into a single DataFrame for easier handling
# # # # # data = pd.concat([X, y], axis=1)
# # # # # 
# # # # # # Streamlit App
# # # # # st.title("California Housing Price Predictor by Location")
# # # # # st.sidebar.header("Model Configuration")
# # # # # 
# # # # # # Test size slider
# # # # # test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.5, step=0.05, value=0.2)
# # # # # 
# # # # # # Random state slider
# # # # # random_state = st.sidebar.number_input("Random State", min_value=0, step=1, value=42)
# # # # # 
# # # # # # Splitting data
# # # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
# # # # # 
# # # # # # Train the Linear Regression model
# # # # # model = LinearRegression()
# # # # # model.fit(X_train.drop("Location", axis=1), y_train)
# # # # # y_pred = model.predict(X_test.drop("Location", axis=1))
# # # # # 
# # # # # # Calculate metrics
# # # # # mae = mean_absolute_error(y_test, y_pred)
# # # # # r2 = r2_score(y_test, y_pred)
# # # # # 
# # # # # # Display metrics
# # # # # st.subheader("Model Performance")
# # # # # st.write("**Mean Absolute Error (MAE):**", mae)
# # # # # st.write("**R² Score:**", r2)
# # # # # 
# # # # # # Visualization
# # # # # st.subheader("Actual vs Predicted Prices")
# # # # # fig, ax = plt.subplots(figsize=(10, 6))
# # # # # ax.scatter(y_test, y_pred, alpha=0.6, label="Predictions")
# # # # # ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Prediction Line")
# # # # # ax.set_title("Actual vs Predicted House Prices")
# # # # # ax.set_xlabel("Actual Prices")
# # # # # ax.set_ylabel("Predicted Prices")
# # # # # ax.set_xlim([0, 5])
# # # # # ax.set_ylim([0, 5])
# # # # # ax.legend()
# # # # # ax.grid()
# # # # # st.pyplot(fig)
# # # # # 
# # # # # # Allow user to input location for prediction
# # # # # st.subheader("Predict Prices for a Location")
# # # # # location_selected = st.selectbox("Select a Location", data["Location"].unique())
# # # # # 
# # # # # # Filter data for the selected location
# # # # # location_data = data[data["Location"] == location_selected]
# # # # # if not location_data.empty:
# # # # #     location_features = location_data.drop(columns=["Location", "Current Price"])
# # # # #     predicted_price = model.predict(location_features)[0]
# # # # #     current_price = location_data["Current Price"].values[0]
# # # # #     
# # # # #     # Display prices
# # # # #     st.write(f"**Selected Location:** {location_selected}")
# # # # #     st.write(f"**Predicted Price:** {predicted_price:.2f}")
# # # # #     st.write(f"**Actual/Current Price:** {current_price:.2f}")
# # # # # else:
# # # # #     st.write("No data available for the selected location.")
# # # 
# # # 
# # # 
# # # 
# # # 
# # # import numpy as np
# # # import pandas as pd
# # # import matplotlib.pyplot as plt
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.linear_model import LinearRegression
# # # from sklearn.metrics import mean_absolute_error, r2_score
# # # import streamlit as st
# # # 
# # # # Streamlit App
# # # st.title("Custom Housing Price Predictor")
# # # 
# # # # Upload dataset
# # # uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
# # # if uploaded_file:
# # #     # Load dataset
# # #     data = pd.read_csv(uploaded_file)
# # #     
# # #     # Display dataset
# # #     st.subheader("Dataset Overview")
# # #     st.write(data.head())
# # # 
# # #     # Select features and target
# # #     st.sidebar.header("Dataset Configuration")
# # #     target_column = st.sidebar.selectbox("Select Target Column (Price)", data.columns)
# # #     feature_columns = st.sidebar.multiselect(
# # #         "Select Feature Columns (Predictors)", [col for col in data.columns if col != target_column]
# # #     )
# # #     location_column = st.sidebar.selectbox(
# # #         "Select Location Column (Optional)", [None] + list(data.columns), index=0
# # #     )
# # # 
# # #     if target_column and feature_columns:
# # #         X = data[feature_columns]
# # #         y = data[target_column]
# # # 
# # #         # Add location if selected
# # #         if location_column:
# # #             X["Location"] = data[location_column]
# # #             locations = X["Location"].unique()
# # #         else:
# # #             locations = None
# # # 
# # #         # Test size slider
# # #         test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.5, step=0.05, value=0.2)
# # # 
# # #         # Random state slider
# # #         random_state = st.sidebar.number_input("Random State", min_value=0, step=1, value=42)
# # # 
# # #         # Split data
# # #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
# # # 
# # #         # Drop location column for training (if applicable)
# # #         if location_column:
# # #             X_train_model = X_train.drop(columns=["Location"])
# # #             X_test_model = X_test.drop(columns=["Location"])
# # #         else:
# # #             X_train_model, X_test_model = X_train, X_test
# # # 
# # #         # Train model
# # #         model = LinearRegression()
# # #         model.fit(X_train_model, y_train)
# # #         y_pred = model.predict(X_test_model)
# # # 
# # #         # Calculate metrics
# # #         mae = mean_absolute_error(y_test, y_pred)
# # #         r2 = r2_score(y_test, y_pred)
# # # 
# # #         # Display metrics
# # #         st.subheader("Model Performance")
# # #         st.write("**Mean Absolute Error (MAE):**", mae)
# # #         st.write("**R² Score:**", r2)
# # # 
# # #         # Visualization
# # #         st.subheader("Actual vs Predicted Prices")
# # #         fig, ax = plt.subplots(figsize=(10, 6))
# # #         ax.scatter(y_test, y_pred, alpha=0.6, label="Predictions")
# # #         ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Prediction Line")
# # #         ax.set_title("Actual vs Predicted House Prices")
# # #         ax.set_xlabel("Actual Prices")
# # #         ax.set_ylabel("Predicted Prices")
# # #         ax.legend()
# # #         ax.grid()
# # #         st.pyplot(fig)
# # # 
# # #         # Prediction by location (if applicable)
# # #         if location_column:
# # #             st.subheader("Predict Prices for a Location")
# # #             location_selected = st.selectbox("Select a Location", locations)
# # #             location_data = X[X["Location"] == location_selected].drop(columns=["Location"])
# # #             if not location_data.empty:
# # #                 location_prediction = model.predict(location_data)
# # #                 actual_prices = y[X["Location"] == location_selected]
# # #                 st.write(f"**Predicted Price for {location_selected}:** {location_prediction[0]:.2f}")
# # #                 st.write(f"**Actual Price for {location_selected}:** {actual_prices.values[0]:.2f}")
# # #             else:
# # #                 st.write("No data available for the selected location.")
# # # 
# # # 
# # # 
# # # 
# # # 
# # # #
# # # # import numpy as np
# # # # import pandas as pd
# # # # import matplotlib.pyplot as plt
# # # # from sklearn.datasets import fetch_california_housing
# # # # from sklearn.model_selection import train_test_split
# # # # from sklearn.linear_model import LinearRegression
# # # # from sklearn.metrics import mean_absolute_error, r2_score
# # # # import streamlit as st
# # # # 
# # # # # Load California housing data
# # # # housing_data = fetch_california_housing()
# # # # X = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
# # # # X['Latitude'] = housing_data.data[:, 6]
# # # # X['Longitude'] = housing_data.data[:, 7]
# # # # y = pd.Series(housing_data.target)
# # # # 
# # # # # Streamlit App
# # # # st.title("California Housing Price Predictor (Location-Based)")
# # # # st.sidebar.header("Model Configuration")
# # # # 
# # # # # Location filters
# # # # st.sidebar.subheader("Location Filter")
# # # # latitude_range = st.sidebar.slider("Latitude Range", float(X['Latitude'].min()), float(X['Latitude'].max()), (32.0, 42.0))
# # # # longitude_range = st.sidebar.slider("Longitude Range", float(X['Longitude'].min()), float(X['Longitude'].max()), (-125.0, -114.0))
# # # # 
# # # # # Filter data based on location
# # # # filtered_data = X[(X['Latitude'] >= latitude_range[0]) & (X['Latitude'] <= latitude_range[1]) &
# # # #                   (X['Longitude'] >= longitude_range[0]) & (X['Longitude'] <= longitude_range[1])]
# # # # filtered_target = y.loc[filtered_data.index]
# # # # 
# # # # # Display filtered data size
# # # # st.sidebar.write(f"Filtered Data Size: {filtered_data.shape[0]} samples")
# # # # 
# # # # # Test size and random state
# # # # test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.5, step=0.05, value=0.2)
# # # # random_state = st.sidebar.number_input("Random State", min_value=0, step=1, value=42)
# # # # 
# # # # # Split the filtered data
# # # # X_train, X_test, y_train, y_test = train_test_split(filtered_data, filtered_target, test_size=test_size, random_state=random_state)
# # # # 
# # # # # Train the model
# # # # model = LinearRegression()
# # # # model.fit(X_train, y_train)
# # # # y_pred = model.predict(X_test)
# # # # 
# # # # # Calculate metrics
# # # # mae = mean_absolute_error(y_test, y_pred)
# # # # r2 = r2_score(y_test, y_pred)
# # # # 
# # # # # Display metrics
# # # # st.subheader("Model Performance")
# # # # st.write("**Mean Absolute Error (MAE):**", mae)
# # # # st.write("**R² Score:**", r2)
# # # # 
# # # # # Visualization
# # # # st.subheader("Actual vs Predicted Prices")
# # # # fig, ax = plt.subplots(figsize=(10, 6))
# # # # ax.scatter(y_test, y_pred, alpha=0.6, label="Predictions")
# # # # ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Prediction Line")
# # # # ax.set_title("Actual vs Predicted House Prices")
# # # # ax.set_xlabel("Actual Prices")
# # # # ax.set_ylabel("Predicted Prices")
# # # # ax.legend()
# # # # ax.grid()
# # # # st.pyplot(fig)
# # # # 
# # # # # Predict house price for specific location
# # # # st.subheader("Make Predictions Based on Location")
# # # # latitude = st.number_input("Latitude", value=34.0)
# # # # longitude = st.number_input("Longitude", value=-118.0)
# # # # input_data = {feature: st.number_input(feature, value=float(X[feature].mean())) for feature in housing_data.feature_names if feature not in ['Latitude', 'Longitude']}
# # # # input_data['Latitude'] = latitude
# # # # input_data['Longitude'] = longitude
# # # # input_df = pd.DataFrame([input_data])
# # # # 
# # # # # Predict new data
# # # # if st.button("Predict"):
# # # #     prediction = model.predict(input_df)
# # # #     st.write("**Predicted House Price:**", prediction[0])
# # # #     st.map(filtered_data.assign(Predicted_Price=model.predict(filtered_data)))
# # 
# # 
# # 
# # 
# # 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

# Predefined dataset and configuration
DATA_PATH = r"C:\Users\Raghav\Desktop\entire_dataset.csv"  # Replace with the actual dataset path
PREDICTORS = ["bath", "balcony", "bhk", "price_per_sqft", "new_total_sqft", "site_location"]  # Replace with your predictors
TARGET = "price"  # Replace with your target column name
LOCATION_COLUMN = "site_location"  # Replace with the location column name

# Streamlit App
st.title("Housing Price Predictor with Location")

# Load the dataset
data = pd.read_csv(DATA_PATH)

# Display dataset overview
st.subheader("Dataset Overview")
st.write(data.head())

# Extract features and target
X = data[PREDICTORS]
y = data[TARGET]

# One-hot encode the location column
encoder = OneHotEncoder(sparse_output=False)
location_encoded = encoder.fit_transform(X[[LOCATION_COLUMN]])

# Replace the location column with encoded values
location_columns = encoder.get_feature_names_out([LOCATION_COLUMN])
X_encoded = pd.DataFrame(location_encoded, columns=location_columns, index=X.index)
X = pd.concat([X.drop(columns=[LOCATION_COLUMN]), X_encoded], axis=1)

# Test size slider
st.sidebar.header("Model Configuration")
test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.5, step=0.05, value=0.2)
random_state = st.sidebar.number_input("Random State", min_value=0, step=1, value=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display metrics
st.subheader("Model Performance")
st.write("**Mean Absolute Error (MAE):**", mae)
st.write("**R² Score:**", r2)

# User interaction for prediction
st.subheader("Predict Prices")
input_data = {}

# Input predictors
for predictor in PREDICTORS:
    if predictor == LOCATION_COLUMN:
        selected_location = st.selectbox("Select Location", encoder.categories_[0])
        input_data[predictor] = selected_location
    else:
        input_data[predictor] = st.number_input(f"Enter {predictor}", value=float(data[predictor].mean()))

# Encode the selected location
location_encoded = encoder.transform([[input_data[LOCATION_COLUMN]]])
location_encoded_df = pd.DataFrame(location_encoded, columns=location_columns)

# Combine user input into a single feature row
input_features = pd.concat([pd.DataFrame([input_data]).drop(columns=[LOCATION_COLUMN]), location_encoded_df], axis=1)

# Predict the price
if st.button("Predict"):
    prediction = model.predict(input_features)
    st.write(f"**Predicted Price:** {prediction[0]:.2f}")

    # Show actual price for the selected location
    actual_price = data[data[LOCATION_COLUMN] == input_data[LOCATION_COLUMN]][TARGET].values[0]
    st.write(f"**Actual Price for {input_data[LOCATION_COLUMN]}:** {actual_price:.2f}")

# # 
# # 
# # 
# # 
# # 
# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LinearRegression
# # from sklearn.metrics import mean_absolute_error, r2_score
# # from sklearn.preprocessing import OneHotEncoder
# # import streamlit as st
# # 
# # # Predefined dataset and configuration
# # DATA_PATH = r"C:\Users\Raghav\Desktop\entire_dataset.csv"  # Replace with the actual dataset path
# # PREDICTORS = ["bath", "balcony", "bhk", "site_location"]  # Removed price_per_sqft and new_total_sqft
# # TARGET = "price"  # Replace with your target column name
# # LOCATION_COLUMN = "site_location"  # Replace with the location column name
# # 
# # # Streamlit App
# # st.title("Housing Price Predictor with Location")
# # 
# # # Load the dataset
# # data = pd.read_csv(DATA_PATH)
# # 
# # # Display dataset overview
# # st.subheader("Dataset Overview")
# # st.write(data.head())
# # 
# # # Extract features and target
# # X = data[PREDICTORS]
# # y = data[TARGET]
# # 
# # # One-hot encode the location column
# # encoder = OneHotEncoder(sparse_output=False)
# # location_encoded = encoder.fit_transform(X[[LOCATION_COLUMN]])
# # 
# # # Replace the location column with encoded values
# # location_columns = encoder.get_feature_names_out([LOCATION_COLUMN])
# # X_encoded = pd.DataFrame(location_encoded, columns=location_columns, index=X.index)
# # X = pd.concat([X.drop(columns=[LOCATION_COLUMN]), X_encoded], axis=1)
# # 
# # # Test size slider
# # st.sidebar.header("Model Configuration")
# # test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.5, step=0.05, value=0.2)
# # random_state = st.sidebar.number_input("Random State", min_value=0, step=1, value=42)
# # 
# # # Split data
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
# # 
# # # Train the model
# # model = LinearRegression()
# # model.fit(X_train, y_train)
# # y_pred = model.predict(X_test)
# # 
# # # Calculate metrics
# # mae = mean_absolute_error(y_test, y_pred)
# # r2 = r2_score(y_test, y_pred)
# # 
# # # Display metrics
# # st.subheader("Model Performance")
# # st.write("**Mean Absolute Error (MAE):**", mae)
# # st.write("**R² Score:**", r2)
# # 
# # # User interaction for prediction
# # st.subheader("Predict Prices")
# # input_data = {}
# # 
# # # Input predictors
# # for predictor in PREDICTORS:
# #     if predictor == LOCATION_COLUMN:
# #         selected_location = st.selectbox("Select Location", encoder.categories_[0])
# #         input_data[predictor] = selected_location
# #     else:
# #         input_data[predictor] = st.number_input(f"Enter {predictor}", value=float(data[predictor].mean()))
# # 
# # # Encode the selected location
# # location_encoded = encoder.transform([[input_data[LOCATION_COLUMN]]])
# # location_encoded_df = pd.DataFrame(location_encoded, columns=location_columns)
# # 
# # # Combine user input into a single feature row
# # input_features = pd.concat([pd.DataFrame([input_data]).drop(columns=[LOCATION_COLUMN]), location_encoded_df], axis=1)
# # 
# # # Predict the price
# # if st.button("Predict"):
# #     prediction = model.predict(input_features)
# #     st.write(f"**Predicted Price:** {prediction[0]:.2f}")
# 
# 
# 
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, r2_score
# from sklearn.preprocessing import OneHotEncoder
# import streamlit as st
# 
# # Predefined dataset and configuration
# DATA_PATH = r"C:\Users\Raghav\Desktop\bang.csv"  # Replace with the actual dataset path
# PREDICTORS = ["bath", "balcony", "bhk", "site_location"]  # Removed price_per_sqft and new_total_sqft
# TARGET = "price"  # Replace with your target column name
# LOCATION_COLUMN = "site_location"  # Replace with the location column name
# 
# # Streamlit App
# st.title("Housing Price Predictor with Location")
# 
# # Load the dataset
# data = pd.read_csv(DATA_PATH)
# 
# # Display dataset overview
# st.subheader("Dataset Overview")
# st.write(data.head())
# 
# # Extract features and target
# X = data[PREDICTORS]
# y = data[TARGET]
# 
# # One-hot encode the location column
# encoder = OneHotEncoder(sparse_output=False)
# location_encoded = encoder.fit_transform(X[[LOCATION_COLUMN]])
# 
# # Replace the location column with encoded values
# location_columns = encoder.get_feature_names_out([LOCATION_COLUMN])
# X_encoded = pd.DataFrame(location_encoded, columns=location_columns, index=X.index)
# X = pd.concat([X.drop(columns=[LOCATION_COLUMN]), X_encoded], axis=1)
# 
# # Test size slider
# st.sidebar.header("Model Configuration")
# test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.5, step=0.05, value=0.2)
# random_state = st.sidebar.number_input("Random State", min_value=0, step=1, value=42)
# 
# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
# 
# # Train the model
# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# 
# # Calculate metrics
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# 
# # Display metrics
# st.subheader("Model Performance")
# st.write("**Mean Absolute Error (MAE):**", mae)
# st.write("**R² Score:**", r2)
# 
# # User interaction for prediction
# st.subheader("Predict Prices")
# input_data = {}
# 
# # Input predictors
# for predictor in PREDICTORS:
#     if predictor == LOCATION_COLUMN:
#         selected_location = st.selectbox("Select Location", encoder.categories_[0])
#         input_data[predictor] = selected_location
#     else:
#         input_data[predictor] = st.number_input(f"Enter {predictor}", value=float(data[predictor].mean()))
# 
# # Encode the selected location
# location_encoded = encoder.transform([[input_data[LOCATION_COLUMN]]])
# location_encoded_df = pd.DataFrame(location_encoded, columns=location_columns)
# 
# # Combine user input into a single feature row
# input_features = pd.concat([pd.DataFrame([input_data]).drop(columns=[LOCATION_COLUMN]), location_encoded_df], axis=1)
# 
# # Predict the price
# if st.button("Predict"):
#     prediction = model.predict(input_features)
#     st.write(f"**Predicted Price:** {prediction[0]:.2f}")
#     
#     # Show actual price for the selected location
#     actual_price = data[data[LOCATION_COLUMN] == input_data[LOCATION_COLUMN]][TARGET].values[0]
#     st.write(f"**Actual Price for {input_data[LOCATION_COLUMN]}:** {actual_price:.2f}")
