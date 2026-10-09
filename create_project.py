from pathlib import Path

files = {
    "requirements.txt": """
pandas
scikit-learn
streamlit
matplotlib
""".strip(),

    "train_model.py": r'''
import pickle
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "housing.csv"
MODEL_FILE = BASE_DIR / "house_price_model.pkl"

# Download a real housing dataset if it is not already present.
# The dataset contains California housing data.
DATA_URL = (
    "https://raw.githubusercontent.com/ageron/handson-ml2/"
    "master/datasets/housing/housing.csv"
)

if not DATA_FILE.exists():
    print("Downloading the housing dataset...")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)

data = pd.read_csv(DATA_FILE)

# Remove records without a target price.
data = data.dropna(subset=["median_house_value"])

features = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity",
]

target = "median_house_value"

X = data[features]
y = data[target]

numeric_features = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

categorical_features = ["ocean_proximity"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression()),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training the machine learning model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

with open(MODEL_FILE, "wb") as file:
    pickle.dump(model, file)

metrics = {
    "mae": float(mae),
    "r2": float(r2),
    "test_rows": int(len(X_test)),
    "train_rows": int(len(X_train)),
    "features": features,
}

with open(BASE_DIR / "model_metrics.pkl", "wb") as file:
    pickle.dump(metrics, file)

print("\nModel training completed!")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"R2 Score: {r2:.3f}")
print(f"Saved model: {MODEL_FILE}")
''',

    "app.py": r'''
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "housing.csv"
MODEL_FILE = BASE_DIR / "house_price_model.pkl"
METRICS_FILE = BASE_DIR / "model_metrics.pkl"

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="wide",
)

st.title("🏡 House Price Prediction")
st.caption(
    "Machine Learning Dashboard | Estimate housing values "
    "using property and location information."
)

if not MODEL_FILE.exists() or not METRICS_FILE.exists():
    st.error("The model has not been trained yet.")
    st.info("Run this command in the terminal: python train_model.py")
    st.stop()

with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)

with open(METRICS_FILE, "rb") as file:
    metrics = pickle.load(file)

data = pd.read_csv(DATA_FILE)

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Predict Price", "Dataset", "Model Performance"],
)

if page == "Dashboard":
    st.subheader("Project Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Housing Records", f"{len(data):,}")
    c2.metric("Model", "Linear Regression")
    c3.metric("Input Features", len(metrics["features"]))

    st.markdown("### Housing Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True)

    st.markdown("### House Value Distribution")
    fig, ax = plt.subplots()
    ax.hist(data["median_house_value"].dropna(), bins=35)
    ax.set_xlabel("Median House Value (USD)")
    ax.set_ylabel("Number of Records")
    ax.set_title("Distribution of Housing Values")
    st.pyplot(fig)
    plt.close(fig)

elif page == "Predict Price":
    st.subheader("Enter Property Details")

    st.info(
        "This model was trained on California housing data. "
        "Predictions represent estimated median house values in USD, "
        "not verified prices for individual homes."
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            longitude = st.number_input(
                "Longitude", min_value=-125.0, max_value=-113.0,
                value=-122.0, step=0.1
            )
            latitude = st.number_input(
                "Latitude", min_value=32.0, max_value=42.0,
                value=37.0, step=0.1
            )
            house_age = st.number_input(
                "Median House Age (years)", min_value=1.0,
                max_value=60.0, value=25.0, step=1.0
            )
            median_income = st.number_input(
                "Median Household Income (in $10,000 units)",
                min_value=0.5, max_value=15.0, value=4.0, step=0.1
            )

        with col2:
            total_rooms = st.number_input(
                "Total Rooms", min_value=1, max_value=100000,
                value=2000, step=100
            )
            total_bedrooms = st.number_input(
                "Total Bedrooms", min_value=1, max_value=50000,
                value=400, step=10
            )
            population = st.number_input(
                "Population", min_value=1, max_value=100000,
                value=1000, step=50
            )
            households = st.number_input(
                "Households", min_value=1, max_value=50000,
                value=350, step=10
            )

        ocean = st.selectbox(
            "Location Category",
            ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"],
        )

        submitted = st.form_submit_button(
            "Predict House Value",
            use_container_width=True,
        )

    if submitted:
        input_data = pd.DataFrame([{
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": house_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean,
        }])

        prediction = float(model.predict(input_data)[0])

        st.markdown("### Prediction Result")
        st.metric("Estimated Median House Value", f"${prediction:,.0f}")
        st.caption(
            "This is a model estimate, not a guaranteed selling price. "
            "It uses California data and is not suitable for directly "
            "estimating property prices in India."
        )

elif page == "Dataset":
    st.subheader("Housing Dataset")

    st.write("Explore the data used to train the model.")
    st.dataframe(data, use_container_width=True)

    st.markdown("### Dataset Summary")
    st.dataframe(data.describe(include="all"), use_container_width=True)

    st.markdown("### Missing Values")
    missing = data.isnull().sum().rename("Missing Values")
    st.dataframe(missing, use_container_width=True)

elif page == "Model Performance":
    st.subheader("Model Evaluation")

    c1, c2 = st.columns(2)
    c1.metric("Mean Absolute Error (USD)", f"${metrics['mae']:,.0f}")
    c2.metric("R² Score", f"{metrics['r2']:.3f}")

    st.write(f"Training records: {metrics['train_rows']:,}")
    st.write(f"Testing records: {metrics['test_rows']:,}")

    st.markdown("### What do these metrics mean?")
    st.write(
        "**MAE:** The average absolute difference between predicted "
        "and actual house values, measured in USD."
    )
    st.write(
        "**R² Score:** Indicates how much of the variation in the "
        "test-set prices is explained by the model. A higher score "
        "is generally better, but it does not guarantee accurate "
        "predictions for every property."
    )

    st.warning(
        "The model uses California housing data. Its performance "
        "metrics do not establish its accuracy for Indian properties."
    )

st.divider()
st.caption("House Price Prediction | Built with Python, Scikit-learn and Streamlit")
'''
}

for name, content in files.items():
    Path(name).write_text(content.strip() + "\n", encoding="utf-8")
    print(f"Created: {name}")

print("\nProject files created successfully!")
print("Next: run python -m pip install -r requirements.txt")
