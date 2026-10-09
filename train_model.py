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
