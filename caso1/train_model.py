import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

CSV_FILE = "housing.csv"
JSON_OUT = "model.json"
TARGET = "median_house_value"
CATEGORICAL = ["ocean_proximity"]

df = pd.read_csv(CSV_FILE)
df = df.dropna().reset_index(drop=True)

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_encoded = pd.get_dummies(X, columns=CATEGORICAL, drop_first=True)
feature_names = list(X_encoded.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

coeffs = {name: float(c) for name, c in zip(feature_names, model.coef_)}
ranges = {}
for name in feature_names:
    if name.startswith("ocean_proximity_"):
        continue
    values = X_encoded[name]
    lo, hi = float(values.min()), float(values.max())
    step = max(1.0, round((hi - lo) / 100, 4))
    ranges[name] = {
        "min": lo,
        "max": hi,
        "step": step,
        "default": round(float(values.mean()), 4),
    }

categorical_options = sorted(X["ocean_proximity"].unique().tolist())

model_json = {
    "info": {
        "dataset": "California Housing (housing.csv)",
        "rows": int(len(df)),
        "features": len(feature_names),
        "method": "Regresión lineal múltiple (scikit-learn LinearRegression)",
        "test_split": 0.2,
    },
    "target": {
        "column": TARGET,
        "min": float(y.min()),
        "max": float(y.max()),
        "mean": float(y.mean()),
        "median": float(y.median()),
    },
    "metrics": {
        "r2": round(r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "n": int(len(df)),
        "n_test": int(len(X_test)),
    },
    "features": feature_names,
    "intercept": float(model.intercept_),
    "coefficients": coeffs,
    "ranges": ranges,
    "categorical": {
        "column": "ocean_proximity",
        "options": categorical_options,
        "dummy_features": [f for f in feature_names if f.startswith("ocean_proximity_")],
        "encoding": {
            c: {
                f"ocean_proximity_{d}": (
                    1.0 if d == c else 0.0
                )
                for d in categorical_options
                if f"ocean_proximity_{d}" in feature_names
            }
            for c in categorical_options
        },
    },
}

with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(model_json, f, indent=2, ensure_ascii=False)

print(f"R² (test): {r2:.4f}")
print(f"MAE (test): ${mae:,.2f}")
print(f"RMSE (test): ${rmse:,.2f}")
print(f"Intercepto: {model.intercept_:,.2f}")
print("Modelo exportado a model.json")