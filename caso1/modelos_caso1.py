"""Fase B — Comparativa de modelos: Regresión Lineal Múltiple vs Regresión Polinomial.

Entrena sobre housing_clean.csv (resultado del EDA: winsorización 1-99%):
  1. Regresión Lineal Múltiple (baseline)
  2. Regresión Polinomial grado 2
  3. Regresión Polinomial grado 3

Preprocesado: StandardScaler sobre las 8 variables numéricas, one-hot
(drop_first) de ocean_proximity. Exporta comparacion_modelos.json,
modelo_polinomial.json y figuras/comparacion_metricas.png.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

CSV_FILE = "housing_clean.csv"
TARGET = "median_house_value"
CATEGORICAL = ["ocean_proximity"]
NUMERIC = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
]
DEGREES = [1, 2, 3]
RANDOM_STATE = 42
TEST_SIZE = 0.2

matplotlib.rcParams["figure.dpi"] = 110


def main() -> None:
    df = pd.read_csv(CSV_FILE)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_encoded = pd.get_dummies(X, columns=CATEGORICAL, drop_first=True)
    dummy_cols = [c for c in X_encoded.columns if c not in NUMERIC]

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    rows = []
    models = {}
    best_poly = None

    for degree in DEGREES:
        if degree == 1:
            name = "Lineal Múltiple"
            pipe = make_pipeline(StandardScaler(), LinearRegression())
        else:
            name = f"Polinomial grado {degree}"
            pipe = make_pipeline(
                StandardScaler(),
                PolynomialFeatures(degree=degree, include_bias=False),
                LinearRegression(),
            )

        pipe.fit(X_train, y_train)

        metrics = {}
        for tag, Xs, ys in [("train", X_train, y_train), ("test", X_test, y_test)]:
            y_pred = pipe.predict(Xs)
            metrics[tag] = {
                "r2": round(float(r2_score(ys, y_pred)), 4),
                "mae": round(float(mean_absolute_error(ys, y_pred)), 2),
                "rmse": round(float(np.sqrt(mean_squared_error(ys, y_pred))), 2),
            }

        rows.append({
            "modelo": name,
            "grado": degree,
            "r2_train": metrics["train"]["r2"],
            "mae_train": metrics["train"]["mae"],
            "rmse_train": metrics["train"]["rmse"],
            "r2_test": metrics["test"]["r2"],
            "mae_test": metrics["test"]["mae"],
            "rmse_test": metrics["test"]["rmse"],
        })
        print(f"{name:22s} | R² train {metrics['train']['r2']:.4f} | R² test "
              f"{metrics['test']['r2']:.4f} | MAE test ${metrics['test']['mae']:,.2f} | "
              f"RMSE test ${metrics['test']['rmse']:,.2f}")

        if degree > 1:
            models[name] = {"pipeline": pipe, "metrics": metrics["test"]}
            if best_poly is None or metrics["test"]["r2"] > models[best_poly]["metrics"]["r2"]:
                best_poly = name

    df_cmp = pd.DataFrame(rows)
    print("\n" + df_cmp.to_string(index=False))

    export_comparison(df_cmp)
    export_poly_model(models[best_poly], best_poly, X_train, dummy_cols)
    plot_comparison(df_cmp)

    print(f"\nMejor polinomial: {best_poly}")
    print("Exportado: comparacion_modelos.json, modelo_polinomial.json, "
          "figuras/comparacion_metricas.png")


def export_comparison(df_cmp: pd.DataFrame) -> None:
    with open("comparacion_modelos.json", "w", encoding="utf-8") as f:
        json.dump(
            {"modelos": df_cmp.to_dict(orient="records")},
            f, indent=2, ensure_ascii=False,
        )


def export_poly_model(entry, name: str, X_train: pd.DataFrame, dummy_cols) -> None:
    pipe = entry["pipeline"]
    scaler = pipe.named_steps["standardscaler"]
    poly = pipe.named_steps["polynomialfeatures"]
    reg = pipe.named_steps["linearregression"]

    feature_names = poly.get_feature_names_out(X_train.columns).tolist()

    payload = {
        "info": {
            "dataset": "California Housing (housing_clean.csv)",
            "modelo": name,
            "rows": int(len(X_train) / 0.8),
            "n_train": int(len(X_train)),
            "metodo": "Regresión polinomial (Pipeline: StandardScaler + "
                      "PolynomialFeatures + LinearRegression)",
            "test_split": 0.2,
            "random_state": 42,
        },
        "degree": int(poly.degree),
        "numeric_features": [c for c in X_train.columns if c not in dummy_cols],
        "dummy_features": dummy_cols,
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "target": {
            "column": TARGET,
            "min": float(entry["metrics"].get("min", 0)),
        },
        "metrics": entry["metrics"],
        "intercept": float(reg.intercept_),
        "coefficients": [float(c) for c in reg.coef_],
        "feature_names": feature_names,
        "n_features": len(feature_names),
    }

    with open("modelo_polinomial.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def plot_comparison(df_cmp: pd.DataFrame) -> None:
    os.makedirs("figuras", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df_cmp))
    w = 0.35
    ax.bar(x - w / 2, df_cmp["r2_train"], width=w, label="R² (train)", color="#4f8ef7")
    ax.bar(x + w / 2, df_cmp["r2_test"], width=w, label="R² (test)", color="#18b45e")
    ax.set_xticks(x)
    ax.set_xticklabels(df_cmp["modelo"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R²")
    ax.set_title("Comparativa: Regresión Lineal vs Polinomial (California Housing)")
    for xi, (a, b) in enumerate(zip(df_cmp["r2_train"], df_cmp["r2_test"])):
        ax.text(xi - w / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=8)
    ax.legend()
    plt.tight_layout()
    plt.savefig("figuras/comparacion_metricas.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()