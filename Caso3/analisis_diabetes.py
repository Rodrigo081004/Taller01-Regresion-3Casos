"""
Caso 3: Diabetes (Scikit-learn)
Fase A: EDA y preprocesamiento
Fase B: Regresión Lineal Múltiple y Regresión Polinomial
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

import json, os

OUT_IMG = "/home/claude/proyecto_ia/caso3_diabetes/img"
OUT_DATA = "/home/claude/proyecto_ia/caso3_diabetes/data"
os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_DATA, exist_ok=True)

# -----------------------------------------------------------------
# 1. CARGA DE DATOS
# -----------------------------------------------------------------
diabetes = load_diabetes(as_frame=True)
df = diabetes.frame.copy()
df.to_csv(f"{OUT_DATA}/diabetes_raw.csv", index=False)

resumen = {}
resumen["forma"] = df.shape
resumen["columnas"] = list(df.columns)
resumen["nulos_por_columna"] = df.isnull().sum().to_dict()
resumen["duplicados"] = int(df.duplicated().sum())

# -----------------------------------------------------------------
# 2. ESTADÍSTICA DESCRIPTIVA
# -----------------------------------------------------------------
describe = df.describe().T
describe.to_csv(f"{OUT_DATA}/estadistica_descriptiva.csv")

# -----------------------------------------------------------------
# 3. DETECCIÓN DE OUTLIERS (IQR)
# -----------------------------------------------------------------
outlier_counts = {}
for col in df.columns[:-1]:  # todas menos target
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_counts[col] = int(((df[col] < lo) | (df[col] > hi)).sum())
resumen["outliers_iqr_por_variable"] = outlier_counts

# -----------------------------------------------------------------
# 4. MATRIZ DE CORRELACIÓN
# -----------------------------------------------------------------
corr = df.corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Matriz de correlación - Dataset Diabetes")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/matriz_correlacion.png", dpi=140)
plt.close()

# -----------------------------------------------------------------
# 5. MULTICOLINEALIDAD (VIF)
# -----------------------------------------------------------------
X_vif = df.drop(columns=["target"])
vif_data = pd.DataFrame()
vif_data["variable"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
vif_data.to_csv(f"{OUT_DATA}/vif.csv", index=False)

# -----------------------------------------------------------------
# 6. TRAIN/TEST SPLIT + ESCALAMIENTO
# -----------------------------------------------------------------
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------
# 7. REGRESIÓN LINEAL MÚLTIPLE
# -----------------------------------------------------------------
lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)
y_pred_lin = lin_model.predict(X_test_scaled)

metrics_lin = {
    "modelo": "Regresión Lineal Múltiple",
    "R2": round(r2_score(y_test, y_pred_lin), 4),
    "MAE": round(mean_absolute_error(y_test, y_pred_lin), 4),
    "RMSE": round(mean_squared_error(y_test, y_pred_lin) ** 0.5, 4),
}

coef_lin = dict(zip(X.columns, np.round(lin_model.coef_, 3)))

# -----------------------------------------------------------------
# 8. REGRESIÓN POLINOMIAL (grado 2)
# -----------------------------------------------------------------
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

metrics_poly = {
    "modelo": "Regresión Polinomial (grado 2)",
    "R2": round(r2_score(y_test, y_pred_poly), 4),
    "MAE": round(mean_absolute_error(y_test, y_pred_poly), 4),
    "RMSE": round(mean_squared_error(y_test, y_pred_poly) ** 0.5, 4),
    "n_features_generadas": X_train_poly.shape[1],
}

# -----------------------------------------------------------------
# 9. GRÁFICO COMPARATIVO: REAL VS PREDICHO
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(y_test, y_pred_lin, alpha=0.6, color="steelblue")
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
axes[0].set_title("Regresión Lineal Múltiple")
axes[0].set_xlabel("Valor real")
axes[0].set_ylabel("Predicción")

axes[1].scatter(y_test, y_pred_poly, alpha=0.6, color="darkorange")
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
axes[1].set_title("Regresión Polinomial (grado 2)")
axes[1].set_xlabel("Valor real")
axes[1].set_ylabel("Predicción")

plt.tight_layout()
plt.savefig(f"{OUT_IMG}/real_vs_prediccion.png", dpi=140)
plt.close()

# -----------------------------------------------------------------
# 10. DISTRIBUCIÓN DE VARIABLES
# -----------------------------------------------------------------
df.hist(figsize=(12, 10), bins=20, color="teal")
plt.tight_layout()
plt.savefig(f"{OUT_IMG}/distribuciones.png", dpi=140)
plt.close()

# -----------------------------------------------------------------
# GUARDAR TODO EN JSON PARA EL INFORME
# -----------------------------------------------------------------
resultado_final = {
    "resumen_dataset": resumen,
    "vif": vif_data.to_dict(orient="records"),
    "metricas_lineal": metrics_lin,
    "coeficientes_lineal": coef_lin,
    "metricas_polinomial": metrics_poly,
}

with open(f"{OUT_DATA}/resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultado_final, f, ensure_ascii=False, indent=2, default=str)

print(json.dumps(resultado_final, ensure_ascii=False, indent=2, default=str))
