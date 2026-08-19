# 🏠 Caso 1 — Simulador de Precios de Casas (California Housing)

Caso 1 del Taller 1.1: **regresión lineal múltiple y regresión polinomial** sobre `housing.csv` (California Housing) para predecir el valor medio de la vivienda (`median_house_value`).

Incluye un simulador interactivo que corre 100 % en el navegador (JS puro, sin servidor).

## 🔗 Enlaces

- Demo del simulador: https://rodrigo081004.github.io/Taller01-Regresion-3Casos/caso1/
- Repositorio: https://github.com/Rodrigo081004/Taller01-Regresion-3Casos

## 📂 Estructura

| Archivo | Descripción |
|---|---|
| `eda_caso1.ipynb` | **Fase A** — EDA completo: nulos, outliers, correlación/VIF, estandarización |
| `train_model.py` | Entrena la regresión lineal múltiple (datos crudos) y exporta `model.json` |
| `modelos_caso1.py` | **Fase B** — Comparativa lineal vs polinomial (grados 2 y 3) |
| `model.json` | Coeficientes del modelo lineal (generado por `train_model.py`) |
| `modelo_polinomial.json` | Coeficientes del polinomial grado 3 + pipeline (generado) |
| `comparacion_modelos.json` | Métricas comparativas de los 3 modelos (generado) |
| `housing_clean.csv` | Dataset con outliers tratados (generado por el EDA) |
| `figuras/` | Gráficos del EDA y de la comparativa (generados) |
| `index.html` + `simulador.js` | Simulador interactivo del modelo lineal |
| `housing.csv` | Dataset original (20,640 filas, 10 columnas) |
| `requirements.txt` | Dependencias de Python |

## 🧠 Fase A — Análisis Exploratorio y Preprocesamiento (`eda_caso1.ipynb`)

1. **Nulos**: 207 valores en `total_bedrooms` (1.0 %) → eliminados (`dropna`) → **20,433** registros.
2. **Outliers**: boxplots + conteo IQR por variable; tratamiento con **winsorización** (clip en percentiles 1 %–99 %) sobre los 8 predictores → `housing_clean.csv`. La variable objetivo conserva su truncamiento oficial a $500,001.
3. **Multicolinealidad**: matriz de correlación (heatmap) + **VIF** calculado manualmente (VIF = 1/(1−R²)). Se detecta colinealidad esperada entre coordenadas y variables demográficas (VIF > 10); se documenta y no se eliminan variables por interpretabilidad.
4. **Estandarización**: `StandardScaler` sobre las variables numéricas (media ≈ 0, desviación ≈ 1), indispensable para la regresión polinomial.

## 🧠 Fase B — Modelamiento y comparativa (`modelos_caso1.py`)

Pipeline común: `StandardScaler` + (opcionalmente `PolynomialFeatures`) + `LinearRegression`. Split 80/20 con `random_state=42`, métricas sobre el conjunto de test:

| Modelo | R² (train) | R² (test) | MAE (test) | RMSE (test) |
|---|---|---|---|---|
| Lineal múltiple | 0.6684 | 0.6675 | $49,131 | $67,435 |
| Polinomial grado 2 | 0.7348 | 0.7262 | $43,149 | $61,185 |
| **Polinomial grado 3** | **0.7845** | **0.7649** | **$39,361** | **$56,700** |

El polinomial de grado 3 mejora el R² de test en **+0.097** (0.6675 → 0.7649) y reduce el MAE en ~$9,770 frente al modelo lineal.

## 🧠 Modelo lineal del simulador (`train_model.py`)

- **Método**: `LinearRegression` de scikit-learn (mínimos cuadrados ordinarios).
- **Predictores (12)**: 8 numéricos + 4 dummies de `ocean_proximity` (one-hot, base `<1H OCEAN`).
- **Preprocesado**: `dropna()` de 207 nulos; split 80/20 con `random_state=42`.
- **Rendimiento (test)**: R² = **0.6488**, MAE = **$50,413**, RMSE = **$69,298**.

La predicción se calcula como `precio = intercepto + Σ (coeficienteᵢ × variableᵢ)` y el navegador replica exactamente la fórmula con los coeficientes de `model.json` (paridad verificada contra scikit-learn).

## 🚀 Uso local

```bash
pip install -r requirements.txt
python train_model.py        # regenera model.json (modelo lineal)
python modelos_caso1.py      # comparativa lineal vs polinomial
jupyter nbconvert --execute --inplace eda_caso1.ipynb   # o ábrelo en Jupyter
python -m http.server 8000   # simulador: http://localhost:8000/caso1/
```

## ⚠️ Notas

- R² ≈ 0.65–0.76 es lo esperado para California Housing sin ingeniería de características profunda; es un modelo educativo, no apto para valoración real.
- El precio máximo del dataset está truncado en $500,001 (dato oficial de California).
- El simulador carga `model.json` de forma relativa (requiere servidor local o GitHub Pages, no abrir como `file://`).