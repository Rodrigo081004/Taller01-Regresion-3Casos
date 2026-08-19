# Caso 2: Wine Quality - Prediccion de Calidad de Vino Tinto

## Que contiene esta carpeta

| Elemento | Descripcion |
|---|---|
| `data/winequality-red.csv` | Dataset de vino tinto (1599 registros, 12 columnas: 11 propiedades fisico-quimicas + `quality`). |
| `notebooks/analisis.ipynb` | Cuaderno Jupyter con el analisis exploratorio (EDA), outliers, correlacion, VIF, estandarizacion y entrenamiento/comparacion de modelos. |
| `modelos/` | Artefactos entrenados guardados con `joblib`: `mejor_modelo.pkl` (modelo ganador + scaler + PolynomialFeatures) y `scaler.pkl`. |
| `app_caso2.py` | Modulo Streamlit con la funcion `render_wine_quality()` para predecir la calidad desde una interfaz web. |
| `requirements.txt` | Dependencias de Python necesarias. |

## Origen del dataset

- **Fuente:** UCI Machine Learning Repository - Wine Quality
- **Enlace:** https://archive.ics.uci.edu/dataset/186/wine+quality
- **Subconjunto usado:** **vino tinto** (`winequality-red`), con 1599 muestras.
- **Target:** `quality` (puntuacion de 0 a 10 asignada por catadores).
- Nota: la copia local del CSV esta separada por **comas** (`,`); el original de UCI usa punto y coma (`;`).

## Como correr el notebook

Con `jupyter` instalado (o desde VS Code), desde la carpeta `caso2`:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/analisis.ipynb
```

El notebook ejecuta todo el flujo y deja guardados los modelos en `modelos/`.

## Como correr la app

Desde la carpeta `caso2`:

```bash
streamlit run app_caso2.py
```

La app carga `modelos/mejor_modelo.pkl` y `modelos/scaler.pkl`, pide las 11 variables fisico-quimicas y muestra la calidad predicha al presionar el boton.

La funcion `render_wine_quality()` puede importarse desde un `app.py` principal (multi-caso) sin ejecutarse sola:

```python
from app_caso2 import render_wine_quality
render_wine_quality()
```

## Resumen del modelado

- **Fase A:** sin valores nulos; outliers detectados con boxplots (IQR); VIF alerta en `fixed acidity` (7.8) y `density` (6.3); predictoras estandarizadas con `StandardScaler` (el target `quality` no se escala).
- **Fase B:** Regresion Lineal Multiple (R² test 0.4032) vs Polinomial grados 2 (0.4071) y 3 (0.4249) sobre las 3 variables mas correlacionadas (`alcohol`, `volatile acidity`, `sulphates`). El **polinomial grado 3** gano en test; la mejora frente al lineal es modesta y con riesgo de overfitting al crecer los terminos.
- **Fase C:** aplicativo Streamlit desplegable en la nube (Streamlit Community Cloud / Hugging Face Spaces / Render).