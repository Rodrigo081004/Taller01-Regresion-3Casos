import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELO_PATH = os.path.join(BASE_DIR, "modelos", "mejor_modelo.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "modelos", "scaler.pkl")

COLUMNAS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

# Valores de referencia (mín, máx, media) extraídos del dataset winequality-red.csv
REFERENCIAS = {
    "fixed acidity": (4.6, 15.9, 8.32),
    "volatile acidity": (0.12, 1.58, 0.53),
    "citric acid": (0.0, 1.0, 0.27),
    "residual sugar": (0.9, 15.5, 2.54),
    "chlorides": (0.012, 0.611, 0.087),
    "free sulfur dioxide": (1.0, 72.0, 15.87),
    "total sulfur dioxide": (6.0, 289.0, 46.47),
    "density": (0.9901, 1.0037, 0.9967),
    "pH": (2.74, 4.01, 3.31),
    "sulphates": (0.33, 2.0, 0.66),
    "alcohol": (8.4, 14.9, 10.42),
}


@st.cache_resource
def cargar_artefactos():
    """Carga el modelo, el scaler y la configuracion guardados por el notebook."""
    artefacto = joblib.load(MODELO_PATH)
    scaler = joblib.load(SCALER_PATH)
    return artefacto, scaler


def render_wine_quality():
    """Pagina del Caso 2: prediccion de calidad de vino tinto (Wine Quality - UCI)."""
    st.header(":wine_glass: Caso 2: Wine Quality (vino tinto)")
    st.caption(
        "Prediccion de la calidad del vino (escala 0-10) a partir de 11 propiedades "
        "fisico-quimicas. Dataset: Wine Quality (UCI)."
    )

    if not (os.path.exists(MODELO_PATH) and os.path.exists(SCALER_PATH)):
        st.error(
            "No se encontraron los artefactos en modelos/. "
            "Ejecuta primero el notebook `notebooks/analisis.ipynb` para entrenar y guardar el modelo."
        )
        return

    artefacto, scaler = cargar_artefactos()

    with st.form("form_wine_quality"):
        st.subheader("Propiedades fisico-quimicas del vino")
        cols = st.columns(2)
        valores = {}
        for i, nombre in enumerate(COLUMNAS):
            minimo, maximo, media = REFERENCIAS[nombre]
            step = 0.01 if maximo - minimo < 3 else 0.1
            valores[nombre] = cols[i % 2].number_input(
                nombre.capitalize(),
                min_value=float(minimo),
                max_value=float(maximo),
                value=float(media),
                step=float(step),
                format="%.4f" if maximo < 10 else "%.2f",
            )

        predecir = st.form_submit_button("Predecir calidad", type="primary", use_container_width=True)

    if predecir:
        fila = pd.DataFrame([valores], columns=COLUMNAS)
        columnas_modelo = artefacto["columnas_modelo"]
        poly = artefacto["poly"]

        X = scaler.transform(fila[columnas_modelo])
        if poly is not None:
            X = poly.transform(X)

        prediccion = artefacto["modelo"].predict(X)[0]
        prediccion_redondeada = int(np.clip(np.round(prediccion), 0, 10))

        st.success(f"Calidad predicha: **{prediccion:.2f}** (~ {prediccion_redondeada}/10)")
        st.info(
            "Nota: la calidad es una etiqueta subjetiva asignada por catadores; "
            "el modelo estima un valor continuo aproximado."
        )


if __name__ == "__main__":
    render_wine_quality()