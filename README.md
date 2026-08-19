# 📊 Taller 1.1 — Modelado Predictivo Multisectorial

Trabajo práctico de **Inteligencia Artificial / Aprendizaje Automático** (Semestre 2026-II).

Aplicativo interactivo con **regresión lineal múltiple** y **regresión polinomial** aplicadas a tres dominios:

| Caso | Dataset | Problema |
|---|---|---|
| [1 — California Housing](caso1/) | `housing.csv` | Predecir el valor medio de la vivienda en distritos de California |
| 2 — Wine Quality | *Próximamente* | Determinar la calidad del vino (0–10) por propiedades físico-químicas |
| 3 — Diabetes (Scikit-learn) | *Próximamente* | Predecir la progresión cuantitativa de la enfermedad |

## 🚀 Demo en vivo (GitHub Pages)

https://rodrigo081004.github.io/Taller01-Regresion-3Casos/

## 📂 Estructura del repositorio

```
├── index.html              ← Portada con menú de navegación de los 3 casos
├── README.md
└── caso1/                  ← Caso 1: California Housing
    ├── eda_caso1.ipynb     ← Fase A: EDA (nulos, outliers, correlación/VIF, escalado)
    ├── train_model.py      ← Entrena regresión lineal múltiple → model.json
    ├── modelos_caso1.py    ← Fase B: comparativa lineal vs polinomial (grados 2 y 3)
    ├── model.json          ← Coeficientes del modelo lineal (generado)
    ├── modelo_polinomial.json ← Coeficientes del modelo polinomial (generado)
    ├── comparacion_modelos.json ← Métricas comparativas (generado)
    ├── housing_clean.csv   ← Dataset con outliers tratados (generado)
    ├── figuras/            ← Gráficos del EDA (generados)
    ├── index.html          ← Simulador interactivo del caso 1
    ├── simulador.js        ← Predicción en el navegador (JS puro)
    ├── requirements.txt    ← Dependencias de Python
    └── README.md           ← Documentación detallada del caso 1
```

## 🧠 Fases desarrolladas

- **Fase A — EDA y preprocesamiento** (`caso1/eda_caso1.ipynb`): limpieza de nulos, tratamiento de outliers (winsorización), análisis de multicolinealidad (matriz de correlación + VIF) y estandarización (`StandardScaler`).
- **Fase B — Modelamiento y comparativa** (`caso1/modelos_caso1.py`): regresión lineal múltiple base y regresión polinomial (grados 2 y 3), con comparativa de R², MAE y RMSE.
- **Fase C — Aplicativo web**: simuladores HTML interactivos (100 % en el navegador, sin servidor), navegables desde `index.html`.

## ▶️ Cómo ejecutar el caso 1 en local

```bash
cd caso1
pip install -r requirements.txt
python train_model.py        # regenera model.json
python modelos_caso1.py      # comparativa lineal vs polinomial
jupyter notebook eda_caso1.ipynb   # o: jupyter nbconvert --execute --to notebook --inplace eda_caso1.ipynb
```

Para ver los simuladores HTML:

```bash
cd ..
python -m http.server 8000
# abre http://localhost:8000
```

## 🌐 Despliegue en GitHub Pages

1. Sube el repo a GitHub (rama `main`).
2. **Settings → Pages** → *Build and deployment* → **Deploy from a branch** → `main` / raíz (`/`).
3. Guarda y espera 1–2 min. El sitio queda publicado en `https://<usuario>.github.io/<repo>/`.

Sin build ni configuración adicional: todos los simuladores corren en el navegador.

## 📝 Informe

Informe técnico consolidado en PDF (entregable del curso) con capturas del aplicativo y discusión comparativa entre los tres dominios.