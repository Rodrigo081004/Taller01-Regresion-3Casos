const LABELS = {
  "fixed acidity": "Acidez fija",
  "volatile acidity": "Acidez volátil",
  "citric acid": "Ácido cítrico",
  "residual sugar": "Azúcar residual",
  chlorides: "Cloruros",
  "free sulfur dioxide": "Dióxido de azufre libre",
  "total sulfur dioxide": "Dióxido de azufre total",
  density: "Densidad",
  pH: "pH",
  sulphates: "Sulfatos",
  alcohol: "Alcohol",
};

let MODEL = null;
const inputs = {};

async function init() {
  const res = await fetch("model.json");
  MODEL = await res.json();
  buildHeader();
  buildFields();
  buildRefs();
  buildTables();
  update();
}

function buildHeader() {
  const { info, metrics } = MODEL;
  document.getElementById("badges").innerHTML =
    `<div class="badge">Filas <span>${info.rows.toLocaleString("en-US")}</span></div>` +
    `<div class="badge">Predictores <span>${info.features}</span></div>` +
    `<div class="badge">R² (test) <span>${metrics.r2}</span></div>` +
    `<div class="badge">RMSE <span>${metrics.rmse}</span></div>` +
    `<div class="badge">MAE <span>${metrics.mae}</span></div>`;
  document.getElementById("note").textContent =
    `Modelo: ${info.method}. Variables usadas por el modelo: ` +
    `${info.variables_modelo.join(", ")}. ` +
    `Rendimiento evaluado sobre ${metrics.n_test.toLocaleString("en-US")} vinos reservados para prueba (20 %).`;
}

function buildFields() {
  const container = document.getElementById("fields");
  const { ranges } = MODEL;
  let html = "";
  for (const [name, r] of Object.entries(ranges)) {
    html += `
      <div class="field">
        <label>${LABELS[name] || name}
          <output id="out_${slug(name)}"></output>
        </label>
        <input type="range" id="in_${slug(name)}"
          min="${r.min}" max="${r.max}" step="${r.step}" value="${r.default}">
      </div>`;
  }
  container.innerHTML = html;
  for (const name of Object.keys(ranges)) {
    const el = document.getElementById(`in_${slug(name)}`);
    inputs[name] = el;
    el.addEventListener("input", update);
  }
}

function buildRefs() {
  const t = MODEL.target;
  document.getElementById("refs").innerHTML =
    `<div class="ref"><span>Mínimo del dataset</span><b>${t.min}</b></div>` +
    `<div class="ref"><span>Mediana del dataset</span><b>${t.median}</b></div>` +
    `<div class="ref"><span>Media del dataset</span><b>${t.mean}</b></div>` +
    `<div class="ref"><span>Máximo del dataset</span><b>${t.max}</b></div>`;
}

function buildTables() {
  const corrBody = document.getElementById("corrRows");
  corrBody.innerHTML = Object.entries(MODEL.corr_quality)
    .map(
      ([k, v]) =>
        `<tr><td>${LABELS[k] || k}</td><td>${v.toFixed(3)}</td></tr>`
    )
    .join("");
  const vifBody = document.getElementById("vifRows");
  vifBody.innerHTML = Object.entries(MODEL.vif)
    .map(
      ([k, v]) =>
        `<tr><td>${LABELS[k] || k}</td><td>${v.toFixed(2)}</td></tr>`
    )
    .join("");
}

function slug(name) {
  return name.replace(/\s+/g, "_");
}

function predict() {
  const cols = MODEL.info.variables_modelo;
  const raw = cols.map((c) => parseFloat(inputs[c].value));
  const { mean, scale } = MODEL.scaler;
  const z = raw.map((v, i) => (v - mean[i]) / scale[i]);
  let acc = MODEL.intercept;
  MODEL.coefficients.forEach((coef, k) => {
    let term = 1;
    const pw = MODEL.poly.powers[k];
    for (let i = 0; i < pw.length; i++) term *= Math.pow(z[i], pw[i]);
    acc += coef * term;
  });
  return acc;
}

function update() {
  const { ranges, target } = MODEL;
  for (const name of Object.keys(ranges)) {
    const el = inputs[name];
    const pct =
      ((parseFloat(el.value) - parseFloat(el.min)) /
        (parseFloat(el.max) - parseFloat(el.min))) *
      100;
    el.style.setProperty("--pct", pct + "%");
    document.getElementById(`out_${slug(name)}`).textContent =
      Number(el.value).toFixed(el.step < 0.1 ? 3 : 2);
  }
  const pred = predict();
  const clipped = Math.max(target.min, Math.min(target.max, pred));
  const pct = ((clipped - target.min) / (target.max - target.min)) * 100;
  document.getElementById("predValue").textContent = pred.toFixed(2);
  document.getElementById("predRounded").textContent = Math.round(clipped) + "/10";
  document.getElementById("bar").style.width = pct + "%";
  const isGood = pred >= 6;
  document.getElementById("bar").style.background = isGood
    ? "linear-gradient(90deg, #34d399, #a3e635)"
    : "linear-gradient(90deg, #fbbf24, #f87171)";
  const emoji = pred >= 7 ? "🍷 Excelente" : pred >= 6 ? "🍷 Buena" : "🥴 Regular";
  document.getElementById("verdict").textContent = emoji;
}

document.addEventListener("DOMContentLoaded", init);