const LABELS = {
  longitude: "Longitud",
  latitude: "Latitud",
  housing_median_age: "Edad mediana de la casa (años)",
  total_rooms: "Total de habitaciones",
  total_bedrooms: "Total de dormitorios",
  population: "Población",
  households: "Hogares",
  median_income: "Ingreso medio (× $10,000)",
  ocean_proximity: "Proximidad al océano",
};

const fmtUSD = (v) =>
  "$" + Math.round(v).toLocaleString("en-US");

let MODEL = null;
const inputs = {};

async function init() {
  const res = await fetch("model.json");
  MODEL = await res.json();
  buildHeader();
  buildFields();
  buildRefs();
  update();
}

function buildHeader() {
  const { info, metrics } = MODEL;
  document.getElementById("badges").innerHTML =
    `<div class="badge">Filas <span>${info.rows.toLocaleString("en-US")}</span></div>` +
    `<div class="badge">Predictores <span>${info.features.length}</span></div>` +
    `<div class="badge">R² (test) <span>${metrics.r2}</span></div>` +
    `<div class="badge">MAE <span>${fmtUSD(metrics.mae)}</span></div>`;
  document.getElementById("note").textContent =
    `Modelo: ${info.method}. Precio = intercepto + Σ (coeficiente × variable). ` +
    `Rendimiento evaluado sobre ${metrics.n_test.toLocaleString("en-US")} casas reservadas para prueba (20 %).`;
}

function buildFields() {
  const container = document.getElementById("fields");
  const { ranges, categorical } = MODEL;
  let html = "";

  for (const [name, r] of Object.entries(ranges)) {
    html += `
      <div class="field">
        <label>${LABELS[name] || name}
          <output id="out_${name}"></output>
        </label>
        <input type="range" id="in_${name}"
          min="${r.min}" max="${r.max}" step="${r.step}" value="${r.default}">
      </div>`;
  }

  html += `
    <div class="field">
      <label>${LABELS.ocean_proximity}</label>
      <select id="in_ocean_proximity">
        ${categorical.options
          .map((o) => `<option value="${o}">${o}</option>`)
          .join("")}
      </select>
    </div>`;

  container.innerHTML = html;

  for (const name of Object.keys(MODEL.ranges)) {
    const el = document.getElementById(`in_${name}`);
    inputs[name] = el;
    el.addEventListener("input", update);
  }
  const sel = document.getElementById("in_ocean_proximity");
  inputs.ocean_proximity = sel;
  sel.addEventListener("change", update);
}

function buildRefs() {
  const t = MODEL.target;
  const mean = (t.min + t.max) / 2;
  document.getElementById("barMin").textContent = fmtUSD(t.min);
  document.getElementById("barMax").textContent = fmtUSD(t.max);
  document.getElementById("refs").innerHTML =
    `<div class="ref"><span>Mínimo del dataset</span><b>${fmtUSD(t.min)}</b></div>` +
    `<div class="ref"><span>Mediana del dataset</span><b>${fmtUSD(t.median)}</b></div>` +
    `<div class="ref"><span>Media del dataset</span><b>${fmtUSD(t.mean)}</b></div>` +
    `<div class="ref"><span>Máximo del dataset</span><b>${fmtUSD(t.max)}</b></div>`;
  window.__rangeMid = mean;
}

function currentVector() {
  const vec = {};
  for (const f of MODEL.features) vec[f] = 0;
  for (const name of Object.keys(MODEL.ranges)) {
    vec[name] = parseFloat(inputs[name].value);
  }
  const cat = inputs.ocean_proximity.value;
  for (const [feature, val] of Object.entries(
    MODEL.categorical.encoding[cat]
  )) {
    vec[feature] = val;
  }
  return vec;
}

function predict() {
  const vec = currentVector();
  let price = MODEL.intercept;
  for (const f of MODEL.features) price += MODEL.coefficients[f] * vec[f];
  return price;
}

function update() {
  const { ranges } = MODEL;
  for (const name of Object.keys(ranges)) {
    const el = inputs[name];
    const pct =
      ((parseFloat(el.value) - parseFloat(el.min)) /
        (parseFloat(el.max) - parseFloat(el.min))) *
      100;
    el.style.setProperty("--pct", pct + "%");
    const formatted =
      name === "longitude" || name === "latitude"
        ? parseFloat(el.value).toFixed(2)
        : parseFloat(el.value) >= 100
        ? Math.round(el.value).toLocaleString("en-US")
        : parseFloat(el.value).toFixed(1);
    document.getElementById(`out_${name}`).textContent = formatted;
  }

  const price = Math.max(0, predict());
  document.getElementById("price").textContent = fmtUSD(price);

  const t = MODEL.target;
  const pct = Math.min(100, Math.max(0, ((price - t.min) / (t.max - t.min)) * 100));
  document.getElementById("bar").style.width = pct + "%";

  const diff = price - t.median;
  const sign = diff >= 0 ? "+" : "−";
  document.getElementById("priceSub").textContent =
    `${sign}${fmtUSD(Math.abs(diff))} respecto a la mediana del dataset ` +
    `(${fmtUSD(t.median)})`;
}

init();