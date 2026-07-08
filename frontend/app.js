const RESULT_NAMES = { H: "Home", D: "Draw", A: "Away" };

let predictions = [];

async function loadJSON(path) {
  const res = await fetch(path);
  return res.json();
}

function pickFor(row, prefix) {
  const probs = { Home: row[`${prefix}_H`], Draw: row[`${prefix}_D`], Away: row[`${prefix}_A`] };
  return Object.keys(probs).reduce((a, b) => (probs[a] >= probs[b] ? a : b));
}

function seasonLabel(year) {
  return `${year}/${String(year + 1).slice(-2)}`;
}

function renderTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });
}

function populateFilters() {
  const seasons = [...new Set(predictions.map((p) => p.season))].sort((a, b) => b - a);
  const seasonSelect = document.getElementById("season-filter");
  seasonSelect.innerHTML = seasons.map((s) => `<option value="${s}">${seasonLabel(s)}</option>`).join("");

  const teams = [...new Set(predictions.flatMap((p) => [p.home, p.away]))].sort();
  const teamSelect = document.getElementById("team-filter");
  teamSelect.innerHTML =
    `<option value="">All teams</option>` + teams.map((t) => `<option value="${t}">${t}</option>`).join("");

  seasonSelect.addEventListener("change", renderTable);
  teamSelect.addEventListener("change", renderTable);
}

function renderTable() {
  const season = Number(document.getElementById("season-filter").value);
  const team = document.getElementById("team-filter").value;

  let rows = predictions.filter((p) => p.season === season);
  if (team) rows = rows.filter((p) => p.home === team || p.away === team);
  rows = [...rows].sort((a, b) => a.date.localeCompare(b.date));

  let modelCorrect = 0;
  let bookCorrect = 0;

  const tbody = document.getElementById("matches-tbody");
  tbody.innerHTML = rows
    .map((r) => {
      const modelPick = pickFor(r, "model");
      const bookPick = pickFor(r, "book");
      const actual = RESULT_NAMES[r.actual];
      const modelOk = modelPick[0] === r.actual;
      const bookOk = bookPick[0] === r.actual;
      if (modelOk) modelCorrect++;
      if (bookOk) bookCorrect++;

      return `<tr>
        <td>${r.date}</td>
        <td>${r.home}</td>
        <td>${r.away}</td>
        <td class="pick-${modelPick}">${modelPick}</td>
        <td class="pick-${bookPick}">${bookPick}</td>
        <td>${actual}</td>
        <td class="${modelOk ? "mark-correct" : "mark-wrong"}">${modelOk ? "✓" : "✗"}</td>
        <td class="${bookOk ? "mark-correct" : "mark-wrong"}">${bookOk ? "✓" : "✗"}</td>
      </tr>`;
    })
    .join("");

  document.getElementById("kpi-count").textContent = rows.length;
  document.getElementById("kpi-model-acc").textContent = rows.length
    ? `${((modelCorrect / rows.length) * 100).toFixed(1)}%`
    : "–";
  document.getElementById("kpi-book-acc").textContent = rows.length
    ? `${((bookCorrect / rows.length) * 100).toFixed(1)}%`
    : "–";
}

function renderSummaryTable(summary) {
  const tbody = document.getElementById("summary-tbody");
  tbody.innerHTML = summary
    .map(
      (r) =>
        `<tr><td>${r.model}</td><td>${r.log_loss.toFixed(4)}</td><td>${r.brier_score.toFixed(4)}</td><td>${r.n_matches}</td></tr>`
    )
    .join("");
}

function renderCiTable(ci) {
  const tbody = document.getElementById("ci-tbody");
  tbody.innerHTML = ci
    .map(
      (r) =>
        `<tr><td>${r.model}</td><td>${r.point_diff.toFixed(4)}</td><td>${r.ci_low.toFixed(4)}</td><td>${r.ci_high.toFixed(4)}</td></tr>`
    )
    .join("");
}

function renderCalibrationChart(calibration) {
  const ctx = document.getElementById("calibration-chart");
  new Chart(ctx, {
    type: "line",
    data: {
      labels: calibration.map((r) => r.bin),
      datasets: [
        {
          label: "Predicted",
          data: calibration.map((r) => r.mean_predicted),
          borderColor: "#0071e3",
          backgroundColor: "#0071e3",
          tension: 0.1,
        },
        {
          label: "Actual",
          data: calibration.map((r) => r.mean_actual),
          borderColor: "#30d158",
          backgroundColor: "#30d158",
          tension: 0.1,
        },
      ],
    },
    options: {
      responsive: true,
      scales: { y: { beginAtZero: true, max: 1 } },
      plugins: { legend: { position: "bottom" } },
    },
  });
}

function renderImportanceChart(importance) {
  const top = importance.slice(0, 10);
  const ctx = document.getElementById("importance-chart");
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: top.map((r) => r.feature),
      datasets: [
        {
          label: "Importance",
          data: top.map((r) => r.importance),
          backgroundColor: "#0071e3",
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: { legend: { display: false } },
    },
  });
}

async function init() {
  renderTabs();

  const [preds, summary, ci, calibration, importance] = await Promise.all([
    loadJSON("data/predictions.json"),
    loadJSON("data/model_summary.json"),
    loadJSON("data/bootstrap_ci.json"),
    loadJSON("data/calibration.json"),
    loadJSON("data/feature_importance.json"),
  ]);

  predictions = preds;
  populateFilters();
  renderTable();
  renderSummaryTable(summary);
  renderCiTable(ci);
  renderCalibrationChart(calibration);
  renderImportanceChart(importance);
}

init();
