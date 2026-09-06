// Dark glass theme: Chart.js defaults to black text / gray gridlines, which
// disappear against the site's dark green background - repoint them once,
// globally, instead of passing color options into every chart call site.
if (typeof Chart !== "undefined") {
  Chart.defaults.color = "#a2c2b1";
  Chart.defaults.borderColor = "rgba(255, 255, 255, 0.12)";
  Chart.defaults.font.family = "'Space Grotesk', sans-serif";
}

const RESULT_NAMES = { H: "Home", D: "Draw", A: "Away" };
const MODEL_LABELS = {
  rf: "Random Forest",
  logreg: "Logistic Regression",
  xgb: "XGBoost",
  ens: "Ensemble",
};
const MODEL_PREFIXES = Object.keys(MODEL_LABELS);
const PICK_COLORS = { Home: "#4ade80", Draw: "#8aa89a", Away: "#7dd3fc" };

let predictions = [];
let upcoming = [];
let teamBadges = {};
let teamStadiums = {};

async function loadJSON(path) {
  const res = await fetch(path);
  return res.json();
}

function teamInitials(name) {
  if (!/\s/.test(name) && name.length <= 4) return name.toUpperCase();
  return name
    .split(/[\s']+/)
    .filter(Boolean)
    .map((w) => w[0])
    .join("")
    .slice(0, 3)
    .toUpperCase();
}

// Each club's real primary color, for the team-page theme accent.
//
// A canvas-based extraction from the badge image (draw it, sample the most
// common saturated pixel) was tried first instead of a hardcoded table, to
// avoid maintaining 40+ entries by hand. Dropped it: it requires
// img.crossOrigin = "anonymous" to read pixel data at all, and
// TheSportsDB's CDN doesn't consistently send CORS headers - some badges
// extracted fine, others silently failed and fell back to an arbitrary
// hash-based hue, so the *same team* could render in the wrong color
// depending on the viewer's browser/session (confirmed live: Liverpool
// rendered blue instead of red on a reload where extraction failed). A
// wrong club color is worse than no theming at all, so a small hardcoded
// map - needing an update only on promotion/relegation, the same
// maintenance cost as src/data/team_badges.py's own name-alias table - is
// the more reliable choice here.
const TEAM_COLORS = {
  "Arsenal": "#EF0107",
  "Aston Villa": "#670E36",
  "Birmingham": "#0000FF",
  "Blackburn": "#009EE0",
  "Blackpool": "#F68712",
  "Bolton": "#1C3F94",
  "Bournemouth": "#DA020E",
  "Brentford": "#E30613",
  "Brighton": "#0057B8",
  "Burnley": "#6C1D45",
  "Cardiff": "#0070B5",
  "Chelsea": "#034694",
  "Coventry": "#78D0F7",
  "Crystal Palace": "#1B458F",
  "Everton": "#003399",
  "Fulham": "#000000",
  "Huddersfield": "#0E63AD",
  "Hull": "#F18A00",
  "Ipswich": "#0044A9",
  "Leeds": "#1D428A",
  "Leicester": "#003090",
  "Liverpool": "#C8102E",
  "Luton": "#F78F1E",
  "Man City": "#6CABDD",
  "Man United": "#DA291C",
  "Middlesbrough": "#DC1122",
  "Newcastle": "#241F20",
  "Norwich": "#FFF200",
  "Nott'm Forest": "#DD0000",
  "QPR": "#1D5BA4",
  "Reading": "#004494",
  "Sheffield United": "#EE2737",
  "Southampton": "#D71920",
  "Stoke": "#E03A3E",
  "Sunderland": "#EB172B",
  "Swansea": "#000000",
  "Tottenham": "#132257",
  "Watford": "#FBEE23",
  "West Brom": "#122F67",
  "West Ham": "#7A263A",
  "Wigan": "#1D59AF",
  "Wolves": "#FDB913",
};

function teamColor(name) {
  if (TEAM_COLORS[name]) return TEAM_COLORS[name];
  // Deterministic (not random) fallback for a team not yet in the map -
  // e.g. a newly promoted club before this table gets updated. Same team
  // name always hashes to the same hue, so it's at least consistent across
  // reloads, just not that team's real color.
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = name.charCodeAt(i) + ((hash << 5) - hash);
  return `hsl(${Math.abs(hash) % 360}, 55%, 42%)`;
}

// Club badges are hotlinked directly from TheSportsDB's own CDN (never
// downloaded/re-hosted or modified - see src/data/team_badges.py and the
// attribution footer), per their free-tier terms of use. A team missing
// from the map, or an image that fails to load, falls back to a colored
// initials badge instead of a broken image. The fallback is wired up via
// attachBadgeFallbacks() after insertion (not an inline onerror string) -
// team names can contain apostrophes ("Nott'm Forest"), which broke a prior
// version of this that built the fallback as a quoted string inside an
// HTML attribute.
function initialsBadgeElement(name) {
  const span = document.createElement("span");
  span.className = "team-initials";
  span.style.background = teamColor(name);
  span.textContent = teamInitials(name);
  return span;
}

function teamCrestHtml(name) {
  const badgeUrl = teamBadges[name];
  if (!badgeUrl) return initialsBadgeElement(name).outerHTML;
  return `<span class="team-crest-wrap">
    <img class="team-badge" data-team="${name}" src="${badgeUrl}" alt="${name} badge" loading="lazy">
  </span>`;
}

function teamRoute(name) {
  return `#team/${encodeURIComponent(name)}`;
}

function teamLinkHtml(name, { withBadge = true } = {}) {
  const badge = withBadge ? teamCrestHtml(name) : "";
  return `<a class="team-link" href="${teamRoute(name)}">${badge}<span class="team">${name}</span></a>`;
}

function attachBadgeFallbacks(container) {
  container.querySelectorAll("img.team-badge").forEach((img) => {
    img.addEventListener("error", () => img.replaceWith(initialsBadgeElement(img.dataset.team)), {
      once: true,
    });
  });
}

// A real photo of the club's home ground (via TheSportsDB), when we have
// one on record; falls back to the generated illustration otherwise, and
// also on a broken/expired image URL at render time.
function stadiumBannerHtml(name, accent) {
  const stadium = teamStadiums[name];
  if (!stadium) return `<div class="stadium-banner">${stadiumBannerSvg(accent)}</div>`;
  const capacity = stadium.capacity ? `${Number(stadium.capacity).toLocaleString()} capacity` : null;
  const caption = [stadium.name, capacity].filter(Boolean).join(" &middot; ");
  return `<div class="stadium-banner stadium-banner-photo">
    <img class="stadium-photo" src="${stadium.image}" alt="${stadium.name}" loading="lazy">
    <div class="stadium-scrim"></div>
    <span class="stadium-caption">${caption}</span>
  </div>`;
}

function attachStadiumFallback(container, accent) {
  const img = container.querySelector(".stadium-photo");
  if (!img) return;
  img.addEventListener(
    "error",
    () => {
      const banner = img.closest(".stadium-banner");
      banner.classList.remove("stadium-banner-photo");
      banner.innerHTML = stadiumBannerSvg(accent);
    },
    { once: true }
  );
}

function renderLeagueTable(table) {
  const tbody = document.getElementById("league-table-tbody");
  const n = table.length;
  tbody.innerHTML = table
    .map((row) => {
      const zoneClass = row.position <= 4 ? "table-row-ucl" : row.position > n - 3 ? "table-row-releg" : "";
      return `<tr class="${zoneClass}">
        <td>${row.position}</td>
        <td>${teamLinkHtml(row.team)}</td>
        <td>${row.played}</td>
        <td>${row.gd > 0 ? "+" : ""}${row.gd}</td>
        <td><strong>${row.points}</strong></td>
      </tr>`;
    })
    .join("");
  attachBadgeFallbacks(tbody);
}

function modelAccuracy(rows, prefix) {
  const evaluated = rows.filter((r) => r.evaluated !== false);
  if (!evaluated.length) return null;
  const correct = evaluated.filter((r) => pickFor(r, prefix)[0] === r.actual).length;
  return correct / evaluated.length;
}

const MODEL_COLORS = { rf: "#60a5fa", logreg: "#4ade80", xgb: "#fbbf24", ens: "#c084fc", book: "#9fb3a8" };
const SHORT_MODEL_LABELS = { rf: "RF", logreg: "LogReg", xgb: "XGB", ens: "Ens", book: "Book" };

function accuracyRows() {
  const currentSeason = Math.max(...predictions.map((p) => p.season));
  const seasonRows = predictions.filter((p) => p.season === currentSeason);
  return {
    currentSeason,
    rows: [...MODEL_PREFIXES, "book"].map((prefix) => ({
      prefix,
      label: prefix === "book" ? "Bookmaker" : MODEL_LABELS[prefix],
      allTime: modelAccuracy(predictions, prefix),
      season: modelAccuracy(seasonRows, prefix),
    })),
  };
}

let accuracyChart = null;

function renderAccuracyPanel() {
  const { currentSeason, rows } = accuracyRows();
  document.getElementById("accuracy-season-header").textContent = seasonLabel(currentSeason);

  document.getElementById("accuracy-table-tbody").innerHTML = rows
    .map(
      (r) => `<tr>
        <td>${r.label}</td>
        <td>${r.allTime === null ? "&ndash;" : formatPct(r.allTime)}</td>
        <td>${r.season === null ? "&ndash;" : formatPct(r.season)}</td>
      </tr>`
    )
    .join("");

  const canvas = document.getElementById("accuracy-chart");
  if (accuracyChart) accuracyChart.destroy();
  accuracyChart = new Chart(canvas, {
    type: "bar",
    data: {
      labels: rows.map((r) => SHORT_MODEL_LABELS[r.prefix]),
      datasets: [
        {
          label: "All-time",
          data: rows.map((r) => (r.allTime === null ? null : Math.round(r.allTime * 1000) / 10)),
          backgroundColor: rows.map((r) => MODEL_COLORS[r.prefix]),
        },
        {
          label: seasonLabel(currentSeason),
          data: rows.map((r) => (r.season === null ? null : Math.round(r.season * 1000) / 10)),
          backgroundColor: rows.map((r) => MODEL_COLORS[r.prefix] + "80"),
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true, ticks: { callback: (v) => `${v}%` } },
        x: { ticks: { maxRotation: 0, minRotation: 0 } },
      },
      plugins: { legend: { position: "bottom" } },
    },
  });
}

function renderHeadlineKpis() {
  const { currentSeason, rows } = accuracyRows();
  const best = rows
    .filter((r) => r.prefix !== "book" && r.allTime !== null)
    .reduce((a, b) => (b.allTime > a.allTime ? b : a));
  const book = rows.find((r) => r.prefix === "book");

  document.getElementById("kpi-total-matches").textContent = predictions.length.toLocaleString();
  document.getElementById("kpi-best-model").textContent = `${best.label} · ${formatPct(best.allTime)}`;
  document.getElementById("kpi-book-overall").textContent = book.allTime === null ? "–" : formatPct(book.allTime);
  document.getElementById("kpi-gameweek-count").textContent = upcoming.length
    ? `Matchweek ${upcoming[0].matchday}`
    : "–";
}

let logLossTrendChart = null;

function renderLogLossTrendChart() {
  const seasons = [...new Set(predictions.map((p) => p.season))].sort((a, b) => a - b);
  const allPrefixes = [...MODEL_PREFIXES, "book"];

  const datasets = allPrefixes.map((prefix) => ({
    label: prefix === "book" ? "Bookmaker" : MODEL_LABELS[prefix],
    data: seasons.map((season) => {
      const rows = predictions.filter((p) => p.season === season);
      const losses = rows.map((r) => {
        const p = Math.min(Math.max(r[`${prefix}_${r.actual}`], 1e-15), 1);
        return -Math.log(p);
      });
      return losses.reduce((a, b) => a + b, 0) / losses.length;
    }),
    borderColor: MODEL_COLORS[prefix],
    backgroundColor: MODEL_COLORS[prefix],
    tension: 0.2,
    fill: false,
  }));

  // Chart.js's default "nice round numbers" auto-scale (e.g. 0.5-1.5) makes
  // this data (which only ever ranges ~0.85-1.1) look almost flat - a tight
  // range around the actual data makes the season-to-season trend visible.
  const allValues = datasets.flatMap((d) => d.data);
  const padding = 0.03;

  const canvas = document.getElementById("logloss-trend-chart");
  if (logLossTrendChart) logLossTrendChart.destroy();
  logLossTrendChart = new Chart(canvas, {
    type: "line",
    data: { labels: seasons.map(seasonLabel), datasets },
    options: {
      responsive: true,
      scales: {
        y: {
          title: { display: true, text: "Log-loss (lower is better)" },
          min: Math.floor((Math.min(...allValues) - padding) * 20) / 20,
          max: Math.ceil((Math.max(...allValues) + padding) * 20) / 20,
        },
      },
      plugins: { legend: { position: "bottom" } },
    },
  });
}

function bestPick(row, prefix) {
  const probs = { Home: row[`${prefix}_H`], Draw: row[`${prefix}_D`], Away: row[`${prefix}_A`] };
  const label = Object.keys(probs).reduce((a, b) => (probs[a] >= probs[b] ? a : b));
  return { label, prob: probs[label] };
}

function pickFor(row, prefix) {
  return bestPick(row, prefix).label;
}

function formatPct(p) {
  return `${Math.round(p * 100)}%`;
}

function seasonLabel(year) {
  return `${year}/${String(year + 1).slice(-2)}`;
}

function predictionsSeasonRangeLabel() {
  const seasons = predictions.map((p) => p.season);
  const min = Math.min(...seasons);
  const max = Math.max(...seasons);
  return `${seasonLabel(min)}–${seasonLabel(max)}`;
}

function formatMatchDate(iso) {
  const d = new Date(`${iso}T00:00:00`);
  return d.toLocaleDateString("en-GB", { weekday: "short", month: "short", day: "numeric" });
}

// football-data.co.uk's own fixtures.csv is only refreshed a couple of
// times a week (per their site: weekend fixtures' odds by Friday afternoon
// UK time, midweek fixtures' odds by Tuesday) - so "not published yet" for
// a weekend fixture on, say, a Wednesday isn't missing data, it's just not
// that day yet. Fri/Sat/Sun/Mon fixtures -> odds expected by Friday;
// Tue/Wed/Thu fixtures -> expected by Tuesday.
function expectedOddsDay(dateStr) {
  const day = new Date(`${dateStr}T00:00:00`).getDay(); // 0=Sun..6=Sat
  return [5, 6, 0, 1].includes(day) ? "Friday" : "Tuesday";
}

// Once the match's own date has arrived with still no odds, "usually by
// Friday" has already turned out to be wrong for this fixture - promising
// a day that's already passed (or is passing) reads as broken, not
// informative, so drop the guess and just say it's unavailable.
function bookmakerUnavailableHtml(dateStr) {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const matchDate = new Date(`${dateStr}T00:00:00`);
  return matchDate <= today
    ? `<span class="odds-unavailable">not available</span>`
    : `<span class="odds-unavailable">not published yet (usually by ${expectedOddsDay(dateStr)})</span>`;
}

const TAB_IDS = ["upcoming", "browse", "performance"];

function activatePanel(panelId) {
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
  document.getElementById(panelId).classList.add("active");
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === panelId));
  document.body.classList.toggle("team-theme", panelId === "team-page");
  window.scrollTo(0, 0);
}

// Hash-based routing so a team gets a real, bookmarkable/shareable URL
// (#team/Liverpool) and the browser's back/forward buttons work, rather
// than a modal popup with no address of its own.
function route() {
  const teamMatch = window.location.hash.match(/^#team\/(.+)$/);
  if (teamMatch) {
    renderTeamPage(decodeURIComponent(teamMatch[1]));
    activatePanel("team-page");
    return;
  }
  const tabId = window.location.hash.replace("#", "");
  activatePanel(TAB_IDS.includes(tabId) ? tabId : "upcoming");
}

function renderTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      window.location.hash = btn.dataset.tab;
    });
  });
  window.addEventListener("hashchange", route);
}

function renderUpcoming(fixtures) {
  const grid = document.getElementById("upcoming-grid");
  const empty = document.getElementById("upcoming-empty");

  if (!fixtures.length) {
    empty.hidden = false;
    grid.innerHTML = "";
    return;
  }
  empty.hidden = true;

  grid.innerHTML = fixtures
    .map((f) => {
      const modelRows = MODEL_PREFIXES.map((prefix) => {
        const { label, prob } = bestPick(f, prefix);
        const segments = ["Home", "Draw", "Away"]
          .map(
            (outcome) =>
              `<span class="prob-seg" style="width:${(f[`${prefix}_${outcome[0]}`] * 100).toFixed(1)}%; background:${PICK_COLORS[outcome]}"></span>`
          )
          .join("");
        return `<div class="model-row">
          <div class="model-row-top">
            <span class="model-name">${MODEL_LABELS[prefix]}</span>
            <span class="model-pick pick-${label}">${label} &middot; ${formatPct(prob)}</span>
          </div>
          <div class="prob-bar">${segments}</div>
        </div>`;
      }).join("");

      const bookmakerLine = f.odds_available
        ? (() => {
            const { label, prob } = bestPick(f, "book");
            return `<span class="pick-${label}">${label} &middot; ${formatPct(prob)}</span>`;
          })()
        : bookmakerUnavailableHtml(f.date);

      const historyNote = f.insufficient_history
        ? `<div class="history-note">One of these teams has limited match history. Treat this prediction as lower-confidence.</div>`
        : "";

      const homeForm = formBeforeDate(f.home, f.date);
      const awayForm = formBeforeDate(f.away, f.date);
      const h2h = headToHead(f.home, f.away, f.date);
      const h2hLine = h2h.meetings.length
        ? `H2H (last ${h2h.meetings.length}): ${h2h.record.W}W&ndash;${h2h.record.D}D&ndash;${h2h.record.L}L`
        : `H2H: no previous meetings on record`;

      return `<article class="match-card">
        <div class="match-date">${formatMatchDate(f.date)}</div>
        <div class="match-teams">
          ${teamLinkHtml(f.home)}
          <span class="vs">vs</span>
          ${teamLinkHtml(f.away)}
        </div>
        <div class="form-row">
          <span class="form-dots">${formDotsHtml(homeForm)}</span>
          <span class="form-dots form-dots-right">${formDotsHtml(awayForm)}</span>
        </div>
        <div class="model-rows">${modelRows}</div>
        <div class="bookmaker-row">
          <span class="bookmaker-label">Bookmaker pick</span>
          ${bookmakerLine}
        </div>
        <div class="h2h-line note">${h2hLine} (${f.home}'s record)</div>
        ${historyNote}
      </article>`;
    })
    .join("");

  attachBadgeFallbacks(grid);
}

function populateModelFilter() {
  const select = document.getElementById("model-filter");
  select.innerHTML = MODEL_PREFIXES.map(
    (p) => `<option value="${p}">${MODEL_LABELS[p]}</option>`
  ).join("");
  select.value = "rf";
  select.addEventListener("change", renderTable);
}

function populateFilters() {
  document.getElementById("browse-subtitle").textContent =
    `${predictions.length.toLocaleString()} out-of-fold walk-forward test matches, Premier League ${predictionsSeasonRangeLabel()}. Every prediction here is on a match the model never saw during training.`;

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
  const modelPrefix = document.getElementById("model-filter").value;
  const season = Number(document.getElementById("season-filter").value);
  const team = document.getElementById("team-filter").value;

  let rows = predictions.filter((p) => p.season === season);
  if (team) rows = rows.filter((p) => p.home === team || p.away === team);
  rows = [...rows].sort((a, b) => a.date.localeCompare(b.date));

  let modelCorrect = 0;
  let bookCorrect = 0;
  let evaluatedCount = 0;

  const tbody = document.getElementById("matches-tbody");
  tbody.innerHTML = rows
    .map((r) => {
      const actual = RESULT_NAMES[r.actual];

      // Cold-start matches (has_min_history == False at the time - see
      // ADR-004/007) never got a walk-forward prediction from any model.
      // Shown with the real result but neutral "n/a" picks, rather than
      // silently dropped from a team's history (the gap a user spotted).
      if (r.evaluated === false) {
        return `<tr>
          <td>${r.date}</td>
          <td>${teamLinkHtml(r.home, { withBadge: false })}</td>
          <td>${teamLinkHtml(r.away, { withBadge: false })}</td>
          <td class="note" colspan="2" title="Not enough match history at the time to generate a prediction">n/a</td>
          <td>${actual}</td>
          <td class="note">&ndash;</td>
          <td class="note">&ndash;</td>
        </tr>`;
      }

      evaluatedCount++;
      const modelPick = pickFor(r, modelPrefix);
      const bookPick = pickFor(r, "book");
      const modelOk = modelPick[0] === r.actual;
      const bookOk = bookPick[0] === r.actual;
      if (modelOk) modelCorrect++;
      if (bookOk) bookCorrect++;

      return `<tr>
        <td>${r.date}</td>
        <td>${teamLinkHtml(r.home, { withBadge: false })}</td>
        <td>${teamLinkHtml(r.away, { withBadge: false })}</td>
        <td class="pick-${modelPick}">${modelPick}</td>
        <td class="pick-${bookPick}">${bookPick}</td>
        <td>${actual}</td>
        <td class="${modelOk ? "mark-correct" : "mark-wrong"}">${modelOk ? "✓" : "✗"}</td>
        <td class="${bookOk ? "mark-correct" : "mark-wrong"}">${bookOk ? "✓" : "✗"}</td>
      </tr>`;
    })
    .join("");

  document.getElementById("kpi-count").textContent = rows.length;
  document.getElementById("kpi-model-acc").textContent = evaluatedCount
    ? `${((modelCorrect / evaluatedCount) * 100).toFixed(1)}%`
    : "–";
  document.getElementById("kpi-book-acc").textContent = evaluatedCount
    ? `${((bookCorrect / evaluatedCount) * 100).toFixed(1)}%`
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

let consensusChart = null;

function renderConsensusChart(fixtures) {
  const canvas = document.getElementById("consensus-chart");
  if (!canvas) return;
  if (!fixtures.length) {
    canvas.closest(".chart-card").hidden = true;
    return;
  }
  canvas.closest(".chart-card").hidden = false;

  const counts = MODEL_PREFIXES.map((prefix) => {
    const tally = { Home: 0, Draw: 0, Away: 0 };
    fixtures.forEach((f) => tally[bestPick(f, prefix).label]++);
    return tally;
  });

  if (consensusChart) consensusChart.destroy();
  consensusChart = new Chart(canvas, {
    type: "bar",
    data: {
      labels: MODEL_PREFIXES.map((p) => MODEL_LABELS[p]),
      datasets: ["Home", "Draw", "Away"].map((outcome) => ({
        label: outcome,
        data: counts.map((c) => c[outcome]),
        backgroundColor: PICK_COLORS[outcome],
      })),
    },
    options: {
      responsive: true,
      scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true, ticks: { stepSize: 1 } } },
      plugins: { legend: { position: "bottom" } },
    },
  });
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

  const [preds, upcomingFixtures, badges, stadiums, table, summary, ci, calibration, importance] = await Promise.all([
    loadJSON("data/predictions.json"),
    loadJSON("data/upcoming.json"),
    loadJSON("data/team_badges.json").catch(() => ({})),
    loadJSON("data/team_stadiums.json").catch(() => ({})),
    loadJSON("data/table.json"),
    loadJSON("data/model_summary.json"),
    loadJSON("data/bootstrap_ci.json"),
    loadJSON("data/calibration.json"),
    loadJSON("data/feature_importance.json"),
  ]);

  predictions = preds;
  upcoming = upcomingFixtures;
  teamBadges = badges;
  teamStadiums = stadiums;

  renderUpcoming(upcoming);
  renderConsensusChart(upcoming);
  renderLeagueTable(table);
  renderAccuracyPanel();
  renderHeadlineKpis();
  renderLogLossTrendChart();
  renderLastGameweekRecap();
  renderFormExtremes(table);
  renderResultsSplitChart();
  populateModelFilter();
  populateFilters();
  renderTable();
  renderSummaryTable(summary);
  renderCiTable(ci);
  renderCalibrationChart(calibration);
  renderImportanceChart(importance);

  route();
}

function average(values) {
  const valid = values.filter((v) => v !== null && v !== undefined && !Number.isNaN(v));
  if (!valid.length) return null;
  return valid.reduce((a, b) => a + b, 0) / valid.length;
}

function teamMatchStats(matches, name) {
  const perTeam = matches.map((p) => {
    const isHome = p.home === name;
    return {
      shots: isHome ? p.home_shots : p.away_shots,
      shotsOnTarget: isHome ? p.home_shots_on_target : p.away_shots_on_target,
      corners: isHome ? p.home_corners : p.away_corners,
      cards: isHome ? p.home_cards : p.away_cards,
      xg: isHome ? p.home_xg : p.away_xg,
    };
  });
  return {
    shots: average(perTeam.map((m) => m.shots)),
    shotsOnTarget: average(perTeam.map((m) => m.shotsOnTarget)),
    corners: average(perTeam.map((m) => m.corners)),
    cards: average(perTeam.map((m) => m.cards)),
    xg: average(perTeam.map((m) => m.xg)),
  };
}

function statCell(value, decimals = 1) {
  return value === null ? "&ndash;" : value.toFixed(decimals);
}

function teamRecord(name) {
  const matches = predictions.filter((p) => p.home === name || p.away === name);
  const record = { W: 0, D: 0, L: 0 };
  matches.forEach((p) => {
    const isHome = p.home === name;
    if (p.actual === "D") record.D++;
    else if ((p.actual === "H") === isHome) record.W++;
    else record.L++;
  });
  return { ...record, total: matches.length };
}

function outcomeFor(match, name) {
  const isHome = match.home === name;
  if (match.actual === "D") return "D";
  return (match.actual === "H") === isHome ? "W" : "L";
}

// Last N *real* results for `name` strictly before `beforeDate` - the same
// leak-free convention the models themselves use (form_pts_N features), so
// this reads as "what the model saw," not next-gameweek results leaking in.
function formBeforeDate(name, beforeDate, n = 5) {
  return [...predictions]
    .filter((p) => (p.home === name || p.away === name) && p.date < beforeDate)
    .sort((a, b) => a.date.localeCompare(b.date))
    .slice(-n)
    .map((p) => outcomeFor(p, name));
}

function formDotsHtml(letters) {
  if (!letters.length) return `<span class="note">no data</span>`;
  const cls = { W: "form-dot-w", D: "form-dot-d", L: "form-dot-l" };
  return letters.map((r) => `<span class="form-dot ${cls[r]}" title="${r}">${r}</span>`).join("");
}

// Head-to-head meetings between two teams, most recent first, strictly
// before `beforeDate`. Record is reported from `teamA`'s perspective.
function headToHead(teamA, teamB, beforeDate, limit = 5) {
  const meetings = [...predictions]
    .filter(
      (p) =>
        ((p.home === teamA && p.away === teamB) || (p.home === teamB && p.away === teamA)) &&
        p.date < beforeDate
    )
    .sort((a, b) => b.date.localeCompare(a.date))
    .slice(0, limit);

  const record = { W: 0, D: 0, L: 0 };
  meetings.forEach((m) => record[outcomeFor(m, teamA)]++);
  return { meetings, record };
}

// Matches from the most recent "gameweek window" on record - a trailing
// few-day slice around the latest date in predictions.json, mirroring how
// src/data/fixtures_openfootball.py groups an upcoming round by matchday
// rather than by an arbitrary fixed count of matches.
function lastGameweekMatches() {
  if (!predictions.length) return [];
  const maxDate = predictions.reduce((a, p) => (p.date > a ? p.date : a), predictions[0].date);
  const windowStart = new Date(`${maxDate}T00:00:00`);
  windowStart.setDate(windowStart.getDate() - 3);
  const windowStartStr = windowStart.toISOString().slice(0, 10);
  return predictions
    .filter((p) => p.date >= windowStartStr && p.date <= maxDate)
    .sort((a, b) => a.date.localeCompare(b.date));
}

function renderLastGameweekRecap() {
  const matches = lastGameweekMatches();
  const subtitle = document.getElementById("last-gameweek-subtitle");
  const list = document.getElementById("last-gameweek-list");

  if (!matches.length) {
    subtitle.textContent = "No recent results on record yet.";
    list.innerHTML = "";
    return;
  }

  const evaluatedMatches = matches.filter((m) => m.evaluated !== false);
  const correct = Object.fromEntries(MODEL_PREFIXES.map((p) => [p, 0]));
  evaluatedMatches.forEach((m) => MODEL_PREFIXES.forEach((p) => {
    if (pickFor(m, p)[0] === m.actual) correct[p]++;
  }));

  const dateRange =
    matches[0].date === matches[matches.length - 1].date
      ? formatMatchDate(matches[0].date)
      : `${formatMatchDate(matches[0].date)} to ${formatMatchDate(matches[matches.length - 1].date)}`;
  subtitle.textContent = evaluatedMatches.length
    ? `${dateRange}: ` +
      MODEL_PREFIXES.map((p) => `${SHORT_MODEL_LABELS[p]} ${correct[p]}/${evaluatedMatches.length}`).join(" · ")
    : `${dateRange}: no predictions available for these matches yet.`;

  list.innerHTML = `<div class="table-wrap"><table>
    <thead><tr><th>Match</th><th>Score</th>${MODEL_PREFIXES.map((p) => `<th>${SHORT_MODEL_LABELS[p]}</th>`).join("")}</tr></thead>
    <tbody>${matches
      .map((m) => {
        const cells =
          m.evaluated === false
            ? MODEL_PREFIXES.map(() => `<td class="note" title="Not enough match history at the time">&ndash;</td>`).join("")
            : MODEL_PREFIXES.map((p) => {
                const ok = pickFor(m, p)[0] === m.actual;
                return `<td class="${ok ? "mark-correct" : "mark-wrong"}">${ok ? "✓" : "✗"}</td>`;
              }).join("");
        return `<tr>
          <td>${teamLinkHtml(m.home, { withBadge: false })} <span class="note">vs</span> ${teamLinkHtml(m.away, { withBadge: false })}</td>
          <td>${m.home_goals}&ndash;${m.away_goals}</td>
          ${cells}
        </tr>`;
      })
      .join("")}</tbody>
  </table></div>`;
  attachBadgeFallbacks(list);
}

// Points-per-game (3/1/0) over a team's last 5 real results on record,
// regardless of date - "current form" as of right now, not tied to any
// specific fixture (unlike formBeforeDate, used for a specific matchup).
function currentForm(name, n = 5) {
  const recent = [...predictions]
    .filter((p) => p.home === name || p.away === name)
    .sort((a, b) => b.date.localeCompare(a.date))
    .slice(0, n);
  if (!recent.length) return null;
  const results = recent.map((m) => outcomeFor(m, name));
  const points = results.reduce((sum, r) => sum + { W: 3, D: 1, L: 0 }[r], 0);
  return { team: name, played: recent.length, points, form: [...results].reverse() };
}

let resultsSplitChart = null;

function renderResultsSplitChart() {
  const currentSeason = Math.max(...predictions.map((p) => p.season));
  const seasonMatches = predictions.filter((p) => p.season === currentSeason);
  document.getElementById("results-split-title").textContent = `Results this season (${seasonLabel(currentSeason)})`;

  const counts = { H: 0, D: 0, A: 0 };
  seasonMatches.forEach((m) => counts[m.actual]++);

  const canvas = document.getElementById("results-split-chart");
  if (resultsSplitChart) resultsSplitChart.destroy();
  resultsSplitChart = new Chart(canvas, {
    type: "doughnut",
    data: {
      labels: ["Home win", "Draw", "Away win"],
      datasets: [
        {
          data: [counts.H, counts.D, counts.A],
          backgroundColor: [PICK_COLORS.Home, PICK_COLORS.Draw, PICK_COLORS.Away],
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: "bottom", labels: { boxWidth: 10, padding: 10 } } },
    },
  });
}

function renderFormExtremes(table) {
  // Restricted to this season's actual top-flight clubs (from the league
  // table), not every team that ever appeared in the historical dataset -
  // otherwise a long-relegated club's last-ever top-flight form (however
  // old) could surface as "coldest form" alongside teams playing today.
  const teams = table.map((row) => row.team);
  const stats = teams.map((t) => currentForm(t)).filter((s) => s && s.played >= 3);
  const container = document.getElementById("form-extremes");
  if (!stats.length) {
    container.innerHTML = `<p class="note">Not enough recent matches on record.</p>`;
    return;
  }

  const hottest = stats.reduce((a, b) => (b.points > a.points ? b : a));
  const coldest = stats.reduce((a, b) => (b.points < a.points ? b : a));

  const block = (label, s, cls) => `
    <div class="form-extreme ${cls}">
      <span class="form-extreme-label">${label}</span>
      ${teamLinkHtml(s.team)}
      <span class="form-dots">${formDotsHtml(s.form)}</span>
      <span class="note">${s.points} pts / last ${s.played}</span>
    </div>`;

  container.innerHTML = block("Hottest form", hottest, "form-extreme-hot") + block("Coldest form", coldest, "form-extreme-cold");
  attachBadgeFallbacks(container);
}

// Perceived brightness (ITU-R BT.601) of a #RRGGBB color, used to pick
// readable white-or-dark text/pattern overlays for an arbitrary club color
// banner - some clubs (Norwich yellow, Watford yellow) are far too light
// for white text.
function isLightColor(hex) {
  if (!hex.startsWith("#")) return false; // hsl(...) fallback colors are all mid-tone by construction
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return (r * 299 + g * 587 + b * 114) / 1000 > 165;
}

function withAlpha(color, alpha) {
  if (color.startsWith("#")) {
    const hexAlpha = Math.round(alpha * 255)
      .toString(16)
      .padStart(2, "0");
    return `${color}${hexAlpha}`;
  }
  return color.replace("hsl(", "hsla(").replace(")", `, ${alpha})`);
}

// A generic "night match" stadium illustration (floodlights, crowd stand,
// pitch edge) tinted in the club's own color via CSS color-mix() - built
// from primitives rather than a photo so it needs no image asset/license
// and re-colors instantly per team.
function stadiumBannerSvg(accent) {
  const w = 900;
  const h = 220;

  const scallops = 26;
  const scallopW = w / scallops;
  let standPath = `M0,168`;
  for (let i = 0; i < scallops; i++) {
    const midX = i * scallopW + scallopW / 2;
    const endX = (i + 1) * scallopW;
    standPath += ` Q${midX.toFixed(1)},148 ${endX.toFixed(1)},168`;
  }
  standPath += ` L${w},${h} L0,${h} Z`;

  const stripeW = 30;
  let stripes = "";
  for (let i = 0; i < Math.ceil(w / stripeW); i++) {
    const shade = i % 2 === 0 ? 42 : 24;
    stripes += `<rect x="${i * stripeW}" y="196" width="${stripeW}" height="24" fill="color-mix(in srgb, ${accent} ${shade}%, #04150d)" />`;
  }

  let crowd = "";
  for (let i = 0; i < 60; i++) {
    const x = (i * 47 + (i % 7) * 13) % w;
    const y = 152 + ((i * 17) % 16);
    const r = 1 + (i % 3) * 0.5;
    const op = (0.12 + (i % 5) * 0.07).toFixed(2);
    crowd += `<circle cx="${x}" cy="${y}" r="${r}" fill="rgba(255,255,255,${op})" />`;
  }

  return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" role="presentation" aria-hidden="true">
    <defs>
      <linearGradient id="stadiumSky" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="color-mix(in srgb, ${accent} 26%, #030f0a)" />
        <stop offset="100%" stop-color="color-mix(in srgb, ${accent} 58%, #04150d)" />
      </linearGradient>
      <radialGradient id="floodGlowL" cx="8%" cy="6%" r="60%">
        <stop offset="0%" stop-color="rgba(255,255,255,0.55)" />
        <stop offset="100%" stop-color="rgba(255,255,255,0)" />
      </radialGradient>
      <radialGradient id="floodGlowR" cx="92%" cy="6%" r="60%">
        <stop offset="0%" stop-color="rgba(255,255,255,0.55)" />
        <stop offset="100%" stop-color="rgba(255,255,255,0)" />
      </radialGradient>
    </defs>
    <rect width="${w}" height="${h}" fill="url(#stadiumSky)" />
    <rect width="${w}" height="${h}" fill="url(#floodGlowL)" />
    <rect width="${w}" height="${h}" fill="url(#floodGlowR)" />
    <rect x="14" y="10" width="34" height="11" rx="2" fill="rgba(255,255,255,0.85)" />
    <rect x="29" y="21" width="5" height="70" fill="rgba(255,255,255,0.5)" />
    <rect x="${w - 48}" y="10" width="34" height="11" rx="2" fill="rgba(255,255,255,0.85)" />
    <rect x="${w - 34}" y="21" width="5" height="70" fill="rgba(255,255,255,0.5)" />
    ${crowd}
    <path d="${standPath}" fill="color-mix(in srgb, ${accent} 16%, #020a06)" opacity="0.92" />
    ${stripes}
  </svg>`;
}

function renderTeamPage(name) {
  const nextFixtures = upcoming.filter((f) => f.home === name || f.away === name);
  const recent = [...predictions]
    .filter((p) => p.home === name || p.away === name)
    .sort((a, b) => b.date.localeCompare(a.date))
    .slice(0, 20);
  const record = teamRecord(name);

  const nextFixtureHtml = nextFixtures.length
    ? nextFixtures
        .map((f) => {
          const opponent = f.home === name ? f.away : f.home;
          const isHome = f.home === name;
          const venue = isHome ? "Home" : "Away";
          const { label, prob } = bestPick(f, "rf");
          const teamOutcome = label === "Draw" ? "Draw" : (label === "Home") === isHome ? "Win" : "Loss";
          const teamOutcomeClass = { Win: "mark-correct", Draw: "", Loss: "mark-wrong" }[teamOutcome];
          const h2h = headToHead(name, opponent, f.date);
          const h2hText = h2h.meetings.length
            ? `H2H (last ${h2h.meetings.length}): ${h2h.record.W}W&ndash;${h2h.record.D}D&ndash;${h2h.record.L}L`
            : `H2H: no previous meetings on record`;
          return `<div class="profile-next-fixture">
            <span class="note">${formatMatchDate(f.date)} &middot; ${venue} vs ${opponent}</span>
            <span class="model-pick ${teamOutcomeClass}">Random Forest: ${teamOutcome} &middot; ${formatPct(prob)}</span>
            <span class="note">${h2hText}</span>
          </div>`;
        })
        .join("")
    : `<p class="note">No upcoming fixture in the current data.</p>`;

  const stats = teamMatchStats(recent, name);

  const recentRowsHtml =
    recent
      .map((p) => {
        const isHome = p.home === name;
        const opponent = isHome ? p.away : p.home;
        const teamGoals = isHome ? p.home_goals : p.away_goals;
        const oppGoals = isHome ? p.away_goals : p.home_goals;
        const teamXg = isHome ? p.home_xg : p.away_xg;
        const outcome = p.actual === "D" ? "D" : (p.actual === "H") === isHome ? "W" : "L";
        const outcomeClass = { W: "mark-correct", D: "", L: "mark-wrong" }[outcome];
        const rfCell =
          p.evaluated === false
            ? `<td class="note" title="Not enough match history at the time">&ndash;</td>`
            : (() => {
                const modelOk = pickFor(p, "rf")[0] === p.actual;
                return `<td class="${modelOk ? "mark-correct" : "mark-wrong"}">${modelOk ? "✓" : "✗"}</td>`;
              })();
        return `<tr>
          <td>${p.date}</td>
          <td>${isHome ? "vs" : "@"} ${opponent}</td>
          <td class="${outcomeClass}">${teamGoals}&ndash;${oppGoals}</td>
          <td>${teamXg === null ? "&ndash;" : teamXg.toFixed(2)}</td>
          ${rfCell}
        </tr>`;
      })
      .join("") || `<tr><td colspan="5" class="note">No historical matches on record.</td></tr>`;

  const progression = seasonPointsProgression(name);
  const progressionHtml =
    progression.points.length >= 2
      ? `<h3>${seasonLabel(progression.season)} points progression</h3>
         <div class="chart-card team-progression-card"><canvas id="team-points-chart"></canvas></div>`
      : "";

  const accent = teamColor(name);
  const textColor = isLightColor(accent) ? "#1c1c1e" : "#ffffff";
  const patternColor = isLightColor(accent) ? "rgba(0,0,0,0.08)" : "rgba(255,255,255,0.14)";

  const page = document.getElementById("team-page");
  const content = document.getElementById("team-page-content");
  for (const el of [document.body, page, content]) {
    el.style.setProperty("--team-accent", accent);
    el.style.setProperty("--team-text", textColor);
    el.style.setProperty("--team-pattern", patternColor);
  }

  const stadium = teamStadiums[name];
  const taglineParts = [];
  if (stadium?.nickname) taglineParts.push(stadium.nickname);
  if (stadium?.founded) taglineParts.push(`Est. ${stadium.founded}`);
  const taglineHtml = taglineParts.length ? `<div class="profile-tagline">${taglineParts.join(" &middot; ")}</div>` : "";

  content.innerHTML = `
    ${stadiumBannerHtml(name, accent)}
    <div class="profile-header">
      <div class="profile-crest-badge">${teamCrestHtml(name)}</div>
      <div>
        <h2>${name}</h2>
        ${taglineHtml}
      </div>
    </div>
    <div class="profile-record">${record.W}W &ndash; ${record.D}D &ndash; ${record.L}L
      <span class="note">(${record.total} matches on record, ${predictionsSeasonRangeLabel()})</span>
    </div>
    <h3>Next fixture</h3>
    ${nextFixtureHtml}
    <h3>Recent averages (last ${recent.length} games)</h3>
    <div class="profile-stats-grid">
      <div class="profile-stat"><span class="stat-value">${statCell(stats.shots)}</span><span class="stat-label">Shots</span></div>
      <div class="profile-stat"><span class="stat-value">${statCell(stats.shotsOnTarget)}</span><span class="stat-label">On target</span></div>
      <div class="profile-stat"><span class="stat-value">${statCell(stats.corners)}</span><span class="stat-label">Corners</span></div>
      <div class="profile-stat"><span class="stat-value">${statCell(stats.cards)}</span><span class="stat-label">Cards</span></div>
      <div class="profile-stat"><span class="stat-value">${statCell(stats.xg, 2)}</span><span class="stat-label">xG</span></div>
    </div>
    ${progressionHtml}
    <h3>Recent form (last ${recent.length})</h3>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Date</th><th>Opponent</th><th>Score</th><th>xG</th><th>RF pick</th></tr></thead>
        <tbody>${recentRowsHtml}</tbody>
      </table>
    </div>
  `;
  attachBadgeFallbacks(content);
  attachStadiumFallback(content, accent);

  if (progression.points.length >= 2) {
    renderTeamPointsChart(progression, accent);
  }
}

function seasonPointsProgression(name) {
  const season = Math.max(...predictions.map((p) => p.season));
  const matches = [...predictions]
    .filter((p) => (p.home === name || p.away === name) && p.season === season)
    .sort((a, b) => a.date.localeCompare(b.date));

  let total = 0;
  const points = matches.map((m) => {
    total += { W: 3, D: 1, L: 0 }[outcomeFor(m, name)];
    return total;
  });
  return { season, labels: matches.map((_, i) => `MD${i + 1}`), points };
}

let teamPointsChart = null;

function renderTeamPointsChart(progression, color) {
  const canvas = document.getElementById("team-points-chart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const gradient = ctx.createLinearGradient(0, 0, 0, 200);
  gradient.addColorStop(0, withAlpha(color, 0.35));
  gradient.addColorStop(1, withAlpha(color, 0));

  if (teamPointsChart) teamPointsChart.destroy();
  teamPointsChart = new Chart(canvas, {
    type: "line",
    data: {
      labels: progression.labels,
      datasets: [
        {
          label: "Cumulative points",
          data: progression.points,
          borderColor: color,
          backgroundColor: gradient,
          pointBackgroundColor: color,
          pointBorderColor: "#fff",
          pointRadius: 4,
          borderWidth: 2.5,
          tension: 0.25,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      scales: { y: { beginAtZero: true, ticks: { stepSize: 3 } } },
      plugins: { legend: { display: false } },
    },
  });
}

init();
