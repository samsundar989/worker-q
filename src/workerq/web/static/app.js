/* Router, data fetching and the shell.
 *
 * Polling rather than server-sent events, at the same 2 s the TUI uses. SSE
 * would be tidier but holds a thread each in ThreadingHTTPServer, and for a
 * single-user local dashboard that trade is not worth making. Only the live
 * view polls; history and accuracy are static until asked to refresh.
 */

import { hideTip } from "./charts.js";
import { banner, h, spinner } from "./ui.js";
import {
  viewAccuracy, viewEfficiency, viewHistory, viewJob, viewMachines, viewNow,
} from "./views.js";

const LIVE_INTERVAL_MS = 2000;

const state = {
  route: { name: "now", params: {} },
  filters: {
    search: "", state: [], project: [], node: [], priority: [],
    sort: "id", dir: "desc", limit: 50, offset: 0,
  },
  facets: {},
  cache: {},
  lastAction: null,
  timer: null,
  inflight: false,
};

/* -- fetching ------------------------------------------------------------- */
async function get(path) {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  const payload = await response.json().catch(() => ({ error: "unreadable reply" }));
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

async function post(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const payload = await response.json().catch(() => ({ error: "unreadable reply" }));
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function queryString(filters) {
  const params = new URLSearchParams();
  for (const key of ["state", "project", "node", "priority"]) {
    for (const value of filters[key] || []) params.append(key, value);
  }
  if (filters.search) params.set("search", filters.search);
  params.set("sort", filters.sort);
  params.set("dir", filters.dir);
  params.set("limit", String(filters.limit));
  params.set("offset", String(filters.offset));
  return params.toString();
}

/* -- routing -------------------------------------------------------------- */
function parseHash() {
  const raw = (location.hash || "#/now").replace(/^#\/?/, "");
  const parts = raw.split("/").filter(Boolean);
  if (!parts.length) return { name: "now", params: {} };
  if (parts[0] === "job" && parts[1]) return { name: "job", params: { id: Number(parts[1]) } };
  const known = ["now", "history", "accuracy", "machines", "efficiency"];
  return { name: known.includes(parts[0]) ? parts[0] : "now", params: {} };
}

const TABS = [
  ["now", "Now"],
  ["history", "History"],
  ["accuracy", "Accuracy"],
  ["machines", "Machines"],
  ["efficiency", "Efficiency"],
];

function renderTabs() {
  const nav = document.getElementById("tabs");
  nav.replaceChildren(...TABS.map(([key, label]) => h("a", {
    href: `#/${key}`,
    class: state.route.name === key ? "active" : "",
    text: label,
  })));
}

/* -- theme ---------------------------------------------------------------- */
function initTheme() {
  const stored = (() => {
    try { return localStorage.getItem("workerq-theme"); } catch { return null; }
  })();
  if (stored === "dark" || stored === "light") {
    document.documentElement.dataset.theme = stored;
  }
  document.getElementById("theme").addEventListener("click", () => {
    const now = document.documentElement.dataset.theme;
    const isDark = now
      ? now === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    const next = isDark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("workerq-theme", next); } catch { /* private mode */ }
  });
}

/* -- rendering ------------------------------------------------------------ */
function setStatus(text, kind = "muted") {
  const node = document.getElementById("status");
  node.className = `small ${kind}`;
  node.textContent = text;
}

async function loadAndRender({ quiet = false } = {}) {
  const main = document.getElementById("main");
  if (state.inflight) return;
  state.inflight = true;
  if (!quiet) {
    setStatus("loading…");
    if (!state.cache[state.route.name]) main.replaceChildren(spinner());
  }
  try {
    const node = await buildView();
    hideTip();
    main.replaceChildren(node);
    setStatus(`updated ${new Date().toLocaleTimeString()}`);
  } catch (error) {
    main.replaceChildren(banner("bad", `Could not load: ${error.message}`));
    setStatus("failed", "verdict-under");
  } finally {
    state.inflight = false;
  }
}

async function buildView() {
  const { name, params } = state.route;
  if (name === "now") {
    const data = await get("/api/overview");
    return viewNow(data, {});
  }
  if (name === "history") {
    if (!state.facets.project) state.facets = await get("/api/facets");
    const data = await get(`/api/jobs?${queryString(state.filters)}`);
    return viewHistory(data, {
      filters: state.filters,
      facets: state.facets,
      setFilter: (patch) => {
        Object.assign(state.filters, patch);
        loadAndRender();
      },
    });
  }
  if (name === "job") {
    const id = params.id;
    const [job, series, log] = await Promise.all([
      get(`/api/jobs/${id}`),
      get(`/api/jobs/${id}/series`).catch(() => null),
      get(`/api/jobs/${id}/log`).catch(() => null),
    ]);
    return viewJob(job, {
      jobId: id,
      series,
      log,
      suggestion: job.suggestion,
      lastAction: state.lastAction,
      reloadLog: () => loadAndRender(),
      act: async (jobId, action, body) => {
        try {
          state.lastAction = await post(`/api/jobs/${jobId}/${action}`, body);
        } catch (error) {
          state.lastAction = { error: error.message };
        }
        loadAndRender();
      },
    });
  }
  if (name === "accuracy") return viewAccuracy(await get("/api/accuracy?limit=600"), {});
  if (name === "machines") return viewMachines(await get("/api/machines"), {});
  if (name === "efficiency") return viewEfficiency(await get("/api/efficiency"), {});
  return banner("bad", "Unknown view.");
}

function schedulePolling() {
  clearInterval(state.timer);
  state.timer = null;
  // Only the live view refreshes on its own. Re-fetching a filtered history
  // page under someone reading it is worse than stale data.
  if (state.route.name === "now") {
    state.timer = setInterval(() => {
      if (document.hidden) return;
      loadAndRender({ quiet: true });
    }, LIVE_INTERVAL_MS);
  }
}

function onRoute() {
  const next = parseHash();
  const changed = next.name !== state.route.name
    || next.params.id !== state.route.params.id;
  state.route = next;
  if (changed) state.lastAction = null;
  renderTabs();
  schedulePolling();
  loadAndRender();
}

window.addEventListener("hashchange", onRoute);
window.addEventListener("DOMContentLoaded", () => {
  initTheme();
  document.getElementById("refresh").addEventListener("click", () => loadAndRender());
  onRoute();
  get("/api/overview").then((d) => {
    document.getElementById("version").textContent = `v${d.version}`;
  }).catch(() => {});
});
