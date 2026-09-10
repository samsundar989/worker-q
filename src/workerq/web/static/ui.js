/* DOM helpers and the pieces more than one view needs. */

import { duration, gib, hideTip, pct, shortTime } from "./charts.js";

export function h(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === "dataset") Object.assign(node.dataset, value);
    else node.setAttribute(key, String(value));
  }
  for (const child of [children].flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function card(title, hint, children) {
  return h("section", { class: "card" }, [
    title ? h("h2", { text: title }) : null,
    hint ? h("p", { class: "hint", text: hint }) : null,
    ...[children].flat(Infinity),
  ]);
}

export function tile(label, value, sub, alarm = false) {
  return h("div", { class: alarm ? "tile alarm" : "tile" }, [
    h("div", { class: "label", text: label }),
    h("div", { class: "value", text: value }),
    sub ? h("div", { class: "sub", text: sub }) : null,
  ]);
}

export function stateBadge(state) {
  return h("span", { class: `badge state-${state}` }, [
    h("span", { class: "dot" }), state,
  ]);
}

/* Verdicts always ship as an icon plus a word. The colour is a reinforcement,
 * never the message - which is both the accessibility rule and, here, the
 * safety one: "over" and "under" must not be distinguishable by hue alone. */
const VERDICT_GLYPH = {
  "under-declared": "▲",
  "severely-over-declared": "▼",
  "over-declared": "▽",
  ok: "●",
  "low-confidence": "?",
  unmeasured: "–",
};

const VERDICT_WORD = {
  "under-declared": "over budget",
  "severely-over-declared": "way under-used",
  "over-declared": "under-used",
  ok: "well sized",
  "low-confidence": "too few samples",
  unmeasured: "not measured",
};

export function verdictBadge(verdict, ratio) {
  return h("span", { class: `badge ${verdictClass(verdict)}`, title: VERDICT_WORD[verdict] }, [
    h("span", { text: VERDICT_GLYPH[verdict] ?? "·" }),
    ratio === null || ratio === undefined ? VERDICT_WORD[verdict] : pct(ratio),
  ]);
}

export function verdictClass(verdict) {
  return {
    "under-declared": "verdict-under",
    "severely-over-declared": "verdict-severely-over",
    "over-declared": "verdict-over",
    ok: "verdict-ok",
    "low-confidence": "verdict-low-confidence",
    unmeasured: "verdict-unmeasured",
  }[verdict] ?? "verdict-unmeasured";
}

export function verdictColor(verdict) {
  return {
    "under-declared": "var(--critical)",
    "severely-over-declared": "var(--warning)",
    "over-declared": "var(--warning)",
    ok: "var(--good)",
  }[verdict] ?? "var(--text-muted)";
}

export function verdictWord(verdict) {
  return VERDICT_WORD[verdict] ?? verdict;
}

export const STATE_COLOR = {
  RUNNING: "var(--good)",
  SUCCEEDED: "var(--good)",
  QUEUED: "var(--warning)",
  PREPARING: "var(--series-1)",
  FAILED: "var(--critical)",
  LOST: "var(--critical)",
  CANCELLED: "var(--text-muted)",
};

export function nodeChip(node) {
  return h("span", { class: "node-chip", text: node || "local" });
}

export function jobLink(id, label) {
  return h("a", {
    class: "jobref",
    href: `#/job/${id}`,
    text: label ?? `#${id}`,
  });
}

export function meterRow(label, used, total, value) {
  const ratio = total ? Math.max(0, Math.min(1, used / total)) : 0;
  const tone = ratio > 0.9 ? "critical" : ratio > 0.75 ? "warning" : "good";
  return h("div", { class: "meter-row" }, [
    h("span", { class: "k", text: label }),
    h("div", { class: `meter ${tone}` }, [
      h("i", { style: `width:${(ratio * 100).toFixed(1)}%` }),
    ]),
    h("span", { class: "v", text: value }),
  ]);
}

export function empty(message) {
  return h("div", { class: "empty", text: message });
}

export function spinner() {
  return h("div", { class: "empty" }, [h("span", { class: "spin" })]);
}

export function banner(kind, text, extra) {
  return h("div", { class: `banner ${kind}` }, [h("div", {}, [text, extra])]);
}

/** The "why does it say that?" disclosure every panel carries.
 *
 * A dashboard where numbers appear without provenance is a dashboard nobody
 * can debug. Each one names the call that produced the data and shows the raw
 * payload, so anything seen here can be reproduced in a terminal. */
export function why(source, payload, command) {
  return h("details", { class: "why" }, [
    h("summary", { text: "where this comes from" }),
    h("div", { class: "small muted", style: "margin:6px 0" }, [
      "Produced by ",
      h("code", { class: "cmd", text: source }),
      command ? " · same data from the CLI: " : null,
      command ? h("code", { class: "cmd", text: command }) : null,
    ]),
    h("pre", { class: "code", text: JSON.stringify(payload, null, 2).slice(0, 40000) }),
  ]);
}

/** The job table shared by the live view and history. */
export function jobTable(jobs, opts = {}) {
  const { onSort, sort, dir, showUsage = true, showWait = true } = opts;
  const th = (label, key, numeric) => {
    if (!onSort || !key) {
      return h("th", { class: numeric ? "num" : "", text: label });
    }
    const active = sort === key;
    return h("th", {
      class: `sortable${numeric ? " num" : ""}`,
      onclick: () => onSort(key),
      text: active ? `${label} ${dir === "asc" ? "↑" : "↓"}` : label,
    });
  };

  const head = h("tr", {}, [
    th("ID", "id", true),
    th("State", "state"),
    th("Project", "project"),
    th("Node", "node"),
    th("What"),
    showWait ? th("Wait", "wait", true) : null,
    th("Runtime", "runtime", true),
    showUsage ? th("Budget", "requested_ram_mib", true) : null,
    showUsage ? th("Peak commit", "peak_ram_mib", true) : null,
    showUsage ? th("Used") : null,
    th("Queued", "queued_at", true),
  ]);

  const rows = jobs.map((job) => {
    const u = job.usage || {};
    const what = job.description || (job.command || []).join(" ");
    const tr = h("tr", {
      class: "clickable",
      onclick: () => { hideTip(); location.hash = `#/job/${job.id}`; },
    }, [
      h("td", { class: "num" }, [jobLink(job.id)]),
      h("td", {}, [stateBadge(job.state)]),
      h("td", { class: "truncate", text: job.project, title: job.project }),
      h("td", {}, [nodeChip(job.node)]),
      h("td", { class: "wide-truncate", text: what, title: what }),
      showWait ? h("td", { class: "num", text: duration(job.wait_seconds) }) : null,
      h("td", { class: "num", text: duration(job.runtime_seconds) }),
      showUsage ? h("td", { class: "num", text: gib(u.commit_budget_mib) }) : null,
      showUsage ? h("td", { class: "num", text: gib(u.peak_commit_mib) }) : null,
      showUsage ? h("td", {}, [
        u.verdict ? verdictBadge(u.verdict, u.commit_ratio) : h("span", { class: "muted", text: "-" }),
      ]) : null,
      h("td", { class: "num nowrap muted", text: shortTime(job.queued_at) }),
    ]);
    return tr;
  });

  return h("div", { class: "table-wrap" }, [
    h("table", {}, [h("thead", {}, [head]), h("tbody", {}, rows)]),
  ]);
}
