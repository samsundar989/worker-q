/* Inline SVG chart primitives.
 *
 * Hand-rolled rather than pulled from a library: there are five chart shapes
 * here, all simple, and the alternative is vendoring a megabyte of JavaScript
 * into a tool whose entire dependency list is Typer and Rich. Everything below
 * is theme-aware through CSS custom properties, so light and dark are one
 * definition rather than two.
 *
 * Rules followed throughout, from the data-viz method:
 *  - one axis, never two scales on one chart;
 *  - categorical hues assigned in fixed order, never cycled;
 *  - status colour never carries meaning alone - it always has a word beside it;
 *  - a hover layer on every plot, because an SVG chart is interactive by nature;
 *  - selective direct labels, never a number on every mark.
 */

const NS = "http://www.w3.org/2000/svg";

export const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];

export function el(name, attrs = {}, children = []) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    node.setAttribute(k, String(v));
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined) continue;
    node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

/* -- tooltip -------------------------------------------------------------- */
let tipNode = null;

function tip() {
  if (!tipNode) {
    tipNode = document.createElement("div");
    tipNode.className = "tooltip";
    tipNode.hidden = true;
    document.body.appendChild(tipNode);
  }
  return tipNode;
}

export function showTip(event, html) {
  const node = tip();
  node.innerHTML = html;
  node.hidden = false;
  const pad = 14;
  const rect = node.getBoundingClientRect();
  let x = event.clientX + pad;
  let y = event.clientY + pad;
  if (x + rect.width > window.innerWidth - 8) x = event.clientX - rect.width - pad;
  if (y + rect.height > window.innerHeight - 8) y = event.clientY - rect.height - pad;
  node.style.left = `${Math.max(8, x)}px`;
  node.style.top = `${Math.max(8, y)}px`;
}

export function hideTip() {
  if (tipNode) tipNode.hidden = true;
}

/** Attach a hover tooltip to any mark. Hit target is the mark plus padding. */
export function hoverable(node, html) {
  node.addEventListener("mousemove", (e) => showTip(e, html));
  node.addEventListener("mouseleave", hideTip);
  node.style.cursor = "default";
  return node;
}

/* -- formatting ----------------------------------------------------------- */
export function gib(mib) {
  if (mib === null || mib === undefined) return "-";
  return `${(mib / 1024).toFixed(mib < 10240 ? 1 : 0)} GiB`;
}

export function pct(value, digits = 0) {
  if (value === null || value === undefined) return "-";
  return `${(value * 100).toFixed(digits)}%`;
}

export function duration(seconds) {
  if (seconds === null || seconds === undefined) return "-";
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  const h = Math.floor(s / 3600);
  const m = Math.round((s % 3600) / 60);
  return `${h}h ${m}m`;
}

export function shortTime(iso) {
  if (!iso) return "-";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "-";
  return d.toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

export function clockTime(iso) {
  if (!iso) return "-";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "-";
  return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

export function tipHtml(title, rows) {
  const body = rows
    .filter(Boolean)
    .map(([k, v]) => `<div class="t-row"><span>${escapeHtml(k)}</span><span class="t-v">${escapeHtml(v)}</span></div>`)
    .join("");
  return `<div class="t-title">${escapeHtml(title)}</div>${body}`;
}

/* -- legend --------------------------------------------------------------- */
export function legend(items) {
  const wrap = document.createElement("div");
  wrap.className = "legend";
  for (const item of items) {
    const node = document.createElement("span");
    node.className = "item";
    node.innerHTML =
      `<span class="swatch" style="background:${item.color}"></span>` +
      `<span>${escapeHtml(item.label)}</span>`;
    wrap.appendChild(node);
  }
  return wrap;
}

function svg(width, height) {
  return el("svg", {
    class: "chart",
    viewBox: `0 0 ${width} ${height}`,
    preserveAspectRatio: "xMidYMid meet",
    role: "img",
  });
}

function niceMax(value) {
  if (!value || value <= 0) return 1;
  const mag = 10 ** Math.floor(Math.log10(value));
  const step = [1, 2, 2.5, 5, 10].find((s) => value <= s * mag) ?? 10;
  return step * mag;
}

/* -- horizontal bars ------------------------------------------------------ */
/**
 * Ranked horizontal bars. One series, so no legend: the title names it.
 * Values are direct-labelled at the end of each bar, which is also the
 * relief the light-mode contrast warning requires.
 */
export function barsH(rows, opts = {}) {
  const {
    height = 22, gap = 8, labelWidth = 190, valueWidth = 92,
    color = "var(--series-1)", format = (v) => String(v),
    // Long labels have to be allowed to be long: two block reasons truncated
    // to the same prefix read as one reason counted twice.
    labelChars = 30,
  } = opts;
  const width = 900;
  const max = niceMax(Math.max(...rows.map((r) => r.value), 0));
  const plotLeft = labelWidth;
  const plotWidth = width - labelWidth - valueWidth;
  const total = rows.length * (height + gap) + 4;
  const node = svg(width, Math.max(total, 30));

  rows.forEach((row, i) => {
    const y = i * (height + gap) + 2;
    const w = max ? Math.max(2, (row.value / max) * plotWidth) : 2;
    node.appendChild(el("text", {
      x: labelWidth - 10, y: y + height / 2 + 4, "text-anchor": "end",
    }, row.label.length > labelChars
      ? `${row.label.slice(0, labelChars - 1)}…`
      : row.label));
    node.appendChild(el("rect", {
      class: "track", x: plotLeft, y, width: plotWidth, height,
    }));
    const bar = el("rect", {
      class: "mark", x: plotLeft, y, width: w, height,
      fill: row.color || color,
    });
    node.appendChild(hoverable(bar, row.tip || tipHtml(row.label, [["value", format(row.value)]])));
    node.appendChild(el("text", {
      class: "direct-label", x: plotLeft + plotWidth + 8, y: y + height / 2 + 4,
    }, format(row.value)));
  });
  return node;
}

/* -- budget vs peak ------------------------------------------------------- */
/**
 * A bullet bar per job: the declared budget as a recessive track, the measured
 * peak as the mark, and a rule at 100% of budget. The whole point of the view
 * is where the mark sits relative to that rule, so the rule is drawn, not left
 * to be inferred from two numbers in a table.
 *
 * Colour comes from the verdict, and every bar is labelled with its ratio, so
 * the status hue is never the only thing saying "this one is over".
 */
export function budgetBars(rows, opts = {}) {
  const { height = 20, gap = 9, labelWidth = 150, valueWidth = 150 } = opts;
  const width = 900;
  const plotLeft = labelWidth;
  const plotWidth = width - labelWidth - valueWidth;
  // Scaled so a bar at budget fills 70% of the track: an over-budget job has
  // somewhere to go, instead of being clipped at the same length as a job that
  // landed exactly on its declaration.
  const maxRatio = Math.max(1.45, ...rows.map((r) => r.ratio ?? 0)) * 1.02;
  const scale = (ratio) => (ratio / maxRatio) * plotWidth;
  const node = svg(width, Math.max(rows.length * (height + gap) + 22, 40));

  const ruleX = plotLeft + scale(1);
  rows.forEach((row, i) => {
    const y = i * (height + gap) + 2;
    node.appendChild(el("text", {
      x: labelWidth - 10, y: y + height / 2 + 4, "text-anchor": "end",
    }, row.label));
    node.appendChild(el("rect", {
      class: "track", x: plotLeft, y, width: plotWidth, height,
    }));
    const w = Math.max(2, scale(row.ratio ?? 0));
    const bar = el("rect", {
      class: "mark", x: plotLeft, y, width: w, height, fill: row.color,
    });
    node.appendChild(hoverable(bar, row.tip));
    node.appendChild(el("text", {
      class: "direct-label", x: plotLeft + plotWidth + 8, y: y + height / 2 + 4,
    }, row.valueLabel));
  });

  // Budget rule, drawn last so it sits above the marks it judges.
  node.appendChild(el("line", {
    x1: ruleX, x2: ruleX, y1: 0, y2: rows.length * (height + gap),
    stroke: "var(--text-secondary)", "stroke-width": 1.5, "stroke-dasharray": "3 3",
  }));
  node.appendChild(el("text", {
    x: ruleX, y: rows.length * (height + gap) + 14, "text-anchor": "middle",
  }, "declared budget"));
  return node;
}

/* -- multi-series line ---------------------------------------------------- */
/**
 * Percentages only, so there is one axis and one scale. Mixing GiB and percent
 * on two y-axes is the most common way to make a chart that cannot be read;
 * normalising instead keeps every series comparable.
 */
export function linesPct(series, opts = {}) {
  const { height = 210, marks = [] } = opts;
  const width = 900;
  const pad = { top: 12, right: 14, bottom: 24, left: 38 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const points = series[0]?.values.length ?? 0;
  const node = svg(width, height);
  if (points < 2) {
    node.appendChild(el("text", { x: width / 2, y: height / 2, "text-anchor": "middle" },
      "not enough samples to plot"));
    return node;
  }

  const x = (i) => pad.left + (i / (points - 1)) * plotW;
  const y = (v) => pad.top + plotH - (Math.max(0, Math.min(100, v)) / 100) * plotH;

  for (const gridValue of [0, 25, 50, 75, 100]) {
    node.appendChild(el("line", {
      class: "grid-line", x1: pad.left, x2: pad.left + plotW,
      y1: y(gridValue), y2: y(gridValue),
    }));
    node.appendChild(el("text", {
      x: pad.left - 7, y: y(gridValue) + 4, "text-anchor": "end",
    }, `${gridValue}%`));
  }

  // Vertical annotations (job start, job end) before the lines, so the data
  // is never hidden behind the annotation.
  for (const mark of marks) {
    const mx = x(mark.index);
    node.appendChild(el("line", {
      x1: mx, x2: mx, y1: pad.top, y2: pad.top + plotH,
      stroke: "var(--text-muted)", "stroke-width": 1, "stroke-dasharray": "2 3",
    }));
    node.appendChild(el("text", { x: mx + 4, y: pad.top + 10 }, mark.label));
  }

  series.forEach((s, index) => {
    const d = s.values
      .map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`)
      .join(" ");
    node.appendChild(el("path", {
      class: "series-line", d, stroke: s.color ?? SERIES[index],
    }));
  });

  // One crosshair for every series at that instant: reading three separate
  // tooltips to compare three lines at one moment is the thing that makes
  // multi-series charts tedious.
  const crosshair = el("line", {
    x1: 0, x2: 0, y1: pad.top, y2: pad.top + plotH,
    stroke: "var(--text-secondary)", "stroke-width": 1, opacity: 0,
  });
  node.appendChild(crosshair);
  const overlay = el("rect", {
    x: pad.left, y: pad.top, width: plotW, height: plotH, fill: "transparent",
  });
  overlay.addEventListener("mousemove", (event) => {
    const box = node.getBoundingClientRect();
    const ratio = (event.clientX - box.left) / box.width;
    const i = Math.max(0, Math.min(points - 1, Math.round(ratio * width / width * (points - 1))));
    const px = x(i);
    crosshair.setAttribute("x1", px);
    crosshair.setAttribute("x2", px);
    crosshair.setAttribute("opacity", "0.5");
    showTip(event, tipHtml(
      opts.labelAt ? opts.labelAt(i) : `sample ${i + 1}`,
      series.map((s) => [s.label, `${s.values[i]?.toFixed(0) ?? "-"}%`]),
    ));
  });
  overlay.addEventListener("mouseleave", () => {
    crosshair.setAttribute("opacity", "0");
    hideTip();
  });
  node.appendChild(overlay);
  return node;
}

/* -- stacked wait/run ----------------------------------------------------- */
/** Two segments per row, with a 2px surface gap between them so the boundary
 *  reads as a boundary rather than a colour change. */
export function stackedH(rows, opts = {}) {
  const { height = 20, gap = 9, labelWidth = 160, valueWidth = 150 } = opts;
  const width = 900;
  const plotLeft = labelWidth;
  const plotWidth = width - labelWidth - valueWidth;
  const max = niceMax(Math.max(...rows.map((r) => r.a + r.b), 0));
  const node = svg(width, Math.max(rows.length * (height + gap) + 4, 30));

  rows.forEach((row, i) => {
    const y = i * (height + gap) + 2;
    node.appendChild(el("text", {
      x: labelWidth - 10, y: y + height / 2 + 4, "text-anchor": "end",
    }, row.label));
    node.appendChild(el("rect", { class: "track", x: plotLeft, y, width: plotWidth, height }));
    const wa = max ? (row.a / max) * plotWidth : 0;
    const wb = max ? (row.b / max) * plotWidth : 0;
    if (wa > 0.5) {
      node.appendChild(hoverable(el("rect", {
        class: "mark", x: plotLeft, y, width: Math.max(2, wa), height,
        fill: SERIES[1],
      }), row.tipA));
    }
    if (wb > 0.5) {
      node.appendChild(hoverable(el("rect", {
        class: "mark", x: plotLeft + wa + 2, y, width: Math.max(2, wb - 2), height,
        fill: SERIES[0],
      }), row.tipB));
    }
    node.appendChild(el("text", {
      class: "direct-label", x: plotLeft + plotWidth + 8, y: y + height / 2 + 4,
    }, row.valueLabel));
  });
  return node;
}

/* -- timeline ------------------------------------------------------------- */
/**
 * Jobs as bars on a shared time axis, one lane per machine.
 *
 * This is the view a flat table cannot give: whether work actually overlapped,
 * whether the second machine was doing anything, and where the gaps are. The
 * lane is the placement decision made visible.
 */
export function timeline(lanes, opts = {}) {
  const { rowHeight = 15, laneGap = 16, labelWidth = 74 } = opts;
  const width = 900;
  const plotLeft = labelWidth;
  const plotWidth = width - labelWidth - 12;

  let min = Infinity;
  let max = -Infinity;
  for (const lane of lanes) {
    for (const bar of lane.bars) {
      min = Math.min(min, bar.start);
      max = Math.max(max, bar.end);
    }
  }
  if (!Number.isFinite(min) || max <= min) {
    const node = svg(width, 60);
    node.appendChild(el("text", { x: width / 2, y: 32, "text-anchor": "middle" },
      "no jobs with a start and finish time in this range"));
    return node;
  }
  const span = max - min;
  const x = (t) => plotLeft + ((t - min) / span) * plotWidth;

  // Pack each lane's bars into as few rows as fit without overlapping, so
  // concurrency is visible as stacked rows rather than bars drawn over
  // each other.
  const laid = lanes.map((lane) => {
    const rows = [];
    for (const bar of [...lane.bars].sort((a, b) => a.start - b.start)) {
      let placed = false;
      for (const row of rows) {
        if (bar.start >= row.until) {
          row.bars.push(bar);
          row.until = bar.end;
          placed = true;
          break;
        }
      }
      if (!placed) rows.push({ until: bar.end, bars: [bar] });
      if (rows.length > 14) break; // a wall of rows stops being readable
    }
    return { name: lane.name, rows };
  });

  const height =
    laid.reduce((sum, lane) => sum + lane.rows.length * (rowHeight + 2) + laneGap, 0) + 26;
  const node = svg(width, height);

  // Time gridlines. Five is enough to orient; more is chartjunk.
  for (let i = 0; i <= 4; i += 1) {
    const t = min + (span * i) / 4;
    const gx = x(t);
    node.appendChild(el("line", {
      class: "grid-line", x1: gx, x2: gx, y1: 14, y2: height - 18,
    }));
    node.appendChild(el("text", {
      x: gx, y: height - 5, "text-anchor": i === 0 ? "start" : i === 4 ? "end" : "middle",
    }, new Date(t).toLocaleString(undefined, {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
    })));
  }

  let y = 16;
  for (const lane of laid) {
    node.appendChild(el("text", {
      x: labelWidth - 10, y: y + 10, "text-anchor": "end",
      style: "font-weight:600;fill:var(--text-secondary)",
    }, lane.name));
    for (const row of lane.rows) {
      for (const bar of row.bars) {
        const bx = x(bar.start);
        const bw = Math.max(2, x(bar.end) - bx);
        const rect = el("rect", {
          class: "mark", x: bx, y, width: bw, height: rowHeight, fill: bar.color,
          // A 2px surface ring, so touching bars still read as two bars.
          stroke: "var(--surface-1)", "stroke-width": 1,
        });
        hoverable(rect, bar.tip);
        rect.style.cursor = "pointer";
        rect.addEventListener("click", () => bar.onClick && bar.onClick());
        node.appendChild(rect);
      }
      y += rowHeight + 2;
    }
    y += laneGap;
  }
  return node;
}

/* -- histogram ------------------------------------------------------------ */
/** Distribution of a ratio. A single series, so no legend. */
export function histogram(values, opts = {}) {
  const { bins = 20, max = 2.0, height = 170, markAt = 1.0 } = opts;
  const width = 900;
  const pad = { top: 10, right: 12, bottom: 30, left: 36 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    const idx = Math.min(bins - 1, Math.max(0, Math.floor((v / max) * bins)));
    counts[idx] += 1;
  }
  const peak = niceMax(Math.max(...counts, 1));
  const node = svg(width, height);
  const bw = plotW / bins;

  node.appendChild(el("line", {
    class: "axis-line", x1: pad.left, x2: pad.left + plotW,
    y1: pad.top + plotH, y2: pad.top + plotH,
  }));
  counts.forEach((count, i) => {
    const h = (count / peak) * plotH;
    if (h <= 0) return;
    const lo = (i / bins) * max;
    const hi = ((i + 1) / bins) * max;
    // Over budget is the dangerous side of the chart, and it is coloured as
    // such - with an axis label saying so, not colour alone.
    const bar = el("rect", {
      class: "mark", x: pad.left + i * bw + 1, y: pad.top + plotH - h,
      width: Math.max(1, bw - 2), height: h,
      fill: lo >= 1 ? "var(--critical)" : "var(--series-1)",
    });
    node.appendChild(hoverable(bar, tipHtml(
      `${(lo * 100).toFixed(0)}–${(hi * 100).toFixed(0)}% of budget`,
      [["jobs", String(count)]],
    )));
  });
  for (let i = 0; i <= 4; i += 1) {
    const value = (max * i) / 4;
    const gx = pad.left + (value / max) * plotW;
    node.appendChild(el("text", { x: gx, y: height - 12, "text-anchor": "middle" },
      `${(value * 100).toFixed(0)}%`));
  }
  const mx = pad.left + (markAt / max) * plotW;
  node.appendChild(el("line", {
    x1: mx, x2: mx, y1: pad.top, y2: pad.top + plotH,
    stroke: "var(--text-secondary)", "stroke-width": 1.5, "stroke-dasharray": "3 3",
  }));
  node.appendChild(el("text", { x: mx + 5, y: pad.top + 10, style: "fill:var(--text-secondary)" },
    "declared budget"));
  node.appendChild(el("text", {
    x: pad.left + plotW / 2, y: height - 0.5, "text-anchor": "middle",
  }, "peak commit as a share of declared budget →"));
  return node;
}
