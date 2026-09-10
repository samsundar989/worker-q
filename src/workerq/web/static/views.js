/* The six views.
 *
 * Each is a function of data to DOM. None of them fetch: the router does that,
 * so a view can be re-rendered from a poll without re-issuing its own request.
 */

import {
  budgetBars, barsH, clockTime, duration, gib, histogram, hoverable, legend,
  linesPct, pct, shortTime, SERIES, stackedH, timeline, tipHtml,
} from "./charts.js";
import {
  banner, card, empty, h, jobLink, jobTable, meterRow, nodeChip, spinner,
  stateBadge, STATE_COLOR, tile, verdictBadge, verdictColor, verdictWord, why,
} from "./ui.js";

/* ========================================================================= */
/* Now                                                                       */
/* ========================================================================= */

export function viewNow(data, ctx) {
  const jobs = data.jobs || [];
  const running = jobs.filter((j) => j.state === "RUNNING");
  const queued = jobs.filter((j) => j.state === "QUEUED" || j.state === "PREPARING");
  const summary = data.summary || {};
  const host = data.host || {};
  const tp = data.throughput || {};

  const blocked = queued.filter((j) => j.wait_reason);
  const wrap = h("div");

  if (summary.daemon_running === false) {
    wrap.appendChild(banner("bad",
      "The dispatcher is not running, so nothing will start. Run "));
    wrap.lastChild.querySelector("div").appendChild(
      h("code", { class: "cmd", text: "workerq restart" }));
  }

  wrap.appendChild(h("div", { class: "tiles", style: "margin-bottom:14px" }, [
    tile("Running", String(running.length),
      `${summary.backend_slots ?? "?"} slots`),
    tile("Queued", String(queued.length),
      blocked.length ? `${blocked.length} blocked` : "none blocked"),
    tile("Host RAM free", pct((host.free_percent ?? 0) / 100),
      `${gib(host.available_mib)} of ${gib(host.total_mib)}`,
      (host.free_percent ?? 100) < 12),
    tile("Commit", pct((host.commit_percent ?? 0) / 100),
      `${gib(host.commit_used_mib)} of ${gib(host.commit_limit_mib)}`,
      (host.commit_percent ?? 0) > 95),
    tile("24h success", tp.success_rate === null || tp.success_rate === undefined
      ? "-" : pct(tp.success_rate / 100),
      `${tp.finished ?? 0} finished · median wait ${duration(tp.median_wait_seconds)}`),
  ]));

  wrap.appendChild(h("div", { class: "grid two" }, [
    machinesCard(data),
    pressureCard(data),
  ]));

  wrap.appendChild(card("Running", null, running.length
    ? runningList(running)
    : empty("Nothing is running.")));

  wrap.appendChild(card("Queue",
    "A blocked job shows what the dispatcher is waiting for, in its own words.",
    queued.length ? queueList(queued) : empty("The queue is empty.")));

  wrap.appendChild(card("Recently finished", null, [
    jobTable(jobs.filter((j) => !["RUNNING", "QUEUED", "PREPARING"].includes(j.state)).slice(0, 12),
      { showUsage: false }),
    why("GPUQService.list_jobs + status_summary + throughput", {
      summary, host, throughput: tp,
    }, "workerq status --json"),
  ]));
  return wrap;
}

function machinesCard(data) {
  const host = data.host || {};
  const gpu = data.gpu || {};
  const rows = [];
  for (const device of gpu.devices || []) {
    rows.push(meterRow(
      `GPU ${device.index}`,
      device.memory_used_mib, device.memory_total_mib,
      `${gib(device.memory_used_mib)} / ${gib(device.memory_total_mib)} · ${device.utilization_percent ?? 0}% util`,
    ));
  }
  rows.push(meterRow("RAM", (host.total_mib || 0) - (host.available_mib || 0),
    host.total_mib, `${gib(host.available_mib)} free`));
  rows.push(meterRow("Commit", host.commit_used_mib, host.commit_limit_mib,
    `${gib(host.commit_used_mib)} / ${gib(host.commit_limit_mib)}`));

  const nodes = [];
  for (const report of data.nodes || []) {
    if (report.is_local) continue;
    const nodeHost = report.host || {};
    const device = (report.gpu && report.gpu.devices && report.gpu.devices[0]) || {};
    nodes.push(h("div", { style: "margin-top:14px" }, [
      h("div", { class: "small", style: "margin-bottom:4px" }, [
        h("strong", { text: report.name }),
        " ",
        h("span", {
          class: report.online ? "muted" : "verdict-under",
          text: report.online
            ? `${(report.running || []).length} running · ${report.queued ?? 0} queued`
            : `offline: ${report.error || "unreachable"}`,
        }),
      ]),
      device.memory_total_mib ? meterRow("VRAM", device.memory_used_mib,
        device.memory_total_mib,
        `${gib(device.memory_used_mib)} / ${gib(device.memory_total_mib)}`) : null,
      nodeHost.total_mib ? meterRow("RAM",
        nodeHost.total_mib - (nodeHost.available_mib || 0), nodeHost.total_mib,
        `${gib(nodeHost.available_mib)} free`) : null,
    ]));
  }

  return card("Machines",
    "Node figures come from the dispatcher's published reports, not a live SSH round trip.",
    [...rows, ...nodes]);
}

function pressureCard(data) {
  const procs = data.processes || [];
  if (!procs.length) return card("Memory pressure", null, empty("No process data."));
  const rows = procs.slice(0, 9).map((p) => ({
    label: `${p.name} (${p.pid})`,
    value: p.memory_mib || 0,
    color: p.ours ? "var(--series-1)" : "var(--series-2)",
    tip: tipHtml(p.name, [
      ["pid", String(p.pid)],
      ["commit", gib(p.memory_mib)],
      ["owner", p.ours ? "worker-q" : "not worker-q"],
    ]),
  }));
  return card("Memory pressure",
    "Everything holding memory, not just what worker-q started. Work outside the queue is what the broker cannot control.",
    [
      legend([
        { label: "worker-q", color: SERIES[0] },
        { label: "outside the queue", color: SERIES[1] },
      ]),
      barsH(rows, { format: (v) => gib(v), labelWidth: 230 }),
    ]);
}

function runningList(running) {
  return h("div", { class: "table-wrap" }, [
    h("table", {}, [
      h("thead", {}, [h("tr", {}, [
        h("th", { text: "ID" }), h("th", { text: "Project" }), h("th", { text: "Node" }),
        h("th", { text: "What" }), h("th", { class: "num", text: "Elapsed" }),
        h("th", { text: "Progress" }), h("th", { class: "num", text: "Left" }),
      ])]),
      h("tbody", {}, running.map((job) => {
        const est = job.estimate || {};
        const frac = job.progress_fraction;
        return h("tr", { class: "clickable", onclick: () => { location.hash = `#/job/${job.id}`; } }, [
          h("td", { class: "num" }, [jobLink(job.id)]),
          h("td", { class: "truncate", text: job.project }),
          h("td", {}, [nodeChip(job.node)]),
          h("td", { class: "wide-truncate", text: job.description || (job.command || []).join(" ") }),
          h("td", { class: "num", text: duration(job.runtime_seconds) }),
          h("td", { style: "min-width:130px" }, [
            frac === null || frac === undefined
              ? h("span", { class: "muted small", text: est.source || "no progress reported" })
              : h("div", { class: "meter good" }, [h("i", { style: `width:${(frac * 100).toFixed(1)}%` })]),
          ]),
          h("td", { class: "num", text: duration(est.remaining_seconds) }),
        ]);
      })),
    ]),
  ]);
}

function queueList(queued) {
  return h("div", { class: "table-wrap" }, [
    h("table", {}, [
      h("thead", {}, [h("tr", {}, [
        h("th", { text: "ID" }), h("th", { text: "Pri" }), h("th", { text: "Project" }),
        h("th", { text: "What" }), h("th", { class: "num", text: "Waiting" }),
        h("th", { class: "num", text: "Asks for" }),
      ])]),
      h("tbody", {}, queued.flatMap((job) => {
        const rows = [h("tr", { class: "clickable", onclick: () => { location.hash = `#/job/${job.id}`; } }, [
          h("td", { class: "num" }, [jobLink(job.id)]),
          h("td", { class: "small", text: job.priority }),
          h("td", { class: "truncate", text: job.project }),
          h("td", { class: "wide-truncate", text: job.description || (job.command || []).join(" ") }),
          h("td", { class: "num", text: duration(job.wait_seconds) }),
          h("td", { class: "num nowrap", text: `${gib(job.requested_ram_mib)}${job.requested_vram_mib ? ` + ${gib(job.requested_vram_mib)} vram` : ""}` }),
        ])];
        if (job.wait_reason) {
          rows.push(h("tr", {}, [
            h("td", {}),
            h("td", { colspan: "5", class: "small", style: "color:var(--warning);padding-top:0" },
              [job.wait_reason]),
          ]));
        }
        return rows;
      })),
    ]),
  ]);
}

/* ========================================================================= */
/* History                                                                   */
/* ========================================================================= */

export function viewHistory(data, ctx) {
  const { filters, facets } = ctx;
  const jobs = data.jobs || [];
  const wrap = h("div");

  wrap.appendChild(card("History",
    "The jobs table is never pruned, so this reaches back to the first job ever submitted.",
    [
      filterBar(ctx, facets),
      timelineCard(jobs),
      jobTable(jobs, {
        sort: filters.sort,
        dir: filters.dir,
        onSort: (key) => ctx.setFilter({
          sort: key,
          dir: filters.sort === key && filters.dir === "desc" ? "asc" : "desc",
          offset: 0,
        }),
      }),
      pager(data, ctx),
      why("Database.query_jobs", { total: data.total, filters },
        "workerq list --all --json"),
    ]));
  return wrap;
}

function timelineCard(jobs) {
  const lanes = new Map();
  for (const job of jobs) {
    if (!job.started_at || !job.finished_at) continue;
    const start = Date.parse(job.started_at);
    const end = Date.parse(job.finished_at);
    if (!Number.isFinite(start) || !Number.isFinite(end)) continue;
    const name = job.node || "local";
    if (!lanes.has(name)) lanes.set(name, []);
    lanes.get(name).push({
      start,
      end: Math.max(end, start + 1000),
      color: STATE_COLOR[job.state] || "var(--text-muted)",
      onClick: () => { location.hash = `#/job/${job.id}`; },
      tip: tipHtml(`#${job.id} ${job.project}`, [
        ["state", job.state],
        ["node", job.node],
        ["ran", duration(job.runtime_seconds)],
        ["waited", duration(job.wait_seconds)],
        job.description ? ["what", job.description.slice(0, 70)] : null,
      ]),
    });
  }
  const laneList = [...lanes.entries()].map(([name, bars]) => ({ name, bars }));
  if (!laneList.length) return h("div");
  return h("div", { style: "margin:6px 0 16px" }, [
    h("p", { class: "hint", text: "One lane per machine. Rows within a lane are concurrent jobs, so a tall lane is a busy machine. Click a bar to open the job." }),
    legend(Object.entries(STATE_COLOR)
      .filter(([s]) => jobs.some((j) => j.state === s && j.started_at && j.finished_at))
      .map(([label, color]) => ({ label, color }))),
    timeline(laneList),
  ]);
}

function filterBar(ctx, facets) {
  const { filters } = ctx;
  const chipGroup = (key, values) => h("div", { class: "chips" },
    values.map((value) => h("button", {
      class: `chip${(filters[key] || []).includes(value) ? " on" : ""}`,
      text: value,
      onclick: () => {
        const current = new Set(filters[key] || []);
        if (current.has(value)) current.delete(value); else current.add(value);
        ctx.setFilter({ [key]: [...current], offset: 0 });
      },
    })));

  const search = h("input", {
    type: "search",
    placeholder: "Search project, description, command…",
    value: filters.search || "",
  });
  let timer = null;
  search.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(() => ctx.setFilter({ search: search.value, offset: 0 }), 280);
  });

  return h("div", {}, [
    h("div", { class: "filters" }, [
      search,
      h("button", {
        class: "ghost", text: "Clear filters",
        onclick: () => ctx.setFilter({
          search: "", state: [], project: [], node: [], priority: [], offset: 0,
        }),
      }),
    ]),
    h("div", { class: "filters" }, [
      h("span", { class: "small muted", text: "state" }), chipGroup("state", facets.state || []),
      h("span", { class: "small muted", style: "margin-left:10px", text: "node" }),
      chipGroup("node", facets.node || []),
    ]),
    h("div", { class: "filters" }, [
      h("span", { class: "small muted", text: "project" }),
      chipGroup("project", (facets.project || []).slice(0, 12)),
    ]),
  ]);
}

function pager(data, ctx) {
  const { offset, limit, total } = data;
  const from = total ? offset + 1 : 0;
  const to = Math.min(offset + limit, total);
  return h("div", { class: "pager" }, [
    h("span", { text: `${from}–${to} of ${total}` }),
    h("button", {
      text: "Newer", disabled: offset <= 0,
      onclick: () => ctx.setFilter({ offset: Math.max(0, offset - limit) }),
    }),
    h("button", {
      text: "Older", disabled: to >= total,
      onclick: () => ctx.setFilter({ offset: offset + limit }),
    }),
    h("select", {
      onchange: (e) => ctx.setFilter({ limit: Number(e.target.value), offset: 0 }),
    }, [25, 50, 100, 200].map((n) => h("option", {
      value: String(n), text: `${n} per page`, selected: n === limit,
    }))),
  ]);
}

/* ========================================================================= */
/* Job detail                                                                */
/* ========================================================================= */

export function viewJob(data, ctx) {
  const job = data;
  const u = job.usage || {};
  const wrap = h("div");

  wrap.appendChild(h("div", { class: "filters", style: "justify-content:space-between" }, [
    h("div", {}, [
      h("h1", { style: "margin:0;font-size:19px" }, [
        `#${job.id} `,
        h("span", { class: "muted", text: job.project }),
      ]),
      h("div", { class: "small muted", text: job.description || (job.command || []).join(" ") }),
    ]),
    h("div", { class: "actions" }, [
      stateBadge(job.state), nodeChip(job.node),
      h("button", { class: "ghost", text: "← history", onclick: () => { location.hash = "#/history"; } }),
    ]),
  ]));

  wrap.appendChild(usageCard(job, u, ctx));
  wrap.appendChild(h("div", { class: "grid two" }, [
    provenanceCard(job),
    timingCard(job),
  ]));
  wrap.appendChild(seriesCard(ctx.series, job));
  wrap.appendChild(siblingsCard(job));
  wrap.appendChild(eventsCard(job));
  wrap.appendChild(logCard(ctx));
  wrap.appendChild(actionsCard(job, ctx));
  return wrap;
}

function usageCard(job, u, ctx) {
  const s = ctx.suggestion || job.suggestion || {};
  const body = [];

  if (u.verdict === "under-declared") {
    body.push(banner("bad",
      `This job used ${pct(u.commit_ratio)} of the footprint it declared. Under-declaring is what takes the machine down: admission control handed out capacity this job then exceeded.`));
  } else if (u.verdict === "low-confidence") {
    body.push(banner("info",
      `Only ${u.samples} usage sample${u.samples === 1 ? "" : "s"} were taken, so this peak is not worth drawing a conclusion from. Usage is sampled every 2s for the first 30s and every 15s after.`));
  } else if (u.verdict === "unmeasured") {
    body.push(banner("info",
      "No usage was measured for this job. That is not the same as using nothing."));
  }

  body.push(h("div", { class: "tiles" }, [
    tile("Declared budget", gib(u.commit_budget_mib),
      job.requested_vram_mib
        ? `${gib(job.requested_ram_mib)} RAM + ${gib(job.requested_vram_mib)} VRAM`
        : `${gib(job.requested_ram_mib)} RAM, no GPU`),
    tile("Peak commit", gib(u.peak_commit_mib),
      `${u.samples || 0} samples · ${u.peak_source || "unmeasured"}`),
    tile("Used", u.commit_ratio === null || u.commit_ratio === undefined ? "-" : pct(u.commit_ratio),
      verdictWord(u.verdict), u.verdict === "under-declared"),
    tile("Peak VRAM", gib(u.peak_vram_mib),
      u.vram_is_device_delta ? "whole-card delta" : (u.vram_source || "not measured")),
    tile("Runtime vs ETA", u.eta_ratio ? pct(u.eta_ratio) : "-",
      job.eta_seconds ? `declared ${duration(job.eta_seconds)}` : "no ETA declared"),
  ]));

  body.push(h("p", { class: "hint", style: "margin-top:12px" }, [
    "Budget is declared RAM plus declared VRAM. The measured peak is commit charge for the whole process tree, and under WDDM a GPU job's commit includes its video memory, so comparing the peak against declared RAM alone reads a well-sized GPU job as dangerously over.",
  ]));

  if (s && s.suggested_ram_gb) {
    body.push(h("div", { class: "actions", style: "margin-top:8px" }, [
      h("span", { class: "small secondary", text: `History suggests --ram ${s.suggested_ram_gb}` }),
      h("code", { class: "cmd", text: `workerq requests ${job.id} --ram ${s.suggested_ram_gb}` }),
      h("span", { class: "small muted", text: `from ${s.runs ?? 0} run(s), ${s.provenance || "unknown"}` }),
    ]));
  }

  body.push(why("workerq.usage.for_job", u, `workerq resources --verify`));
  return card("Declared against actual", null, body);
}

function provenanceCard(job) {
  const kv = [];
  const add = (k, v) => { if (v !== null && v !== undefined && v !== "") kv.push([k, v]); };
  add("command", (job.command || []).join(" "));
  add("signature", job.command_signature);
  add("repo", job.repo_root);
  add("snapshot", job.snapshot_commit
    ? `${job.snapshot_mode} @ ${String(job.snapshot_commit).slice(0, 12)}` : job.snapshot_mode);
  const pass = job.snapshot_passthrough || [];
  add("passthrough", pass.length > 6
    ? `${pass.slice(0, 6).join(", ")} … and ${pass.length - 6} more`
    : pass.join(", "));
  add("execution cwd", job.execution_cwd);
  add("submitted by", job.submitter_agent || job.host);
  add("priority", `${job.priority}${job.preemptible ? " · preemptible" : ""}`);
  add("devices", job.cuda_visible_devices);
  add("blocks", job.blocks);
  add("log", job.log_path);
  if (job.preemption_count) add("preempted", `${job.preemption_count}× · ${job.preempted_reason || ""}`);
  if (job.error) add("error", job.error);

  return card("Provenance", null, [
    h("dl", { class: "kv" }, kv.flatMap(([k, v]) => [
      h("dt", { text: k }), h("dd", { text: String(v) }),
    ])),
  ]);
}

function timingCard(job) {
  const est = job.estimate || {};
  const rows = [
    ["queued", shortTime(job.queued_at)],
    ["started", shortTime(job.started_at)],
    ["finished", shortTime(job.finished_at)],
    ["waited", duration(job.wait_seconds)],
    ["ran", duration(job.runtime_seconds)],
    ["exit code", job.exit_code === null || job.exit_code === undefined ? "-" : String(job.exit_code)],
  ];
  if (job.queue_position !== null && job.queue_position !== undefined) {
    rows.push(["queue position", String(job.queue_position)]);
  }
  if (job.wait_reason) rows.push(["waiting for", job.wait_reason]);
  if (est.remaining_seconds) rows.push(["estimated left", duration(est.remaining_seconds)]);
  if (est.source) rows.push(["estimate from", est.source]);
  return card("Timing", null, [
    h("dl", { class: "kv" }, rows.flatMap(([k, v]) => [
      h("dt", { text: k }), h("dd", { text: v }),
    ])),
  ]);
}

function seriesCard(series, job) {
  if (!series || !series.samples || series.samples.length < 2) {
    return card("The machine while this ran",
      "Machine-wide telemetry for the job's window.",
      empty("No telemetry samples were recorded for this window."));
  }
  const samples = series.samples;
  const commit = samples.map((s) => s.commit_percent ?? 0);
  const freeRam = samples.map((s) => s.host_free_percent ?? 0);
  const gpuUsed = samples.map((s) => 100 - (s.gpu_free_percent ?? 100));

  return card("The machine while this ran", null, [
    banner("info",
      "These are whole-machine figures for the window this job ran in, not this job's own usage. They show the conditions it ran under - which is usually what explains a slow run. Per-job attribution is the peak above."),
    legend([
      { label: "commit used", color: SERIES[0] },
      { label: "host RAM free", color: SERIES[1] },
      { label: "GPU memory used", color: SERIES[2] },
    ]),
    linesPct([
      { label: "commit used", values: commit, color: SERIES[0] },
      { label: "host RAM free", values: freeRam, color: SERIES[1] },
      { label: "GPU memory used", values: gpuUsed, color: SERIES[2] },
    ], {
      labelAt: (i) => clockTime(samples[i].at),
    }),
    why("Telemetry.job_series (samples joined via job_samples)",
      { count: samples.length, scope: series.scope }),
  ]);
}

function siblingsCard(job) {
  const siblings = job.siblings || [];
  if (siblings.length < 2) {
    return card("Other runs of this command",
      "Matched on command signature: flags, paths and numbers are stripped, so the same job with different arguments still matches.",
      empty("This command has not run before."));
  }
  const rows = siblings.slice(0, 18).map((s) => ({
    label: `#${s.id}${s.is_self ? " ←" : ""}`,
    ratio: s.commit_ratio ?? 0,
    color: verdictColor(s.verdict),
    valueLabel: `${s.commit_ratio ? pct(s.commit_ratio) : "-"} · ${duration(s.runtime_seconds)}`,
    tip: tipHtml(`#${s.id} ${s.state}`, [
      ["node", s.node],
      ["peak commit", gib(s.peak_commit_mib)],
      ["budget", gib(s.commit_budget_mib)],
      ["samples", String(s.samples)],
      ["ran", duration(s.runtime_seconds)],
    ]),
  }));
  return card("Other runs of this command",
    "One job being wrong is a typo. The same command being wrong every time is a default worth changing at the source.",
    [
      budgetBars(rows),
      why("Database.query_jobs(signature=…)", { signature: job.command_signature, runs: siblings.length }),
    ]);
}

function eventsCard(job) {
  const events = job.events || [];
  if (!events.length) {
    return card("Lifecycle", null,
      empty("No events retained for this job."));
  }
  return card("Lifecycle", null, [
    h("div", { class: "table-wrap" }, [
      h("table", {}, [
        h("thead", {}, [h("tr", {}, [
          h("th", { text: "When" }), h("th", { text: "Event" }), h("th", { text: "Detail" }),
        ])]),
        h("tbody", {}, events.map((e) => h("tr", {}, [
          h("td", { class: "nowrap mono small", text: shortTime(e.at) }),
          h("td", { class: "small", text: e.kind }),
          h("td", { class: "small muted", text: e.detail || "" }),
        ]))),
      ]),
    ]),
  ]);
}

function logCard(ctx) {
  const log = ctx.log || {};
  const body = log.missing
    ? empty("No log file for this job.")
    : h("pre", { class: "log", text: log.text || "" });
  return card("Log", log.size ? `${(log.size / 1024).toFixed(0)} KiB on disk` : null, [
    body,
    h("div", { class: "actions", style: "margin-top:8px" }, [
      h("button", { class: "ghost", text: "Refresh", onclick: () => ctx.reloadLog() }),
      h("code", { class: "cmd", text: `workerq logs ${ctx.jobId} --follow` }),
    ]),
  ]);
}

function actionsCard(job, ctx) {
  const terminal = ["SUCCEEDED", "FAILED", "CANCELLED", "LOST"].includes(job.state);
  const act = (action, body) => () => ctx.act(job.id, action, body || {});
  const ramInput = h("input", {
    type: "number", min: "1", step: "1", style: "width:88px",
    value: String(Math.round((job.requested_ram_mib || 0) / 1024) || 4),
  });
  return card("Actions",
    "Each button runs the same service call the CLI does, and tells you the command it is equivalent to.",
    [
      h("div", { class: "actions" }, [
        h("button", { text: "Cancel", disabled: terminal, onclick: act("cancel") }),
        h("button", { text: "Cancel (force)", disabled: terminal, onclick: act("cancel", { force: true }) }),
        h("button", { text: "Promote", disabled: job.state !== "QUEUED", onclick: act("promote") }),
        h("button", { text: "Bump to critical", disabled: terminal, onclick: act("bump", { level: "critical" }) }),
      ]),
      h("div", { class: "actions", style: "margin-top:10px" }, [
        h("span", { class: "small secondary", text: "Set RAM (GiB)" }),
        ramInput,
        h("button", {
          class: "primary", text: "Apply", disabled: job.state !== "QUEUED",
          onclick: () => ctx.act(job.id, "requests", { ram_gb: Number(ramInput.value) }),
        }),
        h("span", { class: "small muted", text: job.state === "QUEUED" ? "" : "only a queued job can be re-declared" }),
      ]),
      ctx.lastAction ? h("div", { class: "banner info", style: "margin-top:12px" }, [
        h("div", {}, [
          ctx.lastAction.error
            ? h("span", { class: "verdict-under", text: ctx.lastAction.error })
            : h("span", {}, [
              (ctx.lastAction.result && ctx.lastAction.result.message) || "done",
              " · ",
              h("code", { class: "cmd", text: ctx.lastAction.command }),
            ]),
        ]),
      ]) : null,
    ]);
}

/* ========================================================================= */
/* Accuracy                                                                  */
/* ========================================================================= */

export function viewAccuracy(data, ctx) {
  const summary = data.summary || {};
  const rows = data.rows || [];
  const counts = summary.counts || {};
  const conclusive = rows.filter((r) => !["unmeasured", "low-confidence"].includes(r.verdict));
  const wrap = h("div");

  wrap.appendChild(h("div", { class: "tiles", style: "margin-bottom:14px" }, [
    tile("Median use of budget",
      summary.median_commit_ratio ? pct(summary.median_commit_ratio) : "-",
      `${summary.conclusive} job(s) with enough samples`),
    tile("Reserved, never touched",
      `${Math.round(summary.unused_gib_hours || 0).toLocaleString()}`,
      "GiB-hours of queue time spent waiting for nothing"),
    tile("Over budget", String(counts["under-declared"] || 0),
      "exceeded what they declared", (counts["under-declared"] || 0) > 0),
    tile("Median runtime vs ETA",
      summary.median_eta_ratio ? pct(summary.median_eta_ratio) : "-",
      "under 100% means ETAs are pessimistic"),
    tile("Too few samples", String(counts["low-confidence"] || 0),
      "no conclusion drawn from these"),
  ]));

  if ((counts["under-declared"] || 0) > 0) {
    wrap.appendChild(card("Went over its declaration",
      "The short, loud list. Over-declaring makes other people wait; under-declaring takes the machine down, so these are ranked separately and never averaged together.",
      [
        underTable(summary.worst_under || []),
        why("usage.summarise → worst_under", { count: (summary.worst_under || []).length }),
      ]));
  }

  wrap.appendChild(card("Distribution",
    "Peak commit as a share of declared budget, across every measured job in range.",
    conclusive.length
      ? [histogram(conclusive.map((r) => Math.min(r.commit_ratio ?? 0, 1.99)))]
      : empty("Nothing measured with enough confidence yet.")));

  wrap.appendChild(card("Costliest over-declarations",
    "Ranked by GiB-hours held and unused, not by ratio: a job that over-declares by 10x for four seconds costs the queue nothing.",
    [
      overTable(summary.worst_over || []),
    ]));

  wrap.appendChild(card("By command",
    "The actionable grouping. A command that is mis-declared every time it runs is a default worth fixing at the source.",
    groupTable(data.by_signature || [], "signature")));

  wrap.appendChild(card("By project", null, [
    groupTable(data.by_project || [], "project"),
    why("report.declared_vs_observed / usage.group_by", { considered: data.considered },
      "workerq resources --verify --json"),
  ]));
  return wrap;
}

function underTable(rows) {
  if (!rows.length) return empty("Nothing over budget.");
  return h("div", { class: "table-wrap" }, [
    h("table", {}, [
      h("thead", {}, [h("tr", {}, [
        h("th", { text: "Job" }), h("th", { text: "Project" }), h("th", { text: "Node" }),
        h("th", { class: "num", text: "Budget" }), h("th", { class: "num", text: "Peak commit" }),
        h("th", { class: "num", text: "Over by" }), h("th", { class: "num", text: "Samples" }),
        h("th", { text: "Fix" }),
      ])]),
      h("tbody", {}, rows.map((r) => h("tr", {}, [
        h("td", {}, [jobLink(r.job_id)]),
        h("td", { class: "truncate", text: r.project }),
        h("td", {}, [nodeChip(r.node)]),
        h("td", { class: "num", text: gib(r.commit_budget_mib) }),
        h("td", { class: "num", text: gib(r.peak_commit_mib) }),
        h("td", { class: "num verdict-under", text: pct((r.commit_ratio || 1) - 1) }),
        h("td", { class: "num", text: String(r.samples) }),
        h("td", {}, [h("code", { class: "cmd", text: `workerq requests ${r.job_id} --ram ${r.suggested_ram_gb ?? "?"}` })]),
      ]))),
    ]),
  ]);
}

function overTable(rows) {
  if (!rows.length) return empty("Nothing measured yet.");
  return h("div", { class: "table-wrap" }, [
    h("table", {}, [
      h("thead", {}, [h("tr", {}, [
        h("th", { text: "Job" }), h("th", { text: "Project" }),
        h("th", { class: "num", text: "Budget" }), h("th", { class: "num", text: "Peak commit" }),
        h("th", { class: "num", text: "Used" }), h("th", { class: "num", text: "Wasted" }),
        h("th", { text: "Fix" }),
      ])]),
      h("tbody", {}, rows.map((r) => h("tr", {}, [
        h("td", {}, [jobLink(r.job_id)]),
        h("td", { class: "truncate", text: r.project }),
        h("td", { class: "num", text: gib(r.commit_budget_mib) }),
        h("td", { class: "num", text: gib(r.peak_commit_mib) }),
        h("td", { class: "num" }, [verdictBadge(r.verdict, r.commit_ratio)]),
        h("td", { class: "num", text: `${(r.unused_gib_hours || 0).toFixed(0)} GiB·h` }),
        h("td", {}, [h("code", { class: "cmd", text: `workerq requests ${r.job_id} --ram ${r.suggested_ram_gb ?? "?"}` })]),
      ]))),
    ]),
  ]);
}

/** Mirrors workerq.usage.classify, so a bar never disagrees with a badge. */
function groupVerdict(ratio) {
  if (ratio === null || ratio === undefined) return "unmeasured";
  if (ratio > 1) return "under-declared";
  if (ratio < 0.25) return "severely-over-declared";
  if (ratio < 0.5) return "over-declared";
  return "ok";
}

function labelFor(group) {
  const text = group.label || group.key;
  return text.length > 30 ? `${text.slice(0, 29)}…` : text;
}

function groupTable(groups, kind) {
  if (!groups.length) return empty("Nothing to group yet.");
  const rows = groups.slice(0, 20).map((g) => ({
    label: `${labelFor(g)} (${g.runs})`,
    ratio: g.median_commit_ratio ?? 0,
    color: verdictColor(groupVerdict(g.median_commit_ratio)),
    valueLabel: `${g.median_commit_ratio ? pct(g.median_commit_ratio) : "-"} · ${Math.round(g.unused_gib_hours)} GiB·h`,
    tip: tipHtml(g.label || g.key, [
      ["signature", g.key],
      ["projects", (g.projects || []).join(", ")],
      ["runs", String(g.runs)],
      ["median use of budget", g.median_commit_ratio ? pct(g.median_commit_ratio) : "-"],
      ["median runtime vs ETA", g.median_eta_ratio ? pct(g.median_eta_ratio) : "-"],
      ["held unused", `${Math.round(g.unused_gib_hours)} GiB-hours`],
      ["over budget", String((g.counts || {})["under-declared"] || 0)],
    ]),
  }));
  return [
    h("p", { class: "hint", text: "Bar is the median share of declared budget actually used. The dashed rule is the declaration; short bars are wasted headroom, bars past the rule went over." }),
    budgetBars(rows, { labelWidth: 210, valueWidth: 190 }),
  ];
}

/* ========================================================================= */
/* Machines                                                                  */
/* ========================================================================= */

export function viewMachines(data, ctx) {
  const machines = data.machines || [];
  const wrap = h("div");

  wrap.appendChild(card("Where the work ran", null, [
    h("div", { class: "table-wrap" }, [
      h("table", {}, [
        h("thead", {}, [h("tr", {}, [
          h("th", { text: "Machine" }), h("th", { class: "num", text: "Jobs" }),
          h("th", { class: "num", text: "Succeeded" }), h("th", { class: "num", text: "Failed" }),
          h("th", { class: "num", text: "Success rate" }),
          h("th", { class: "num", text: "Median runtime" }),
          h("th", { class: "num", text: "Median wait" }),
          h("th", { class: "num", text: "Busy" }),
          h("th", { class: "num", text: "Measured" }),
        ])]),
        h("tbody", {}, machines.map((m) => h("tr", {}, [
          h("td", {}, [nodeChip(m.node)]),
          h("td", { class: "num", text: String(m.jobs) }),
          h("td", { class: "num", text: String(m.succeeded) }),
          h("td", { class: "num", text: String(m.failed) }),
          h("td", { class: "num", text: m.success_rate === null ? "-" : pct(m.success_rate) }),
          h("td", { class: "num", text: duration(m.median_runtime_seconds) }),
          h("td", { class: "num", text: duration(m.median_wait_seconds) }),
          h("td", { class: "num", text: duration(m.busy_seconds) }),
          h("td", { class: "num", text: `${m.measured}/${m.jobs}` }),
        ]))),
      ]),
    ]),
  ]));

  const withIdle = machines.filter((m) => m.idle_while_queued
    && m.idle_while_queued.fraction !== null);
  if (withIdle.length) {
    wrap.appendChild(card("Idle while the queue had work",
      "Placement only pays when moving a job lets a different job start sooner. This is the test: of the samples where something was waiting to run, how often was each machine doing nothing. Compare the machines against each other - a second machine idle far more than the first is not earning its place.",
      withIdle.map((m) => {
        const iw = m.idle_while_queued;
        return h("div", { style: "margin-bottom:10px" }, [
          h("div", { class: "small", style: "margin-bottom:5px" }, [
            h("strong", { text: m.node }),
            " idle for ",
            h("span", {
              class: (iw.fraction ?? 0) > 0.9 ? "verdict-under" : "secondary",
              text: iw.fraction === null ? "unknown" : pct(iw.fraction),
            }),
            ` of the ${iw.queued_samples.toLocaleString()} samples where the queue had work waiting`,
          ]),
          h("div", { class: `meter ${(iw.fraction ?? 0) > 0.9 ? "critical" : "warning"}` }, [
            h("i", { style: `width:${((iw.fraction ?? 0) * 100).toFixed(1)}%` }),
          ]),
        ]);
      })));
  }

  const unmeasured = machines.filter((m) => m.node !== "local" && m.measured < m.jobs);
  if (unmeasured.length) {
    wrap.appendChild(banner("info",
      "Remote jobs only carry usage measurements home from the point that merge shipped. Older travelled jobs will read as unmeasured for good, because the node's own record is what held them."));
  }

  wrap.appendChild(card("Live", null,
    (data.reports || []).length
      ? (data.reports || []).map((r) => reportCard(r))
      : empty("No node reports published.")));

  wrap.appendChild(why("Database.query_jobs grouped by node + nodes.published_reports",
    { machines }, "workerq node list --json"));
  return wrap;
}

function reportCard(report) {
  const host = report.host || {};
  const device = (report.gpu && report.gpu.devices && report.gpu.devices[0]) || {};
  return h("div", { style: "margin-bottom:14px" }, [
    h("div", { class: "small", style: "margin-bottom:6px" }, [
      h("strong", { text: report.name }),
      " ",
      h("span", { class: "muted", text: report.hostname || "" }),
      " ",
      h("span", {
        class: report.online ? "muted" : "verdict-under",
        text: report.online
          ? `online · ${report.cpus ?? "?"} cpus · ${report.slots ?? "?"} slots · ${(report.age_seconds ?? 0).toFixed(0)}s ago`
          : `offline: ${report.error || "unreachable"}`,
      }),
    ]),
    device.memory_total_mib ? meterRow("VRAM", device.memory_used_mib, device.memory_total_mib,
      `${gib(device.memory_used_mib)} / ${gib(device.memory_total_mib)}`) : null,
    host.total_mib ? meterRow("RAM", host.total_mib - (host.available_mib || 0), host.total_mib,
      `${gib(host.available_mib)} free of ${gib(host.total_mib)}`) : null,
    host.commit_limit_mib ? meterRow("Commit", host.commit_used_mib, host.commit_limit_mib,
      `${gib(host.commit_used_mib)} / ${gib(host.commit_limit_mib)}`) : null,
  ]);
}

/* ========================================================================= */
/* Efficiency                                                                */
/* ========================================================================= */

export function viewEfficiency(data, ctx) {
  const wrap = h("div");
  const byProject = data.by_project || [];
  const tp = data.throughput || {};

  wrap.appendChild(h("div", { class: "tiles", style: "margin-bottom:14px" }, [
    tile("24h utilisation", tp.utilisation_percent === null || tp.utilisation_percent === undefined
      ? "-" : `${tp.utilisation_percent.toFixed(0)}%`,
      "share of wall time a job was running"),
    tile("Median wait", duration(tp.median_wait_seconds), "queued to started"),
    tile("Median runtime", duration(tp.median_runtime_seconds), "started to finished"),
    tile("Preemptions", String((data.preemptions || []).length), "in retained history"),
  ]));

  wrap.appendChild(card("Waiting against computing",
    "If a project waits longer than it computes, the declarations are the bottleneck, not the code.",
    byProject.length ? [
      legend([
        { label: "waiting in queue", color: SERIES[1] },
        { label: "running", color: SERIES[0] },
      ]),
      stackedH(byProject
        .filter((p) => (p.wait_seconds + p.run_seconds) > 60)
        .slice(0, 14)
        .map((p) => ({
        label: p.project.length > 22 ? `${p.project.slice(0, 21)}…` : p.project,
        a: p.wait_seconds,
        b: p.run_seconds,
        valueLabel: `${duration(p.wait_seconds)} waiting · ${pct(p.wait_fraction ?? 0)}`,
        tipA: tipHtml(p.project, [["waiting", duration(p.wait_seconds)], ["runs", String(p.runs)]]),
        tipB: tipHtml(p.project, [["running", duration(p.run_seconds)], ["runs", String(p.runs)]]),
      })), { valueWidth: 230 }),
      h("p", { class: "hint", text: "Bar length is real time, so a project with a short bar is not the problem however high its percentage. Projects with under a minute of total time are left out." }),
    ] : empty("No jobs in range.")));

  wrap.appendChild(card("What the queue is actually waiting for",
    "Blocked reasons with their live numbers stripped, so the same condition counts once per occurrence rather than once per tick.",
    (data.block_reasons || []).length ? [
      barsH((data.block_reasons || []).slice(0, 12).map((r) => ({
        label: r.reason.replace(/\s*\|\s*/g, " / ").slice(0, 58),
        value: r.count,
        tip: tipHtml("blocked", [["reason", r.reason.slice(0, 200)], ["occurrences", String(r.count)]]),
      })), { labelWidth: 430, valueWidth: 70, labelChars: 58,
              format: (v) => String(v) }),
    ] : empty("Nothing has been blocked recently.")));

  wrap.appendChild(card("Recent lifecycle", null, [
    h("div", { class: "table-wrap" }, [
      h("table", {}, [
        h("thead", {}, [h("tr", {}, [
          h("th", { text: "When" }), h("th", { text: "Event" }),
          h("th", { text: "Job" }), h("th", { text: "Detail" }),
        ])]),
        h("tbody", {}, (data.events || []).slice(0, 40).map((e) => h("tr", {}, [
          h("td", { class: "nowrap mono small", text: shortTime(e.at) }),
          h("td", { class: "small", text: e.kind }),
          h("td", {}, [e.job_id ? jobLink(e.job_id) : h("span", { class: "muted small", text: e.backend_job_id ? `b${e.backend_job_id}` : "-" })]),
          h("td", { class: "small muted wide-truncate", text: e.detail || "", title: e.detail || "" }),
        ]))),
      ]),
    ]),
    why("Telemetry.recent_events + Database.query_jobs", { projects: byProject.length },
      "workerq report --json"),
  ]));
  return wrap;
}
