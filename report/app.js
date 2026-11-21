(() => {
  "use strict";

  const BUNDLE_FORMAT = "nomad-results-bundle";
  const BUNDLE_VERSION = 1;
  const CONDITIONS = {
    unchanged: { label: "Unchanged Day-0 transfer", color: "#617f9c" },
    normalization_only: { label: "Normalization-only (target-training calibrated)", color: "#aeb8c0" },
    alignment_unsupervised: { label: "NoMAD alignment (unsupervised)", color: "#15978b" },
    alignment_supervised: { label: "NoMAD alignment (supervised)", color: "#ab774d" },
    day0: { label: "Day-0 reference", color: "#526173" },
  };
  const CONDITION_ORDER = ["unchanged", "normalization_only", "alignment_unsupervised", "alignment_supervised", "day0"];
  const state = { records: [], pending: null, split: "test", method: "Wiener on generator", metric: "force", expanded: new Set() };
  const $ = (selector) => document.querySelector(selector);
  const svgNS = "http://www.w3.org/2000/svg";

  function conditionFor(label) {
    const text = String(label).toLowerCase();
    if (text.includes("unchanged day-0 transfer")) return "unchanged";
    if (text.includes("normalization-only")) return "normalization_only";
    if (text.includes("supervised alignment")) return "alignment_supervised";
    if (text.includes("unsupervised alignment")) return "alignment_unsupervised";
    if (text.startsWith("day-0")) return "day0";
    return `other:${label}`;
  }

  function finiteScore(value, context) {
    if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`Score for ${context} must be a finite number.`);
    return value;
  }

  function normalizeEvaluation(report, label, seedOverride, source) {
    if (!report || typeof report !== "object" || Array.isArray(report)) throw new Error("Evaluation JSON must be an object.");
    const seed = seedOverride ?? report.args?.seed;
    if (!Number.isSafeInteger(seed)) throw new Error("Enter an integer run seed; args.seed was missing or invalid.");
    if (!label.trim()) throw new Error("Enter a Day-K label.");
    if (!["test", "validation"].includes(report.split)) throw new Error("Report split must be test or validation.");
    if (!Array.isArray(report.results)) throw new Error("Evaluation JSON has no results list.");
    const rows = report.results.map((row, index) => {
      if (!row || typeof row.method !== "string" || typeof row.split !== "string" || !row.scores || typeof row.scores !== "object") {
        throw new Error(`Result row ${index + 1} is incomplete.`);
      }
      const condition = conditionFor(row.split);
      const scores = {};
      for (const [metric, value] of Object.entries(row.scores)) {
        scores[metric] = finiteScore(value, `${row.method}/${row.split}/${metric}`);
      }
      if (!Object.keys(scores).length) throw new Error(`Result row ${index + 1} has no scores.`);
      return {
        method: row.method.trim(), condition,
        condition_label: CONDITIONS[condition]?.label ?? row.split,
        split_label: row.split, scores,
      };
    });
    return {
      dayk: label.trim(), target_sha256: report.target_sha256 || null,
      reference_sha256: report.reference_sha256 || null,
      reference_model_id: report.reference_model_id || null,
      target_model_id: report.target_model_id || null,
      seed, split: report.split, source: source || "browser import",
      alignment: report.alignment || "unspecified", trial_counts: report.trial_counts || {}, rows,
    };
  }

  function normalizeSummary(summary, source) {
    if (!summary || !Array.isArray(summary.days)) throw new Error("Summary JSON must contain a days list.");
    return summary.days.map((entry, index) => {
      if (!entry || !entry.report) throw new Error(`Summary entry ${index + 1} has no evaluation report.`);
      const label = entry.target || entry.archive?.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, "") || `Day-K ${index + 1}`;
      return normalizeEvaluation(entry.report, label, undefined, entry.archive || source);
    });
  }

  function validateRecords(records) {
    if (!Array.isArray(records) || !records.length) throw new Error("No result records were supplied.");
    const refs = new Set(records.map((record) => record.reference_sha256).filter(Boolean));
    if (refs.size > 1) throw new Error("These results use different reference data hashes; use one reference dataset per report.");
    const labelByIdentity = new Map();
    const identityByLabel = new Map();
    const seen = new Set();
    for (const record of records) {
      if (!record || typeof record.dayk !== "string" || !Number.isSafeInteger(record.seed) ||
          !["test", "validation"].includes(record.split) || !Array.isArray(record.rows)) {
        throw new Error("A normalized result has an invalid Day-K, seed, split, or rows field.");
      }
      if (!record.dayk.trim() || record.seed < 0 || !record.rows.length) {
        throw new Error("A normalized result needs a Day-K label, a nonnegative run seed, and at least one score row.");
      }
      for (const row of record.rows) {
        if (!row || typeof row.method !== "string" || typeof row.condition !== "string" ||
            !row.scores || typeof row.scores !== "object" || !Object.keys(row.scores).length) {
          throw new Error(`A normalized result for ${record.dayk} contains an invalid score row.`);
        }
        for (const [metric, value] of Object.entries(row.scores)) {
          finiteScore(value, `${record.dayk}/${row.method}/${metric}`);
        }
      }
      const identity = record.target_sha256 || `label:${record.dayk}`;
      if (labelByIdentity.has(identity) && labelByIdentity.get(identity) !== record.dayk) {
        throw new Error(`Target ${identity} has conflicting labels. Use the same Day-K label for its seeds.`);
      }
      if (identityByLabel.has(record.dayk) && identityByLabel.get(record.dayk) !== identity) {
        throw new Error(`Day-K label “${record.dayk}” refers to different target data. Use distinct labels.`);
      }
      labelByIdentity.set(identity, record.dayk);
      identityByLabel.set(record.dayk, identity);
      const key = `${identity}\u001f${record.seed}\u001f${record.split}`;
      if (seen.has(key)) throw new Error(`Duplicate ${record.split} report for ${record.dayk}, seed ${record.seed}.`);
      seen.add(key);
    }
    return records;
  }

  function stats(values) {
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    if (values.length < 2) return { mean, sd: null, n: values.length };
    const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1);
    return { mean, sd: Math.sqrt(variance), n: values.length };
  }

  function aggregate(records, split) {
    const selected = records.filter((record) => record.split === split);
    const daysMap = new Map();
    const definitions = new Map();
    const byDaySeries = new Map();
    for (const record of selected) {
      const identity = record.target_sha256 || `label:${record.dayk}`;
      if (!daysMap.has(identity)) daysMap.set(identity, { id: identity, label: record.dayk, target_sha256: record.target_sha256 || null, seeds: new Set(), records: [] });
      const day = daysMap.get(identity);
      day.seeds.add(record.seed);
      day.records.push(record);
      for (const row of record.rows) {
        for (const [metric, value] of Object.entries(row.scores)) {
          const key = [row.method, row.condition, metric].join("\u001f");
          definitions.set(key, { key, method: row.method, condition: row.condition,
            condition_label: row.condition_label || CONDITIONS[row.condition]?.label || row.split_label,
            metric });
          if (!byDaySeries.has(identity)) byDaySeries.set(identity, new Map());
          const series = byDaySeries.get(identity);
          if (!series.has(key)) series.set(key, new Map());
          series.get(key).set(record.seed, value);
        }
      }
    }
    const days = [...daysMap.values()].sort((a, b) => a.label.localeCompare(b.label));
    for (const day of days) {
      const series = {};
      const valuesByKey = byDaySeries.get(day.id) || new Map();
      for (const [key, observations] of valuesByKey) {
        const result = stats([...observations.values()]);
        result.seeds = [...observations.entries()].sort((a, b) => a[0] - b[0]).map(([seed, value]) => ({ seed, value }));
        series[key] = result;
      }
      day.seed_count = day.seeds.size;
      day.seed_list = [...day.seeds].sort((a, b) => a - b);
      day.series = series;
    }
    const overall = {};
    for (const [key] of definitions) {
      const dayMeans = days.filter((day) => day.series[key]).map((day) => day.series[key].mean);
      if (dayMeans.length) {
        const result = stats(dayMeans);
        result.day_count = dayMeans.length;
        result.seed_count = days.filter((day) => day.series[key]).reduce((sum, day) => sum + day.series[key].n, 0);
        overall[key] = result;
      }
    }
    return { split, days, series: [...definitions.values()], overall, record_count: selected.length,
      reference_sha256: selected.find((record) => record.reference_sha256)?.reference_sha256 || null };
  }

  function format(value, digits = 3) {
    return Number.isFinite(value) ? value.toFixed(digits) : "—";
  }
  function formatMeanSd(value, sd) {
    return `${format(value)} ± ${sd == null ? "—" : format(sd)}`;
  }
  function metricLabel(metric) {
    if (metric.toLowerCase() === "force") return "Force R²";
    if (metric.toLowerCase() === "emg") return "EMG R²";
    return `${metric} R²`;
  }
  function seriesKey(method, condition, metric) { return [method, condition, metric].join("\u001f"); }

  function fillSelect(select, options, selected) {
    select.replaceChildren(...options.map((option) => {
      const element = document.createElement("option");
      element.value = option.value;
      element.textContent = option.label;
      return element;
    }));
    if (options.some((option) => option.value === selected)) select.value = selected;
  }

  function activeData() {
    return aggregate(state.records, state.split);
  }

  function selectableSeries(data) {
    return data.series.filter((item) => item.condition !== "day0");
  }

  function updateControls(data) {
    const series = selectableSeries(data);
    const methods = [...new Set(series.map((item) => item.method))];
    if (!methods.length) {
      state.method = "";
      state.metric = "";
      fillSelect($("#method-select"), [], "");
      fillSelect($("#metric-select"), [], "");
      return;
    }
    if (!methods.includes(state.method)) state.method = methods.includes("Wiener on generator") ? "Wiener on generator" : methods[0];
    fillSelect($("#method-select"), methods.map((method) => ({ value: method, label: method })), state.method);
    const metrics = [...new Set(series.filter((item) => item.method === state.method).map((item) => item.metric))];
    if (!metrics.includes(state.metric)) state.metric = metrics.includes("force") ? "force" : metrics[0];
    fillSelect($("#metric-select"), metrics.map((metric) => ({ value: metric, label: metricLabel(metric) })), state.metric);
  }

  function seriesForSelection(data) {
    return data.series.filter((item) => item.method === state.method && item.metric === state.metric && item.condition !== "day0");
  }

  function svgElement(name, attrs = {}) {
    const node = document.createElementNS(svgNS, name);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
    return node;
  }

  function drawChart(data) {
    const svg = $("#performance-chart");
    const empty = $("#chart-empty");
    svg.replaceChildren();
    const series = seriesForSelection(data);
    const conditions = CONDITION_ORDER.filter((condition) => series.some((item) => item.condition === condition));
    const days = data.days;
    if (!conditions.length || !days.length) {
      svg.hidden = true;
      empty.hidden = false;
      $("#chart-legend").replaceChildren();
      return;
    }
    svg.hidden = false;
    empty.hidden = true;
    const viewW = 1080, viewH = 340;
    const margin = { top: 18, right: 20, bottom: 54, left: 62 };
    const plotW = viewW - margin.left - margin.right;
    const plotH = viewH - margin.top - margin.bottom;
    const chartSeries = new Map(series.map((item) => [item.condition, item]));
    const samples = [];
    for (const day of days) {
      for (const condition of conditions) {
        const value = day.series[chartSeries.get(condition).key];
        if (value) samples.push([value.mean - (value.sd || 0), value.mean + (value.sd || 0)]);
      }
    }
    let low = Math.min(0, ...samples.map((range) => range[0]));
    let high = Math.max(1, ...samples.map((range) => range[1]));
    const span = high - low || 1;
    low -= span * 0.08;
    high += span * 0.08;
    const y = (value) => margin.top + ((high - value) / (high - low)) * plotH;
    const title = svgElement("title", { id: "chart-title" });
    title.textContent = `${metricLabel(state.metric)} by Day-K and transfer condition`;
    const description = svgElement("desc", { id: "chart-description" });
    description.textContent = `Bars show the mean across seeds with seed standard deviation error bars for ${days.length} target days.`;
    svg.append(title, description);

    const tickCount = 5;
    for (let i = 0; i <= tickCount; i += 1) {
      const value = low + ((high - low) * i) / tickCount;
      const yy = y(value);
      svg.append(svgElement("line", { x1: margin.left, y1: yy, x2: viewW - margin.right, y2: yy, class: "gridline" }));
      const label = svgElement("text", { x: margin.left - 10, y: yy + 4, "text-anchor": "end", class: "axis-label" });
      label.textContent = value.toFixed(2);
      svg.append(label);
    }
    svg.append(svgElement("line", { x1: margin.left, y1: margin.top, x2: margin.left, y2: margin.top + plotH, class: "axisline" }));
    svg.append(svgElement("line", { x1: margin.left, y1: y(0), x2: viewW - margin.right, y2: y(0), class: "axisline" }));
    const axisTitle = svgElement("text", { x: 17, y: margin.top + plotH / 2, transform: `rotate(-90 17 ${margin.top + plotH / 2})`, "text-anchor": "middle", class: "axis-title" });
    axisTitle.textContent = metricLabel(state.metric);
    svg.append(axisTitle);

    const groupWidth = plotW / days.length;
    const barWidth = Math.min(36, groupWidth / (conditions.length + 1.4));
    days.forEach((day, dayIndex) => {
      const groupStart = margin.left + dayIndex * groupWidth;
      const totalWidth = conditions.length * barWidth * 1.35;
      const startX = groupStart + (groupWidth - totalWidth) / 2;
      conditions.forEach((condition, conditionIndex) => {
        const item = chartSeries.get(condition);
        const value = day.series[item.key];
        if (!value) return;
        const x = startX + conditionIndex * barWidth * 1.35;
        const yy = y(value.mean);
        const y0 = y(0);
        const bar = svgElement("rect", { x, y: Math.min(yy, y0), width: barWidth, height: Math.max(1, Math.abs(y0 - yy)), fill: CONDITIONS[condition]?.color || "#788797", class: "bar" });
        const titleNode = svgElement("title");
        titleNode.textContent = `${day.label} · ${CONDITIONS[condition]?.label || item.condition_label}: ${format(value.mean)} mean, ${format(value.sd)} SD across ${value.n} seeds`;
        bar.append(titleNode);
        svg.append(bar);
        if (value.sd != null) {
          const cap = Math.min(8, barWidth / 2);
          const y1 = y(value.mean + value.sd), y2 = y(value.mean - value.sd);
          const middle = x + barWidth / 2;
          svg.append(
            svgElement("line", { x1: middle, y1, x2: middle, y2, class: "errorbar" }),
            svgElement("line", { x1: middle - cap, y1, x2: middle + cap, y2: y1, class: "errorbar" }),
            svgElement("line", { x1: middle - cap, y1: y2, x2: middle + cap, y2, class: "errorbar" }),
          );
        }
      });
      const label = svgElement("text", { x: groupStart + groupWidth / 2, y: viewH - 23, "text-anchor": "middle", class: "day-label" });
      label.textContent = day.label;
      svg.append(label);
    });
    $("#chart-legend").replaceChildren(...conditions.map((condition) => {
      const entry = document.createElement("span");
      entry.className = "legend-item";
      const swatch = document.createElement("span");
      swatch.className = "legend-swatch";
      swatch.style.backgroundColor = CONDITIONS[condition]?.color || "#788797";
      const text = document.createElement("span");
      text.textContent = CONDITIONS[condition]?.label || chartSeries.get(condition).condition_label;
      entry.append(swatch, text);
      return entry;
    }));
    $("#chart-count").textContent = `${days.length} Day-K · ${data.record_count} runs`;
    $("#performance-title").textContent = `Mean ${metricLabel(state.metric).replace(" R²", "")} by target day`;
    $("#chart-caption").textContent = `${state.method} · seed mean ± SD for each Day-K`;
  }

  function selectedSeries(data) {
    return seriesForSelection(data).sort((a, b) => CONDITION_ORDER.indexOf(a.condition) - CONDITION_ORDER.indexOf(b.condition));
  }

  function renderOverall(data) {
    const series = selectedSeries(data);
    const aligned = series.find((item) => item.condition === "alignment_unsupervised") ||
      series.find((item) => item.condition === "alignment_supervised");
    const alignedStats = aligned ? data.overall[aligned.key] : null;
    $("#overall-title").textContent = aligned?.condition === "alignment_supervised" ? "Overall NoMAD · supervised" : "Overall NoMAD";
    $("#overall-nomad-value").textContent = alignedStats ? formatMeanSd(alignedStats.mean, alignedStats.sd) : "—";
    $("#overall-nomad-label").textContent = alignedStats
      ? `equal-weight mean ± SD across ${alignedStats.day_count} Day-K means`
      : "No aligned result for this readout and target";
    $("#overall-method-note").textContent = alignedStats
      ? `${state.method} · ${metricLabel(state.metric)} · each Day-K contributes once after its seeds are averaged.`
      : "Add a report containing a NoMAD alignment row to populate this summary.";
    const comparison = series.map((item) => {
      const result = data.overall[item.key];
      return { item, result };
    });
    $("#overall-comparison").replaceChildren(...comparison.map(({ item, result }) => {
      const cell = document.createElement("div");
      cell.className = "overall-cell";
      const name = document.createElement("span");
      name.textContent = CONDITIONS[item.condition]?.label || item.condition_label;
      const value = document.createElement("strong");
      value.textContent = result ? formatMeanSd(result.mean, result.sd) : "—";
      cell.append(name, value);
      return cell;
    }));
  }

  function makeSeedTable(day, series) {
    const table = document.createElement("table");
    table.className = "seed-table";
    const head = document.createElement("thead");
    const headerRow = document.createElement("tr");
    for (const label of ["Seed", ...series.map((item) => CONDITIONS[item.condition]?.label || item.condition_label), ""] ) {
      const th = document.createElement("th");
      th.textContent = label;
      headerRow.append(th);
    }
    head.append(headerRow);
    const body = document.createElement("tbody");
    for (const seed of day.seed_list) {
      const tr = document.createElement("tr");
      const seedCell = document.createElement("td");
      seedCell.textContent = String(seed);
      tr.append(seedCell);
      for (const item of series) {
        const seedValues = day.series[item.key]?.seeds || [];
        const observation = seedValues.find((entry) => entry.seed === seed);
        const td = document.createElement("td");
        td.textContent = observation ? format(observation.value) : "—";
        tr.append(td);
      }
      const actionCell = document.createElement("td");
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "remove-seed";
      remove.textContent = "Remove run";
      remove.setAttribute("aria-label", `Remove ${day.label} seed ${seed} ${state.split} run`);
      remove.addEventListener("click", () => {
        const identity = day.target_sha256 || `label:${day.label}`;
        state.records = state.records.filter((record) =>
          !((record.target_sha256 || `label:${record.dayk}`) === identity && record.seed === seed && record.split === state.split));
        state.expanded.delete(day.id);
        render();
        announce(`Removed ${day.label}, seed ${seed}, ${state.split}.`);
      });
      actionCell.append(remove);
      tr.append(actionCell);
      body.append(tr);
    }
    table.append(head, body);
    return table;
  }

  function renderTable(data) {
    const series = selectedSeries(data);
    const head = $("#day-table thead");
    const body = $("#day-table tbody");
    const headerRow = document.createElement("tr");
    for (const label of ["Day-K", "Seeds", ...series.map((item) => `${CONDITIONS[item.condition]?.label || item.condition_label} · mean ± SD`), "Inspect"]) {
      const th = document.createElement("th");
      th.textContent = label;
      headerRow.append(th);
    }
    head.replaceChildren(headerRow);
    body.replaceChildren();
    for (const day of data.days) {
      const row = document.createElement("tr");
      const label = document.createElement("td");
      const strong = document.createElement("strong");
      strong.textContent = day.label;
      label.append(strong);
      row.append(label);
      const count = document.createElement("td");
      count.textContent = String(day.seed_count);
      row.append(count);
      for (const item of series) {
        const cell = document.createElement("td");
        const value = day.series[item.key];
        cell.textContent = value ? formatMeanSd(value.mean, value.sd) : "—";
        row.append(cell);
      }
      const inspectCell = document.createElement("td");
      const inspect = document.createElement("button");
      inspect.type = "button";
      inspect.className = "inspect-button";
      inspect.textContent = state.expanded.has(day.id) ? "Hide seeds" : "Inspect seeds";
      inspect.setAttribute("aria-expanded", String(state.expanded.has(day.id)));
      inspect.addEventListener("click", () => {
        if (state.expanded.has(day.id)) state.expanded.delete(day.id);
        else state.expanded.add(day.id);
        render();
      });
      inspectCell.append(inspect);
      row.append(inspectCell);
      body.append(row);
      if (state.expanded.has(day.id)) {
        const details = document.createElement("tr");
        details.className = "seed-row";
        const cell = document.createElement("td");
        cell.colSpan = 3 + series.length;
        const wrapper = document.createElement("div");
        wrapper.className = "seed-details";
        const title = document.createElement("h3");
        title.textContent = `${day.label} · ${state.split} run results`;
        wrapper.append(title, makeSeedTable(day, series));
        cell.append(wrapper);
        details.append(cell);
        body.append(details);
      }
    }
    const overallRow = document.createElement("tr");
    overallRow.className = "overall-row";
    const label = document.createElement("td");
    label.textContent = "Overall · equal Day-K weight";
    overallRow.append(label);
    const seeds = document.createElement("td");
    seeds.textContent = String(data.days.reduce((sum, day) => sum + day.seed_count, 0));
    overallRow.append(seeds);
    for (const item of series) {
      const cell = document.createElement("td");
      const value = data.overall[item.key];
      cell.textContent = value ? formatMeanSd(value.mean, value.sd) : "—";
      overallRow.append(cell);
    }
    const blank = document.createElement("td");
    blank.textContent = "—";
    overallRow.append(blank);
    body.append(overallRow);
    $("#table-count").textContent = `${data.days.length} target days`;
  }

  function render() {
    const hasRecords = state.records.length > 0;
    $("#empty-state").hidden = hasRecords;
    $("#report-content").hidden = !hasRecords;
    $("#export-button").disabled = !hasRecords;
    if (!hasRecords) {
      $("#report-summary").textContent = "No report data loaded";
      return;
    }
    const splits = [...new Set(state.records.map((record) => record.split))];
    if (!splits.includes(state.split)) state.split = splits.includes("test") ? "test" : splits[0];
    fillSelect($("#split-select"), splits.map((split) => ({ value: split, label: split[0].toUpperCase() + split.slice(1) })), state.split);
    const data = activeData();
    updateControls(data);
    const selected = seriesForSelection(data);
    const dayCount = data.days.length;
    const seedRuns = data.days.reduce((sum, day) => sum + day.seed_count, 0);
    $("#report-summary").textContent = `${dayCount} target ${dayCount === 1 ? "day" : "days"} · ${seedRuns} seed runs · ${state.split} split`;
    $("#filter-note").textContent = "Statistics use the selected split only.";
    $("#performance-title").textContent = `Mean ${metricLabel(state.metric).replace(" R²", "")} by target day`;
    $("#chart-caption").textContent = `${state.method} · seed mean ± SD for each Day-K`;
    $("#chart-count").textContent = `${dayCount} Day-K · ${data.record_count} runs`;
    $("#provenance-reference").textContent = data.reference_sha256
      ? `Reference data SHA-256: ${data.reference_sha256}`
      : "Reference data hash was not included in every input report.";
    drawChart(data);
    renderOverall(data);
    renderTable(data);
    if (!selected.length) $("#chart-empty").hidden = false;
  }

  function announce(message) { $("#footer-status").textContent = message; }

  async function parseFile(file) {
    const text = await file.text();
    try { return JSON.parse(text); }
    catch (error) { throw new Error(`${file.name} is not valid JSON: ${error.message}`); }
  }

  function defaultLabel(fileName) {
    return fileName.replace(/\.json$/i, "").replace(/(^|[-_])evaluation$/i, "").replace(/[-_]+/g, " ").trim() || "Day-K";
  }

  function addBatch(additions) {
    const next = validateRecords([...state.records, ...additions]);
    state.records = next;
    render();
  }

  function openImport(file) {
    const modal = $("#import-dialog");
    $("#import-error").hidden = true;
    $("#import-filename").textContent = file.name;
    $("#dayk-input").value = defaultLabel(file.name);
    $("#seed-input").value = Number.isSafeInteger(state.pending.report?.args?.seed) ? state.pending.report.args.seed : "";
    $("#seed-input").disabled = false;
    $("#dayk-field").hidden = false;
    modal.showModal();
    $("#dayk-input").focus();
  }

  async function handleFile(file) {
    try {
      const parsed = await parseFile(file);
      if (parsed?.format === BUNDLE_FORMAT) {
        if (parsed.version !== BUNDLE_VERSION || !Array.isArray(parsed.records)) throw new Error("Unsupported normalized results bundle.");
        addBatch(parsed.records);
        announce(`Imported ${parsed.records.length} normalized result records.`);
        return;
      }
      if (Array.isArray(parsed?.days)) {
        addBatch(normalizeSummary(parsed, file.name));
        announce(`Imported results from ${parsed.days.length} target days in ${file.name}.`);
        return;
      }
      state.pending = { report: parsed, file };
      openImport(file);
    } catch (error) {
      announce(error.message);
    }
  }

  function exportBundle() {
    const bundle = { format: BUNDLE_FORMAT, version: BUNDLE_VERSION, records: state.records };
    const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "nomad-results-bundle.json";
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    announce("Downloaded the combined results bundle.");
  }

  function init() {
    const initial = window.NOMAD_REPORT_BUNDLE;
    if (initial) {
      try {
        if (initial.format !== BUNDLE_FORMAT || initial.version !== BUNDLE_VERSION) throw new Error("Unsupported generated report bundle.");
        if ((initial.records || []).length) state.records = validateRecords(initial.records);
      } catch (error) {
        announce(error.message);
      }
    }
    $("#add-results").addEventListener("click", () => $("#result-file").click());
    $("#empty-add-results").addEventListener("click", () => $("#result-file").click());
    $("#result-file").addEventListener("change", (event) => {
      const file = event.target.files?.[0];
      if (file) handleFile(file);
      event.target.value = "";
    });
    $("#export-button").addEventListener("click", exportBundle);
    $("#import-form").addEventListener("submit", (event) => {
      if (event.submitter?.value === "cancel") {
        state.pending = null;
        return;
      }
      event.preventDefault();
      if (!state.pending) return;
      const rawSeed = $("#seed-input").value.trim();
      const seed = rawSeed === "" ? null : Number(rawSeed);
      try {
        if (!Number.isSafeInteger(seed) || seed < 0) throw new Error("Enter the run seed as a nonnegative whole number.");
        const record = normalizeEvaluation(state.pending.report, $("#dayk-input").value, seed, state.pending.file.name);
        addBatch([record]);
        $("#import-dialog").close();
        announce(`Added ${record.dayk}, seed ${record.seed}, ${record.split}.`);
        state.pending = null;
      } catch (error) {
        $("#import-error").textContent = error.message;
        $("#import-error").hidden = false;
      }
    });
    $("#import-dialog").addEventListener("close", () => { state.pending = null; });
    $("#split-select").addEventListener("change", (event) => { state.split = event.target.value; render(); });
    $("#method-select").addEventListener("change", (event) => { state.method = event.target.value; state.metric = ""; render(); });
    $("#metric-select").addEventListener("change", (event) => { state.metric = event.target.value; render(); });
    render();
  }

  window.aggregateNomadRecords = aggregate;
  init();
})();
