"""
Generate a self-contained HTML review page for the regenerated course metadata.

Built for non-technical reviewers: every course is one card showing the generated
content beside the original, in plain language, with Approve / Needs work /
Reject buttons and a comment box. Decisions are saved in the browser as the
reviewer works and can be exported as CSV or JSON to hand back.

The output is a single file with no external assets, so it can be emailed, put on
a share, or opened straight off disk.

    python tools/review_report.py                          # 50 newest courses
    python tools/review_report.py --limit 200 --out qa.html
    python tools/review_report.py --tier metadata_only     # weakest-evidence set
    python tools/review_report.py --with-issues            # only flagged records
    python tools/review_report.py --area Domain
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import asyncpg  # noqa: E402

FETCH_SQL = """
SELECT course_id, regenerated_json, original_metadata_json, validation_issues,
       evidence_tier, declared_competency_area, llm_model, llm_prompt_version,
       regenerated_at, processing_duration_seconds
  FROM course_metadata_regenerated
 WHERE llm_prompt_version = $1
   AND ($2::text IS NULL OR evidence_tier = $2)
   AND ($3::text IS NULL OR regenerated_json->'PrimaryCompetencyArea'->>'name' = $3)
   AND ($4::bool IS FALSE OR jsonb_array_length(COALESCE(validation_issues, '[]'::jsonb)) > 0)
 ORDER BY regenerated_at DESC
 LIMIT $5
"""


def _j(value: Any) -> Any:
    """asyncpg returns jsonb as str; normalise to Python objects."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def build_records(rows: List[asyncpg.Record]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        gen = _j(r["regenerated_json"]) or {}
        orig = _j(r["original_metadata_json"]) or {}
        issues = _j(r["validation_issues"]) or []
        explain = gen.get("explain") or {}
        rubric = gen.get("RubricScoring") or {}
        ta = gen.get("TranscriptAnalysis") or {}

        out.append({
            "id": r["course_id"],
            "tier": r["evidence_tier"],
            "declared": r["declared_competency_area"],
            "model": r["llm_model"],
            "version": r["llm_prompt_version"],
            "at": r["regenerated_at"].strftime("%Y-%m-%d %H:%M") if r["regenerated_at"] else "",
            "secs": float(r["processing_duration_seconds"] or 0),

            # generated
            "name": gen.get("CourseName"),
            "summary": gen.get("CourseSummary"),
            "description": gen.get("CourseDescription"),
            "outcomes": gen.get("LearningOutcomes") or [],
            "level": gen.get("LearningLevel"),
            "mode": gen.get("LearningMode"),
            "duration": gen.get("Duration"),
            "language": gen.get("Language"),
            "tags": gen.get("Tags") or [],
            "area": (gen.get("PrimaryCompetencyArea") or {}).get("name"),
            "areaReason": (gen.get("PrimaryCompetencyArea") or {}).get("reason"),
            "areaConf": (gen.get("PrimaryCompetencyArea") or {}).get("confidence"),
            "sector": gen.get("Sector"),
            "subSector": gen.get("SubSector"),
            "subSectorTheme": gen.get("SubSectorTheme"),
            "functional": gen.get("FunctionalCompetencies") or [],
            "behavioural": gen.get("BehaviouralCompetencies")
                           or gen.get("BehavioralCompetencies") or [],
            "domain": gen.get("DomainCompetencies") or [],
            "bands": gen.get("TargetEmployeeGroups") or [],
            "roles": gen.get("Targetroles") or [],
            "priorDeclared": gen.get("PriorKnowledgeDeclared") or {},
            "priorSuggested": gen.get("SuggestivePriorKnowledge") or {},
            "rubric": rubric,
            "score": rubric.get("TotalScore"),
            "keywords": ((ta.get("KeywordsExtracted") or {}).get("Values")) or [],
            "tone": ((ta.get("LearningTone") or {}).get("Value")),

            # provenance / audit
            "issues": issues,
            "agrees": explain.get("agrees_with_declared_area"),
            "explain": explain,

            # original, for comparison
            "origName": orig.get("name"),
            "origDescription": orig.get("description"),
            "origKeywords": orig.get("keywords") or [],
            "origInstructions": orig.get("instructions"),
            "origOrg": orig.get("organisation"),
        })
    return out


# --------------------------------------------------------------------- template
# Colours are the validated reference palette: three categorical slots for the
# competency areas (that trio clears the all-pairs CVD and normal-vision floors
# in both modes) and the fixed status palette for review state. Every colour is
# paired with a text label, so nothing is carried by hue alone.
HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    color-scheme: light;
    --plane:#f9f9f7; --surface:#fcfcfb;
    --ink:#0b0b0b; --ink-2:#52514e; --ink-3:#898781;
    --rule:#e1e0d9; --ring:rgba(11,11,11,.10);
    --domain:#2a78d6; --functional:#eb6834; --behavioural:#1baf7a;
    --good:#0ca30c; --warning:#fab219; --critical:#d03b3b;
    --shadow:0 1px 2px rgba(11,11,11,.04), 0 4px 12px rgba(11,11,11,.04);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme=light]) {
      color-scheme: dark;
      --plane:#0d0d0d; --surface:#1a1a19;
      --ink:#fff; --ink-2:#c3c2b7; --ink-3:#898781;
      --rule:#2c2c2a; --ring:rgba(255,255,255,.10);
      --domain:#3987e5; --functional:#d95926; --behavioural:#199e70;
      --shadow:none;
    }
  }
  :root[data-theme=dark] {
    color-scheme: dark;
    --plane:#0d0d0d; --surface:#1a1a19;
    --ink:#fff; --ink-2:#c3c2b7; --ink-3:#898781;
    --rule:#2c2c2a; --ring:rgba(255,255,255,.10);
    --domain:#3987e5; --functional:#d95926; --behavioural:#199e70;
    --shadow:none;
  }
  * { box-sizing:border-box; }
  body {
    margin:0; background:var(--plane); color:var(--ink);
    font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    -webkit-font-smoothing:antialiased;
  }
  .wrap { max-width:1080px; margin:0 auto; padding:24px 20px 96px; }
  h1 { font-size:22px; margin:0 0 4px; letter-spacing:-.01em; }
  .sub { color:var(--ink-2); font-size:13.5px; margin:0 0 20px; }
  a { color:inherit; }

  /* ---- sticky header ---- */
  header {
    position:sticky; top:0; z-index:20; background:var(--plane);
    border-bottom:1px solid var(--rule); margin:0 -20px 20px; padding:14px 20px 12px;
  }
  .hrow { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  .grow { flex:1 1 220px; }
  input[type=search], select {
    font:inherit; color:var(--ink); background:var(--surface);
    border:1px solid var(--rule); border-radius:8px; padding:7px 10px; width:100%;
  }
  button {
    font:inherit; color:var(--ink); background:var(--surface); cursor:pointer;
    border:1px solid var(--rule); border-radius:8px; padding:7px 12px;
  }
  button:hover { border-color:var(--ink-3); }
  .btn-primary { background:var(--ink); color:var(--plane); border-color:var(--ink); }

  /* ---- KPI strip: a row of stat tiles, not a chart ---- */
  .kpis { display:flex; gap:10px; flex-wrap:wrap; margin:0 0 18px; }
  .kpi {
    background:var(--surface); border:1px solid var(--ring); border-radius:10px;
    padding:10px 14px; min-width:96px; box-shadow:var(--shadow);
  }
  .kpi b { display:block; font-size:22px; font-variant-numeric:tabular-nums; letter-spacing:-.02em; }
  .kpi span { font-size:11.5px; color:var(--ink-2); text-transform:uppercase; letter-spacing:.04em; }
  .bar { height:6px; border-radius:3px; background:var(--rule); overflow:hidden; margin-top:8px; }
  .bar i { display:block; height:100%; background:var(--good); }

  /* ---- course card ---- */
  .card {
    background:var(--surface); border:1px solid var(--ring); border-radius:12px;
    padding:18px 20px; margin:0 0 16px; box-shadow:var(--shadow);
  }
  .card.is-approved { border-left:3px solid var(--good); }
  .card.is-needswork { border-left:3px solid var(--warning); }
  .card.is-rejected  { border-left:3px solid var(--critical); }
  .cardhead { display:flex; gap:12px; align-items:flex-start; justify-content:space-between; }
  .cardhead h2 { font-size:17px; margin:0 0 6px; line-height:1.35; letter-spacing:-.01em; }
  .cid { font:12px/1 ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--ink-3); }

  .chips { display:flex; gap:6px; flex-wrap:wrap; margin:10px 0 0; }
  .chip {
    font-size:12px; padding:3px 9px; border-radius:999px;
    border:1px solid var(--ring); color:var(--ink-2); background:transparent; white-space:nowrap;
  }
  .chip.area { color:#fff; border:0; font-weight:600; }
  .chip.area.Domain { background:var(--domain); }
  .chip.area.Functional { background:var(--functional); }
  .chip.area.Behavioural, .chip.area.Behavioral { background:var(--behavioural); color:#08130e; }
  .chip.flag { border-color:var(--warning); color:var(--ink); }

  .field { margin:16px 0 0; }
  .label {
    font-size:11.5px; text-transform:uppercase; letter-spacing:.05em;
    color:var(--ink-3); margin:0 0 5px; font-weight:600;
  }
  .field p { margin:0; }
  ul.out { margin:0; padding-left:20px; }
  ul.out li { margin:0 0 4px; }
  .two { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  @media (max-width:760px){ .two { grid-template-columns:1fr; } }

  table.kv { width:100%; border-collapse:collapse; font-size:14px; }
  table.kv td { padding:4px 0; vertical-align:top; border-bottom:1px solid var(--rule); }
  table.kv td:first-child { color:var(--ink-2); width:38%; padding-right:10px; }
  .roles { margin:0; padding:0; list-style:none; }
  .roles li { padding:7px 0; border-bottom:1px solid var(--rule); }
  .roles .rn { font-weight:600; }
  .roles .rr { color:var(--ink-2); font-size:13.5px; }
  .conf { font:12px/1 ui-monospace,monospace; color:var(--ink-3); }

  details { margin:14px 0 0; border-top:1px solid var(--rule); padding-top:10px; }
  summary { cursor:pointer; font-size:13.5px; color:var(--ink-2); }
  summary:hover { color:var(--ink); }
  pre {
    background:var(--plane); border:1px solid var(--rule); border-radius:8px;
    padding:12px; overflow-x:auto; font:12px/1.5 ui-monospace,monospace; margin:10px 0 0;
  }
  .orig { color:var(--ink-2); font-size:14px; }

  .issues { margin:12px 0 0; padding:10px 12px; border-radius:8px;
            border:1px solid var(--warning); background:transparent; }
  .issues .label { color:var(--ink); }
  .issues ul { margin:6px 0 0; padding-left:18px; font-size:13.5px; }

  /* ---- review controls ---- */
  .review { margin:16px 0 0; padding:14px 0 0; border-top:2px solid var(--rule); }
  .verdicts { display:flex; gap:8px; flex-wrap:wrap; margin:0 0 10px; }
  .v { border-width:1.5px; }
  .v[aria-pressed=true][data-v=approved]  { background:var(--good); border-color:var(--good); color:#fff; }
  .v[aria-pressed=true][data-v=needswork] { background:var(--warning); border-color:var(--warning); color:#1a1400; }
  .v[aria-pressed=true][data-v=rejected]  { background:var(--critical); border-color:var(--critical); color:#fff; }
  textarea {
    font:inherit; width:100%; min-height:64px; resize:vertical; color:var(--ink);
    background:var(--plane); border:1px solid var(--rule); border-radius:8px; padding:9px 10px;
  }
  .saved { font-size:12px; color:var(--good); margin-left:8px; }
  .empty { text-align:center; color:var(--ink-2); padding:48px 0; }
  footer { position:fixed; inset:auto 0 0 0; background:var(--surface);
           border-top:1px solid var(--rule); padding:10px 20px; z-index:30; }
  .fwrap { max-width:1080px; margin:0 auto; display:flex; gap:12px; align-items:center;
           justify-content:space-between; flex-wrap:wrap; font-size:13.5px; }
  .vh { position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); }
</style>
</head>
<body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <p class="sub">__SUBTITLE__</p>

  <header>
    <div class="hrow">
      <div class="grow">
        <label class="vh" for="q">Search courses</label>
        <input id="q" type="search" placeholder="Search course name, sector, role or tag…">
      </div>
      <div>
        <label class="vh" for="fArea">Competency area</label>
        <select id="fArea"><option value="">All areas</option></select>
      </div>
      <div>
        <label class="vh" for="fTier">Evidence</label>
        <select id="fTier"><option value="">All evidence levels</option></select>
      </div>
      <div>
        <label class="vh" for="fState">Review state</label>
        <select id="fState">
          <option value="">All review states</option>
          <option value="unreviewed">Not yet reviewed</option>
          <option value="approved">Approved</option>
          <option value="needswork">Needs work</option>
          <option value="rejected">Rejected</option>
          <option value="flagged">Auto-flagged only</option>
        </select>
      </div>
      <button id="theme" title="Toggle light/dark">◐</button>
    </div>
  </header>

  <div class="kpis" id="kpis"></div>
  <div id="list"></div>
  <div class="empty" id="empty" hidden>No courses match these filters.</div>
</div>

<footer>
  <div class="fwrap">
    <div id="progress">—</div>
    <div>
      <button id="exportCsv">Download comments (CSV)</button>
      <button id="exportJson">Download comments (JSON)</button>
      <button class="btn-primary" id="clear">Clear my review</button>
    </div>
  </div>
</footer>

<script type="application/json" id="data">__DATA__</script>
<script>
(function () {
  "use strict";
  const DATA = JSON.parse(document.getElementById("data").textContent);
  const KEY  = "igot-review-" + (DATA.reportId || "default");

  // Reviews live in this browser only, so a reviewer can stop and come back.
  // Nothing is uploaded anywhere; the export buttons are how findings travel.
  let review = {};
  try { review = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { review = {}; }
  const save = () => { try { localStorage.setItem(KEY, JSON.stringify(review)); } catch (e) {} };

  const esc = s => String(s == null ? "" : s).replace(/[&<>"']/g,
    c => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c]));
  const el = (h) => { const t = document.createElement("template"); t.innerHTML = h.trim(); return t.content.firstChild; };
  const TIER_LABEL = {
    transcript:"Full transcript", thin_transcript:"Short transcript",
    pdf_only:"Documents only", links_only:"Links only", metadata_only:"No content — title & description only"
  };

  /* ---------- filters ---------- */
  const fill = (sel, vals, labeller) => {
    vals.filter(Boolean).sort().forEach(v => {
      const o = document.createElement("option");
      o.value = v; o.textContent = labeller ? (labeller[v] || v) : v;
      sel.appendChild(o);
    });
  };
  fill(document.getElementById("fArea"), [...new Set(DATA.courses.map(c => c.area))]);
  fill(document.getElementById("fTier"), [...new Set(DATA.courses.map(c => c.tier))], TIER_LABEL);

  const state = () => ({
    q: document.getElementById("q").value.trim().toLowerCase(),
    area: document.getElementById("fArea").value,
    tier: document.getElementById("fTier").value,
    rev: document.getElementById("fState").value
  });

  function matches(c, s) {
    if (s.area && c.area !== s.area) return false;
    if (s.tier && c.tier !== s.tier) return false;
    const r = review[c.id] || {};
    if (s.rev === "unreviewed" && r.verdict) return false;
    if (s.rev === "flagged" && !(c.issues || []).length) return false;
    if (["approved","needswork","rejected"].includes(s.rev) && r.verdict !== s.rev) return false;
    if (s.q) {
      const hay = [c.name, c.origName, c.sector, c.subSector, c.subSectorTheme, c.id,
                   (c.tags||[]).join(" "), (c.roles||[]).map(x=>x.Name).join(" "),
                   (c.functional||[]).concat(c.behavioural||[],c.domain||[])
                     .map(x=>x.Theme+" "+x.SubTheme).join(" ")]
                  .join(" ").toLowerCase();
      if (!hay.includes(s.q)) return false;
    }
    return true;
  }

  /* ---------- card ---------- */
  function pairs(list) {
    if (!list || !list.length) return '<p class="orig">None</p>';
    return '<ul class="out">' + list.map(x =>
      `<li><b>${esc(x.Theme)}</b> › ${esc(x.SubTheme)}` +
      (x.Confidence != null ? ` <span class="conf">${(+x.Confidence).toFixed(2)}</span>` : "") +
      `</li>`).join("") + "</ul>";
  }

  function card(c) {
    const r = review[c.id] || {};
    const areaCls = (c.area || "").replace(/[^A-Za-z]/g, "");
    const comps = c.area === "Domain" ? c.domain
                : c.area === "Functional" ? c.functional : c.behavioural;

    const sgos = c.sector
      ? `<table class="kv">
           <tr><td>Sector</td><td>${esc(c.sector)}</td></tr>
           <tr><td>Sub-sector</td><td>${esc(c.subSector)}</td></tr>
           <tr><td>Theme</td><td>${esc(c.subSectorTheme)}</td></tr>
         </table>`
      : '<p class="orig">Not a sector-specific course, so no sector was assigned.</p>';

    const roles = (c.roles || []).length
      ? '<ul class="roles">' + c.roles.map(x =>
          `<li><span class="rn">${esc(x.Name)}</span>
               <span class="conf">${x.Confidence != null ? (+x.Confidence).toFixed(2) : ""}</span>
               <div class="rr">${esc(x.Rationale)}</div></li>`).join("") + "</ul>"
      : '<p class="orig">No matching designations were found.</p>';

    const issues = (c.issues || []).length
      ? `<div class="issues"><div class="label">⚠ Automatically flagged (${c.issues.length})</div>
           <ul>${c.issues.map(i => `<li>${esc(i)}</li>`).join("")}</ul></div>`
      : "";

    const agree = c.declared && c.agrees === false
      ? `<span class="chip flag">⚠ AI says ${esc(c.area)}, source said ${esc(c.declared)}</span>` : "";

    const node = el(`<article class="card ${r.verdict ? "is-" + r.verdict : ""}" data-id="${esc(c.id)}">
      <div class="cardhead">
        <div>
          <h2>${esc(c.name || "(no title generated)")}</h2>
          <div class="cid">${esc(c.id)}</div>
        </div>
      </div>

      <div class="chips">
        <span class="chip area ${areaCls}">${esc(c.area || "?")}</span>
        <span class="chip">${esc(TIER_LABEL[c.tier] || c.tier || "")}</span>
        <span class="chip">${esc(c.level || "")}${c.score != null ? " · score " + c.score : ""}</span>
        <span class="chip">${esc(c.duration || "")}</span>
        <span class="chip">${esc(c.language || "")}</span>
        ${agree}
      </div>

      ${issues}

      <div class="field">
        <div class="label">Summary</div>
        <p>${esc(c.summary)}</p>
      </div>

      <div class="field">
        <div class="label">Description</div>
        <p>${esc(c.description)}</p>
      </div>

      <div class="field">
        <div class="label">What a learner will be able to do</div>
        <ul class="out">${(c.outcomes || []).map(o => `<li>${esc(o)}</li>`).join("")}</ul>
      </div>

      <div class="two">
        <div class="field">
          <div class="label">Competencies (${esc(c.area || "")})</div>
          ${pairs(comps)}
        </div>
        <div class="field">
          <div class="label">Sector classification</div>
          ${sgos}
        </div>
      </div>

      <div class="two">
        <div class="field">
          <div class="label">Who it is for</div>
          <p>${(c.bands || []).map(b => esc(b)).join(", ") || "—"}</p>
          ${roles}
        </div>
        <div class="field">
          <div class="label">Search tags</div>
          <div class="chips">${(c.tags || []).map(t => `<span class="chip">${esc(t)}</span>`).join("")}</div>
        </div>
      </div>

      <details>
        <summary>Compare with the original course record</summary>
        <div class="field"><div class="label">Original title</div><p class="orig">${esc(c.origName) || "—"}</p></div>
        <div class="field"><div class="label">Original description</div><p class="orig">${esc(c.origDescription) || "—"}</p></div>
        <div class="field"><div class="label">Original objectives written by the author</div><p class="orig">${esc(c.origInstructions) || "—"}</p></div>
        <div class="field"><div class="label">Provider</div><p class="orig">${esc(c.origOrg) || "—"}</p></div>
      </details>

      <details>
        <summary>Why the AI decided this (audit trail)</summary>
        <div class="field"><div class="label">Competency area reasoning</div><p class="orig">${esc(c.areaReason)}</p></div>
        <pre>${esc(JSON.stringify(c.explain, null, 2))}</pre>
      </details>

      <div class="review">
        <div class="label">Your review</div>
        <div class="verdicts">
          <button class="v" data-v="approved"  aria-pressed="${r.verdict === "approved"}">✓ Looks good</button>
          <button class="v" data-v="needswork" aria-pressed="${r.verdict === "needswork"}">△ Needs changes</button>
          <button class="v" data-v="rejected"  aria-pressed="${r.verdict === "rejected"}">✕ Not acceptable</button>
          <span class="saved" hidden>Saved</span>
        </div>
        <label class="vh" for="c-${esc(c.id)}">Comment</label>
        <textarea id="c-${esc(c.id)}" placeholder="What is wrong, and what should it say instead?">${esc(r.comment || "")}</textarea>
      </div>
    </article>`);

    const flash = node.querySelector(".saved");
    const ping = () => { flash.hidden = false; setTimeout(() => { flash.hidden = true; }, 1200); };

    node.querySelectorAll(".v").forEach(btn => {
      btn.addEventListener("click", () => {
        const v = btn.dataset.v;
        const cur = review[c.id] || {};
        cur.verdict = cur.verdict === v ? null : v;   // click again to undo
        cur.at = new Date().toISOString();
        review[c.id] = cur; save(); ping();
        node.querySelectorAll(".v").forEach(b =>
          b.setAttribute("aria-pressed", String(b.dataset.v === cur.verdict)));
        node.className = "card" + (cur.verdict ? " is-" + cur.verdict : "");
        kpis();
      });
    });

    let t;
    node.querySelector("textarea").addEventListener("input", e => {
      clearTimeout(t);
      t = setTimeout(() => {
        const cur = review[c.id] || {};
        cur.comment = e.target.value;
        cur.at = new Date().toISOString();
        review[c.id] = cur; save(); ping(); kpis();
      }, 400);
    });

    return node;
  }

  /* ---------- KPI tiles ---------- */
  function kpis() {
    const cs = DATA.courses;
    const done = cs.filter(c => (review[c.id] || {}).verdict).length;
    const counts = { approved:0, needswork:0, rejected:0 };
    cs.forEach(c => { const v = (review[c.id] || {}).verdict; if (v) counts[v]++; });
    const flagged = cs.filter(c => (c.issues || []).length).length;
    const pct = cs.length ? Math.round(done / cs.length * 100) : 0;

    document.getElementById("kpis").innerHTML = `
      <div class="kpi"><b>${cs.length}</b><span>Courses</span></div>
      <div class="kpi"><b>${done}</b><span>Reviewed</span>
        <div class="bar"><i style="width:${pct}%"></i></div></div>
      <div class="kpi"><b>${counts.approved}</b><span>✓ Looks good</span></div>
      <div class="kpi"><b>${counts.needswork}</b><span>△ Needs changes</span></div>
      <div class="kpi"><b>${counts.rejected}</b><span>✕ Not acceptable</span></div>
      <div class="kpi"><b>${flagged}</b><span>⚠ Auto-flagged</span></div>`;
    document.getElementById("progress").textContent =
      `${done} of ${cs.length} reviewed (${pct}%) · ${cs.length - done} to go`;
  }

  /* ---------- render ---------- */
  function render() {
    const s = state();
    const list = document.getElementById("list");
    list.innerHTML = "";
    const shown = DATA.courses.filter(c => matches(c, s));
    shown.forEach(c => list.appendChild(card(c)));
    document.getElementById("empty").hidden = shown.length > 0;
    kpis();
  }
  ["q","fArea","fTier","fState"].forEach(id =>
    document.getElementById(id).addEventListener("input", render));

  /* ---------- export ---------- */
  const dl = (name, text, type) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type }));
    a.download = name; a.click(); URL.revokeObjectURL(a.href);
  };
  document.getElementById("exportJson").addEventListener("click", () => {
    dl(`review-${DATA.reportId}.json`, JSON.stringify({
      report: DATA.reportId, reviewedAt: new Date().toISOString(), reviews: review
    }, null, 2), "application/json");
  });
  document.getElementById("exportCsv").addEventListener("click", () => {
    const q = v => '"' + String(v == null ? "" : v).replace(/"/g, '""') + '"';
    const rows = [["course_id","course_name","verdict","comment","reviewed_at",
                   "competency_area","evidence_tier","auto_flagged"].join(",")];
    DATA.courses.forEach(c => {
      const r = review[c.id] || {};
      if (!r.verdict && !r.comment) return;      // only what the reviewer touched
      rows.push([c.id, c.name, r.verdict || "", r.comment || "", r.at || "",
                 c.area, c.tier, (c.issues || []).length].map(q).join(","));
    });
    if (rows.length === 1) { alert("No reviews recorded yet."); return; }
    dl(`review-${DATA.reportId}.csv`, rows.join("\\n"), "text/csv");
  });
  document.getElementById("clear").addEventListener("click", () => {
    if (!confirm("Delete every verdict and comment you have entered on this page?")) return;
    review = {}; save(); render();
  });

  /* ---------- theme ---------- */
  document.getElementById("theme").addEventListener("click", () => {
    const dark = matchMedia("(prefers-color-scheme: dark)").matches;
    const cur = document.documentElement.getAttribute("data-theme") || (dark ? "dark" : "light");
    document.documentElement.setAttribute("data-theme", cur === "dark" ? "light" : "dark");
  });

  render();
})();
</script>
</body>
</html>
"""


def render_html(records: List[Dict[str, Any]], report_id: str, subtitle: str) -> str:
    payload = {"reportId": report_id, "courses": records}
    # Escaping < prevents a "</script>" inside any string from ending the block.
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    return (
        HTML.replace("__TITLE__", "Course metadata review")
        .replace("__SUBTITLE__", subtitle)
        .replace("__DATA__", data)
    )


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=os.environ.get("PROMPT_VERSION", "v3.5-advanced"))
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--tier", default=None)
    ap.add_argument("--area", default=None, choices=["Domain", "Functional", "Behavioural"])
    ap.add_argument("--with-issues", action="store_true",
                    help="only courses the validator flagged")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    conn = await asyncpg.connect(dsn=os.environ["DB_DSN"])
    try:
        rows = await conn.fetch(
            FETCH_SQL, args.version, args.tier, args.area, args.with_issues, args.limit
        )
    finally:
        await conn.close()

    if not rows:
        print(f"No records for {args.version} with those filters.")
        return 1

    records = build_records(rows)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or Path(os.environ.get("LOG_DIR", "logs")) / f"review-{stamp}.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    filters = [f"{len(records)} courses", args.version]
    if args.tier:
        filters.append(f"evidence: {args.tier}")
    if args.area:
        filters.append(f"area: {args.area}")
    if args.with_issues:
        filters.append("auto-flagged only")
    subtitle = (
        " · ".join(filters)
        + f" · generated {datetime.now().strftime('%d %b %Y %H:%M')}"
        + ". Your verdicts and comments are saved in this browser as you go — "
          "use the buttons at the bottom to download them when you are done."
    )

    out.write_text(render_html(records, f"{args.version}-{stamp}", subtitle), encoding="utf-8")
    flagged = sum(1 for r in records if r["issues"])
    print(f"wrote {out}  ({len(records)} courses, {flagged} auto-flagged, "
          f"{out.stat().st_size/1024:.0f} KB, self-contained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
