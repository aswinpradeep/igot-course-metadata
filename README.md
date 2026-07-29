# iGOT Course Metadata Regeneration — Framework v3.5 (Advanced)

Regenerates standardised metadata for iGOT Karmayogi Bharat courses using Vertex AI Gemini, grounded
in three master datasets (KCM competencies, SGOS sectors, the iGOT designation master) and in each
course's own transcripts and documents.

This replaces the earlier `v3.4-advanced-extended` implementation. v3.4 is superseded but its output
is preserved for comparison — see [Version coexistence](#version-coexistence).

---

## Contents

- [What it does](#what-it-does)
- [Inputs you must supply](#inputs-you-must-supply)
- [Quick start](#quick-start)
- [How a course is processed](#how-a-course-is-processed)
- [The input data, as it actually is](#the-input-data-as-it-actually-is)
- [Changes from v3.4](#changes-from-v34)
- [Output examples](#output-examples)
- [Output contract](#output-contract)
- [Operations](#operations)
- [Files](#files)
- [Known gaps](#known-gaps)

---

## What it does

For each course folder it produces one JSON record containing a regenerated title, summary and
description; learning outcomes; competency classification (exactly one of Domain / Functional /
Behavioural); SGOS sector mapping; prior-knowledge analysis; transcript analysis; a seven-parameter
rubric score; target role bands and specific designations; and a full `explain` audit trail.

Records are written to Postgres (`course_metadata_regenerated`), with a per-run outcome CSV and log.

---

## Inputs you must supply

**This repository contains code only.** The reference masters, course content and framework
specifications are internal iGOT artefacts and are deliberately not tracked here. Obtain them from the
iGOT/Karmayogi team, place them anywhere on disk, and point `.env` at them.

| Input | What it is | `.env` variable |
|---|---|---|
| `competencies*.json` | KCM master — the **only** valid source of Functional and Behavioural competencies. 114 Theme/SubTheme pairs (73 Functional, 41 Behavioural). Identical to the `cbp-ai-service` service's `data/competencies.json` (md5 `b021990a65365834cfb60bb702fcb0a4`). | `KCM_PATH` |
| `SGOS*.json` | Sector → SubSector → Theme master — the **only** valid source of Domain classification. 11 sectors, 92 subsectors, 460 themes. KCM has no Domain entries, so SGOS *is* the Domain source. | `SGOS_PATH` |
| `igot_designations*.csv` | Official iGOT designation master, columns `id,name` (19,936 rows). The only permitted source of `Targetroles`. | `DESIGNATIONS_PATH` |
| `non-scorm-content-*.zip` | Course content: an outer zip of per-batch zips of course folders (~4.3 GB). | `COURSES_BASE_PATH` after extraction |
| `scorm-content-*.zip` | SCORM course content (~1.2 GB). Not yet wired up — see [Known gaps](#known-gaps). | — |
| Vertex AI service-account JSON | Needs the Vertex AI User role on the project. | `GOOGLE_APPLICATION_CREDENTIALS` |
| Postgres | One writable database for output; optionally a second, read-only, for enrichment. Tables are created automatically. | `DB_DSN`, `ENRICHMENT_DSN` |

Derived artefacts, generated locally and also untracked: `manifest.jsonl` (from `tools/inventory.py`)
and `data/designation_index.npz` (from `tools/build_designation_index.py`).

The framework specification PDFs (TPT v3.4 / v3.5 / AI Output Schema / AI Prompts) are the source of
the output contract but are internal documents and are not published here. The contract they define is
described under [Output contract](#output-contract) and [Changes from v3.4](#changes-from-v34).

---

## Quick start

```bash
# 1. Dependencies
pip install asyncpg httpx pymupdf tenacity python-dotenv google-genai numpy

# 2. Configuration
cp .env.example .env      # then fill in real values

# 3. Unpack course content and build the deduplicated manifest
#    (the outer zip contains nested per-batch zips)
mkdir -p course_data/extracted/non-scorm/_staging
unzip course_data/non-scorm-content-11-july.zip -d course_data/extracted/non-scorm/_batches
for z in course_data/extracted/non-scorm/_batches/*/*.zip; do
  unzip -q -o "$z" -d "course_data/extracted/non-scorm/_staging/$(basename "$z" .zip)"
done
python tools/inventory.py \
  --staging course_data/extracted/non-scorm/_staging \
  --out     course_data/extracted/non-scorm/manifest.jsonl

# 4. Build the designation embedding index (one-off, ~90s)
python tools/build_designation_index.py

# 5. Dry run first — no AI calls, no writes
python meta_gen.py --limit 10

# 6. Then for real
python meta_gen.py --limit 10 --execute
```

**Dry run is the default.** Nothing is called or written until you pass `--execute`. This follows the
convention in the legacy service's `bulk_scripts/README.md`.

---

## How a course is processed

1. **Read metadata** — `metadata.json` at the course root. `competencies_v6` is *removed* from what
   the model sees (the framework requires the competency area to be judged from content, and the
   declared value is frequently wrong) but retained for audit comparison.
2. **Extract content** — transcripts are concatenated from **module subfolders**, labelled per module;
   PDFs are read with PyMuPDF; reference URLs are collected from `pdf_links.txt` / `video.txt`.
   Placeholder files are recognised and treated as absent.
3. **Assign an evidence tier** — `transcript`, `thin_transcript`, `pdf_only`, `links_only`, or
   `metadata_only`. This is passed to the model and caps how confident it is allowed to be.
4. **Gather authoritative facts** — language, duration and provider. Duration and language come from
   `course_metadata_v3` where available; the model is told these override its own inference.
5. **Shortlist designations** — the course text is embedded and matched against 19,936 pre-embedded
   designation names; the top 150 are passed to the model, which may only choose from them.
6. **Generate** — one Gemini call with a structured response schema.
7. **Validate and repair** — see below. Every correction is recorded.
8. **Persist** — record, validation issues, evidence tier, token usage and declared-area comparison.

### The validation layer

The response schema constrains *shape* only. These are enforced afterwards, in `validation.py`:

| Check | Action |
|---|---|
| Theme/SubTheme not a real KCM pair | dropped, issue recorded |
| Competency branch contradicts the primary area | cleared (one area per course) |
| SGOS sector/subsector/theme not a real path | cleared |
| Designation not in the iGOT master | rejected |
| `TotalScore` | recomputed in Python from the weights — never trusted from the model |
| `Classification` / `LearningLevel` | recomputed and forced to agree |
| Placeholder strings (`"Not Applicable"`, `"N/A"`, …) | nulled |
| Provider name appearing in tags | dropped |
| Transcript-derived evidence on a `metadata_only` course | cleared as impossible |
| Non-URL entries in `ReferenceResources` | dropped |

Repairs never fail a course — the record is stored with its issues alongside, so systemic prompt
problems are visible in aggregate.

---

## The input data, as it actually is

Measured across all 3,707 non-SCORM courses (`tools/inventory.py`):

| | |
|---|---|
| Course folders in the zips | 3,801 |
| Distinct courses after dedup | **3,707** (94 duplicates across overlapping batch zips) |
| Courses with a usable transcript | 2,139 (57.7%), median 19,114 chars |
| Courses with no transcript and no PDF | 1,526 (41.2%) — `metadata_only` |
| Courses with local PDFs | 567 (2,184 PDFs) |
| **Course-root `english_subtitles.vtt` that is a placeholder** | **3,707 / 3,707 (100%)** |
| Total transcript text | 67.2M chars |
| Present in `course_metadata_v3` (language/duration) | 3,395 (91.6%) |

Course folders are **nested**, and the real content is *not* at the course root:

```
do_<courseid>/
├── metadata.json            ← identifier, name, description, keywords,
├── english_subtitles.vtt    ←   organisation, competencies_v6, instructions,
├── video.txt                ←   courseCategory, scorm
├── pdf_links.txt
└── do_<moduleid>/
    ├── <Module Name>.pdf              ← PDFs live here (depth 2)
    └── <Module Name>/
        ├── metadata.json
        ├── english_subtitles.vtt      ← the real transcript
        └── pdf_links.txt
```

Two consequences worth stating plainly:

- The course-root VTT is the literal string `// No English subtitles found` in **every** non-SCORM
  course. Reading only the course root — as v3.4 did — yields no transcript at all, and injects that
  sentinel string into the prompt as if it were the transcript.
- PDFs sit at `do_<moduleid>/<Name>.pdf`, a *sibling* of the module directory. A top-level
  `glob('*.pdf')` finds none of them.

Other observations that shaped the implementation:

- `competencies_v6` is a newline-joined value with one token per module (e.g.
  `"Functional\nFunctional\nFunctional"`). Normalised: Domain 2,463, Functional 862, Behavioural 365,
  mixed 16, empty 1. So "one competency area per course" is already 99.6% true in the source.
- Content is multilingual — English 1,865, Hindi 865, plus Kannada, Telugu, Marathi, Malayalam,
  Punjabi, Gujarati. Roughly 45% is non-English. Folder names are non-ASCII.
- `english_subtitles.vtt` is often an *English translation* of non-English audio, so the transcript's
  language does not indicate the course language. The prompt says so explicitly.
- `difficulty_level` in `course_metadata_v3` is "Beginner" for 77% of courses — consistent with the
  ticket note that the incoming difficulty level is not correct. It is never given to the model as a
  fact; it is kept for comparison only.
- VTT cue spans **underestimate** duration (median 0.59× the platform value; only 61% within 0.5–2×),
  so platform duration is primary and VTT-derived duration is a labelled fallback.

---

## Changes from v3.4

### 1. Transcripts and PDFs are actually read

v3.4 read only the course root, so for practically every course it sent an empty transcript, no PDFs,
and the placeholder sentinel as content. It now recurses into module folders, filters placeholders,
and finds PDFs at depth 2. This is the single largest change in output quality — 67.2M characters of
transcript that were previously invisible.

### 2. The `explain` audit trail is no longer silently discarded

v3.4 declared `explain`'s sub-objects as bare `{"type": "object"}` with no `properties`. Gemini's
structured output therefore returned `{}` for each. **All 368 rows of the v3.4 run have an empty
audit trail** — `mapping_reasons`, `designation_inference`, `competency_mapping_triggers` and
`learning_outcomes_validation` are empty in 368/368. Since auditability is the framework's stated
purpose, this was the most serious defect. Every sub-object is now fully typed and populates.

The same class of bug affected `TranscriptAnalysis`: the prompt asked for per-marker `verb`,
`example_phrase`, `estimated_level` and `confidence`, while the schema declared an array of strings,
so the requested evidence was thrown away. Now typed.

### 3. New v3.5 fields

`Targetroles` (the headline v3.5 addition), `DomainCompetencies`, `SuggestiveCompetencies.Domain`,
`ReferenceResources`, `Language`, `EmbeddingVectorID`, `Version`, `GeneratedOn`, `Generator`, and
`TranscriptAnalysis.TranscriptPath`.

### 4. `Targetroles` grounded in the designation master

19,936 designations cannot go in a prompt (~250k tokens of noise). The master is embedded once and
cached; per course the top 150 candidates are retrieved and the model chooses from those only.
Anything outside the master is rejected in Python — an unconstrained model invents plausible ids like
`desig_001`.

### 5. Rubric arithmetic is computed, not requested

v3.4 asked the model for `TotalScore` and trusted it. Scores are now recomputed in Python from the
weights, `Classification` is derived from the total, and `LearningLevel` is forced to match. In the
v3.4 baseline 11 of 368 rows had `LearningLevel` disagreeing with `Classification`.

Weights follow the v3.5 spec table and differ from the v3.4 code:

| Parameter | v3.5 spec (now) | v3.4 code |
|---|---|---|
| LearningObjectives | 10 | 5 |
| PriorKnowledge | 5 | 5 |
| BloomTaxonomy | 10 | 25 |
| ComplexityOfContent | 15 | 35 |
| ExpectedRoleOutcome | 20 | 10 |
| ExtentOfLearning | 25 | 20 |
| TargetAudienceAlignment | 15 | **0** |

Bands are now `≤45` Beginner, `46–75` Intermediate, `>75` Advanced (v3.4 used 55/75). The spec's own
bands overlap at exactly 75; resolved in favour of Intermediate. Both weights and bands are
configurable (`RUBRIC_WEIGHTS`, `RUBRIC_BEGINNER_MAX`, `RUBRIC_INTERMEDIATE_MAX`) and the same
constants feed both the prompt text and the Python calculation, so they cannot drift apart.

The v3.4 zero-base evidence rules — which are more specific than anything in the spec — are kept.

### 6. `Sector` is nullable instead of `"Not Applicable"`

v3.4 marked `Sector`, `SubSector` and `SubSectorTheme` as *required* while the prompt forbade
populating them for Functional/Behavioural courses. The model complied by writing the string
`"Not Applicable"` into a field typed as a sector name — **214 of 368 rows (58%)**. They are now
nullable, not required, and validated against the SGOS master when the area is Domain.

Related: KCM contains no Domain entries (73 Functional + 41 Behavioural only), which confirms the
ticket's note that SGOS *is* the Domain source. When the area is Domain the SGOS choice is mirrored
into `DomainCompetencies` so both shapes in the spec schema are satisfied from one decision.

### 7. `scorm` is read per course

v3.4 took a single `SCORM_FLAG` environment variable and applied it to every course. The flag is per
course in `metadata.json`.

### 8. `competencies_v6` no longer destroys the record

v3.4 did `del current_metadata['competencies_v6']`. A missing key raised `KeyError`, which a bare
`except` caught by setting the metadata to `None` *and* the audit copy to `'null'` — so the course was
processed with no metadata and the original was lost. It is now `pop(..., None)`, and the declared
value is kept for audit: every record carries `declared_competency_area` and
`explain.agrees_with_declared_area`.

### 9. Reliability and operations

- **A request timeout.** v3.4 had none; a single call could hang forever and wedge the run (observed).
- **Retries only on transient errors.** v3.4 retried on `Exception`, so permanent failures (bad
  credentials, rejected schema) burned three backed-off attempts each.
- **Stale claims are reclaimed.** A killed worker left rows in `in_progress`, which the pending query
  does not select — those courses were stranded permanently and a restart silently skipped them.
- **Version-scoped storage.** The table was keyed on `course_id` alone, so a v3.5 run would have
  overwritten the v3.4 baseline.
- **Terminates.** v3.4 looped forever sleeping 30s. It now drains and exits; `--serve` opts back in.
- **Dry-run default, outcome CSV, `--force`, `--batch-size`, up-front env validation** — matching the
  house conventions in the legacy `bulk_scripts/README.md`.
- Removed a debug `prompt.txt` write that all concurrent workers raced on, dead `data/kcm_book.pdf`
  code, and an unused `httpx` import; `LOG_LEVEL` was read but never applied.

### 10. Prompt hardening taken from the production service

`src/prompts/v3/prompts.py` in the legacy service encodes failure modes learned in production, now
adopted here:

- **No cross-mixing between KCM entries** — a `theme` from one row must not be paired with a
  `sub_theme` from another, even when both values exist somewhere in the dataset.
- **No field-swapping** — `theme` → Theme and `sub_theme` → SubTheme, never interchanged.
- **Copy character-for-character**; locate the exact entry before writing it.

`validation.py` already rejected these structurally, which meant the model's mistakes showed up as
silently dropped competencies. Warning about them in the prompt reduces the drops.

Also adopted: the AB/CD seniority criteria from `DESIGNNATION_GROUP_SYSTEM_PROMPT` for grounding
Group A/B/C/D role-band assignment, which the designation master cannot validate (it has only
`id,name` — no band column).

### 11. Prompt structure for cacheability

The static system prompt and both masters (~55k tokens) now form a byte-identical prefix on every
request, with per-course content strictly after it. v3.4 interleaved per-course data first, which
makes prefix caching impossible.

### What deliberately did *not* change

**Learning outcomes stay unlabelled.** Spec section 8 asks for outcomes prefixed `"Why:"`, `"How:"`,
`"What:"`. v3.4 overrode this and produced clean learner-facing outcomes; labelled strings would be a
visible regression on the platform. Why/how/what coverage is asserted in
`explain.learning_outcomes_validation` instead, so the validation intent is still met. Set
`LEARNING_OUTCOME_STYLE=why_how_what` for the literal spec behaviour.

**`Behavioural` keeps the British spelling**, matching the `type` values in the KCM master. The spec
document writes `Behavioral`. Set `BEHAVIOURAL_SPELLING=Behavioral` if a consumer needs it.

---

## Output examples

All excerpts below are **real output** — v3.4 rows from the 368-record baseline, v3.5 rows from the
pilot run. Long prose fields are elided as `…`; nothing else is edited.

### New in v3.5

`Targetroles` — the headline addition. Objects, not strings, so each row joins back to the designation
master by id. Every id here was verified present in the 19,936-row master:

```json
"Targetroles": [
  { "DesignationId": "DESG-001578", "Name": "Food Analyst",
    "Confidence": 0.7,
    "Rationale": "Directly responsible for analyzing food samples, aligning with the course's focus on grain analysis." },
  { "DesignationId": "DESG-005003", "Name": "Assistant Director (Quality Control)",
    "Confidence": 0.7,
    "Rationale": "Oversees quality control processes, which involves understanding and utilizing AI-based grain analysers." },
  { "DesignationId": "DESG-016550", "Name": "Junior Analyst (Food)",
    "Confidence": 0.7,
    "Rationale": "Performs routine food quality testing and would operate such analytical equipment." },
  { "DesignationId": "DESG-004264", "Name": "Laboratory-In-Charge (Food Processing)",
    "Confidence": 0.65,
    "Rationale": "Manages laboratory equipment and testing protocols for food processing and quality assessment." }
]
```

`PrimaryCompetencyArea`, `DomainCompetencies`, `Language`, `Version`, `Generator`,
`ReferenceResources` — from the same record ("Operating the AI-Based Grain Analyser for Quality
Control"). Note the SGOS triple and `DomainCompetencies` are one decision expressed two ways, and the
KCM branches are empty because the area is Domain:

```json
"PrimaryCompetencyArea": {
  "name": "Domain",
  "confidence": 0.7,
  "reason": "The course focuses on the operation of an AI-based grain analyser for food grain procurement and quality control, which directly aligns with the domain-specific mandate of maintaining buffer stocks and price stabilization under the Food and Public Distribution sector."
},
"Sector":         "Rural and Agriculture",
"SubSector":      "Food and Public Distribution",
"SubSectorTheme": "Buffer Stocks & Price Stabilization",
"DomainCompetencies": [
  { "Theme": "Food and Public Distribution", "SubTheme": "Buffer Stocks & Price Stabilization" }
],
"FunctionalCompetencies":  [],
"BehaviouralCompetencies": [],

"Language":  "English",
"Duration":  "7 Minutes",
"Version":   "v3.5-advanced",
"Generator": "iGOT Metadata Regeneration Engine (Vertex AI / Gemini)",
"ReferenceResources": { "ExtendedLearning": [], "AssignmentsAndPracticeLinks": [] }
```

### Changed: `explain` went from empty to a usable audit trail

**v3.4** — all four sub-objects are `{}`, in **368 of 368 rows**:

```json
"explain": {
  "sgos_reason": "The course's primary focus is on developing the cognitive and strategic skills of game theory …",
  "missing_info": "No transcript or PDF text was provided. All text-dependent fields are empty …",
  "mapping_reasons":              {},
  "designation_inference":        {},
  "competency_mapping_triggers":  {},
  "learning_outcomes_validation": {},
  "version_adherence_check": "1.0",
  "prior_knowledge_confidence": 0.85
}
```

**v3.5** — same fields, now typed and populated:

```json
"explain": {
  "primary_area_reason": "The course focuses on secretarial duties, office management, meeting coordination, and file handling, which directly align with the Functional competency area.",
  "sgos_reason": "Primary area is Functional, hence no SGOS domain mapping is applicable.",
  "mapping_reasons": [
    { "Field": "LearningObjectives",
      "Reason": "Clear, measurable outcomes starting with action verbs are provided in the instructions.",
      "EvidenceQuote": "Identify the primary duties and responsibilities…" },
    { "Field": "ExpectedRoleOutcome",
      "Reason": "Directly trains Personal Assistants and Private Secretaries on their core job responsibilities.",
      "EvidenceQuote": "equip Personal Assistants (PAs) with the necessary skills" },
    { "Field": "ExtentOfLearning",
      "Reason": "Course duration is approximately 1.5 hours with multiple modules.",
      "EvidenceQuote": "5688.0 seconds" }
    // … one entry per rubric parameter
  ],
  "designation_inference": [
    { "DesignationId": "DESG-015195", "Name": "Personal Assistant (Secretarial)",
      "Evidence": "Explicitly mentions 'duties and responsibilities of personal assistant'.",
      "Confidence": 0.95 }
  ],
  "learning_outcomes_validation": {
    "CoversPurpose": true, "CoversProcess": true, "CoversApplication": true,
    "BloomVerbsUsed": ["Identify", "Apply", "Demonstrate", "Utilize", "Develop", "Manage"]
  },
  "evidence_tier_used": "transcript",
  "declared_competency_area_for_audit": "Functional",
  "agrees_with_declared_area": true,
  "validation_issues": [
    "BehaviouralCompetencies: cleared - primary area is Functional, cross-category mapping not allowed"
  ]
}
```

`validation_issues`, `evidence_tier_used`, `declared_competency_area_for_audit` and
`agrees_with_declared_area` are new — they make each record self-describing about how it was produced
and whether the AI agreed with the source's declared area.

### Changed: `TranscriptAnalysis` keeps the evidence it was asked for

**v3.4** — the prompt asked for per-marker verb, phrase, Bloom level and confidence; the schema
declared arrays of strings, so it was all discarded:

```json
"TranscriptAnalysis": {
  "LearningTone":      { "Value": "Not Available", "Confidence": 0.0 },
  "CognitiveMarkers":  { "Values": [], "Confidence": 0.0 },
  "CompetencySignals": { "Values": [], "Confidence": 0.0 },
  "KeywordsExtracted": { "Values": [], "Confidence": 0.0 }
}
```

**v3.5** — typed, and populated because transcripts are now actually read:

```json
"TranscriptAnalysis": {
  "LearningTone": { "Value": "Instructional", "Confidence": 0.9 },
  "CognitiveMarkers": [
    { "Verb": "Identify",    "EstimatedLevel": 1, "Confidence": 0.9,
      "ExamplePhrase": "identify the main duties and responsibilities of a personal assistant" },
    { "Verb": "Apply",       "EstimatedLevel": 3, "Confidence": 0.9,
      "ExamplePhrase": "Apply systematic procedures for organizing and managing official meetings" },
    { "Verb": "Demonstrate", "EstimatedLevel": 3, "Confidence": 0.9,
      "ExamplePhrase": "Demonstrate professional etiquette in handling telephone calls" }
  ],
  "CompetencySignals": [
    { "Category": "Functional", "Theme": "Office Management",
      "SubTheme": "Office Procedures", "Confidence": 0.95,
      "TriggerPhrases": ["duties and responsibilities of personal assistant",
                         "Manual of Office Procedure"] },
    { "Category": "Functional", "Theme": "Office Management",
      "SubTheme": "Noting & Drafting of official Communications", "Confidence": 0.9,
      "TriggerPhrases": ["prepare minutes of the meeting",
                         "draft of a minute should be in a crisp manner"] }
  ]
}
```

### Changed: `Sector` is null instead of `"Not Applicable"`

**v3.4**, in 214 of 368 rows — a sector-typed field carrying a sentinel string, because the field was
required while the prompt forbade populating it for non-Domain courses:

```json
"Sector": "Not Applicable", "SubSector": "Not Applicable", "SubSectorTheme": "Not Applicable"
```

**v3.5**, for the same situation:

```json
"Sector": null, "SubSector": null, "SubSectorTheme": null, "DomainCompetencies": []
```

### Changed: `TargetEmployeeGroups` split into bands + roles

**v3.4** — one object, with the designations as bare strings; the confidence and rationale the prompt
asked for were dropped on the floor:

```json
"TargetEmployeeGroups": {
  "RoleBands": ["Group A", "Group B"],
  "Designations": ["Deputy Secretary", "Under Secretary", "Section Officer"]
}
```

**v3.5** — a flat band array per the spec, with designations moved to master-validated `Targetroles`:

```json
"TargetEmployeeGroups": ["Group B", "Group C"]
```

### Changed: rubric totals are computed, not requested

Real v3.5 record. The seven sub-scores are the model's; `TotalScore` and `Classification` are
recomputed in Python from the configured weights, and `LearningLevel` is forced to match:

```json
"RubricScoring": {
  "LearningObjectives": 60, "PriorKnowledge": 0, "BloomTaxonomy": 60,
  "ComplexityOfContent": 30, "ExpectedRoleOutcome": 50, "ExtentOfLearning": 10,
  "TargetAudienceAlignment": 40,
  "TotalScore": 35, "Classification": "Beginner"
},
"LearningLevel": "Beginner"
```

`60(.10) + 0(.05) + 60(.10) + 30(.15) + 50(.20) + 10(.25) + 40(.15) = 35.0` → `35`, and `35 ≤ 45` →
Beginner. Under the v3.4 weights the same sub-scores total `35.5` → `36` — a similar number, but
reached with `TargetAudienceAlignment` weighted 0, so its score of 40 contributed nothing at all.

Where the model's own arithmetic or a field disagrees, the correction is applied and recorded, e.g.:

```
Duration: model said '9 Minutes 34 Seconds', forced to platform value '9 Minutes'
FunctionalCompetencies: empty although primary area is Functional - nothing in KCM matched
```

---

## Output contract

Top-level keys: `Do_ID`, `CourseName`, `CourseSummary`, `CourseDescription`, `LearningOutcomes`,
`LearningLevel`, `LearningMode`, `Duration`, `Language`, `Sector`, `SubSector`, `SubSectorTheme`,
`DomainCompetencies`, `PrimaryCompetencyArea`, `FunctionalCompetencies`, `BehaviouralCompetencies`,
`SuggestiveCompetencies`, `Tags`, `TargetEmployeeGroups`, `Targetroles`, `PriorKnowledgeDeclared`,
`SuggestivePriorKnowledge`, `RubricScoring`, `TranscriptAnalysis`, `ReferenceResources`,
`EmbeddingVectorID`, `Version`, `GeneratedOn`, `Generator`, `explain`.

Branch rules, enforced:

| Primary area | Sector/SubSector/SubSectorTheme | DomainCompetencies | Functional | Behavioural |
|---|---|---|---|---|
| `Domain` | from SGOS | mirrors the SGOS choice | `[]` | `[]` |
| `Functional` | `null` | `[]` | from KCM | `[]` |
| `Behavioural` | `null` | `[]` | `[]` | from KCM |

`Targetroles` entries are objects (`DesignationId`, `Name`, `Confidence`, `Rationale`), not bare
strings, so they join back to the designation master. v3.4 emitted strings and discarded the
confidence and rationale the prompt asked for.

---

## Operations

```bash
python meta_gen.py --limit 10                       # dry run (default)
python meta_gen.py --limit 10 --execute             # pilot
python meta_gen.py --execute                        # drain everything, then exit
python meta_gen.py --execute --serve                # stay resident, poll for work
python meta_gen.py --course-id do_123 --execute     # one course
python meta_gen.py --tier metadata_only --execute   # only courses with no content
python meta_gen.py --execute --force                # redo completed courses
python meta_gen.py --execute --batch-size 8         # concurrency
```

### Parallelism

Courses run concurrently under a semaphore — `--batch-size` (default `MAX_CONCURRENCY=4`). The worker
claims `WORKER_BATCH_SIZE` rows per round and processes up to `--batch-size` at once. Concurrency
affects throughput only, never which courses are processed.

### Latency, throughput and cost

Every run prints averages and a projection for the outstanding queue. For a standalone estimate from
whatever has been generated so far:

```bash
python tools/estimate.py                  # projects from measured data in the DB
python tools/estimate.py --concurrency 8
```

**Measured on 20 courses** (`gemini-3.1-pro-preview`, `--batch-size 4`):

| Tier | n | mean latency | median | input tokens | output | thinking | $/course |
|---|---|---|---|---|---|---|---|
| `transcript` | 11 | 204 s | 180 s | 74,189 | 3,239 | 4,275 | $0.2385 |
| `metadata_only` | 9 | 154 s | 86 s | 54,605 | 2,340 | 3,914 | $0.1843 |
| blended | 20 | 181 s | — | 65,376 | — | — | **$0.2141** |

Two things are easy to get wrong here:

- **Thinking tokens are billed at the output rate**, and there are *more* of them than of the visible
  answer (≈4,200 vs ≈3,100 per course). Costing only `candidates_token_count` understates the bill by
  well over half of the output charge. Both the run summary and the estimator include them.
- **Per-course latency is not wall-clock ÷ concurrency.** Latency rises as concurrency rises, so the
  two diverge:

| | |
|---|---|
| Mean latency per course, at concurrency 4 | 250 s |
| Effective wall-clock per course | 74 s |
| **Measured throughput** | **~50 courses/hour** |
| Full 3,707 courses at that rate | **~74 hours (3.1 days)** |
| What `latency ÷ 4` would predict | 47 hours |

The measured rate is **1.6× slower** than the arithmetic implies, so `--batch-size` buys much less
than proportionally — check the project's quota for the model before raising it. In the 8-course
timed run, 3 of 8 calls stalled and hit `LLM_TIMEOUT_SECONDS_META` (300 s); the retry then succeeded
in ~120 s. That is the timeout earning its keep, but it inflates p90 latency to ~420 s.

**Full-run cost: ≈$800** at $2.00/$12.00 per 1M tokens (the `gemini-3.1-pro-preview` ≤200k-prompt
tier; prompts here run 55k–140k so the higher $4/$18 tier does not apply). Cost is per-course and
unaffected by concurrency. Prices are configurable — `PRICE_INPUT_PER_M`, `PRICE_OUTPUT_PER_M`,
`PRICE_CACHED_INPUT_PER_M` — and are only as current as whoever last edited `.env`.

**The single biggest saving available is caching.** The static prefix (system prompt + KCM + SGOS) is
byte-identical on every call and is most of the 65k-token input. Served from cache it would save
≈$436 on a full run — **55% of total spend**. `cached_content_token_count` has been 0 in every call so
far, so this is not yet happening.

### Resuming

Safe to re-run. Progress lives in `course_processing_checkpoint`:

- `done` — skipped on re-run unless `--force`.
- `pending` / `failed` — picked up, oldest and fewest-attempts first.
- `failed` past `MAX_ATTEMPTS` becomes `dead` and is excluded; the reason is in
  `course_processing_errors` and `last_error`.
- `in_progress` older than `STALE_CLAIM_MINUTES` (30) is reclaimed at startup — this is what makes a
  killed run recoverable.

A restart therefore processes only what is outstanding. Interrupting with Ctrl-C finishes in-flight
courses and exits cleanly.

### What gets recorded

| Where | Contents |
|---|---|
| `logs/<date>-<version>.log` | per-course line (timing, tier, area, role count, issue count, token counts), every validation issue at INFO, full tracebacks, run summary |
| `logs/meta_gen_<version>_<timestamp>.csv` | one row per course attempted — status, tier, areas, counts, score, tokens, issues, error |
| `course_metadata_regenerated` | the record, `validation_issues`, `evidence_tier`, `declared_competency_area`, `llm_usage_json`, original metadata |
| `course_processing_errors` | one row per failure (message capped at 4,000 chars; full trace in the log) |

Useful queries:

```sql
-- progress
SELECT status, count(*) FROM course_processing_checkpoint
 WHERE content_set='non-scorm' GROUP BY 1;

-- does the AI agree with the declared competency area?
SELECT declared_competency_area,
       regenerated_json->'PrimaryCompetencyArea'->>'name' AS ai_area,
       count(*)
  FROM course_metadata_regenerated
 WHERE llm_prompt_version='v3.5-advanced' GROUP BY 1,2 ORDER BY 3 DESC;

-- most frequent validation issues (systemic prompt problems)
SELECT jsonb_array_elements_text(validation_issues) AS issue, count(*)
  FROM course_metadata_regenerated
 WHERE llm_prompt_version='v3.5-advanced' GROUP BY 1 ORDER BY 2 DESC LIMIT 20;

-- token spend
SELECT sum((llm_usage_json->>'prompt_token_count')::bigint)     AS input,
       sum((llm_usage_json->>'candidates_token_count')::bigint) AS output
  FROM course_metadata_regenerated WHERE llm_prompt_version='v3.5-advanced';
```

### Version coexistence

`course_metadata_regenerated` is keyed on `(course_id, llm_prompt_version)`, so the 368-row v3.4
baseline and the v3.5 output sit side by side and can be diffed per course. The migration that widens
the key is idempotent and runs at startup.

---

## Files

| File | Purpose |
|---|---|
| `meta_gen.py` | worker: discovery, queue, LLM calls, persistence, CLI |
| `prompts_v35.py` | system prompt, response schema, rubric weights, prompt assembly |
| `course_io.py` | reading course folders: nested transcripts, PDFs, placeholder filtering |
| `validation.py` | master-membership enforcement, rubric recomputation, repair |
| `designations.py` | designation embedding index, hybrid retrieval, master validation |
| `tools/inventory.py` | dedup + coverage census, produces `manifest.jsonl` |
| `tools/build_designation_index.py` | one-off designation index builder |
| `.env.example` | every variable, documented |

Reference material: the four TPT framework PDFs, and `legacy_code/` + `legacy_updated_2/` (the
`cbp-ai-service` production service, 4.8.38 and 4.8.39).

---

## Known gaps

- **SCORM content is not wired up yet.** `CONTENT_SET` and the schema support it; only the non-SCORM
  set has been ingested and piloted. SCORM courses genuinely do sometimes carry transcripts and PDFs,
  so `scorm=true` must not skip extraction.
- **41% of courses are `metadata_only`.** For those, transcript analysis is legitimately empty and
  confidence is capped at 0.70. No amount of prompting creates evidence that is not there — the
  honest options are to accept lower-confidence records or to source the missing content.
- **Prompt caching is not yet confirmed.** The prefix is structured for it, but
  `cached_content_token_count` has been 0 in every observed call. At ~55k static tokens per course
  across 3,707 courses this is the dominant cost, so explicit context caching is the obvious next
  optimisation.
- **Role bands cannot be validated.** The designation master has only `id,name` — no group/band
  column — so Group A/B/C/D assignment stays model-inferred and unchecked.
- **312 courses are absent from `course_metadata_v3`**, so they have no platform language or duration
  and fall back to inference. The legacy `scripts/course_metadata_script.py` shows the iGOT content
  API (`/api/content/v1/search`) that could backfill them.
- **`EmbeddingVectorID` is declared but never populated** — no embedding is stored for the course
  record itself yet.
- Only the `--tier`-filtered and small `--limit` paths have been exercised end-to-end; a full
  3,707-course run has not been done.
