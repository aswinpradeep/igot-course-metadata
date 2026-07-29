"""
Prompt + output-schema definitions for the iGOT course metadata regeneration
engine, Framework v3.5 (Advanced).

Separated out of meta_gen.py so the contract can be versioned and reviewed on
its own. v3.4 is superseded; where v3.4-advanced-extended behaved *better* than
the written spec, that behaviour is preserved deliberately and the reason is
noted inline (see NOTES below).

NOTES ON DELIBERATE DIVERGENCE FROM THE WRITTEN v3.5 SPEC
---------------------------------------------------------
1. Learning Outcomes are NOT prefixed with literal "Why:" / "How:" / "What:"
   labels. Spec section 8 asks for the labels; the v3.4 implementation
   overrode this and produced clean learner-facing outcomes instead. Labelled
   strings would be a visible regression on the platform, so outcomes stay
   unlabelled and the why/how/what coverage is asserted in
   `explain.learning_outcomes_validation` instead. Flip LO_STYLE to change.

2. Rubric weights follow the v3.5 spec table (which sums to 100), not the v3.4
   code's weights (which zeroed TargetAudienceAlignment). Weights and bands are
   constants here and TotalScore is recomputed in Python, never trusted from
   the model.

3. TranscriptAnalysis keeps the richer per-item structure the v3.4 *prompt*
   asked for. The v3.4 *schema* flattened these to arrays of strings, so the
   requested evidence was silently discarded on every call.

4. `explain` sub-objects are fully typed. In v3.4 they were declared as bare
   {"type": "object"} with no properties, which made Gemini emit `{}` -- all
   368 rows of the previous run have an empty audit trail as a result.

5. Sector / SubSector / SubSectorTheme are nullable and NOT required. In v3.4
   they were required while the prompt forbade populating them for
   Functional/Behavioural courses, so 58% of rows carry the string
   "Not Applicable" in a field typed as a sector name.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "v3.5-advanced")
SCHEMA_VERSION = PROMPT_VERSION
GENERATOR_NAME = os.environ.get(
    "GENERATOR_NAME", "iGOT Metadata Regeneration Engine (Vertex AI / Gemini)"
)

# "Behavioural" (British) matches the KCM master data's `type` values; the v3.5
# spec document writes "Behavioral". The KCM spelling wins by default so output
# keys and master-data values agree. Set BEHAVIOURAL_SPELLING=Behavioral if a
# downstream consumer needs the spec form.
BEHAVIOURAL_LABEL = os.environ.get("BEHAVIOURAL_SPELLING", "Behavioural")
BEHAVIOURAL_KEY = f"{BEHAVIOURAL_LABEL}Competencies"

# "unlabelled" keeps the v3.4 behaviour (clean learner-facing outcomes);
# "why_how_what" follows the literal v3.5 spec section 8. See NOTES above.
LO_STYLE = os.environ.get("LEARNING_OUTCOME_STYLE", "unlabelled")

# ---------------------------------------------------------------- rubric config
# v3.5 spec section 10. Weights are percentages and sum to 100. Overridable as
# RUBRIC_WEIGHTS="LO:10,PK:5,BT:10,CC:15,ERO:20,EOL:25,TAA:15".
_DEFAULT_WEIGHTS: Dict[str, int] = {
    "LearningObjectives": 10,
    "PriorKnowledge": 5,
    "BloomTaxonomy": 10,
    "ComplexityOfContent": 15,
    "ExpectedRoleOutcome": 20,
    "ExtentOfLearning": 25,
    "TargetAudienceAlignment": 15,
}
_WEIGHT_ALIASES = {
    "LO": "LearningObjectives", "PK": "PriorKnowledge", "BT": "BloomTaxonomy",
    "CC": "ComplexityOfContent", "ERO": "ExpectedRoleOutcome",
    "EOL": "ExtentOfLearning", "TAA": "TargetAudienceAlignment",
}


def _load_weights() -> Dict[str, int]:
    raw = os.environ.get("RUBRIC_WEIGHTS", "").strip()
    if not raw:
        return dict(_DEFAULT_WEIGHTS)
    weights = dict(_DEFAULT_WEIGHTS)
    for part in raw.split(","):
        if ":" not in part:
            continue
        key, _, value = part.partition(":")
        key = _WEIGHT_ALIASES.get(key.strip().upper(), key.strip())
        if key in weights:
            weights[key] = int(value)
    total = sum(weights.values())
    if total != 100:
        raise ValueError(f"RUBRIC_WEIGHTS must sum to 100, got {total}: {weights}")
    return weights


RUBRIC_WEIGHTS: Dict[str, int] = _load_weights()

# Spec lists "<=45 Beginner / 46-75 Intermediate / 75 Advanced"; the 75 overlap
# is resolved in favour of Intermediate.
BEGINNER_MAX = int(os.environ.get("RUBRIC_BEGINNER_MAX", "45"))
INTERMEDIATE_MAX = int(os.environ.get("RUBRIC_INTERMEDIATE_MAX", "75"))

RUBRIC_KEYS: List[str] = list(RUBRIC_WEIGHTS)


def compute_total_score(rubric: Dict[str, Any]) -> int:
    """Weighted total, computed in Python. Models are unreliable at arithmetic."""
    total = 0.0
    for key, weight in RUBRIC_WEIGHTS.items():
        try:
            value = float(rubric.get(key) or 0)
        except (TypeError, ValueError):
            value = 0.0
        total += max(0.0, min(100.0, value)) * (weight / 100.0)
    return int(round(total))


def classify(total_score: int) -> str:
    if total_score <= BEGINNER_MAX:
        return "Beginner"
    if total_score <= INTERMEDIATE_MAX:
        return "Intermediate"
    return "Advanced"


def _rubric_formula() -> str:
    parts = [f"({k}*{w/100:.2f})" for k, w in RUBRIC_WEIGHTS.items()]
    return "TotalScore = " + " + ".join(parts)


def _rubric_table() -> str:
    rows = [f"    | {k:<24} | {w:>6} |" for k, w in RUBRIC_WEIGHTS.items()]
    head = f"    | {'Parameter':<24} | {'Weight':>6} |\n    |{'-'*26}|{'-'*8}|"
    return head + "\n" + "\n".join(rows)


# --------------------------------------------------------------- system prompt
# Tokens (__NAME__) are substituted in build_prompt so the JSON examples below
# can keep their literal braces without f-string escaping.
SYSTEM_PROMPT_TEMPLATE = """
You are an *auditable metadata regeneration engine* for the iGOT Karmayogi Bharat platform.
Your objective is to process the provided inputs and return a single, valid JSON object representing a
complete, standardized metadata record for a learning course.

## Inputs
You must use ONLY these sources. Do not use outside knowledge to add facts about the course.
- course_metadata: (JSON) the raw original metadata record.
- authoritative_facts: (JSON) platform-verified values (language, duration, provider). These OVERRIDE
  anything you might infer. Never contradict them.
- transcript_text: (String, optional) transcript assembled from the course's module subtitle files.
- pdf_texts: (String, optional) extracted text from associated PDF documents.
- kcm_json: (JSON) the master Karmayogi Competency Model - the ONLY valid source of Functional and
  Behavioural competencies.
- sgos_json: (JSON) the master Sector-SubSector-Theme mapping - the ONLY valid source of Domain
  classification.
- designation_candidates: (JSON, optional) a pre-filtered shortlist from the official iGOT designation
  master. You may ONLY return designations present in this shortlist.
- scorm_flag: (Boolean) whether the package is SCORM, which often lacks transcript and PDFs.
- evidence_tier: (String) how much real content you were given. Calibrate confidence to this.

## Global rules
- JSON output only. No prose, no markdown fences, no trailing commas.
- Never invent Themes, SubThemes, Sectors, or Designations. Every such value must appear verbatim in the
  corresponding master input. If nothing qualifies, return an empty array - never a placeholder string.
- Language: write ALL generated prose (title, summary, description, outcomes, tags, designations) in the
  language given by authoritative_facts.language. Note the transcript is often an English translation of
  non-English audio, so the transcript's language does NOT determine the output language. If
  authoritative_facts.language is absent, follow the language of the course metadata.
- Never emit the strings "Not Applicable", "N/A", "None", or "Unknown" as a value for a sector,
  competency, designation, or role field. Use null for absent single values and [] for absent lists.
- Strip the provider/organisation name given in authoritative_facts.provider from every generated field,
  including Tags, CourseName, CourseSummary and CourseDescription.
- Auditability: populate `explain` fully. Every mapping you make must be traceable to a phrase in the
  input. An empty `explain` object is a failed response.
- Calibrate all confidence values honestly to evidence_tier. If evidence_tier is "metadata_only" you have
  no transcript and no PDFs: confidence for any content-derived field must not exceed 0.70, and
  TranscriptAnalysis must be returned with empty values and 0.0 confidences.

## Field generation rules

### 1. Core text fields

    **CourseName:**
    - A clear, professional title of 6-12 words.
    - Preserve original meaning; improve clarity, capitalization, precision.
    - No provider names, organisation names, or promotional terms.

    **CourseSummary:**
    - 3-4 sentences, 80-100 words, single plain-text paragraph.
    - Capture the main topic, its relevance, and the key benefit to the learner.
    - Formal, professional tone. No provider or organisation names.

    **CourseDescription:**
    - 150-250 words, plain text, no bullets or markdown.
    - Cover: why the topic matters for the target audience; what is covered; how learners apply it.
    - Optionally close with one outcome-focused line.
    - Informative and neutral - suitable for official government training material. No marketing tone.

    **Tags:**
    - 10-15 descriptive keyword strings derived from the course content.
    - No organisation names, provider names, or course codes.
    - If a transcript is present, draw from its most significant terms.
    - Add relevant cross-domain conceptual tags where genuinely applicable.

### 2. Primary competency area (decide this FIRST)
    - Determine the single primary competency area from CONTENT ANALYSIS ONLY: one of
      "Domain", "Functional", "__BEHAVIOURAL_LABEL__".
    - Ignore any competency area declared in the incoming metadata. It is frequently wrong and is
      withheld from you deliberately.
    - Exactly ONE primary area per course. A course must never be assigned multiple primary areas.
    - Record the deciding evidence in `PrimaryCompetencyArea.reason` and a 0-1
      `PrimaryCompetencyArea.confidence`.

### 3. Functional and Behavioural competencies
    - Populate ONLY if the primary area is "Functional" or "__BEHAVIOURAL_LABEL__".
    - Read each theme_description and sub_theme_description; match on meaning, not keyword overlap.
    - Map every pair that is genuinely relevant - typically 2 to 6. Do not pad to reach a count, and do
      not stop early if more genuinely apply.

    KCM-ONLY, and three specific ways this goes wrong. Each is a hard constraint:
    - ABSOLUTE RULE: every Theme/SubTheme you output MUST exist verbatim in `kcm_json`. Before writing a
      competency, locate its exact `type`, `theme` and `sub_theme` in `kcm_json`. If you cannot find it
      there, do not include it. Never invent, paraphrase, rename or approximate. Do not use general
      knowledge to name a competency - `kcm_json` is the only permitted source. Copy
      character-for-character.
    - NO CROSS-MIXING BETWEEN ENTRIES: the `theme` and `sub_theme` of one output competency must come
      from the SAME object in `kcm_json`. Never pair a theme from one entry with a sub_theme from another,
      even when each value exists somewhere in the dataset. Before finalising each competency, verify that
      the exact theme + sub_theme combination appears together in one single entry.
    - NO FIELD-SWAPPING: even within one entry, put each value in its matching field. `theme` -> Theme and
      `sub_theme` -> SubTheme, never interchanged.
    - NO CROSS-CATEGORY MAPPING: a Behavioural entry may never appear under FunctionalCompetencies or vice
      versa. The `type` field in `kcm_json` is authoritative.
    - Any competency violating the above is discarded downstream, so a sloppy pair is not a partial
      credit - it is a lost competency.

    - High-confidence mappings (>= 0.85) go in the main arrays. Anything from 0.70 to 0.85 goes to
      `SuggestiveCompetencies` with Confidence and Rationale. Below 0.70, omit entirely.
    - If the primary area is "Domain", both arrays must be [].

### 4. Domain competencies, Sector, SubSector, SubSectorTheme
    - In this framework the Domain competency IS the SGOS classification. `kcm_json` contains no Domain
      entries; `sgos_json` is the only Domain source.
    - Populate ONLY if the primary area is "Domain". Otherwise: Sector, SubSector and SubSectorTheme must
      be null and DomainCompetencies must be [].
    - When the primary area IS "Domain":
        - Choose Sector, SubSector and SubSectorTheme as an existing path in `sgos_json`. The theme must
          actually belong to the chosen subsector, which must belong to the chosen sector.
        - Mirror the same decision into DomainCompetencies as
          [{"Theme": "<SubSector>", "SubTheme": "<SubSectorTheme>"}].
        - Require BOTH: (a) semantic alignment of the course's key concepts with the SGOS theme, and
          (b) contextual alignment - the course's actual purpose falls within that sector's real-world
          mandate (role relevance, ministry function, operational outcome).
        - If only one of the two holds, do not force a mapping: leave the fields null and explain why in
          `explain.sgos_reason`.
        - Record evidence phrases and confidence in `explain.sgos_reason`.

### 5. Prior knowledge declared by author/speaker
    - Covers prerequisites *explicitly stated* in metadata, the `instructions` field, the course
      introduction, speaker narration, presentation text, or documents.
    - If found, set InstructorDeclared true and fill DeclaredLevel (None/Basic/Moderate/Advanced),
      DeclaredTopics, DeclaredNotes (1-2 sentences), Required (Yes/No), Confidence.
    - If not found, set InstructorDeclared false, leave the other fields empty, and populate
      SuggestivePriorKnowledge instead.

### 6. Suggestive prior knowledge (AI-inferred)
    - Populate ONLY when PriorKnowledgeDeclared.InstructorDeclared is false.
    - Emit only if your confidence is >= 0.75; otherwise set AIRecommended false and leave empty.
    - Fields: AIRecommended, SuggestedLevel (Basic/Moderate/Advanced), SuggestedTopics (3-6),
      Required (Yes/No), SuggestedLearningPath (ordered prerequisite course titles that logically precede
      this one, drawn only from titles present in the inputs), Confidence, Rationale.
    - Never mix declared and suggested topics. If both objects apply, Declared takes priority.

### 7. Learning outcomes
    __LO_RULES__

### 8. Transcript analysis
    - If no transcript was provided, return empty values with 0.0 confidences. Do not fabricate.
    - KeywordsExtracted: 10-15 ranked keywords/phrases (1-4 words), with `Method`
      ("term-frequency" | "semantic-importance") and Confidence.
    - LearningTone: one of Instructional / Awareness / Persuasive / Narrative / Technical, plus Confidence.
    - CognitiveMarkers: Bloom markers actually observed. For each: Verb, ExamplePhrase (quoted from the
      transcript), EstimatedLevel 1-6, Confidence. Bloom mapping: Remember/Understand 1-2, Apply 3,
      Analyze 4, Evaluate 5, Create/Design 6.
    - CompetencySignals: probable KCM matches. For each: Category, Theme, SubTheme, Confidence and
      TriggerPhrases (2-6 phrases from the transcript that caused the match). Include only >= 0.70.

### 9. Target employee groups and target roles
    - TargetEmployeeGroups: 1-3 cadre bands from exactly {"Group A", "Group B", "Group C", "Group D"}.
      Assign at least one. Avoid listing all four unless the course is genuinely broad and foundational
      (e.g. public service ethics).
    - Classify by the seniority the content actually addresses, using Indian government service
      classification rules:
        - Group A / Group B - gazetted and senior officers: policymakers, managers, specialists.
          Indicative: IAS, IPS, Secretary, Joint Secretary, Director, Deputy Secretary, Under Secretary,
          Section Officer, Engineers, Doctors, Scientists.
        - Group C / Group D - supporting, clerical and operational staff. Indicative: Clerk, Assistant,
          Stenographer, Personal Assistant, Data Entry Operator, Technician, Constable, Driver, MTS,
          Helper.
      Decide from the designation seniority the course targets and the cognitive depth of the content,
      not from the subject area alone. Indicative sector leaning:
        Governance/Policy -> A,B | Technology/IT -> A,B,C | Finance -> A,B | Education -> A,B,C
        Infrastructure -> A,B | Health -> A,B | Social/Welfare -> B,C,D | Administration -> all
    - Targetroles: 3-6 specific designations, chosen ONLY from `designation_candidates`. Copy
      DesignationId and Name exactly as given. Anything not in the shortlist will be rejected.
      Each entry needs Confidence (0-1) and Rationale citing the phrases/competencies that justify it.
    - If `designation_candidates` is empty or nothing in it genuinely fits, return Targetroles: [].
      Do not invent designations to fill the quota.
    - Output role and designation prose in __COURSE_LANGUAGE__.

### 10. Rubric scoring
    - Score each of the seven parameters 0-100 under a strict ZERO-BASE policy: every parameter starts at
      0 and only rises on real, explicit, or confidently inferred evidence from the supplied inputs.
      No default or buffer minimums.
    - Award points only where evidence is clear or your confidence is >= 0.75. Otherwise leave 0.
    - Every score must be justifiable by a phrase, transcript segment, or metadata element, recorded in
      `explain.mapping_reasons`.

    1. LearningObjectives (LO)
       +10 per measurable, clearly written outcome starting with a Bloom verb (max 100).
       Vague phrasing ("learn about", "understand") = +2 only. Missing = 0.
    2. PriorKnowledge (PK)
       0-100 according to how explicitly prerequisites are stated and how well they match the content.
       No evidence = 0.
    3. BloomTaxonomy (BT)
       Per outcome verb: Remember/Identify 20, Understand 40, Apply 60, Analyze 80, Evaluate/Create 100.
       Average across all outcome verbs. No outcomes = 0.
    4. ComplexityOfContent (CC)
       +15 moderate domain-term density (5-10 per 1000 words); +30 high (10-20 per 1000);
       +50 advanced analytical content, formulas or policy instruments;
       +70-100 very complex, technical or data-heavy. Purely awareness-level or very short = 0.
    5. ExpectedRoleOutcome (ERO)
       +20 clear job-related tasks; +30-50 maps to 1-3 designations;
       +60-80 maps to 6+ designations with role-level use cases; +100 highly specialised with measurable
       job outcomes. No job/role connection = 0.
    6. ExtentOfLearning (EOL)
       By duration: <30 min +10; 30-90 min +20-40; >90 min +50-70; multi-module >3 hrs +80-100.
       Practical components (exercises, labs, activities) add +10-20. Single short video = 0-10.
       Use authoritative_facts.duration_seconds, not your own guess.
    7. TargetAudienceAlignment (TAA)
       +10 general public-servant audience is clear; +20-40 matched to a specific cadre or designation at
       confidence >= 0.8; +50-70 content difficulty aligns with designation level;
       +100 sector + role + competency all align. Unspecified or mismatched = 0.

    - Report the seven scores. Also report TotalScore and Classification, computed as:
__RUBRIC_TABLE__
      __RUBRIC_FORMULA__
      Classification: TotalScore <= __BEGINNER_MAX__ -> Beginner; <= __INTERMEDIATE_MAX__ -> Intermediate;
      otherwise Advanced.
    - LearningLevel MUST equal RubricScoring.Classification.

### 11. Learning mode and duration
    - LearningMode: one of Self-paced / Instructor-led / Blended, inferred from structure and delivery.
    - Duration: render authoritative_facts.duration_seconds as a human-readable string (e.g. "2 Hours
      15 Minutes"). If duration_seconds is null, infer from module count and transcript length and lower
      your ExtentOfLearning confidence accordingly.

### 12. Reference resources
    - ReferenceResources.AssignmentsAndPracticeLinks and .ExtendedLearning: only URLs actually present in
      the supplied inputs. Never fabricate a URL. Empty arrays if none.

### 13. SCORM and missing content
    - If scorm_flag is true, or transcript and PDFs are absent, still generate all required fields by
      inference from the metadata and `instructions`.
    - State the limitation explicitly in `explain.missing_info` and cap affected confidences at 0.70.

### 14. Explain object (mandatory - this is the audit trail)
    - primary_area_reason: why this competency area, citing evidence.
    - sgos_reason: the SGOS decision, evidence phrases, and confidence - or why no mapping was made.
    - competency_mapping_triggers: one entry per mapped competency with the phrases that triggered it.
    - mapping_reasons: one entry per rubric parameter and per generated field group, with the evidence.
    - designation_inference: one entry per returned designation with its evidence.
    - learning_outcomes_validation: whether purpose, process and application are each covered, and the
      Bloom verbs used.
    - prior_knowledge_confidence, missing_or_low_confidence_fields, missing_info, evidence_tier_used.

### 15. Output
    - A single valid JSON object, UTF-8, ISO8601 timestamps, no markdown.
"""

_LO_RULES_UNLABELLED = """    - Generate 5-6 clear, measurable outcomes that implicitly cover purpose (why), process (how) and
      application (what).
    - Do NOT prefix outcomes with "Why:", "How:" or "What:" - the labels are recorded separately in
      `explain.learning_outcomes_validation`, not shown to learners.
    - Begin each with a strong action verb (Identify, Apply, Analyze, Evaluate, Develop, Implement,
      Interpret). Avoid "Learn about" / "Understand about".
    - Use professional instructional-design language suitable for government and public-sector learning.
    - Cover a cognitive progression from foundational understanding to applied or strategic capability.
    - Ground every outcome in the supplied content. If the `instructions` field of the metadata already
      states objectives, use it as the primary source and refine it - do not discard it."""

_LO_RULES_WHW = """    - Generate 3-5 outcomes. Each must be prefixed with exactly one of "Why: ", "How: " or "What: ".
    - The set must contain at least one "Why:", one "How:" and one "What:" outcome.
    - Ground every outcome in the supplied content, using the `instructions` field where available."""


# ---------------------------------------------------------------- output schema
# Gemini/Vertex structured output accepts an OpenAPI-3.0 subset: type, format,
# description, nullable, enum, items, properties, required, min/maxItems,
# propertyOrdering, anyOf. Union types (["string","null"]) are NOT accepted --
# use "nullable": True instead.

_COMPETENCY_PAIR = {
    "type": "object",
    "properties": {
        "Theme": {"type": "string"},
        "SubTheme": {"type": "string"},
    },
    "required": ["Theme", "SubTheme"],
}

_SUGGESTIVE_PAIR = {
    "type": "object",
    "properties": {
        "Theme": {"type": "string"},
        "SubTheme": {"type": "string"},
        "Confidence": {"type": "number"},
        "Rationale": {"type": "string"},
    },
    "required": ["Theme", "SubTheme", "Confidence", "Rationale"],
}

METADATA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "Do_ID": {"type": "string", "description": "Unique course identifier."},
        "CourseName": {"type": "string"},
        "CourseSummary": {"type": "string"},
        "CourseDescription": {"type": "string"},
        "LearningOutcomes": {"type": "array", "items": {"type": "string"}},
        "LearningLevel": {
            "type": "string",
            "enum": ["Beginner", "Intermediate", "Advanced"],
        },
        "LearningMode": {
            "type": "string",
            "enum": ["Self-paced", "Instructor-led", "Blended"],
        },
        "Duration": {"type": "string"},
        "Language": {"type": "string"},
        # --- Domain classification (SGOS). Null unless primary area is Domain.
        "Sector": {"type": "string", "nullable": True},
        "SubSector": {"type": "string", "nullable": True},
        "SubSectorTheme": {"type": "string", "nullable": True},
        "DomainCompetencies": {"type": "array", "items": _COMPETENCY_PAIR},
        # --- KCM classification
        "PrimaryCompetencyArea": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": ["Domain", "Functional", BEHAVIOURAL_LABEL],
                },
                "reason": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["name", "reason", "confidence"],
        },
        "FunctionalCompetencies": {"type": "array", "items": _COMPETENCY_PAIR},
        BEHAVIOURAL_KEY: {"type": "array", "items": _COMPETENCY_PAIR},
        "SuggestiveCompetencies": {
            "type": "object",
            "properties": {
                "Functional": {"type": "array", "items": _SUGGESTIVE_PAIR},
                BEHAVIOURAL_LABEL: {"type": "array", "items": _SUGGESTIVE_PAIR},
                "Domain": {"type": "array", "items": _SUGGESTIVE_PAIR},
            },
        },
        "Tags": {"type": "array", "items": {"type": "string"}},
        # --- Audience
        "TargetEmployeeGroups": {"type": "array", "items": {"type": "string"}},
        "Targetroles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "DesignationId": {"type": "string"},
                    "Name": {"type": "string"},
                    "Confidence": {"type": "number"},
                    "Rationale": {"type": "string"},
                },
                "required": ["DesignationId", "Name", "Confidence", "Rationale"],
            },
        },
        # --- Prior knowledge
        "PriorKnowledgeDeclared": {
            "type": "object",
            "properties": {
                "InstructorDeclared": {"type": "boolean"},
                "DeclaredLevel": {
                    "type": "string",
                    "enum": ["None", "Basic", "Moderate", "Advanced"],
                },
                "DeclaredTopics": {"type": "array", "items": {"type": "string"}},
                "DeclaredNotes": {"type": "string"},
                "Required": {"type": "string", "enum": ["Yes", "No"]},
                "Confidence": {"type": "number"},
            },
            "required": ["InstructorDeclared"],
        },
        "SuggestivePriorKnowledge": {
            "type": "object",
            "properties": {
                "AIRecommended": {"type": "boolean"},
                "SuggestedLevel": {
                    "type": "string",
                    "enum": ["Basic", "Moderate", "Advanced"],
                },
                "SuggestedTopics": {"type": "array", "items": {"type": "string"}},
                "Required": {"type": "string", "enum": ["Yes", "No"]},
                "SuggestedLearningPath": {"type": "array", "items": {"type": "string"}},
                "Confidence": {"type": "number"},
                "Rationale": {"type": "string"},
            },
            "required": ["AIRecommended"],
        },
        # --- Scoring
        "RubricScoring": {
            "type": "object",
            "properties": {
                # minimum/maximum are enforced by the response schema, so a model
                # that drifts onto a 0-10 scale is rejected rather than silently
                # producing scores an order of magnitude low.
                **{
                    k: {"type": "integer", "minimum": 0, "maximum": 100}
                    for k in RUBRIC_KEYS
                },
                "TotalScore": {"type": "integer", "minimum": 0, "maximum": 100},
                "Classification": {
                    "type": "string",
                    "enum": ["Beginner", "Intermediate", "Advanced"],
                },
            },
            "required": RUBRIC_KEYS + ["TotalScore", "Classification"],
        },
        # --- Transcript analysis (typed, so the evidence survives)
        "TranscriptAnalysis": {
            "type": "object",
            "properties": {
                "KeywordsExtracted": {
                    "type": "object",
                    "properties": {
                        "Values": {"type": "array", "items": {"type": "string"}},
                        "Method": {"type": "string"},
                        "Confidence": {"type": "number"},
                    },
                },
                "LearningTone": {
                    "type": "object",
                    "properties": {
                        "Value": {
                            "type": "string",
                            "enum": [
                                "Instructional",
                                "Awareness",
                                "Persuasive",
                                "Narrative",
                                "Technical",
                            ],
                            "nullable": True,
                        },
                        "Confidence": {"type": "number"},
                    },
                },
                "CognitiveMarkers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Verb": {"type": "string"},
                            "ExamplePhrase": {"type": "string"},
                            "EstimatedLevel": {"type": "integer"},
                            "Confidence": {"type": "number"},
                        },
                        "required": ["Verb", "EstimatedLevel", "Confidence"],
                    },
                },
                "CompetencySignals": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Category": {
                                "type": "string",
                                "enum": ["Functional", BEHAVIOURAL_LABEL, "Domain"],
                            },
                            "Theme": {"type": "string"},
                            "SubTheme": {"type": "string"},
                            "Confidence": {"type": "number"},
                            "TriggerPhrases": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["Category", "Theme", "Confidence"],
                    },
                },
            },
        },
        "ReferenceResources": {
            "type": "object",
            "properties": {
                "AssignmentsAndPracticeLinks": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "ExtendedLearning": {"type": "array", "items": {"type": "string"}},
            },
        },
        # --- Audit trail. Typed on purpose: bare {"type":"object"} yields {}.
        "explain": {
            "type": "object",
            "properties": {
                "primary_area_reason": {"type": "string"},
                "sgos_reason": {"type": "string"},
                "competency_mapping_triggers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Category": {
                                "type": "string",
                                "enum": ["Functional", BEHAVIOURAL_LABEL, "Domain"],
                            },
                            "Theme": {"type": "string"},
                            "SubTheme": {"type": "string"},
                            "TriggerPhrases": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "Confidence": {"type": "number"},
                        },
                        "required": ["Category", "Theme", "Confidence"],
                    },
                },
                "mapping_reasons": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Field": {"type": "string"},
                            "Reason": {"type": "string"},
                            "EvidenceQuote": {"type": "string"},
                        },
                        "required": ["Field", "Reason"],
                    },
                },
                "designation_inference": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "DesignationId": {"type": "string"},
                            "Name": {"type": "string"},
                            "Evidence": {"type": "string"},
                            "Confidence": {"type": "number"},
                        },
                        "required": ["Name", "Evidence", "Confidence"],
                    },
                },
                "learning_outcomes_validation": {
                    "type": "object",
                    "properties": {
                        "CoversPurpose": {"type": "boolean"},
                        "CoversProcess": {"type": "boolean"},
                        "CoversApplication": {"type": "boolean"},
                        "BloomVerbsUsed": {"type": "array", "items": {"type": "string"}},
                        "Notes": {"type": "string"},
                    },
                },
                "prior_knowledge_confidence": {"type": "number"},
                "missing_or_low_confidence_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "missing_info": {"type": "string"},
                "evidence_tier_used": {"type": "string"},
            },
            "required": ["primary_area_reason", "missing_info"],
        },
        # --- Provenance. Overwritten in Python; declared so the model can fill.
        "EmbeddingVectorID": {"type": "string", "nullable": True},
        "Version": {"type": "string"},
        "GeneratedOn": {"type": "string"},
        "Generator": {"type": "string"},
    },
    "required": [
        "Do_ID",
        "CourseName",
        "CourseSummary",
        "CourseDescription",
        "LearningOutcomes",
        "LearningLevel",
        "LearningMode",
        "Duration",
        "PrimaryCompetencyArea",
        "FunctionalCompetencies",
        BEHAVIOURAL_KEY,
        "DomainCompetencies",
        "Tags",
        "TargetEmployeeGroups",
        "Targetroles",
        "PriorKnowledgeDeclared",
        "SuggestivePriorKnowledge",
        "RubricScoring",
        "TranscriptAnalysis",
        "explain",
    ],
}


def render_system_prompt() -> str:
    """
    Render the system prompt. Deliberately takes no per-course arguments: it must
    be byte-identical for every course so it, plus the masters, form a stable
    cacheable prefix. Per-course values arrive in the input section instead.
    """
    lo_rules = _LO_RULES_WHW if LO_STYLE == "why_how_what" else _LO_RULES_UNLABELLED
    return (
        SYSTEM_PROMPT_TEMPLATE
        .replace("__LO_RULES__", lo_rules)
        .replace("__RUBRIC_TABLE__", _rubric_table())
        .replace("__RUBRIC_FORMULA__", _rubric_formula())
        .replace("__BEGINNER_MAX__", str(BEGINNER_MAX))
        .replace("__INTERMEDIATE_MAX__", str(INTERMEDIATE_MAX))
        .replace("__BEHAVIOURAL_LABEL__", BEHAVIOURAL_LABEL)
    )


def build_static_prefix(kcm_json: Any, sgos_json: Any) -> str:
    """
    The invariant head of every prompt: system rules + both masters.

    Kept first and identical across courses so the ~55k tokens of KCM + SGOS can
    be served from cache rather than re-billed per course. Reordering this so
    per-course text comes first would silently disable that.
    """
    return f"""{render_system_prompt()}

=================
REFERENCE MASTERS (identical for every course)
=================

[kcm_json] (Karmayogi Competency Model - only valid source of Functional/{BEHAVIOURAL_LABEL})
{json.dumps(kcm_json, indent=2, ensure_ascii=False, sort_keys=True)}

[sgos_json] (Sector-SubSector-Theme mapping - only valid source of Domain)
{json.dumps(sgos_json, indent=2, ensure_ascii=False, sort_keys=True)}
"""


def _truncate(text: str, limit: int, label: str) -> str:
    if not text:
        return "None provided."
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[...{label} truncated at {limit:,} characters...]"


def build_course_section(
    course_id: str,
    current_metadata: Optional[Dict[str, Any]],
    authoritative_facts: Dict[str, Any],
    transcript: str,
    pdf_snippets: str,
    designation_candidates: List[Dict[str, str]],
    scorm_flag: bool,
    evidence_tier: str,
    transcript_char_limit: int = 200_000,
    pdf_char_limit: int = 120_000,
) -> str:
    """The per-course tail of the prompt. Must follow build_static_prefix()."""
    return f"""
=================
THIS COURSE
=================

[course_id]
{course_id}

[authoritative_facts] (platform-verified; these override anything you infer)
{json.dumps(authoritative_facts, indent=2, ensure_ascii=False)}

[evidence_tier]
{evidence_tier}

[scorm_flag]
{'true' if scorm_flag else 'false'}

[course_metadata]
{json.dumps(current_metadata, indent=2, ensure_ascii=False)}

[designation_candidates] (shortlist from the official iGOT master - Targetroles MUST come from here,
copying DesignationId and Name verbatim; anything else is rejected downstream)
{json.dumps(designation_candidates, indent=2, ensure_ascii=False)}

[transcript_text]
<BeginTranscript>
{_truncate(transcript, transcript_char_limit, "transcript")}
<EndTranscript>

[pdf_texts]
<BeginSnippets>
{_truncate(pdf_snippets, pdf_char_limit, "PDF text")}
<EndSnippets>

──────────────────────────────────────────────
TASK
──────────────────────────────────────────────
Using only the above inputs, produce a single JSON object conforming to schema {SCHEMA_VERSION}.
Decide the primary competency area first, then populate only the branches that area permits.
Rubric sub-scores are on a 0-100 scale, not 0-10.
Populate `explain` fully - it is the audit record and must not be empty.
"""


def build_prompt(
    course_id: str,
    current_metadata: Optional[Dict[str, Any]],
    authoritative_facts: Dict[str, Any],
    transcript: str,
    pdf_snippets: str,
    kcm_json: Any,
    sgos_json: Any,
    designation_candidates: List[Dict[str, str]],
    scorm_flag: bool,
    evidence_tier: str,
    transcript_char_limit: int = 200_000,
    pdf_char_limit: int = 120_000,
) -> str:
    """Full single-string prompt (static prefix + course section)."""
    return build_static_prefix(kcm_json, sgos_json) + build_course_section(
        course_id=course_id,
        current_metadata=current_metadata,
        authoritative_facts=authoritative_facts,
        transcript=transcript,
        pdf_snippets=pdf_snippets,
        designation_candidates=designation_candidates,
        scorm_flag=scorm_flag,
        evidence_tier=evidence_tier,
        transcript_char_limit=transcript_char_limit,
        pdf_char_limit=pdf_char_limit,
    )
