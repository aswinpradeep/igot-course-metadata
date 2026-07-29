"""
Post-generation validation and repair for Framework v3.5 output.

The response schema passed to Vertex constrains *shape* only. It cannot check
that a Theme actually exists in the KCM master, that an SGOS sector/subsector/
theme triple is a real path, that the weighted TotalScore is arithmetically
right, or that the competency-area branch rules were respected. Those are
enforced here.

Everything is repair-then-report: the record is corrected where a correction is
unambiguous, and every change is recorded as an issue so the run can be audited
and systemic prompt problems spotted.
"""

from __future__ import annotations

import copy
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import prompts_v35 as P

logger = logging.getLogger("course-regenerator.validation")

# Values models reach for instead of leaving a field empty. In the v3.4 run 58%
# of rows carried "Not Applicable" in Sector.
PLACEHOLDERS = {
    "not applicable", "n/a", "na", "none", "null", "unknown", "not specified",
    "not available", "-", "--", "tbd", "nil",
}

VALID_AREAS = {"Domain", "Functional", P.BEHAVIOURAL_LABEL}


def _is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in PLACEHOLDERS


class MasterData:
    """Membership lookups for the KCM and SGOS masters."""

    def __init__(self, kcm_json: Any, sgos_json: Any) -> None:
        # (category, theme_lower, subtheme_lower) -> (Theme, SubTheme) as written
        self.kcm_pairs: Dict[Tuple[str, str, str], Tuple[str, str]] = {}
        self.kcm_themes: Dict[Tuple[str, str], str] = {}
        for row in kcm_json or []:
            if not isinstance(row, dict):
                continue
            ctype = str(row.get("type") or "").strip()
            theme = str(row.get("theme") or "").strip()
            sub = str(row.get("sub_theme") or "").strip()
            if not ctype or not theme:
                continue
            self.kcm_themes[(ctype, theme.lower())] = theme
            if sub:
                self.kcm_pairs[(ctype, theme.lower(), sub.lower())] = (theme, sub)

        # sector -> subsector -> {theme}
        self.sgos: Dict[str, Dict[str, Set[str]]] = {}
        self._sector_ci: Dict[str, str] = {}
        for sector, subs in (sgos_json or {}).items():
            sector_map: Dict[str, Set[str]] = {}
            for subsector, themes in (subs or {}).items():
                names: Set[str] = set()
                for t in themes or []:
                    if isinstance(t, dict) and t.get("theme"):
                        names.add(str(t["theme"]).strip())
                    elif isinstance(t, str):
                        names.add(t.strip())
                sector_map[str(subsector).strip()] = names
            self.sgos[str(sector).strip()] = sector_map
            self._sector_ci[str(sector).strip().lower()] = str(sector).strip()

    def resolve_kcm(self, category: str, theme: str, subtheme: str) -> Optional[Tuple[str, str]]:
        return self.kcm_pairs.get(
            (category, (theme or "").strip().lower(), (subtheme or "").strip().lower())
        )

    def sgos_path_valid(self, sector: str, subsector: str, theme: str) -> bool:
        canon = self._sector_ci.get((sector or "").strip().lower())
        if not canon:
            return False
        subs = self.sgos.get(canon, {})
        for sub_name, themes in subs.items():
            if sub_name.lower() == (subsector or "").strip().lower():
                return any(t.lower() == (theme or "").strip().lower() for t in themes)
        return False

    def canonical_sgos(
        self, sector: str, subsector: str, theme: str
    ) -> Optional[Tuple[str, str, str]]:
        canon = self._sector_ci.get((sector or "").strip().lower())
        if not canon:
            return None
        for sub_name, themes in self.sgos.get(canon, {}).items():
            if sub_name.lower() == (subsector or "").strip().lower():
                for t in themes:
                    if t.lower() == (theme or "").strip().lower():
                        return (canon, sub_name, t)
        return None


def _clean_pairs(
    raw: Any,
    category: str,
    masters: MasterData,
    issues: List[str],
    field: str,
) -> List[Dict[str, str]]:
    """Keep only Theme/SubTheme pairs that exist in the KCM master."""
    out: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        theme = str(item.get("Theme") or "").strip()
        sub = str(item.get("SubTheme") or "").strip()
        if _is_placeholder(theme) or _is_placeholder(sub) or not theme:
            issues.append(f"{field}: dropped placeholder entry {theme!r}/{sub!r}")
            continue
        resolved = masters.resolve_kcm(category, theme, sub)
        if not resolved:
            issues.append(
                f"{field}: dropped {theme!r}/{sub!r} - not a {category} pair in KCM master"
            )
            continue
        key = (resolved[0].lower(), resolved[1].lower())
        if key in seen:
            continue
        seen.add(key)
        entry = {"Theme": resolved[0], "SubTheme": resolved[1]}
        for extra in ("Confidence", "Rationale"):
            if extra in item:
                entry[extra] = item[extra]
        out.append(entry)
    return out


def _seconds_to_human(seconds: Optional[float]) -> Optional[str]:
    if not seconds or seconds <= 0:
        return None
    total = int(round(float(seconds)))
    hours, rem = divmod(total, 3600)
    minutes = rem // 60
    if hours and minutes:
        return f"{hours} Hour{'s' if hours != 1 else ''} {minutes} Minute{'s' if minutes != 1 else ''}"
    if hours:
        return f"{hours} Hour{'s' if hours != 1 else ''}"
    if minutes:
        return f"{minutes} Minute{'s' if minutes != 1 else ''}"
    return f"{total} Seconds"


def validate_and_repair(
    record: Dict[str, Any],
    *,
    course_id: str,
    masters: MasterData,
    designation_index: Any,
    authoritative_facts: Dict[str, Any],
    evidence_tier: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Returns (repaired_record, issues). Never raises on content problems -- a
    course with issues is still persisted, with the issues recorded alongside.
    """
    rec = copy.deepcopy(record)
    issues: List[str] = []

    # ---- identity and provenance (authoritative, never model-supplied) -------
    rec["Do_ID"] = str(course_id)
    rec["Version"] = P.SCHEMA_VERSION
    rec["Generator"] = P.GENERATOR_NAME
    rec["GeneratedOn"] = datetime.now(timezone.utc).isoformat()

    lang = authoritative_facts.get("language")
    if lang:
        if rec.get("Language") and str(rec["Language"]).strip().lower() != str(lang).strip().lower():
            issues.append(
                f"Language: model said {rec['Language']!r}, forced to platform value {lang!r}"
            )
        rec["Language"] = lang

    secs = authoritative_facts.get("duration_seconds")
    human = _seconds_to_human(secs)
    if human:
        if rec.get("Duration") and rec["Duration"] != human:
            issues.append(
                f"Duration: model said {rec['Duration']!r}, forced to platform value {human!r}"
            )
        rec["Duration"] = human
    elif not rec.get("Duration") or _is_placeholder(rec.get("Duration")):
        rec["Duration"] = None
        issues.append("Duration: no platform value and none inferred")

    # ---- primary competency area --------------------------------------------
    area_obj = rec.get("PrimaryCompetencyArea") or {}
    area = str(area_obj.get("name") or "").strip()
    if area == "Behavioral":  # spec spelling -> master-data spelling
        area = P.BEHAVIOURAL_LABEL
        area_obj["name"] = area
    if area not in VALID_AREAS:
        issues.append(f"PrimaryCompetencyArea: invalid {area!r}; defaulted to Domain")
        area = "Domain"
        area_obj["name"] = area
    rec["PrimaryCompetencyArea"] = area_obj

    # ---- competency membership ----------------------------------------------
    rec["FunctionalCompetencies"] = _clean_pairs(
        rec.get("FunctionalCompetencies"), "Functional", masters, issues,
        "FunctionalCompetencies",
    )
    rec[P.BEHAVIOURAL_KEY] = _clean_pairs(
        rec.get(P.BEHAVIOURAL_KEY), P.BEHAVIOURAL_LABEL, masters, issues,
        P.BEHAVIOURAL_KEY,
    )

    sugg = rec.get("SuggestiveCompetencies") or {}
    sugg["Functional"] = _clean_pairs(
        sugg.get("Functional"), "Functional", masters, issues,
        "SuggestiveCompetencies.Functional",
    )
    sugg[P.BEHAVIOURAL_LABEL] = _clean_pairs(
        sugg.get(P.BEHAVIOURAL_LABEL) or sugg.get("Behavioral"),
        P.BEHAVIOURAL_LABEL, masters, issues,
        f"SuggestiveCompetencies.{P.BEHAVIOURAL_LABEL}",
    )
    sugg.pop("Behavioral", None)
    rec["SuggestiveCompetencies"] = sugg

    # ---- SGOS / Domain ------------------------------------------------------
    for key in ("Sector", "SubSector", "SubSectorTheme"):
        if _is_placeholder(rec.get(key)):
            rec[key] = None

    sector, subsector, theme = (
        rec.get("Sector"), rec.get("SubSector"), rec.get("SubSectorTheme"),
    )
    if area == "Domain":
        canon = (
            masters.canonical_sgos(sector, subsector, theme)
            if sector and subsector and theme
            else None
        )
        if canon:
            rec["Sector"], rec["SubSector"], rec["SubSectorTheme"] = canon
            rec["DomainCompetencies"] = [{"Theme": canon[1], "SubTheme": canon[2]}]
        else:
            if sector or subsector or theme:
                issues.append(
                    f"SGOS: {sector!r}/{subsector!r}/{theme!r} is not a valid path in the "
                    "SGOS master; cleared"
                )
            rec["Sector"] = rec["SubSector"] = rec["SubSectorTheme"] = None
            rec["DomainCompetencies"] = []
        # Domain courses carry no KCM competencies (v3.5: one area per course).
        for field in ("FunctionalCompetencies", P.BEHAVIOURAL_KEY):
            if rec.get(field):
                issues.append(
                    f"{field}: cleared - primary area is Domain, so KCM branches must be empty"
                )
                rec[field] = []
    else:
        if sector or subsector or theme:
            issues.append(
                f"SGOS: cleared Sector/SubSector/SubSectorTheme - primary area is {area}"
            )
        rec["Sector"] = rec["SubSector"] = rec["SubSectorTheme"] = None
        rec["DomainCompetencies"] = []
        rec["SuggestiveCompetencies"]["Domain"] = []
        # Cross-category guard: the non-primary KCM branch must be empty.
        other = P.BEHAVIOURAL_KEY if area == "Functional" else "FunctionalCompetencies"
        if rec.get(other):
            issues.append(
                f"{other}: cleared - primary area is {area}, cross-category mapping not allowed"
            )
            rec[other] = []
        primary_field = (
            "FunctionalCompetencies" if area == "Functional" else P.BEHAVIOURAL_KEY
        )
        if not rec.get(primary_field):
            issues.append(
                f"{primary_field}: empty although primary area is {area} - nothing in KCM matched"
            )

    # ---- target roles -------------------------------------------------------
    kept, rejected = designation_index.validate(rec.get("Targetroles"))
    if rejected:
        issues.append(
            "Targetroles: rejected "
            + ", ".join(f"{r.get('Name')!r}" for r in rejected[:6])
            + (" ..." if len(rejected) > 6 else "")
            + " - not present in the iGOT designation master"
        )
    rec["Targetroles"] = kept

    bands = [
        b for b in (rec.get("TargetEmployeeGroups") or [])
        if isinstance(b, str) and not _is_placeholder(b)
    ]
    if not bands:
        issues.append("TargetEmployeeGroups: empty after cleaning")
    rec["TargetEmployeeGroups"] = bands

    # ---- tags ---------------------------------------------------------------
    provider = str(authoritative_facts.get("provider") or "").strip()
    tags: List[str] = []
    provider_tokens = {t for t in re.findall(r"[A-Za-z]{4,}", provider.lower())}
    for tag in rec.get("Tags") or []:
        if not isinstance(tag, str) or _is_placeholder(tag):
            continue
        tag_clean = tag.strip()
        if not tag_clean:
            continue
        low = tag_clean.lower()
        if provider and (low in provider.lower() or low in provider_tokens):
            issues.append(f"Tags: dropped provider-derived tag {tag_clean!r}")
            continue
        if low not in {t.lower() for t in tags}:
            tags.append(tag_clean)
    rec["Tags"] = tags

    # ---- rubric -------------------------------------------------------------
    rubric = rec.get("RubricScoring") or {}
    for key in P.RUBRIC_KEYS:
        try:
            val = int(round(float(rubric.get(key) or 0)))
        except (TypeError, ValueError):
            val = 0
            issues.append(f"RubricScoring.{key}: non-numeric, set to 0")
        if val < 0 or val > 100:
            issues.append(f"RubricScoring.{key}: {val} out of range, clamped")
            val = max(0, min(100, val))
        rubric[key] = val

    recomputed = P.compute_total_score(rubric)
    claimed = rubric.get("TotalScore")
    if not isinstance(claimed, int) or abs(claimed - recomputed) > 1:
        issues.append(
            f"RubricScoring.TotalScore: model said {claimed!r}, recomputed {recomputed}"
        )
    rubric["TotalScore"] = recomputed

    classification = P.classify(recomputed)
    if rubric.get("Classification") != classification:
        issues.append(
            f"RubricScoring.Classification: model said {rubric.get('Classification')!r}, "
            f"recomputed {classification}"
        )
    rubric["Classification"] = classification
    rec["RubricScoring"] = rubric

    if rec.get("LearningLevel") != classification:
        issues.append(
            f"LearningLevel: model said {rec.get('LearningLevel')!r}, forced to "
            f"{classification} to match Classification"
        )
    rec["LearningLevel"] = classification

    # ---- evidence-tier honesty ---------------------------------------------
    ta = rec.get("TranscriptAnalysis") or {}
    if evidence_tier == "metadata_only":
        markers = ta.get("CognitiveMarkers") or []
        signals = ta.get("CompetencySignals") or []
        kws = (ta.get("KeywordsExtracted") or {}).get("Values") or []
        if markers or signals:
            issues.append(
                "TranscriptAnalysis: cleared CognitiveMarkers/CompetencySignals - no "
                "transcript was supplied, so transcript-derived evidence cannot exist"
            )
            ta["CognitiveMarkers"] = []
            ta["CompetencySignals"] = []
        if kws:
            # Keywords can legitimately come from metadata; keep but mark the method.
            ke = ta.get("KeywordsExtracted") or {}
            ke["Method"] = "metadata-only"
            ta["KeywordsExtracted"] = ke
    rec["TranscriptAnalysis"] = ta

    # ---- reference resources: URLs only ------------------------------------
    refs = rec.get("ReferenceResources") or {}
    for key in ("AssignmentsAndPracticeLinks", "ExtendedLearning"):
        urls = [
            u for u in (refs.get(key) or [])
            if isinstance(u, str) and re.match(r"^https?://", u.strip())
        ]
        dropped = len(refs.get(key) or []) - len(urls)
        if dropped:
            issues.append(f"ReferenceResources.{key}: dropped {dropped} non-URL entries")
        refs[key] = urls
    rec["ReferenceResources"] = refs

    # ---- explain must not be empty -----------------------------------------
    explain = rec.get("explain") or {}
    if not explain or not any(explain.get(k) for k in explain):
        issues.append("explain: empty audit trail returned")
    explain["evidence_tier_used"] = evidence_tier
    explain["validation_issues"] = issues[:]
    rec["explain"] = explain

    # ---- required-field presence -------------------------------------------
    for field in P.METADATA_SCHEMA["required"]:
        if field not in rec:
            issues.append(f"{field}: missing from model output")
        explain["validation_issues"] = issues[:]

    return rec, issues
