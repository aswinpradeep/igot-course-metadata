"""
iGOT course metadata regeneration worker -- Framework v3.5 (Advanced).

Supersedes the v3.4-advanced-extended implementation. Prompt and output schema
now live in prompts_v35.py; content extraction in course_io.py; output checking
in validation.py; target-role retrieval in designations.py.

Follows the conventions in legacy bulk_scripts/README.md: dry-run by default,
`--execute` to make real calls and writes, `--batch-size` for concurrency, an
outcome CSV plus a log file every run, mandatory env vars enforced up front, and
safe to re-run (finished courses are skipped unless `--force`).

Run:
    python meta_gen.py --limit 10                     # dry run: no AI calls, no writes
    python meta_gen.py --limit 10 --execute           # pilot for real
    python meta_gen.py --course-id do_... --execute   # single course
    python meta_gen.py --execute                      # drain all pending, then exit
    python meta_gen.py --execute --serve              # stay resident, poll for work
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

import course_io
import prompts_v35 as P
from designations import DesignationIndex, build_retrieval_query
from validation import MasterData, validate_and_repair

# ---------------------------------------------------------------- configuration
DB_DSN = os.environ.get("DB_DSN", "postgresql://localhost:5432/karmayogi_db")
ENRICHMENT_DSN = os.environ.get("ENRICHMENT_DSN") or None

COURSES_BASE_PATH = Path(os.environ.get("COURSES_BASE_PATH", "course_data"))
COURSE_MANIFEST = os.environ.get("COURSE_MANIFEST") or None
CONTENT_SET = os.environ.get("CONTENT_SET", "non-scorm")

KCM_PATH = Path(os.environ.get("KCM_PATH", "competencies 5.json"))
SGOS_PATH = Path(os.environ.get("SGOS_PATH", "SGOS 1.json"))
DESIGNATIONS_PATH = Path(os.environ.get("DESIGNATIONS_PATH", "igot_designations.csv"))
DESIGNATION_INDEX_PATH = Path(
    os.environ.get("DESIGNATION_INDEX_PATH", "data/designation_index.npz")
)

# META_GEN_* deliberately shadow GENAI_MODEL_NAME / GOOGLE_LOCATION, which in the
# shared .env point at gemini-2.5-flash in asia-south1. No pro model is served
# from asia-south1 for this project, and flash would regress quality against the
# v3.4 baseline, which ran on gemini-2.5-pro.
GENAI_MODEL_NAME = os.environ.get("META_GEN_MODEL") or os.environ.get(
    "GENAI_MODEL_NAME", "gemini-3.1-pro-preview"
)
GOOGLE_LOCATION = os.environ.get("META_GEN_LOCATION") or os.environ.get(
    "GOOGLE_LOCATION", "global"
)
GOOGLE_PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID")

MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "4"))
WORKER_BATCH_SIZE = int(os.environ.get("WORKER_BATCH_SIZE", "20"))
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "3"))
DESIGNATION_CANDIDATES = int(os.environ.get("DESIGNATION_CANDIDATES", "150"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.1"))
LLM_MAX_OUTPUT_TOKENS = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "32768"))
# Without a ceiling a single generate_content call can hang indefinitely and wedge
# the whole run (observed: a worker stuck >10 min with no progress and no error).
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS_META", "300"))

# Courses left 'in_progress' by a killed worker are reclaimed after this long.
STALE_CLAIM_MINUTES = int(os.environ.get("STALE_CLAIM_MINUTES", "30"))

# Published USD per 1M tokens, for run cost reporting. Defaults are
# gemini-3.1-pro-preview at the <=200k-prompt tier ($4/$18 above 200k; our
# prompts run ~55k-140k so the low tier applies). Verify against
# cloud.google.com/vertex-ai/generative-ai/pricing -- these are only as current
# as whoever last edited .env.
PRICE_INPUT_PER_M = float(os.environ.get("PRICE_INPUT_PER_M", "2.00"))
PRICE_OUTPUT_PER_M = float(os.environ.get("PRICE_OUTPUT_PER_M", "12.00"))
PRICE_CACHED_INPUT_PER_M = float(os.environ.get("PRICE_CACHED_INPUT_PER_M", "0.20"))

LLM_PROMPT_VERSION = P.PROMPT_VERSION

# ---------------------------------------------------------------------- logging
LOG_DIR = Path(os.environ.get("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_LEVEL = os.environ.get("META_GEN_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}-{P.PROMPT_VERSION}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
# The shared .env sets LOG_LEVEL=DEBUG for another service; honour it for our own
# loggers only via META_GEN_LOG_LEVEL, and keep the SDK's chatter down.
for noisy in ("google_genai", "google.genai", "httpx", "httpcore", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("course-regenerator")

_shutdown = asyncio.Event()

# ------------------------------------------------------------------------- SQL
DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS course_processing_checkpoint (
      course_id TEXT PRIMARY KEY,
      source_folder TEXT,
      status TEXT,
      last_updated TIMESTAMP WITH TIME ZONE,
      attempts INTEGER DEFAULT 0,
      last_error TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS course_metadata_regenerated (
      course_id TEXT NOT NULL,
      regenerated_json JSONB NOT NULL,
      source_folder TEXT,
      llm_model TEXT,
      llm_prompt_version TEXT,
      processing_duration_seconds NUMERIC,
      llm_usage_json JSONB,
      original_metadata_json JSONB,
      status TEXT,
      regenerated_at TIMESTAMP WITH TIME ZONE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS course_processing_errors (
      id BIGSERIAL PRIMARY KEY,
      course_id TEXT,
      error_message TEXT,
      created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
    )
    """,
    # v3.5 additions. Separate ALTERs so a partially-migrated DB still converges.
    "ALTER TABLE course_processing_checkpoint ADD COLUMN IF NOT EXISTS content_set TEXT",
    "ALTER TABLE course_processing_checkpoint ADD COLUMN IF NOT EXISTS prompt_version TEXT",
    "ALTER TABLE course_metadata_regenerated ADD COLUMN IF NOT EXISTS validation_issues JSONB",
    "ALTER TABLE course_metadata_regenerated ADD COLUMN IF NOT EXISTS evidence_tier TEXT",
    "ALTER TABLE course_metadata_regenerated ADD COLUMN IF NOT EXISTS declared_competency_area TEXT",
    "ALTER TABLE course_metadata_regenerated ADD COLUMN IF NOT EXISTS content_set TEXT",
]

# The v3.4 table was keyed on course_id alone, so re-running would overwrite the
# 368-row v3.4 baseline we want to compare against. Widen the key to include the
# prompt version so both generations coexist.
MIGRATE_PK_SQL = """
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'course_metadata_regenerated_pkey'
      AND conrelid = 'course_metadata_regenerated'::regclass
      AND array_length(conkey, 1) = 1
  ) THEN
    ALTER TABLE course_metadata_regenerated
      DROP CONSTRAINT course_metadata_regenerated_pkey;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'course_metadata_regenerated_pkey'
      AND conrelid = 'course_metadata_regenerated'::regclass
  ) THEN
    UPDATE course_metadata_regenerated
       SET llm_prompt_version = COALESCE(llm_prompt_version, 'unknown')
     WHERE llm_prompt_version IS NULL;
    ALTER TABLE course_metadata_regenerated
      ALTER COLUMN llm_prompt_version SET NOT NULL;
    ALTER TABLE course_metadata_regenerated
      ADD CONSTRAINT course_metadata_regenerated_pkey
      PRIMARY KEY (course_id, llm_prompt_version);
  END IF;
END $$;
"""

UPSERT_CHECKPOINT_SQL = """
INSERT INTO course_processing_checkpoint
  (course_id, source_folder, status, last_updated, attempts, content_set, prompt_version)
VALUES ($1, $2, 'pending', now(), 0, $3, $4)
ON CONFLICT (course_id) DO UPDATE
  SET source_folder = EXCLUDED.source_folder,
      content_set   = EXCLUDED.content_set,
      status        = CASE
                        WHEN course_processing_checkpoint.prompt_version IS DISTINCT FROM EXCLUDED.prompt_version
                        THEN 'pending'
                        ELSE course_processing_checkpoint.status
                      END,
      attempts      = CASE
                        WHEN course_processing_checkpoint.prompt_version IS DISTINCT FROM EXCLUDED.prompt_version
                        THEN 0
                        ELSE course_processing_checkpoint.attempts
                      END,
      prompt_version = EXCLUDED.prompt_version
"""

CLAIM_COURSE_SQL = """
UPDATE course_processing_checkpoint
   SET status = 'in_progress',
       attempts = course_processing_checkpoint.attempts + 1,
       last_updated = now()
 WHERE course_id = $1
   AND status IN ('pending', 'failed')
RETURNING course_id, source_folder
"""

SELECT_PENDING_SQL = """
SELECT course_id, source_folder
  FROM course_processing_checkpoint
 WHERE status IN ('pending', 'failed')
   AND attempts < $2
   AND content_set = $3
   AND ($4::text[] IS NULL OR course_id = ANY($4))
 ORDER BY attempts, last_updated
 LIMIT $1
"""

# A worker killed mid-flight leaves rows in 'in_progress'. Those are not in the
# pending set, so without this they are stranded permanently and a restart
# silently skips them.
RECLAIM_STALE_SQL = """
UPDATE course_processing_checkpoint
   SET status = 'pending', last_updated = now()
 WHERE status = 'in_progress'
   AND content_set = $1
   AND last_updated < now() - ($2 || ' minutes')::interval
RETURNING course_id
"""

SAVE_RESULT_SQL = """
INSERT INTO course_metadata_regenerated
  (course_id, regenerated_json, source_folder, llm_model, llm_prompt_version,
   processing_duration_seconds, llm_usage_json, original_metadata_json, status,
   regenerated_at, validation_issues, evidence_tier, declared_competency_area, content_set)
VALUES ($1, $2::jsonb, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9, now(),
        $10::jsonb, $11, $12, $13)
ON CONFLICT (course_id, llm_prompt_version) DO UPDATE
  SET regenerated_json = EXCLUDED.regenerated_json,
      source_folder = EXCLUDED.source_folder,
      llm_model = EXCLUDED.llm_model,
      processing_duration_seconds = EXCLUDED.processing_duration_seconds,
      llm_usage_json = EXCLUDED.llm_usage_json,
      original_metadata_json = EXCLUDED.original_metadata_json,
      status = EXCLUDED.status,
      regenerated_at = now(),
      validation_issues = EXCLUDED.validation_issues,
      evidence_tier = EXCLUDED.evidence_tier,
      declared_competency_area = EXCLUDED.declared_competency_area,
      content_set = EXCLUDED.content_set
"""

MARK_DONE_SQL = """
UPDATE course_processing_checkpoint
   SET status = 'done', last_updated = now(), last_error = NULL
 WHERE course_id = $1
"""

MARK_FAILED_SQL = """
UPDATE course_processing_checkpoint
   SET status = CASE WHEN attempts >= $3 THEN 'dead' ELSE 'failed' END,
       last_updated = now(),
       last_error = $2
 WHERE course_id = $1
"""

LOG_ERROR_SQL = """
INSERT INTO course_processing_errors (course_id, error_message, created_at)
VALUES ($1, $2, now())
"""


# --------------------------------------------------------------------- helpers
def load_json_file(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required reference file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Could not parse JSON from {path}: {exc}") from exc


def _is_transient(exc: BaseException) -> bool:
    """
    Only transient failures are worth retrying.

    The v3.4 code retried on `Exception`, so permanent errors (bad credentials, a
    rejected schema, an unknown model) burned three backed-off attempts each.
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return True
    if isinstance(exc, json.JSONDecodeError):
        return True  # truncated candidate; a re-roll often succeeds
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "code", None) in (408, 409, 425, 429)
    if isinstance(exc, genai_errors.APIError):
        return getattr(exc, "code", None) in (408, 409, 425, 429, 500, 502, 503, 504)
    return False


class EnrichmentStore:
    """
    Platform-verified facts from cbp_tpc_ai.course_metadata_v3.

    Supplies language, duration and difficulty_level, which the folder-level
    metadata.json does not carry. Covers 3395 of the 3707 non-SCORM courses.
    Read-only: this worker never writes to the enrichment database.
    """

    def __init__(self, rows: Dict[str, Dict[str, Any]]) -> None:
        self.rows = rows

    @classmethod
    async def load(cls, dsn: Optional[str]) -> "EnrichmentStore":
        if not dsn:
            logger.warning("ENRICHMENT_DSN not set; language/duration will be inferred")
            return cls({})
        rows: Dict[str, Dict[str, Any]] = {}
        try:
            conn = await asyncpg.connect(dsn=dsn)
        except Exception as exc:
            logger.warning("enrichment DB unavailable (%s); continuing without it", exc)
            return cls({})
        try:
            records = await conn.fetch(
                "SELECT identifier, difficulty_level, language, duration, organisation "
                "FROM course_metadata_v3"
            )
            for rec in records:
                ident = rec["identifier"]
                if not ident:
                    continue
                langs = list(rec["language"] or [])
                orgs = list(rec["organisation"] or [])
                seconds: Optional[float] = None
                raw_duration = rec["duration"]
                if raw_duration:
                    try:
                        seconds = float(str(raw_duration).strip())
                    except (TypeError, ValueError):
                        seconds = None
                rows[ident] = {
                    "language": langs[0] if langs else None,
                    "all_languages": langs,
                    "duration_seconds": seconds,
                    "declared_difficulty": rec["difficulty_level"] or None,
                    "organisation": orgs[0] if orgs else None,
                }
        finally:
            await conn.close()
        logger.info("loaded platform facts for %d courses", len(rows))
        return cls(rows)

    def facts_for(
        self,
        course_id: str,
        folder_meta: Optional[Dict[str, Any]],
        content: Dict[str, Any],
    ) -> Dict[str, Any]:
        row = self.rows.get(course_id, {})
        meta = folder_meta or {}

        seconds = row.get("duration_seconds")
        duration_source = "platform"
        if not seconds:
            # VTT cue spans underestimate real duration (median ratio 0.59 against
            # platform values), so this is a fallback only and is labelled as such.
            seconds = content.get("transcript_seconds") or None
            duration_source = "transcript-derived (approximate)" if seconds else "unavailable"

        return {
            "language": row.get("language"),
            "all_languages": row.get("all_languages") or [],
            "duration_seconds": seconds,
            "duration_source": duration_source,
            "provider": meta.get("organisation") or row.get("organisation"),
            "module_count": content.get("modules_total"),
            "modules_with_transcript": content.get("modules_with_transcript"),
            "pdf_count": content.get("pdf_count"),
            "course_category": meta.get("courseCategory"),
            # Present for audit only. v3.5 states the incoming difficulty level is
            # not correct, so it is never offered to the model as a fact.
            "_declared_difficulty_not_authoritative": row.get("declared_difficulty"),
        }


# ------------------------------------------------------------------ LLM calling
class LLMClient:
    def __init__(self, client: Any, static_prefix: str) -> None:
        self.client = client
        self.static_prefix = static_prefix

    async def generate(self, course_section: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        prompt = self.static_prefix + course_section
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=P.METADATA_SCHEMA,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
        )
        contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
        ]

        last_exc: Optional[BaseException] = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            start = time.time()
            try:
                resp = await asyncio.wait_for(
                    self.client.aio.models.generate_content(
                        model=GENAI_MODEL_NAME, contents=contents, config=config
                    ),
                    timeout=LLM_TIMEOUT_SECONDS,
                )
                if not resp.text:
                    raise json.JSONDecodeError("empty candidate text", "", 0)
                parsed = json.loads(resp.text)
                usage = resp.usage_metadata.to_json_dict() if resp.usage_metadata else {}
                usage["_elapsed_seconds"] = round(time.time() - start, 2)
                usage["_attempts"] = attempt
                return parsed, usage
            except BaseException as exc:  # noqa: BLE001 - classified below
                last_exc = exc
                if not _is_transient(exc) or attempt == MAX_ATTEMPTS:
                    raise
                backoff = min(30.0, 2 ** attempt) + random.uniform(0, 1.0)
                logger.warning(
                    "LLM attempt %d/%d failed (%s: %s); retrying in %.1fs",
                    attempt, MAX_ATTEMPTS, type(exc).__name__, str(exc)[:160], backoff,
                )
                await asyncio.sleep(backoff)
        raise last_exc if last_exc else RuntimeError("LLM call failed")


# ------------------------------------------------------------------- processing
class Pipeline:
    def __init__(
        self,
        pool: asyncpg.Pool,
        llm: LLMClient,
        masters: MasterData,
        designation_index: DesignationIndex,
        enrichment: EnrichmentStore,
        genai_client: Any,
        dry_run: bool = False,
    ) -> None:
        self.pool = pool
        self.llm = llm
        self.masters = masters
        self.designations = designation_index
        self.enrichment = enrichment
        self.genai_client = genai_client
        self.dry_run = dry_run
        self.stats = {
            "succeeded": 0, "failed": 0, "with_issues": 0, "skipped": 0,
            "prompt_tokens": 0, "output_tokens": 0, "cached_tokens": 0,
            # Billed as output by thinking models, and easy to miss: it runs
            # 3.9k-4.6k per course, more than the visible answer itself.
            "thinking_tokens": 0,
        }
        self.latencies: List[float] = []
        self.outcomes: List[Dict[str, Any]] = []
        self._outcomes_lock = asyncio.Lock()
        self.started_at = time.time()
        # Position within the whole selection, so every course line carries
        # "[n/total]" and a tail -f shows how far along the run is.
        self.total_scope = 0
        self.completed = 0

    def _tick(self) -> str:
        self.completed += 1
        if self.total_scope:
            return f"[{self.completed}/{self.total_scope}]"
        return f"[{self.completed}]"

    async def _record(self, row: Dict[str, Any]) -> None:
        async with self._outcomes_lock:
            self.outcomes.append(row)

    async def process(self, course_id: str, course_dir: Path) -> bool:
        start = time.time()
        original_raw = "null"
        declared_area: Optional[str] = None
        tier = "unknown"
        try:
            if not course_dir.is_dir():
                raise FileNotFoundError(f"course folder not found: {course_dir}")

            folder_meta, original_raw, declared_area = course_io.load_course_metadata(
                course_dir
            )
            content = await course_io.extract_course_content(course_dir)
            tier = content["evidence_tier"]

            scorm_flag = bool((json.loads(original_raw) or {}).get("scorm")) if (
                original_raw and original_raw != "null"
            ) else False

            facts = self.enrichment.facts_for(course_id, folder_meta, content)

            query = build_retrieval_query(
                folder_meta,
                transcript_head=content["transcript"][:2000],
            )
            candidates = await self.designations.shortlist(
                self.genai_client, query, k=DESIGNATION_CANDIDATES
            )

            section = P.build_course_section(
                course_id=course_id,
                current_metadata=folder_meta,
                authoritative_facts=facts,
                transcript=content["transcript"],
                pdf_snippets=content["pdf_text"],
                designation_candidates=candidates,
                scorm_flag=scorm_flag,
                evidence_tier=tier,
            )

            if self.dry_run:
                logger.info(
                    "[dry-run] %s tier=%s transcript=%d pdf=%d candidates=%d prompt=%d chars",
                    course_id, tier, content["transcript_chars"],
                    content["pdf_count"], len(candidates),
                    len(self.llm.static_prefix) + len(section),
                )
                self.stats["skipped"] += 1
                await self._record({
                    "course_id": course_id,
                    "status": "dry_run",
                    "evidence_tier": tier,
                    "declared_area": declared_area,
                    "transcript_chars": content["transcript_chars"],
                    "pdf_count": content["pdf_count"],
                    "seconds": round(time.time() - start, 1),
                })
                return True

            raw_record, usage = await self.llm.generate(section)

            record, issues = validate_and_repair(
                raw_record,
                course_id=course_id,
                masters=self.masters,
                designation_index=self.designations,
                authoritative_facts=facts,
                evidence_tier=tier,
            )
            # Reference links found on disk are authoritative; the model must not
            # invent URLs, so seed them here rather than asking for them.
            if content["reference_links"]:
                refs = record.setdefault("ReferenceResources", {})
                existing = set(refs.get("ExtendedLearning") or [])
                refs["ExtendedLearning"] = list(existing) + [
                    u for u in content["reference_links"] if u not in existing
                ]

            record["explain"]["declared_competency_area_for_audit"] = declared_area
            record["explain"]["agrees_with_declared_area"] = (
                declared_area is not None
                and not declared_area.startswith("MIXED")
                and record.get("PrimaryCompetencyArea", {}).get("name") == declared_area
            )

            duration = time.time() - start
            self.stats["succeeded"] += 1
            if issues:
                self.stats["with_issues"] += 1
            self.stats["prompt_tokens"] += usage.get("prompt_token_count") or 0
            self.stats["output_tokens"] += usage.get("candidates_token_count") or 0
            self.stats["cached_tokens"] += usage.get("cached_content_token_count") or 0
            self.stats["thinking_tokens"] += usage.get("thoughts_token_count") or 0
            self.latencies.append(duration)

            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        SAVE_RESULT_SQL,
                        course_id,
                        json.dumps(record, sort_keys=True, ensure_ascii=False),
                        str(course_dir),
                        GENAI_MODEL_NAME,
                        LLM_PROMPT_VERSION,
                        duration,
                        json.dumps(usage),
                        original_raw if original_raw != "null" else None,
                        "success_with_issues" if issues else "success",
                        json.dumps(issues, ensure_ascii=False),
                        tier,
                        declared_area,
                        CONTENT_SET,
                    )
                    await conn.execute(MARK_DONE_SQL, course_id)

            logger.info(
                "%s %s ok in %.1fs tier=%s area=%s roles=%d issues=%d "
                "tokens(in/cached/out)=%s/%s/%s",
                self._tick(), course_id, duration, tier,
                record.get("PrimaryCompetencyArea", {}).get("name"),
                len(record.get("Targetroles") or []), len(issues),
                usage.get("prompt_token_count"),
                usage.get("cached_content_token_count") or 0,
                usage.get("candidates_token_count"),
            )
            # At INFO, not DEBUG: these are the audit record for the run and the
            # signal for systemic prompt problems. Also stored in
            # course_metadata_regenerated.validation_issues.
            for issue in issues:
                logger.info("  %s issue: %s", course_id, issue)

            rubric = record.get("RubricScoring") or {}
            await self._record({
                "course_id": course_id,
                "status": "success_with_issues" if issues else "success",
                "evidence_tier": tier,
                "primary_area": record.get("PrimaryCompetencyArea", {}).get("name"),
                "declared_area": declared_area,
                "agrees_with_declared": record["explain"].get("agrees_with_declared_area"),
                "functional": len(record.get("FunctionalCompetencies") or []),
                "behavioural": len(record.get(P.BEHAVIOURAL_KEY) or []),
                "domain": len(record.get("DomainCompetencies") or []),
                "targetroles": len(record.get("Targetroles") or []),
                "total_score": rubric.get("TotalScore"),
                "classification": rubric.get("Classification"),
                "transcript_chars": content["transcript_chars"],
                "pdf_count": content["pdf_count"],
                "issue_count": len(issues),
                "issues": " | ".join(issues)[:2000],
                "prompt_tokens": usage.get("prompt_token_count"),
                "cached_tokens": usage.get("cached_content_token_count") or 0,
                "output_tokens": usage.get("candidates_token_count"),
                "seconds": round(duration, 1),
            })
            return True

        except Exception as exc:
            self.stats["failed"] += 1
            message = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "%s %s FAILED (tier=%s): %s", self._tick(), course_id, tier, message[:300]
            )
            await self._record({
                "course_id": course_id,
                "status": "failed",
                "evidence_tier": tier,
                "declared_area": declared_area,
                "seconds": round(time.time() - start, 1),
                # Full traceback is in the log; the CSV/DB keep a bounded copy.
                "error": message[:1000],
            })
            if not self.dry_run:
                try:
                    async with self.pool.acquire() as conn:
                        async with conn.transaction():
                            await conn.execute(
                                MARK_FAILED_SQL, course_id, message[:4000], MAX_ATTEMPTS
                            )
                            await conn.execute(LOG_ERROR_SQL, course_id, message[:4000])
                except Exception:
                    logger.exception("could not record failure for %s", course_id)
            return False


# ----------------------------------------------------------------- work sourcing
def discover_courses() -> List[Tuple[str, Path]]:
    """
    Course id -> folder, preferring the deduplicated manifest from
    tools/inventory.py. Falls back to scanning COURSES_BASE_PATH.
    """
    if COURSE_MANIFEST and Path(COURSE_MANIFEST).is_file():
        out: List[Tuple[str, Path]] = []
        with open(COURSE_MANIFEST, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                out.append((rec["course_id"], Path(rec["path"])))
        logger.info("discovered %d courses from manifest %s", len(out), COURSE_MANIFEST)
        return out

    if not COURSES_BASE_PATH.exists():
        logger.error("COURSES_BASE_PATH does not exist: %s", COURSES_BASE_PATH)
        return []
    found = [(d.name, d) for d in course_io.find_course_dirs(COURSES_BASE_PATH)]
    logger.info("discovered %d courses under %s", len(found), COURSES_BASE_PATH)
    return found


async def populate_checkpoints(pool: asyncpg.Pool, courses: List[Tuple[str, Path]]) -> None:
    async with pool.acquire() as conn:
        async with conn.transaction():
            for course_id, path in courses:
                await conn.execute(
                    UPSERT_CHECKPOINT_SQL, course_id, str(path),
                    CONTENT_SET, LLM_PROMPT_VERSION,
                )
    logger.info("checkpoint table populated for %d courses", len(courses))


async def setup_database(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        for stmt in DDL_STATEMENTS:
            await conn.execute(stmt)
        await conn.execute(MIGRATE_PK_SQL)
    logger.info("database schema ready (v3.5 columns + version-scoped primary key)")


async def run(args: argparse.Namespace) -> None:
    kcm_json = load_json_file(KCM_PATH)
    sgos_json = load_json_file(SGOS_PATH)
    masters = MasterData(kcm_json, sgos_json)
    logger.info(
        "masters: %d KCM pairs, %d SGOS sectors",
        len(masters.kcm_pairs), len(masters.sgos),
    )

    if DESIGNATION_INDEX_PATH.is_file():
        designation_index = DesignationIndex.load(DESIGNATION_INDEX_PATH)
        logger.info("designation index: %d entries (vector)", len(designation_index.ids))
    else:
        designation_index = DesignationIndex.lexical_only(DESIGNATIONS_PATH)
        logger.warning(
            "no embedding index at %s; using lexical-only retrieval "
            "(build it with tools/build_designation_index.py)",
            DESIGNATION_INDEX_PATH,
        )

    if not GOOGLE_PROJECT_ID:
        raise RuntimeError("GOOGLE_PROJECT_ID is not set")
    genai_client = genai.Client(
        project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION, vertexai=True
    )
    logger.info("model %s @ %s", GENAI_MODEL_NAME, GOOGLE_LOCATION)

    static_prefix = P.build_static_prefix(kcm_json, sgos_json)
    logger.info("static prompt prefix: %d chars (cacheable)", len(static_prefix))
    llm = LLMClient(genai_client, static_prefix)

    enrichment = await EnrichmentStore.load(ENRICHMENT_DSN)

    concurrency = max(1, args.batch_size)
    pool = await asyncpg.create_pool(
        dsn=DB_DSN, min_size=1, max_size=max(2, concurrency)
    )
    try:
        await setup_database(pool)

        dry_run = not args.execute
        if dry_run:
            logger.warning(
                "DRY RUN: no AI calls and no writes. Re-run with --execute to apply."
            )

        pipeline = Pipeline(
            pool, llm, masters, designation_index, enrichment, genai_client,
            dry_run=dry_run,
        )

        if args.course_id:
            courses = [(c, p) for c, p in discover_courses() if c in set(args.course_id)]
            if not courses:
                logger.error("no matching course folders for %s", args.course_id)
                return
        else:
            courses = discover_courses()
            if args.tier:
                courses = _filter_by_tier(courses, args.tier)

        if args.force and not dry_run:
            await reset_courses(pool, [cid for cid, _ in courses])

        if not dry_run:
            await populate_checkpoints(pool, courses)
            await reclaim_stale(pool)
            done = await already_done(pool, [cid for cid, _ in courses])
            if done:
                logger.info(
                    "%d of %d course(s) already done for %s; skipping "
                    "(use --force to redo)",
                    len(done), len(courses), LLM_PROMPT_VERSION,
                )
            courses = [(c, p) for c, p in courses if c not in done]

        if args.limit:
            courses = courses[: args.limit]
        pipeline.total_scope = len(courses)
        logger.info("%d course(s) to process, concurrency=%d", len(courses), concurrency)
        if not dry_run:
            await log_progress(pool, 0)

        if dry_run:
            # Walk the list directly: the DB queue is claim-based, and a dry run
            # neither claims nor completes, so draining would re-serve forever.
            sem = asyncio.Semaphore(concurrency)

            async def _one(cid: str, path: Path) -> bool:
                async with sem:
                    return await pipeline.process(cid, path)

            await asyncio.gather(
                *(_one(cid, path) for cid, path in courses), return_exceptions=True
            )
        else:
            await drain(
                pool, pipeline,
                serve=args.serve,
                limit=args.limit,
                only_ids=[cid for cid, _ in courses],
                concurrency=concurrency,
            )

        remaining = await count_outstanding(pool) if not dry_run else None
        _report(
            pipeline.stats,
            pipeline.latencies,
            wall_seconds=time.time() - pipeline.started_at,
            concurrency=concurrency,
            remaining=remaining,
        )
        _write_outcome_csv(pipeline.outcomes, args.out)
    finally:
        await pool.close()


ALREADY_DONE_SQL = """
SELECT c.course_id
  FROM course_processing_checkpoint c
  JOIN course_metadata_regenerated r
    ON r.course_id = c.course_id AND r.llm_prompt_version = $2
 WHERE c.course_id = ANY($1) AND c.status = 'done'
"""

RESET_COURSES_SQL = """
UPDATE course_processing_checkpoint
   SET status = 'pending', attempts = 0, last_error = NULL, last_updated = now()
 WHERE course_id = ANY($1)
"""


COUNT_OUTSTANDING_SQL = """
SELECT count(*) FROM course_processing_checkpoint
 WHERE content_set = $1 AND status IN ('pending', 'failed') AND attempts < $2
"""

# Overall position, so a long run reports "done X of Y" rather than only a
# count of what this invocation happened to touch.
PROGRESS_SQL = """
SELECT count(*)                                            AS total,
       count(*) FILTER (WHERE status = 'done')             AS done,
       count(*) FILTER (WHERE status = 'failed')           AS failed,
       count(*) FILTER (WHERE status = 'dead')             AS dead,
       count(*) FILTER (WHERE status = 'in_progress')      AS in_progress,
       count(*) FILTER (WHERE status = 'pending')          AS pending
  FROM course_processing_checkpoint
 WHERE content_set = $1
"""


async def log_progress(pool: asyncpg.Pool, this_run: int, per_hour: float = 0.0) -> None:
    async with pool.acquire() as conn:
        r = await conn.fetchrow(PROGRESS_SQL, CONTENT_SET)
    if not r or not r["total"]:
        return
    done, total = int(r["done"]), int(r["total"])
    pct = done / total * 100 if total else 0.0
    remaining = total - done - int(r["dead"])
    eta = ""
    if per_hour > 0 and remaining > 0:
        eta = f", eta {_fmt_duration(remaining / per_hour * 3600)} at {per_hour:.0f}/hr"
    logger.info(
        "PROGRESS %s/%s done (%.1f%%) | %d this run | pending %d, failed %d, dead %d%s",
        f"{done:,}", f"{total:,}", pct, this_run,
        int(r["pending"]), int(r["failed"]), int(r["dead"]), eta,
    )


async def count_outstanding(pool: asyncpg.Pool) -> int:
    """Courses still to do for this content set — the basis for the projection."""
    async with pool.acquire() as conn:
        return int(await conn.fetchval(COUNT_OUTSTANDING_SQL, CONTENT_SET, MAX_ATTEMPTS) or 0)


async def already_done(pool: asyncpg.Pool, ids: List[str]) -> set:
    """Courses with a stored result for this prompt version. Re-runs skip these."""
    if not ids:
        return set()
    async with pool.acquire() as conn:
        rows = await conn.fetch(ALREADY_DONE_SQL, ids, LLM_PROMPT_VERSION)
    return {r["course_id"] for r in rows}


async def reset_courses(pool: asyncpg.Pool, ids: List[str]) -> None:
    if not ids:
        return
    async with pool.acquire() as conn:
        await conn.execute(RESET_COURSES_SQL, ids)
    logger.warning("--force: reset %d course(s) back to pending", len(ids))


def _write_outcome_csv(outcomes: List[Dict[str, Any]], out: Optional[Path]) -> None:
    """One row per course attempted, per the bulk_scripts convention."""
    if not outcomes:
        return
    path = out or LOG_DIR / (
        f"meta_gen_{P.PROMPT_VERSION}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "course_id", "status", "evidence_tier", "primary_area",
        "declared_area", "agrees_with_declared", "functional", "behavioural",
        "domain", "targetroles", "total_score", "classification",
        "transcript_chars", "pdf_count", "issue_count", "issues",
        "prompt_tokens", "cached_tokens", "output_tokens", "seconds", "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in outcomes:
            writer.writerow(row)
    logger.info("outcome CSV -> %s (%d rows)", path, len(outcomes))


def _filter_by_tier(courses: List[Tuple[str, Path]], tiers: List[str]) -> List[Tuple[str, Path]]:
    if not (COURSE_MANIFEST and Path(COURSE_MANIFEST).is_file()):
        logger.warning("--tier requires COURSE_MANIFEST; ignoring")
        return courses
    wanted = set(tiers)
    keep = set()
    with open(COURSE_MANIFEST, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                if rec.get("tier") in wanted:
                    keep.add(rec["course_id"])
    filtered = [(c, p) for c, p in courses if c in keep]
    logger.info("tier filter %s -> %d courses", sorted(wanted), len(filtered))
    return filtered


async def reclaim_stale(pool: asyncpg.Pool) -> None:
    """Return rows stranded in 'in_progress' by a killed worker to the queue."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(RECLAIM_STALE_SQL, CONTENT_SET, str(STALE_CLAIM_MINUTES))
    if rows:
        logger.warning(
            "reclaimed %d course(s) stranded in 'in_progress' for >%d min: %s",
            len(rows), STALE_CLAIM_MINUTES,
            ", ".join(r["course_id"] for r in rows[:5])
            + (" ..." if len(rows) > 5 else ""),
        )


async def drain(
    pool: asyncpg.Pool,
    pipeline: Pipeline,
    serve: bool,
    limit: Optional[int],
    only_ids: Optional[List[str]] = None,
    concurrency: int = MAX_CONCURRENCY,
) -> None:
    """
    Claim and process pending work until drained (or forever with --serve).

    only_ids scopes the queue to a specific selection. Without it, --tier/--limit
    would appear to work but drain would still pick up any other pending row for
    this content set, including leftovers from earlier runs.
    """
    processed = 0
    idle_rounds = 0
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(cid: str, path: Path) -> bool:
        async with semaphore:
            if _shutdown.is_set():
                return False
            return await pipeline.process(cid, path)

    while not _shutdown.is_set():
        batch_size = WORKER_BATCH_SIZE
        if limit:
            remaining = limit - processed
            if remaining <= 0:
                break
            batch_size = min(batch_size, remaining)

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                SELECT_PENDING_SQL, batch_size, MAX_ATTEMPTS, CONTENT_SET, only_ids
            )

        if not rows:
            if not serve:
                logger.info("no pending courses remain; draining complete")
                break
            idle_rounds += 1
            logger.info("idle (round %d); sleeping 30s", idle_rounds)
            try:
                await asyncio.wait_for(_shutdown.wait(), timeout=30)
            except asyncio.TimeoutError:
                pass
            continue

        idle_rounds = 0
        tasks = []
        for row in rows:
            cid = row["course_id"]
            source_folder = row["source_folder"]
            if not pipeline.dry_run:
                # A dry run must not claim work: claiming increments attempts and
                # leaves rows stuck in 'in_progress', since nothing ever completes.
                async with pool.acquire() as conn:
                    claimed = await conn.fetchrow(CLAIM_COURSE_SQL, cid)
                if not claimed:
                    continue
                source_folder = claimed["source_folder"]
            folder = Path(source_folder) if source_folder else COURSES_BASE_PATH / cid
            tasks.append(guarded(cid, folder))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            processed += len(tasks)
            elapsed = time.time() - pipeline.started_at
            per_hour = (
                pipeline.stats["succeeded"] / (elapsed / 3600) if elapsed > 0 else 0.0
            )
            await log_progress(pool, processed, per_hour)


def token_cost(prompt_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    """
    USD for one or more calls.

    `output` must already include thinking tokens: thinking is billed at the
    output rate, and on this workload it exceeds the visible answer.
    Cached prompt tokens are billed at the cached rate instead of the full one.
    """
    uncached = max(0, prompt_tokens - cached_tokens)
    return (
        uncached / 1_000_000 * PRICE_INPUT_PER_M
        + cached_tokens / 1_000_000 * PRICE_CACHED_INPUT_PER_M
        + output_tokens / 1_000_000 * PRICE_OUTPUT_PER_M
    )


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _report(
    stats: Dict[str, Any],
    latencies: List[float],
    wall_seconds: float,
    concurrency: int,
    remaining: Optional[int] = None,
) -> None:
    done = stats["succeeded"]
    billed_output = stats["output_tokens"] + stats["thinking_tokens"]

    logger.info("=" * 66)
    logger.info("RUN SUMMARY (%s, model %s)", LLM_PROMPT_VERSION, GENAI_MODEL_NAME)
    logger.info("  succeeded / with issues / failed : %d / %d / %d",
                done, stats["with_issues"], stats["failed"])
    if stats["skipped"]:
        logger.info("  dry-run previews                 : %d", stats["skipped"])

    if not done:
        logger.info("=" * 66)
        return

    ordered = sorted(latencies)
    mean = sum(ordered) / len(ordered)
    median = ordered[len(ordered) // 2]
    p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]

    # Per-course latency is what one course costs in time; throughput is what the
    # run achieves with `concurrency` of them overlapping. Only throughput
    # predicts wall-clock, and the two differ by roughly the concurrency factor.
    per_hour = done / (wall_seconds / 3600) if wall_seconds > 0 else 0.0

    logger.info("  -- latency (per course) --")
    logger.info("    mean / median / p90            : %.1fs / %.1fs / %.1fs", mean, median, p90)
    logger.info("    fastest / slowest              : %.1fs / %.1fs", ordered[0], ordered[-1])
    logger.info("  -- throughput (concurrency %d) --", concurrency)
    logger.info("    wall clock                     : %s", _fmt_duration(wall_seconds))
    logger.info("    courses/hour                   : %.0f", per_hour)
    logger.info("    effective seconds/course       : %.1fs", wall_seconds / done)

    logger.info("  -- tokens (mean per course) --")
    logger.info("    input                          : %s", f"{stats['prompt_tokens'] // done:,}")
    logger.info("    cached input                   : %s", f"{stats['cached_tokens'] // done:,}")
    logger.info("    output (answer)                : %s", f"{stats['output_tokens'] // done:,}")
    logger.info("    output (thinking, billed)      : %s", f"{stats['thinking_tokens'] // done:,}")

    cost = token_cost(stats["prompt_tokens"], billed_output, stats["cached_tokens"])
    logger.info("  -- cost @ $%.2f/$%.2f per 1M (cached $%.2f) --",
                PRICE_INPUT_PER_M, PRICE_OUTPUT_PER_M, PRICE_CACHED_INPUT_PER_M)
    logger.info("    this run                       : $%.2f", cost)
    logger.info("    mean per course                : $%.4f", cost / done)

    if remaining:
        eta = remaining / per_hour * 3600 if per_hour > 0 else 0
        logger.info("  -- projection for %s remaining courses --", f"{remaining:,}")
        logger.info("    at this throughput             : %s", _fmt_duration(eta))
        logger.info("    estimated cost                 : $%.2f", cost / done * remaining)
        if not stats["cached_tokens"]:
            # The static prefix is identical on every call, so if it were served
            # from cache the saving is the whole prefix at the cached rate.
            prefix = stats["prompt_tokens"] // done
            saving_per_course = token_cost(prefix, 0) - token_cost(prefix, 0, prefix)
            logger.info("    no cache hits observed; caching the static prefix")
            logger.info("    could save up to               : $%.2f", saving_per_course * remaining)
    logger.info("=" * 66)


def _install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    def handler() -> None:
        if not _shutdown.is_set():
            logger.warning("shutdown requested; finishing in-flight courses")
            _shutdown.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handler)
        except (NotImplementedError, RuntimeError):
            pass


REQUIRED_ENV = ["DB_DSN", "GOOGLE_PROJECT_ID", "GOOGLE_APPLICATION_CREDENTIALS"]


def check_env() -> None:
    """Fail immediately naming what is missing, rather than part-way through a run."""
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise SystemExit(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + "\nSet them in .env (see the iGOT COURSE METADATA REGENERATION section)."
        )
    creds = Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    if not creds.is_file():
        raise SystemExit(f"GOOGLE_APPLICATION_CREDENTIALS points at a missing file: {creds}")
    for label, path in (("KCM_PATH", KCM_PATH), ("SGOS_PATH", SGOS_PATH)):
        if not path.is_file():
            raise SystemExit(f"{label} points at a missing file: {path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Regenerate iGOT course metadata (Framework v3.5)"
    )
    ap.add_argument("--limit", type=int, help="process at most N courses")
    ap.add_argument("--course-id", action="append", help="process specific course id(s)")
    ap.add_argument(
        "--tier", action="append",
        choices=["transcript", "thin_transcript", "pdf_only", "links_only", "metadata_only"],
        help="restrict to courses in the given evidence tier(s)",
    )
    ap.add_argument("--serve", action="store_true", help="stay resident and poll for work")
    ap.add_argument(
        "--execute", action="store_true",
        help="make real AI calls and DB writes. Without this the run is a dry run: "
             "prompts and shortlists are built, nothing is called or written.",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="reprocess courses already marked done for this prompt version",
    )
    ap.add_argument(
        "--batch-size", type=int, default=MAX_CONCURRENCY,
        help=f"concurrent courses in flight (default {MAX_CONCURRENCY})",
    )
    ap.add_argument("--out", type=Path, help="outcome CSV path (default under logs/)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    check_env()

    async def _main() -> None:
        _install_signal_handlers(asyncio.get_running_loop())
        await run(args)

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        logger.info("interrupted")
    except Exception as exc:
        logger.critical("fatal: %s: %s", type(exc).__name__, exc, exc_info=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
