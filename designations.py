"""
Target-role (designation) retrieval for Framework v3.5.

v3.5 adds `Targetroles`, which must be drawn from the official iGOT designation
master. That master has 19,936 rows -- far too many to put in a prompt (roughly
250k tokens of noise per course, which would also wreck accuracy). So:

  1. Embed the master once (Vertex text-embedding-005, 768d) and cache to disk.
  2. Per course, retrieve a shortlist by cosine similarity, unioned with a
     lexical token-overlap pass to catch literal title matches that embeddings
     sometimes rank low.
  3. Give the model only the shortlist.
  4. Reject anything the model returns that is not in the master. Membership is
     enforced here, never trusted from the model -- an unconstrained run
     fabricates plausible ids like "desig_001".

The index is held in-process with numpy rather than pgvector: 19,936 x 768
float32 is ~61 MB, and karmayogi_db has no vector extension installed.

An existing cbp_tpc_ai.designation_embeddings table covers the same 19,936 ids.
Per the production service (src/services/designation_matcher_service.py) those
vectors were built with GOOGLE_EMBEDDING_MODEL=gemini-embedding-2 at
output_dimensionality=768, over the text "task: sentence similarity | query:
<designation>", called through the Gemini Developer API (vertexai=False) with a
Redis cache in front.

They are not reused here, because matching that setup would require a
GOOGLE_API_KEY, Gemini-API access to gemini-embedding-2, and a Redis dependency,
and would couple this batch job to another service's database. Cosine similarity
is only meaningful between vectors from the same model, so this module builds its
own self-contained Vertex index instead. To align with production later, switch
EMBED_MODEL/EMBED_DIM here and prefix queries with the same string.
"""

from __future__ import annotations

import asyncio
import csv
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("course-regenerator.designations")

# Changing EMBED_MODEL or EMBED_DIM invalidates any cached index: cosine
# similarity is only meaningful between vectors from the same model at the same
# dimensionality. Rebuild with tools/build_designation_index.py --force.
EMBED_MODEL = os.environ.get("DESIGNATION_EMBED_MODEL", "text-embedding-005")
EMBED_DIM = int(os.environ.get("DESIGNATION_EMBED_DIM", "768"))
EMBED_BATCH = int(os.environ.get("DESIGNATION_EMBED_BATCH", "250"))

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Tokens that carry no discriminating signal across a government designation list.
_STOPWORDS = {
    "the", "of", "and", "for", "to", "in", "a", "an", "grade", "group", "level",
    "class", "cadre", "officer", "official", "staff", "assistant", "senior",
    "junior", "deputy", "chief", "i", "ii", "iii", "iv", "v",
}


def _tokens(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS]


def load_master(csv_path: Path) -> List[Dict[str, str]]:
    """Read the id,name designation master."""
    rows: List[Dict[str, str]] = []
    seen: set = set()
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        for rec in csv.DictReader(fh):
            did = (rec.get("id") or "").strip()
            name = (rec.get("name") or "").strip()
            if not did or not name or did in seen:
                continue
            seen.add(did)
            rows.append({"id": did, "name": name})
    logger.info("loaded %d designations from %s", len(rows), csv_path)
    return rows


def _normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class DesignationIndex:
    """Hybrid (vector + lexical) shortlist over the designation master."""

    def __init__(
        self,
        ids: Sequence[str],
        names: Sequence[str],
        matrix: Optional[np.ndarray] = None,
    ) -> None:
        self.ids: List[str] = list(ids)
        self.names: List[str] = list(names)
        self.matrix = matrix  # normalised, or None for lexical-only operation
        self._by_id = {d: i for i, d in enumerate(self.ids)}
        self._by_name = {n.strip().lower(): i for i, n in enumerate(self.names)}

        # Inverted index for the lexical pass.
        self._postings: Dict[str, List[int]] = defaultdict(list)
        for i, name in enumerate(self.names):
            for tok in set(_tokens(name)):
                self._postings[tok].append(i)

    # ---------------------------------------------------------------- building

    @staticmethod
    async def _embed_batches(
        client: Any,
        texts: Sequence[str],
        task_type: str,
        concurrency: int = 4,
    ) -> np.ndarray:
        from google.genai import types

        batches = [
            (i, texts[i : i + EMBED_BATCH]) for i in range(0, len(texts), EMBED_BATCH)
        ]
        out: List[Optional[np.ndarray]] = [None] * len(batches)
        sem = asyncio.Semaphore(concurrency)
        done = 0

        async def run(slot: int, batch: Sequence[str]) -> None:
            nonlocal done
            async with sem:
                for attempt in range(4):
                    try:
                        resp = await client.aio.models.embed_content(
                            model=EMBED_MODEL,
                            contents=list(batch),
                            config=types.EmbedContentConfig(
                                task_type=task_type,
                                output_dimensionality=EMBED_DIM,
                            ),
                        )
                        out[slot] = np.array(
                            [e.values for e in resp.embeddings], dtype=np.float32
                        )
                        break
                    except Exception as exc:  # transient quota/5xx
                        if attempt == 3:
                            raise
                        await asyncio.sleep(2 ** attempt)
                        logger.warning("embed batch %d retry %d: %s", slot, attempt + 1, exc)
                done += 1
                if done % 20 == 0 or done == len(batches):
                    logger.info("  embedded %d/%d batches", done, len(batches))

        await asyncio.gather(*(run(i, b) for i, (_, b) in enumerate(batches)))
        return np.vstack([o for o in out if o is not None])

    @classmethod
    async def build(
        cls,
        csv_path: Path,
        client: Any,
        cache_path: Path,
        concurrency: int = 4,
    ) -> "DesignationIndex":
        master = load_master(csv_path)
        names = [m["name"] for m in master]
        logger.info("embedding %d designation names with %s...", len(names), EMBED_MODEL)
        mat = await cls._embed_batches(
            client, names, task_type="RETRIEVAL_DOCUMENT", concurrency=concurrency
        )
        if mat.shape[0] != len(names):
            raise RuntimeError(
                f"embedding count mismatch: got {mat.shape[0]}, expected {len(names)}"
            )
        mat = _normalise(mat)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            ids=np.array([m["id"] for m in master]),
            names=np.array(names),
            matrix=mat,
            model=np.array([EMBED_MODEL]),
        )
        logger.info("designation index cached -> %s (%s)", cache_path, mat.shape)
        return cls([m["id"] for m in master], names, mat)

    @classmethod
    def load(cls, cache_path: Path) -> "DesignationIndex":
        data = np.load(cache_path, allow_pickle=False)
        return cls(
            [str(x) for x in data["ids"]],
            [str(x) for x in data["names"]],
            data["matrix"],
        )

    @classmethod
    def lexical_only(cls, csv_path: Path) -> "DesignationIndex":
        """Fallback when no embedding index has been built."""
        master = load_master(csv_path)
        return cls([m["id"] for m in master], [m["name"] for m in master], None)

    # --------------------------------------------------------------- retrieval

    def _lexical(self, query: str, k: int) -> List[Tuple[int, float]]:
        q = set(_tokens(query))
        if not q:
            return []
        scores: Dict[int, float] = defaultdict(float)
        for tok in q:
            postings = self._postings.get(tok)
            if not postings:
                continue
            # Rare tokens are far more informative than common ones.
            weight = 1.0 / (1.0 + np.log1p(len(postings)))
            for idx in postings:
                scores[idx] += weight
        # Normalise by designation length so short exact names win over long ones.
        ranked = sorted(
            ((i, s / (1.0 + 0.15 * len(_tokens(self.names[i])))) for i, s in scores.items()),
            key=lambda x: -x[1],
        )
        return ranked[:k]

    async def shortlist(
        self,
        client: Any,
        query_text: str,
        k: int = 150,
        lexical_k: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Return up to k candidates as [{"id","name"}], vector-ranked where an
        embedding index is available, unioned with a lexical pass.
        """
        picked: List[Tuple[int, float, str]] = []
        seen: set = set()

        if self.matrix is not None and query_text.strip():
            try:
                qv = await self._embed_batches(
                    client, [query_text[:8000]], task_type="RETRIEVAL_QUERY", concurrency=1
                )
                qv = _normalise(qv)[0]
                sims = self.matrix @ qv
                top = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
                for i in top[np.argsort(-sims[top])]:
                    i = int(i)
                    seen.add(i)
                    picked.append((i, float(sims[i]), "vector"))
            except Exception as exc:
                logger.warning("vector shortlist failed, falling back to lexical: %s", exc)

        for i, score in self._lexical(query_text, lexical_k):
            if i not in seen:
                seen.add(i)
                picked.append((i, float(score), "lexical"))

        picked.sort(key=lambda x: -x[1])
        return [
            {"id": self.ids[i], "name": self.names[i]}
            for i, _, _ in picked[:k]
        ]

    # -------------------------------------------------------------- validation

    def validate(
        self, entries: Iterable[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split model-returned Targetroles into (kept, rejected).

        An entry is kept only if its id exists in the master, or its name matches
        a master name exactly (case-insensitive) -- in which case the id is
        corrected to the master's. Everything else is rejected.
        """
        kept: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            did = str(entry.get("DesignationId") or "").strip()
            name = str(entry.get("Name") or "").strip()

            idx = self._by_id.get(did)
            if idx is None:
                idx = self._by_name.get(name.lower())
            if idx is None:
                rejected.append({**entry, "_reason": "not in designation master"})
                continue

            fixed = dict(entry)
            fixed["DesignationId"] = self.ids[idx]
            fixed["Name"] = self.names[idx]
            kept.append(fixed)
        return kept, rejected


def build_retrieval_query(
    metadata: Optional[Dict[str, Any]],
    transcript_keywords: Sequence[str] = (),
    sector_hint: str = "",
    transcript_head: str = "",
) -> str:
    """
    Compose the text used to retrieve designation candidates.

    Deliberately excludes the provider/organisation name: it pulls the shortlist
    toward that body's own job titles rather than the roles the course serves.
    """
    meta = metadata or {}
    parts: List[str] = [
        str(meta.get("name") or ""),
        " ".join(meta.get("keywords") or []),
        str(meta.get("instructions") or "")[:1500],
        str(meta.get("description") or "")[:1500],
        " ".join(transcript_keywords),
        sector_hint,
        transcript_head[:2000],
    ]
    return "\n".join(p for p in parts if p).strip()
