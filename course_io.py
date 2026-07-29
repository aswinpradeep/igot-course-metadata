"""
Reading course content off disk.

Course folders are nested and the exporter writes placeholder files when it finds
nothing, so naive reading produces empty or -- worse -- misleading input:

    do_<courseid>/
    |-- metadata.json
    |-- english_subtitles.vtt   <- placeholder in 3707/3707 non-SCORM courses
    |-- video.txt               <- "No video URLs found"
    |-- pdf_links.txt
    `-- do_<moduleid>/
        |-- <Module Name>.pdf           <- PDFs live here, depth 2
        `-- <Module Name>/
            |-- metadata.json
            `-- english_subtitles.vtt   <- the real transcript

The course-root VTT is a sentinel in every non-SCORM course measured, so reading
only the course root yields no transcript at all and injects the literal string
"// No English subtitles found" as if it were content.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("course-regenerator.course_io")

VTT_SENTINELS = ("// no english subtitles found",)
VIDEO_SENTINELS = ("no video urls found",)

CUE_TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
)

# Guardrails so one pathological course cannot blow up a prompt. One observed
# course carries a 290k-char transcript across 32 PDFs.
MAX_TRANSCRIPT_CHARS = int(os.environ.get("MAX_TRANSCRIPT_CHARS", "400000"))
MAX_PDF_CHARS = int(os.environ.get("MAX_PDF_CHARS", "200000"))
MAX_PDFS_PER_COURSE = int(os.environ.get("MAX_PDFS_PER_COURSE", "40"))


def is_course_dir(path: Path) -> bool:
    """
    A course folder, as opposed to a batch folder or a module folder.

    All three are named `do_*` or sit in the same tree, so name alone cannot tell
    them apart. The discriminator is that only a course carries metadata.json
    directly: a batch dir has none, and a module keeps its own one level deeper,
    at do_<moduleid>/<Module Name>/metadata.json.
    """
    return path.is_dir() and path.name.startswith("do_") and (path / "metadata.json").is_file()


def find_course_dirs(base: Path) -> List[Path]:
    """
    Course folders under `base`, accepting either layout:

        base/do_<id>/...                  (flat, e.g. the shareable input bundle)
        base/<batch>/do_<id>/...          (as extracted from the batch zips)

    Never descends into a course, so a module is never mistaken for a course --
    which silently inflated an early manifest from 12 courses to 93.
    """
    if not base.is_dir():
        return []
    found: List[Path] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if is_course_dir(child):
            found.append(child)
        elif not child.name.startswith("do_"):
            found.extend(c for c in sorted(child.iterdir()) if is_course_dir(c))
    return found


def read_text(path: Path) -> str:
    """Read a text file, tolerating the mixed encodings in this export."""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, ValueError):
            continue
        except OSError:
            return ""
    return ""


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(read_text(path))
    except (json.JSONDecodeError, ValueError):
        logger.warning("unparseable JSON: %s", path)
        return None


def is_sentinel(text: str, sentinels: tuple = VTT_SENTINELS) -> bool:
    stripped = (text or "").strip().lower()
    if not stripped:
        return True
    return any(s in stripped for s in sentinels)


def vtt_is_real(text: str) -> bool:
    """A usable VTT must not be a sentinel and must contain a cue timing."""
    return not is_sentinel(text, VTT_SENTINELS) and "-->" in text


def vtt_span_seconds(text: str) -> float:
    """
    Duration proxy for one VTT: last cue end minus first cue start.

    Cue timelines do not reliably start at 00:00 (observed a module whose first
    cue begins at 00:07:30), so an absolute max-end would overstate length.
    Measured against platform durations this underestimates (median ratio 0.59),
    so it is only a fallback where no platform duration exists.
    """
    starts: List[float] = []
    ends: List[float] = []
    for m in CUE_TIME_RE.finditer(text):
        sh, sm, ss, sms, eh, em, es, ems = (int(g) for g in m.groups())
        starts.append(sh * 3600 + sm * 60 + ss + sms / 1000)
        ends.append(eh * 3600 + em * 60 + es + ems / 1000)
    if not starts:
        return 0.0
    return max(0.0, max(ends) - min(starts))


def clean_vtt_text(text: str) -> str:
    """Strip WEBVTT headers, NOTE blocks, cue ids and timings; keep spoken text."""
    lines: List[str] = []
    prev = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("WEBVTT") or upper.startswith("NOTE"):
            continue
        if "-->" in line or line.isdigit():
            continue
        # Auto-generated captions frequently repeat a line verbatim.
        if line == prev:
            continue
        prev = line
        lines.append(line)
    return "\n".join(lines)


def _module_label(module_dir: Path, course_dir: Path) -> str:
    meta = load_json(module_dir / "metadata.json") or {}
    name = str(meta.get("name") or "").strip()
    if name:
        return name
    try:
        return str(module_dir.relative_to(course_dir))
    except ValueError:
        return module_dir.name


def extract_pdf_text_sync(pdf_path: Path, max_chars: int = MAX_PDF_CHARS) -> str:
    """Extract text from one PDF with PyMuPDF, stopping at max_chars."""
    import fitz  # imported lazily so course_io stays importable without PyMuPDF

    parts: List[str] = []
    total = 0
    try:
        with fitz.open(str(pdf_path)) as doc:
            for page in doc:
                text = (page.get_text() or "").strip()
                if not text:
                    continue
                parts.append(text)
                total += len(text)
                if total >= max_chars:
                    break
    except Exception as exc:
        logger.warning("PDF extraction failed for %s: %s", pdf_path, exc)
        return ""
    return "\n\n".join(parts)[:max_chars]


async def extract_course_content(course_dir: Path) -> Dict[str, Any]:
    """
    Gather everything usable from one course folder.

    Returns transcript text (module-labelled, in stable order), PDF text,
    reference links, a duration fallback, and an evidence tier.
    """
    # ---- transcripts from module subfolders ---------------------------------
    module_vtts: List[Tuple[str, Path]] = []
    for vtt_path in sorted(course_dir.rglob("english_subtitles.vtt")):
        if vtt_path.parent == course_dir:
            continue  # course-root placeholder
        module_vtts.append((_module_label(vtt_path.parent, course_dir), vtt_path))

    def _read_transcripts() -> Tuple[str, float, int]:
        chunks: List[str] = []
        seconds = 0.0
        used = 0
        total = 0
        for label, path in module_vtts:
            raw = read_text(path)
            if not vtt_is_real(raw):
                continue
            body = clean_vtt_text(raw)
            if not body.strip():
                continue
            seconds += vtt_span_seconds(raw)
            used += 1
            block = f"## Module: {label}\n{body}"
            if total + len(block) > MAX_TRANSCRIPT_CHARS:
                chunks.append(block[: max(0, MAX_TRANSCRIPT_CHARS - total)])
                chunks.append("\n[...remaining modules truncated...]")
                break
            chunks.append(block)
            total += len(block)
        return "\n\n".join(chunks), seconds, used

    transcript, transcript_seconds, modules_with_transcript = await asyncio.to_thread(
        _read_transcripts
    )

    # ---- PDFs (depth 2, siblings of the module dirs) -----------------------
    # A "<name>.pdf.txt" sidecar is used in place of the PDF when present. The
    # shareable input bundle ships sidecars instead of the binaries: the PDFs are
    # 97% of the raw bytes (3.8 GB), and the largest are scanned images that yield
    # almost no text (one 321 MB / 25-page file gave ~2,900 characters), while
    # extraction is capped at MAX_PDF_CHARS regardless.
    pdf_paths = [p for p in sorted(course_dir.rglob("*.pdf")) if p.is_file()]
    sidecars = [p for p in sorted(course_dir.rglob("*.pdf.txt")) if p.is_file()]

    # Don't read a PDF that already has a sidecar.
    sidecar_stems = {p.name[: -len(".txt")] for p in sidecars}
    pdf_paths = [p for p in pdf_paths if p.name not in sidecar_stems]

    documents: List[Tuple[str, Path, bool]] = (
        [(p.stem, p, False) for p in pdf_paths]
        + [(Path(p.name[: -len(".pdf.txt")]).name, p, True) for p in sidecars]
    )
    documents.sort(key=lambda d: d[0])
    truncated_pdfs = len(documents) > MAX_PDFS_PER_COURSE
    documents = documents[:MAX_PDFS_PER_COURSE]

    pdf_text = ""
    if documents:
        budget = max(1, MAX_PDF_CHARS // len(documents))
        texts = await asyncio.gather(
            *(
                asyncio.to_thread(read_text, path)
                if is_sidecar
                else asyncio.to_thread(extract_pdf_text_sync, path, budget)
                for _, path, is_sidecar in documents
            )
        )
        blocks = [
            f"## Document: {label}\n{t[:budget]}"
            for (label, _, _), t in zip(documents, texts)
            if t and t.strip()
        ]
        pdf_text = "\n\n---\n\n".join(blocks)[:MAX_PDF_CHARS]
        if truncated_pdfs:
            pdf_text += "\n\n[...additional documents omitted...]"

    # ---- reference links ---------------------------------------------------
    links: List[str] = []
    for link_file in sorted(course_dir.rglob("pdf_links.txt")):
        for line in read_text(link_file).splitlines():
            m = re.search(r"https?://\S+", line)
            if m and m.group(0) not in links:
                links.append(m.group(0))
    for vf in sorted(course_dir.rglob("video.txt")):
        raw = read_text(vf)
        if is_sentinel(raw, VIDEO_SENTINELS):
            continue
        for line in raw.splitlines():
            m = re.search(r"https?://\S+", line)
            if m and m.group(0) not in links:
                links.append(m.group(0))

    # ---- evidence tier -----------------------------------------------------
    if len(transcript) >= 2000:
        tier = "transcript"
    elif pdf_text:
        tier = "pdf_only"
    elif transcript:
        tier = "thin_transcript"
    elif links:
        tier = "links_only"
    else:
        tier = "metadata_only"

    return {
        "transcript": transcript,
        "transcript_chars": len(transcript),
        "transcript_seconds": round(transcript_seconds, 1),
        "modules_total": len(list(course_dir.glob("do_*"))),
        "modules_with_transcript": modules_with_transcript,
        "pdf_text": pdf_text,
        # Counts documents, not binaries: a bundle ships text sidecars instead of
        # PDFs, and counting only *.pdf there would report 0 and drop the course
        # out of the pdf_only tier.
        "pdf_count": len(documents),
        "reference_links": links,
        "evidence_tier": tier,
    }


def load_course_metadata(course_dir: Path) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
    """
    Load a course's metadata.json.

    Returns (metadata_for_prompt, raw_text_for_audit, declared_competency_area).

    `competencies_v6` is removed from what the model sees, because the framework
    requires the competency area to be decided from content rather than from the
    frequently-wrong declared value. It is returned separately so the run can be
    audited against it. The v3.4 code did `del metadata['competencies_v6']`,
    which raised KeyError whenever the key was absent and -- via a bare except --
    silently discarded the entire metadata record and the audit copy with it.
    """
    path = course_dir / "metadata.json"
    raw = read_text(path)
    if not raw.strip():
        return None, "null", None

    try:
        meta = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("unparseable metadata.json for %s; keeping raw copy", course_dir.name)
        return None, raw, None

    declared_raw = meta.pop("competencies_v6", None)
    declared: Optional[str] = None
    if isinstance(declared_raw, str) and declared_raw.strip():
        # Stored as one newline-joined token per module, e.g. "Functional\nFunctional".
        values = {v.strip() for v in declared_raw.split("\n") if v.strip()}
        if len(values) == 1:
            declared = values.pop()
        elif values:
            declared = "MIXED:" + "+".join(sorted(values))

    meta.pop("scorm", None)  # passed to the model separately as scorm_flag
    return meta, raw, declared
