"""
Build a deduplicated course inventory + content-coverage census for the
extracted iGOT course folders.

The course folders are nested; the real learning content (transcripts, PDFs)
lives in module sub-folders, not in the course root:

    do_<courseid>/
    |-- metadata.json          <- course-level metadata
    |-- english_subtitles.vtt  <- almost always a placeholder sentinel
    |-- video.txt
    |-- pdf_links.txt
    `-- do_<moduleid>/
        |-- <Module Name>.pdf          <- PDFs live here (depth 2)
        `-- <Module Name>/
            |-- metadata.json
            |-- english_subtitles.vtt  <- real transcript lives here
            `-- pdf_links.txt

Usage:
    python tools/inventory.py --staging <dir> --out <manifest.jsonl>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Extraction primitives are shared with meta_gen.py via course_io so the two
# cannot drift apart on sentinel handling or VTT cleaning.
from course_io import (  # noqa: E402
    VIDEO_SENTINELS,
    VTT_SENTINELS,
    clean_vtt_text,
    find_course_dirs,
    is_sentinel,
    load_json,
    read_text as _read_text,
    vtt_is_real,
    vtt_span_seconds,
)


def scan_course(course_dir: Path) -> Dict[str, Any]:
    """Walk one course folder and summarise everything usable inside it."""
    meta = load_json(course_dir / "metadata.json")

    modules: List[Dict[str, Any]] = []
    transcript_chars = 0
    transcript_seconds = 0.0

    # Module transcripts: any VTT below the course root, excluding the course-root
    # placeholder itself. Sorted for stable module ordering across runs.
    for vtt_path in sorted(course_dir.rglob("english_subtitles.vtt")):
        if vtt_path.parent == course_dir:
            continue
        text = _read_text(vtt_path)
        if not vtt_is_real(text):
            continue
        cleaned = clean_vtt_text(text)
        span = vtt_span_seconds(text)
        transcript_chars += len(cleaned)
        transcript_seconds += span
        modules.append(
            {
                "module_dir": str(vtt_path.parent.relative_to(course_dir)),
                "module_meta": load_json(vtt_path.parent / "metadata.json"),
                "transcript_chars": len(cleaned),
                "span_seconds": round(span, 1),
            }
        )

    # PDFs sit at do_<moduleid>/<Name>.pdf -- a sibling of the module dir, so a
    # top-level glob('*.pdf') finds nothing. rglob is required. A shareable input
    # bundle carries "<name>.pdf.txt" sidecars instead of the binaries, so count
    # those too or every such course drops out of the pdf_only tier.
    pdfs = [p for p in sorted(course_dir.rglob("*.pdf")) if p.is_file()]
    sidecars = [p for p in sorted(course_dir.rglob("*.pdf.txt")) if p.is_file()]
    have = {p.name for p in sidecars}
    pdfs = [p for p in pdfs if f"{p.name}.txt" not in have] + sidecars

    # pdf_links.txt holds remote portal URLs for PDFs not shipped in the zip.
    pdf_links: List[str] = []
    for link_file in sorted(course_dir.rglob("pdf_links.txt")):
        for line in _read_text(link_file).splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.search(r"https?://\S+", line)
            if m:
                pdf_links.append(m.group(0))

    root_vtt = _read_text(course_dir / "english_subtitles.vtt")
    video_txt = _read_text(course_dir / "video.txt")

    return {
        "course_id": course_dir.name,
        "path": str(course_dir),
        "metadata_present": meta is not None,
        "identifier": (meta or {}).get("identifier"),
        "name": (meta or {}).get("name"),
        "organisation": (meta or {}).get("organisation"),
        "competencies_v6": (meta or {}).get("competencies_v6"),
        "course_category": (meta or {}).get("courseCategory"),
        "scorm": (meta or {}).get("scorm"),
        "has_instructions": bool((meta or {}).get("instructions")),
        "keyword_count": len((meta or {}).get("keywords") or []),
        "description_chars": len((meta or {}).get("description") or ""),
        "module_count": len(list(course_dir.glob("do_*"))),
        "modules_with_transcript": len(modules),
        "transcript_chars": transcript_chars,
        "transcript_seconds": round(transcript_seconds, 1),
        "pdf_count": len(pdfs),
        "pdf_link_count": len(pdf_links),
        "root_vtt_is_sentinel": not vtt_is_real(root_vtt),
        "video_txt_is_sentinel": is_sentinel(video_txt, VIDEO_SENTINELS),
        "modules": modules,
    }


def content_tier(rec: Dict[str, Any]) -> str:
    """How much real evidence the LLM will have for this course."""
    if rec["transcript_chars"] >= 2000:
        return "transcript"
    if rec["pdf_count"] > 0:
        return "pdf_only"
    if rec["transcript_chars"] > 0:
        return "thin_transcript"
    if rec["pdf_link_count"] > 0:
        return "links_only"
    return "metadata_only"


def richness(rec: Dict[str, Any]) -> tuple:
    """Sort key for choosing between duplicate copies of the same course."""
    return (
        rec["transcript_chars"],
        rec["pdf_count"],
        rec["modules_with_transcript"],
        rec["description_chars"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    course_dirs = find_course_dirs(args.staging)
    print(f"scanning {len(course_dirs)} course folders...")

    records: List[Dict[str, Any]] = []
    for i, d in enumerate(course_dirs, 1):
        records.append(scan_course(d))
        if i % 500 == 0:
            print(f"  {i}/{len(course_dirs)}")

    # Deduplicate: same course id can appear in several overlapping batch zips
    # (batch_1_to_79.zip and "batch_1_to_79 (1).zip" etc). Keep the richest copy.
    best: Dict[str, Dict[str, Any]] = {}
    dupes: Counter = Counter()
    for rec in records:
        cid = rec["course_id"]
        dupes[cid] += 1
        if cid not in best or richness(rec) > richness(best[cid]):
            best[cid] = rec

    deduped = sorted(best.values(), key=lambda r: r["course_id"])
    for rec in deduped:
        rec["tier"] = content_tier(rec)
        rec["duplicate_copies"] = dupes[rec["course_id"]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for rec in deduped:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- census ----
    tiers = Counter(r["tier"] for r in deduped)
    total = len(deduped)
    print(f"\n{'='*62}\nCOVERAGE CENSUS  (deduped courses: {total})\n{'='*62}")
    print(f"course folders scanned      : {len(records)}")
    print(f"duplicate ids collapsed     : {sum(v-1 for v in dupes.values() if v > 1)}")
    print(f"\ncontent tier:")
    for tier, n in tiers.most_common():
        print(f"  {tier:<16} {n:>6}  ({n/total*100:5.1f}%)")

    with_tr = [r for r in deduped if r["transcript_chars"] > 0]
    print(f"\ntranscript:")
    print(f"  courses with any transcript : {len(with_tr)} ({len(with_tr)/total*100:.1f}%)")
    if with_tr:
        chars = sorted(r["transcript_chars"] for r in with_tr)
        print(f"  median chars                : {chars[len(chars)//2]:,}")
        print(f"  total chars                 : {sum(chars):,}")
        hrs = sum(r['transcript_seconds'] for r in with_tr) / 3600
        print(f"  total derived duration      : {hrs:,.1f} hours")
    print(f"  root VTT was a sentinel     : {sum(r['root_vtt_is_sentinel'] for r in deduped)}/{total}")

    print(f"\npdf:")
    print(f"  courses with local PDFs     : {sum(1 for r in deduped if r['pdf_count'])}")
    print(f"  courses with only pdf links : {sum(1 for r in deduped if not r['pdf_count'] and r['pdf_link_count'])}")
    print(f"  total local PDFs            : {sum(r['pdf_count'] for r in deduped):,}")

    print(f"\nmetadata:")
    print(f"  metadata.json missing       : {sum(1 for r in deduped if not r['metadata_present'])}")
    print(f"  has instructions field      : {sum(1 for r in deduped if r['has_instructions'])}")
    print(f"  scorm=true (in non-scorm!)  : {sum(1 for r in deduped if r['scorm'] is True)}")
    print(f"  competencies_v6 values      : {dict(Counter(r['competencies_v6'] for r in deduped).most_common(8))}")
    print(f"  courseCategory values       : {dict(Counter(r['course_category'] for r in deduped).most_common(8))}")
    print(f"\nmanifest written -> {args.out}")


if __name__ == "__main__":
    main()
