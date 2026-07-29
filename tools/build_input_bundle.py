"""
Assemble every input the pipeline needs into one tidy, self-describing folder and
zip it, so it can be handed to someone offline who just wants to run the script.

    input/
    ├── README.md                     how to run, written for the recipient
    ├── env.template                  copy to .env at the repo root
    ├── masters/
    │   ├── kcm_competencies.json     Functional + Behavioural competency master
    │   ├── sgos_sectors.json         Sector -> SubSector -> Theme master
    │   └── igot_designations.csv     official designation master
    ├── manifests/
    │   ├── non-scorm.jsonl           deduplicated course inventory + coverage
    │   └── scorm.jsonl
    └── courses/
        ├── non-scorm/do_<id>/...     course folders, module structure intact
        └── scorm/do_<id>/...

Two modes:

  raw (default)  Every file from every course folder, verbatim -- PDF binaries,
                 all subtitle files including the per-video "*.mp4.vtt" ones,
                 exporter placeholders, everything. Only duplicate course folders
                 are collapsed, since the same course id appears in several
                 overlapping batch zips. ~5 GB: a faithful copy of the source.

  text-only      PDFs replaced by "<name>.pdf.txt" sidecars holding their
                 extracted text; redundant "*.mp4.vtt" files and placeholders
                 dropped. ~180 MB. A run against it behaves identically -- the
                 pipeline reads english_subtitles.vtt and caps PDF text at
                 MAX_PDF_CHARS regardless -- but it is not a faithful copy.

    python tools/build_input_bundle.py                    # raw, build + zip
    python tools/build_input_bundle.py --mode text-only   # slim, ~180 MB
    python tools/build_input_bundle.py --no-zip
    python tools/build_input_bundle.py --limit 50         # small sample bundle
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

import course_io  # noqa: E402

# Files the pipeline reads from a course folder. Anything else is left behind.
COURSE_FILES = {"metadata.json", "english_subtitles.vtt", "pdf_links.txt", "video.txt"}

MASTERS = [
    ("KCM_PATH", "kcm_competencies.json",
     "Functional + Behavioural competency master (KCM). The only valid source of "
     "those two competency types."),
    ("SGOS_PATH", "sgos_sectors.json",
     "Sector -> SubSector -> Theme master (SGOS). The only valid source of Domain "
     "classification."),
    ("DESIGNATIONS_PATH", "igot_designations.csv",
     "Official iGOT designation master (id,name). The only permitted source of "
     "Targetroles."),
]


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def discover(staging: Path) -> List[Path]:
    """Course folders under a staging dir, whether nested in batches or not."""
    return course_io.find_course_dirs(staging)


def _place(item: Path, target: Path, stats: Dict[str, int]) -> None:
    """
    Put a file in the bundle. Hardlinks when possible -- the staging tree and the
    bundle sit on one filesystem, so this is instant and costs no extra disk,
    while the zip reads content identically. Falls back to a real copy across
    devices.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(item, target)
        stats["linked"] += 1
    except (OSError, NotImplementedError):
        shutil.copy2(item, target)
        stats["copied"] += 1
    stats["files_copied"] += 1
    stats["bytes_copied"] += item.stat().st_size


def copy_course_raw(src: Path, dst: Path, stats: Dict[str, int]) -> None:
    """Copy one course folder verbatim -- every file, nothing filtered."""
    for item in src.rglob("*"):
        if item.is_file():
            _place(item, dst / item.relative_to(src), stats)


def copy_course(src: Path, dst: Path, include_pdfs: bool,
                stats: Dict[str, int]) -> None:
    """Copy one course folder, keeping only what the pipeline reads."""
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        if item.is_dir():
            continue

        target = dst / rel
        name = item.name

        if name in COURSE_FILES:
            # Drop placeholder files entirely -- they carry no information and the
            # extractor treats them as absent anyway.
            raw = course_io.read_text(item)
            if name == "english_subtitles.vtt" and not course_io.vtt_is_real(raw):
                stats["placeholders_dropped"] += 1
                continue
            if name == "video.txt" and course_io.is_sentinel(raw, course_io.VIDEO_SENTINELS):
                stats["placeholders_dropped"] += 1
                continue
            if name == "pdf_links.txt" and not raw.strip():
                stats["placeholders_dropped"] += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            stats["files_copied"] += 1
            stats["bytes_copied"] += item.stat().st_size

        elif name.lower().endswith(".pdf"):
            if include_pdfs:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
                stats["pdfs_copied"] += 1
                stats["bytes_copied"] += item.stat().st_size
            else:
                text = course_io.extract_pdf_text_sync(item, course_io.MAX_PDF_CHARS)
                if not text.strip():
                    # Scanned/image-only: no text to carry, so carry nothing.
                    stats["pdfs_no_text"] += 1
                    continue
                sidecar = target.parent / f"{name}.txt"
                sidecar.parent.mkdir(parents=True, exist_ok=True)
                sidecar.write_text(text, encoding="utf-8")
                stats["pdf_sidecars"] += 1
                stats["bytes_copied"] += len(text.encode("utf-8"))
                stats["pdf_bytes_saved"] += max(0, item.stat().st_size - len(text.encode()))


README = """# iGOT course metadata regeneration — input bundle

Everything the pipeline needs to run, in one folder. Nothing here needs the internet
except the Vertex AI calls the script itself makes.

## Contents

| Path | What it is |
|---|---|
| `masters/kcm_competencies.json` | Functional + Behavioural competency master (KCM) — the only valid source of those two competency types. |
| `masters/sgos_sectors.json` | Sector → SubSector → Theme master (SGOS) — the only valid source of Domain classification. |
| `masters/igot_designations.csv` | Official iGOT designation master (`id,name`) — the only permitted source of `Targetroles`. |
| `manifests/non-scorm.jsonl` | Deduplicated course inventory with per-course content coverage. Makes `--tier` work. |
| `manifests/scorm.jsonl` | The same, for the SCORM set. |
| `courses/non-scorm/do_<id>/` | Course folders, module structure intact. |
| `courses/scorm/do_<id>/` | The same, for SCORM courses. |
| `env.template` | Copy to `.env` at the repo root and fill in the two credential lines. |

__PDF_NOTE__

Placeholder files the exporter emits (`// No English subtitles found`,
`No video URLs found`, empty `pdf_links.txt`) have been dropped. The pipeline treats
them as absent, so their removal changes nothing and keeps the bundle clean.

## Running it

```bash
# 1. Get the code
git clone https://github.com/aswinpradeep/igot-course-metadata.git
cd igot-course-metadata
pip install asyncpg httpx pymupdf tenacity python-dotenv google-genai numpy

# 2. Point the config at this folder
cp /path/to/input/env.template .env
#    then edit .env and set:
#      GOOGLE_APPLICATION_CREDENTIALS  — your Vertex AI service-account JSON
#      GOOGLE_PROJECT_ID               — your GCP project
#      DB_DSN                          — a Postgres you can write to
#      INPUT_DIR                       — the absolute path to this input folder

# 3. Build the designation index (one-off, ~90 seconds)
python tools/build_designation_index.py

# 4. Dry run first — no AI calls, no writes, free
python meta_gen.py --limit 10

# 5. Then for real
python meta_gen.py --limit 10 --execute
```

`--execute` is required for anything to actually happen; without it every run is a
dry run. Re-running is safe: finished courses are skipped.

## Switching between the two course sets

`env.template` is set up for the non-SCORM set. For SCORM, change these two lines:

```
CONTENT_SET=scorm
COURSE_MANIFEST=${INPUT_DIR}/manifests/scorm.jsonl
```

`CONTENT_SET` keeps the two sets separate in the database, so they can be run
independently and their progress tracked separately.

## Reviewing the output

```bash
python tools/review_report.py --limit 100
```

This writes a single self-contained HTML file that can be opened straight off disk
and shared with non-technical reviewers — one card per course, with Looks good /
Needs changes / Not acceptable buttons and a comment box, exportable to CSV.

See the repository README for what each field means and how v3.5 differs from v3.4.
"""

ENV_TEMPLATE = """# =============================================================================
# iGOT Course Metadata Regeneration — configuration
# Copy this file to .env at the root of the igot-course-metadata repository.
# =============================================================================

# ── The one path you must set ────────────────────────────────────────────────
# Absolute path to the unzipped input folder that contained this template.
INPUT_DIR=/absolute/path/to/input

# ── Credentials you must supply ──────────────────────────────────────────────
# Vertex AI service-account JSON, needs the "Vertex AI User" role.
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
GOOGLE_PROJECT_ID=your-gcp-project-id

# A Postgres you can write to. Tables are created automatically on first run.
# Must be a plain libpq DSN — not the SQLAlchemy "postgresql+asyncpg://" form.
DB_DSN=postgresql://user:password@localhost:5432/igot_meta

# ── Model ────────────────────────────────────────────────────────────────────
# gemini-3.1-pro-preview is served ONLY from the "global" endpoint; a regional
# location returns 404. Do not drop below pro class — flash-class models are a
# marked quality regression on the ~55k-token master prompt.
META_GEN_MODEL=gemini-3.1-pro-preview
META_GEN_LOCATION=global

# ── Everything below resolves from INPUT_DIR; no need to edit ────────────────
KCM_PATH=${INPUT_DIR}/masters/kcm_competencies.json
SGOS_PATH=${INPUT_DIR}/masters/sgos_sectors.json
DESIGNATIONS_PATH=${INPUT_DIR}/masters/igot_designations.csv

COURSES_BASE_PATH=${INPUT_DIR}/courses/non-scorm
COURSE_MANIFEST=${INPUT_DIR}/manifests/non-scorm.jsonl
CONTENT_SET=non-scorm

DESIGNATION_INDEX_PATH=data/designation_index.npz

# ── Optional: platform-verified language / duration / difficulty ─────────────
# Leave blank unless you have access to the enrichment database. Without it,
# duration falls back to subtitle timings, which underestimate.
ENRICHMENT_DSN=

# ── Throughput ───────────────────────────────────────────────────────────────
MAX_CONCURRENCY=4
WORKER_BATCH_SIZE=20
MAX_ATTEMPTS=3
DESIGNATION_CANDIDATES=150
LLM_TIMEOUT_SECONDS_META=300
STALE_CLAIM_MINUTES=30

# ── Generation tuning ────────────────────────────────────────────────────────
LLM_TEMPERATURE=0.1
LLM_MAX_OUTPUT_TOKENS=32768
MAX_TRANSCRIPT_CHARS=400000
MAX_PDF_CHARS=200000
MAX_PDFS_PER_COURSE=40

# PDFs whose text layer is unusable (scanned, or legacy non-Unicode Indic fonts
# that extract as ASCII gibberish) are sent to the model as files instead, which
# reads them correctly. Rendering pages costs ~30x a text layer, so bound it.
MAX_NATIVE_PDFS=3
MAX_NATIVE_PDF_MB=10
MAX_NATIVE_PDF_PAGES=50

DESIGNATION_EMBED_MODEL=text-embedding-005
DESIGNATION_EMBED_DIM=768
DESIGNATION_EMBED_BATCH=250

# ── Output contract ──────────────────────────────────────────────────────────
PROMPT_VERSION=v3.5-advanced
GENERATOR_NAME=iGOT Metadata Regeneration Engine (Vertex AI / Gemini)
BEHAVIOURAL_SPELLING=Behavioural
LEARNING_OUTCOME_STYLE=unlabelled
RUBRIC_WEIGHTS=
RUBRIC_BEGINNER_MAX=45
RUBRIC_INTERMEDIATE_MAX=75

# ── Pricing, for the run cost report only ────────────────────────────────────
PRICE_INPUT_PER_M=2.00
PRICE_OUTPUT_PER_M=12.00
PRICE_CACHED_INPUT_PER_M=0.20

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR=logs
META_GEN_LOG_LEVEL=INFO
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "input")
    ap.add_argument("--mode", choices=["raw", "text-only"], default="raw",
                    help="raw (default): every file verbatim, ~5 GB. "
                         "text-only: PDF text sidecars, placeholders dropped, ~180 MB")
    ap.add_argument("--limit", type=int, default=None,
                    help="courses per set, for a small sample bundle")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--zip-out", type=Path, default=None)
    ap.add_argument("--compress-level", type=int, default=None,
                    help="deflate level 0-9 (default 1 for raw, 6 for text-only)")
    args = ap.parse_args()

    sets: List[Tuple[str, Path]] = []
    for name, env in (("non-scorm", "COURSES_BASE_PATH"), ("scorm", "SCORM_BASE_PATH")):
        # Both sets live beside each other under course_data/extracted/<set>/_staging.
        default = ROOT / "course_data" / "extracted" / name / "_staging"
        path = Path(os.environ.get(env, "")) if name == "non-scorm" else default
        if name == "non-scorm" and not discover(path):
            path = default
        if discover(path):
            sets.append((name, path))
        else:
            print(f"  ! no courses found for '{name}' at {path} — skipping")
    if not sets:
        print("No extracted course folders found. Extract the zips first "
              "(see the repository README, Quick start).")
        return 1

    out = args.out
    if out.exists():
        print(f"removing previous bundle at {out}")
        shutil.rmtree(out)
    (out / "masters").mkdir(parents=True)
    (out / "manifests").mkdir(parents=True)

    # ---- masters -----------------------------------------------------------
    print("\nmasters:")
    for env, target, _desc in MASTERS:
        src = Path(os.environ.get(env, ""))
        if not src.is_file():
            print(f"  ! {env} not found ({src}) — bundle will be incomplete")
            continue
        shutil.copy2(src, out / "masters" / target)
        print(f"  {target:<26} {human(src.stat().st_size):>10}  <- {src.name}")

    # ---- courses -----------------------------------------------------------
    totals: Dict[str, int] = {}
    for name, staging in sets:
        courses = discover(staging)
        if args.limit:
            courses = courses[: args.limit]
        dest = out / "courses" / name
        dest.mkdir(parents=True, exist_ok=True)
        stats: Dict[str, int] = {k: 0 for k in (
            "files_copied", "bytes_copied", "pdfs_copied", "pdf_sidecars",
            "pdfs_no_text", "pdf_bytes_saved", "placeholders_dropped",
            "linked", "copied")}

        # Deduplicate by course id: the same course appears in several
        # overlapping batch zips, and writing each into dest/<id> would otherwise
        # just overwrite. Keep one copy per id, chosen deterministically.
        unique: Dict[str, Path] = {}
        for c in courses:
            unique.setdefault(c.name, c)
        dupes = len(courses) - len(unique)

        print(f"\ncourses/{name}: {len(courses)} folder(s) -> {len(unique)} unique"
              + (f" ({dupes} duplicate id(s) collapsed)" if dupes else ""))
        for i, (cid, c) in enumerate(sorted(unique.items()), 1):
            if args.mode == "raw":
                copy_course_raw(c, dest / cid, stats)
            else:
                copy_course(c, dest / cid, False, stats)
            if i % 500 == 0:
                print(f"  {i}/{len(unique)}")

        print(f"  files included    : {stats['files_copied']:,} "
              f"({human(stats['bytes_copied'])})")
        if stats["linked"]:
            print(f"  hardlinked        : {stats['linked']:,} (no extra disk used)")
        if args.mode == "text-only":
            print(f"  placeholders drop : {stats['placeholders_dropped']:,}")
            print(f"  PDF text sidecars : {stats['pdf_sidecars']:,}")
            print(f"  PDFs with no text : {stats['pdfs_no_text']:,} (scanned/image-only, omitted)")
            print(f"  bytes saved       : {human(stats['pdf_bytes_saved'])}")
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v

    # ---- manifests ---------------------------------------------------------
    print("\nmanifests:")
    for name, _ in sets:
        target = out / "manifests" / f"{name}.jsonl"
        rc = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "inventory.py"),
             "--staging", str(out / "courses" / name), "--out", str(target)],
            capture_output=True, text=True,
        )
        if rc.returncode != 0:
            print(f"  ! inventory failed for {name}: {rc.stderr.strip()[:200]}")
            continue
        n = sum(1 for _ in target.open(encoding="utf-8"))
        tail = [l for l in rc.stdout.splitlines() if "content tier" in l or "%)" in l]
        print(f"  {name}.jsonl: {n:,} courses")
        for line in tail[:6]:
            print(f"    {line.strip()}")

    # ---- docs --------------------------------------------------------------
    pdf_note = (
        "**This is a raw bundle: every file from every course folder, verbatim** — PDF\n"
        "binaries, all subtitle files including the per-video `*.mp4.vtt` ones, and the\n"
        "exporter's placeholder files. The only thing removed is duplicate course folders:\n"
        "the same course id appears in several of the overlapping source batch zips, so one\n"
        "copy of each is kept.\n\n"
        "The pipeline itself reads `metadata.json`, `english_subtitles.vtt`, `*.pdf` and\n"
        "`pdf_links.txt`, and ignores the rest — but nothing has been withheld."
        if args.mode == "raw" else
        "**PDFs are shipped as `<name>.pdf.txt` sidecars holding their extracted text,**\n"
        "not as the original binaries, and the redundant per-video `*.mp4.vtt` subtitle\n"
        "files and exporter placeholders have been dropped. The PDFs are 97% of the raw\n"
        "bytes and the largest are scanned images yielding almost no text — one 321 MB,\n"
        "25-page file produced about 2,900 characters. The pipeline caps PDF text at\n"
        "`MAX_PDF_CHARS` regardless and prefers a sidecar when it finds one, so a run\n"
        "behaves identically. Rebuild without `--mode text-only` for a faithful copy."
    )
    (out / "README.md").write_text(README.replace("__PDF_NOTE__", pdf_note), encoding="utf-8")
    (out / "env.template").write_text(ENV_TEMPLATE, encoding="utf-8")

    size = dir_size(out)
    print(f"\nbundle: {out}  ({human(size)}, "
          f"{sum(1 for f in out.rglob('*') if f.is_file()):,} files)")

    # ---- zip ---------------------------------------------------------------
    if args.no_zip:
        return 0
    zip_path = args.zip_out or out.with_suffix(".zip")
    # Level 1 for a raw bundle: it is nearly all PDF, which is already compressed
    # internally, so higher levels cost minutes of CPU for almost no gain. The
    # text-only bundle is all text, where level 6 roughly halves it.
    level = args.compress_level if args.compress_level is not None else (
        1 if args.mode == "raw" else 6
    )
    print(f"zipping -> {zip_path} (deflate level {level}; this takes a few minutes)")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=level) as z:
        files = [f for f in sorted(out.rglob("*")) if f.is_file()]
        for i, f in enumerate(files, 1):
            z.write(f, f.relative_to(out.parent))
            if i % 20000 == 0:
                print(f"  {i:,}/{len(files):,}")
    zsize = zip_path.stat().st_size
    print(f"\n{zip_path.name}: {human(zsize)} "
          f"({zsize/size*100:.0f}% of folder size) — share this file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
