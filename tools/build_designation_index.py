"""
One-off builder for the designation embedding index used by Targetroles (v3.5).

Embeds all 19,936 rows of the iGOT designation master with Vertex
text-embedding-005 and caches them to an .npz. Takes about 90 seconds and only
needs re-running when the master CSV changes.

    python tools/build_designation_index.py
    python tools/build_designation_index.py --out data/designation_index.npz --force
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from designations import DesignationIndex  # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=None, help="designation master CSV")
    ap.add_argument("--out", type=Path, default=Path("data/designation_index.npz"))
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("build-designation-index")

    csv_path = args.csv or Path(os.environ.get("DESIGNATIONS_PATH", ""))
    if not csv_path or not csv_path.is_file():
        log.error("designation master not found: %r (set DESIGNATIONS_PATH or pass --csv)", str(csv_path))
        return 1

    out = args.out
    if out.is_file() and not args.force:
        existing = DesignationIndex.load(out)
        log.info("index already exists at %s with %d entries; pass --force to rebuild",
                 out, len(existing.ids))
        return 0

    project = os.environ.get("GOOGLE_PROJECT_ID")
    if not project:
        log.error("GOOGLE_PROJECT_ID is not set")
        return 1
    location = os.environ.get("META_GEN_LOCATION") or os.environ.get("GOOGLE_LOCATION", "global")

    from google import genai

    client = genai.Client(project=project, location=location, vertexai=True)
    index = await DesignationIndex.build(
        csv_path=csv_path, client=client, cache_path=out, concurrency=args.concurrency
    )
    log.info("done: %d designations, matrix %s -> %s", len(index.ids), index.matrix.shape, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
