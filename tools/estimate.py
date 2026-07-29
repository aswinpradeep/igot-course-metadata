"""
Project run time and cost for the remaining courses, from measured data.

Reads what has actually been generated so far out of course_metadata_regenerated
and combines it with the tier mix in the manifest, so the projection is grounded
in observed latency and token counts rather than guesses. Re-run it as more
courses complete and the estimate tightens.

    python tools/estimate.py
    python tools/estimate.py --concurrency 8
    python tools/estimate.py --version v3.5-advanced

Note on the two different "per course" times:
  * latency      — how long one course takes end to end.
  * effective    — wall clock divided by courses done, i.e. what actually
                   governs how long the full run takes at a given concurrency.
Latency divided by concurrency only equals effective time if nothing upstream is
throttling. Comparing the two is the quickest way to see whether adding
concurrency is buying anything.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import asyncpg  # noqa: E402


def _f(env: str, default: str) -> float:
    return float(os.environ.get(env, default))


PRICE_IN = _f("PRICE_INPUT_PER_M", "2.00")
PRICE_OUT = _f("PRICE_OUTPUT_PER_M", "12.00")
PRICE_CACHED = _f("PRICE_CACHED_INPUT_PER_M", "0.20")

# Completion timestamps, used to measure real wall-clock throughput. Dividing
# latency by concurrency assumes perfect scaling, which is not what happens here:
# per-course latency inflates as concurrency rises (upstream throttling and
# stalled calls), so the arithmetic understates the full-run time. Measured
# throughput is the honest basis.
THROUGHPUT_QUERY = """
SELECT regenerated_at
  FROM course_metadata_regenerated
 WHERE llm_prompt_version = $1
 ORDER BY regenerated_at
"""

QUERY = """
SELECT evidence_tier,
       count(*)                                                          AS n,
       avg(processing_duration_seconds)                                  AS avg_latency,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY processing_duration_seconds) AS med_latency,
       avg((llm_usage_json->>'prompt_token_count')::numeric)             AS avg_in,
       avg((llm_usage_json->>'candidates_token_count')::numeric)         AS avg_out,
       avg(coalesce((llm_usage_json->>'thoughts_token_count')::numeric, 0))       AS avg_think,
       avg(coalesce((llm_usage_json->>'cached_content_token_count')::numeric, 0)) AS avg_cached
  FROM course_metadata_regenerated
 WHERE llm_prompt_version = $1
 GROUP BY 1
"""


def cost_per_course(avg_in: float, avg_out: float, avg_think: float, avg_cached: float) -> float:
    uncached = max(0.0, avg_in - avg_cached)
    return (
        uncached / 1e6 * PRICE_IN
        + avg_cached / 1e6 * PRICE_CACHED
        + (avg_out + avg_think) / 1e6 * PRICE_OUT
    )


def fmt_hours(seconds: float) -> str:
    h = seconds / 3600
    if h < 1:
        return f"{seconds/60:.0f} min"
    if h < 48:
        return f"{h:.1f} h"
    return f"{h/24:.1f} days ({h:.0f} h)"


def tier_mix(manifest: Optional[str]) -> Counter:
    if not manifest or not Path(manifest).is_file():
        return Counter()
    counts: Counter = Counter()
    with open(manifest, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                counts[json.loads(line).get("tier", "unknown")] += 1
    return counts


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=os.environ.get("PROMPT_VERSION", "v3.5-advanced"))
    ap.add_argument("--concurrency", type=int,
                    default=int(os.environ.get("MAX_CONCURRENCY", "4")))
    ap.add_argument("--manifest", default=os.environ.get("COURSE_MANIFEST"))
    args = ap.parse_args()

    conn = await asyncpg.connect(dsn=os.environ["DB_DSN"])
    try:
        rows = await conn.fetch(QUERY, args.version)
        stamps = [r["regenerated_at"] for r in await conn.fetch(THROUGHPUT_QUERY, args.version)]
        outstanding = int(await conn.fetchval(
            "SELECT count(*) FROM course_processing_checkpoint "
            " WHERE content_set = $1 AND status IN ('pending','failed') "
            "   AND attempts < $2",
            os.environ.get("CONTENT_SET", "non-scorm"),
            int(os.environ.get("MAX_ATTEMPTS", "3")),
        ) or 0)
    finally:
        await conn.close()

    if not rows:
        print(f"No completed courses for {args.version} yet — nothing to project from.")
        return 1

    measured = {r["evidence_tier"] or "unknown": r for r in rows}
    total_done = sum(int(r["n"]) for r in rows)

    print(f"\nMEASURED so far — {args.version}, {total_done} course(s)")
    print(f"{'tier':<16}{'n':>5}{'latency':>10}{'median':>9}"
          f"{'in':>10}{'out':>8}{'think':>8}{'$/course':>10}")
    print("-" * 76)
    for tier, r in sorted(measured.items()):
        c = cost_per_course(float(r["avg_in"]), float(r["avg_out"]),
                            float(r["avg_think"]), float(r["avg_cached"]))
        print(f"{tier:<16}{int(r['n']):>5}{float(r['avg_latency']):>9.0f}s"
              f"{float(r['med_latency']):>8.0f}s{float(r['avg_in']):>10,.0f}"
              f"{float(r['avg_out']):>8,.0f}{float(r['avg_think']):>8,.0f}{c:>10.4f}")

    # Blended figures, weighted by how many of each tier were actually measured.
    blend_latency = sum(float(r["avg_latency"]) * int(r["n"]) for r in rows) / total_done
    blend_cost = sum(
        cost_per_course(float(r["avg_in"]), float(r["avg_out"]),
                        float(r["avg_think"]), float(r["avg_cached"])) * int(r["n"])
        for r in rows
    ) / total_done

    print(f"\nblended: {blend_latency:.0f}s per course, ${blend_cost:.4f} per course")

    mix = tier_mix(args.manifest)
    if mix:
        print(f"\nTIER MIX in the manifest ({sum(mix.values()):,} courses)")
        for tier, n in mix.most_common():
            marker = "" if tier in measured else "   <- not yet measured; blended figures used"
            print(f"  {tier:<16}{n:>6,} ({n/sum(mix.values())*100:4.1f}%){marker}")

    # Cost is per-course and additive, so the tier mix refines it; time depends on
    # concurrency and is taken from measured latency.
    if mix:
        projected_cost = sum(
            n * cost_per_course(
                float(measured[t]["avg_in"]), float(measured[t]["avg_out"]),
                float(measured[t]["avg_think"]), float(measured[t]["avg_cached"]),
            ) if t in measured else n * blend_cost
            for t, n in mix.items()
        )
        full_n = sum(mix.values())
    else:
        projected_cost, full_n = blend_cost * outstanding, outstanding

    # Split completions into runs on a >5 min idle gap, then take the largest run
    # as the throughput sample (small runs are dominated by startup).
    runs: List[List[Any]] = []
    for ts in stamps:
        if runs and (ts - runs[-1][-1]).total_seconds() <= 300:
            runs[-1].append(ts)
        else:
            runs.append([ts])
    best = max(runs, key=len) if runs else []
    measured_per_hour = None
    if len(best) >= 4:
        span = (best[-1] - best[0]).total_seconds()
        if span > 0:
            # n-1 intervals between n completions.
            measured_per_hour = (len(best) - 1) / (span / 3600)

    print(f"\nPROJECTION")
    print(f"  outstanding in queue        : {outstanding:,}")

    if measured_per_hour:
        secs = full_n / measured_per_hour * 3600
        print(f"  MEASURED throughput         : {measured_per_hour:.0f} courses/hour "
              f"(from a {len(best)}-course run)")
        print(f"  full {full_n:,} at that rate    : {fmt_hours(secs):>18}   <- use this")
        print()
        print("  Theoretical, if concurrency scaled perfectly (it does not — see below):")
    for c in sorted({1, 2, 4, 8, args.concurrency}):
        secs = blend_latency * full_n / c
        print(f"    full {full_n:,} @ concurrency {c:<3}: {fmt_hours(secs):>18}"
              + ("   <- current setting" if c == args.concurrency else ""))

    if measured_per_hour:
        implied = blend_latency * full_n / args.concurrency / 3600
        actual = full_n / measured_per_hour
        if actual > implied * 1.25:
            print(f"\n  Measured throughput is {actual/implied:.1f}x slower than concurrency "
                  f"{args.concurrency} predicts.")
            print("  Per-course latency rises with concurrency, so raising --batch-size buys")
            print("  much less than proportionally. Check project quota for the model before")
            print("  scaling up, and expect stalled calls to hit LLM_TIMEOUT_SECONDS_META.")

    print(f"\n  full-run cost               : ${projected_cost:,.2f}")
    print(f"  outstanding-only cost       : ${blend_cost * outstanding:,.2f}")
    print("  (cost is per-course and unaffected by concurrency)")

    cached = sum(float(r["avg_cached"]) * int(r["n"]) for r in rows) / total_done
    if cached < 1:
        avg_in = sum(float(r["avg_in"]) * int(r["n"]) for r in rows) / total_done
        saving = (avg_in / 1e6 * PRICE_IN) - (avg_in / 1e6 * PRICE_CACHED)
        print(f"\n  No cache hits measured. The static prefix (system prompt + KCM + SGOS)")
        print(f"  is byte-identical on every call and is most of the {avg_in:,.0f}-token input.")
        print(f"  Serving it from cache would save up to ${saving * full_n:,.2f} on a full run")
        print(f"  ({saving / blend_cost * 100:.0f}% of spend).")

    print(f"\n  Prices used: ${PRICE_IN:.2f} in / ${PRICE_OUT:.2f} out / "
          f"${PRICE_CACHED:.2f} cached per 1M tokens.")
    print("  Thinking tokens are billed at the output rate and are included above.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
