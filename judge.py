"""
Step 2: an LLM judge scores every response against every rubric criterion.

Output: results/judge_scores.csv, one row per (response, criterion) with
PASS/FAIL and a one-line reason.

The judge is blind: it sees the task, the response and the rubric, never the
model name. Safe to re-run; responses already judged are skipped.

Usage:
    python judge.py
"""

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.llm import RESULTS_DIR, generate, load_config, load_prompts

RESPONSES = RESULTS_DIR / "responses.csv"
OUT = RESULTS_DIR / "judge_scores.csv"
COLUMNS = ["response_id", "prompt_id", "category", "model", "criterion_id",
           "criterion", "judge_verdict", "judge_reason", "judge_model"]
PARALLEL_CALLS = 4

JUDGE_SYSTEM = """You are an expert reviewer of consumer insights and brand strategy work \
(survey design, segmentation, customer journey mapping, brand tracking).

You grade a response against a rubric of independent pass/fail criteria.

Rules:
- Judge each criterion on its own. A response can be excellent overall and still FAIL a criterion, or weak overall and PASS one.
- PASS only if the response clearly and explicitly satisfies the criterion as written. If it is ambiguous, implied, or only partly met, FAIL.
- Judge only what is in the response. Do not give credit for what the author probably meant.
- Length, polish and formatting are not criteria. Ignore them unless a criterion mentions them.
- The reason must be one sentence (max 30 words), pointing to the specific part of the response (or its absence) that decided the verdict."""

JUDGE_TEMPLATE = """<task>
{prompt}
</task>

<response>
{response}
</response>

<rubric>
{rubric}
</rubric>

Grade the response against every rubric criterion."""


def verdict_schema(criterion_ids: list[str]) -> dict:
    """JSON shape the judge must return: one verdict per criterion."""
    return {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion_id": {"type": "string", "enum": criterion_ids},
                        "reason": {"type": "string"},
                        "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
                    },
                    # reason comes before verdict so the judge explains first, then decides
                    "required": ["criterion_id", "reason", "verdict"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["verdicts"],
        "additionalProperties": False,
    }


def judge_one(resp: dict, prompt: dict, judge_cfg: dict) -> list[dict]:
    rubric_text = "\n".join(f"[{c['id']}] {c['criterion']}" for c in prompt["rubric"])
    ids = [c["id"] for c in prompt["rubric"]]
    out = generate(
        judge_cfg["provider"], judge_cfg["model_id"],
        JUDGE_TEMPLATE.format(prompt=prompt["prompt"].strip(), response=resp["response"], rubric=rubric_text),
        system=JUDGE_SYSTEM, effort=judge_cfg.get("effort"),
        max_output_tokens=16000, json_schema=verdict_schema(ids),
    )
    if out["stop_reason"] not in ("end_turn",):
        raise RuntimeError(f"judge stopped early: {out['stop_reason']}")

    verdicts = {v["criterion_id"]: v for v in json.loads(out["text"])["verdicts"]}
    missing = set(ids) - set(verdicts)
    if missing:
        raise RuntimeError(f"judge skipped criteria {sorted(missing)}")

    return [dict(
        response_id=resp["response_id"], prompt_id=resp["prompt_id"],
        category=resp["category"], model=resp["model"],
        criterion_id=c["id"], criterion=c["criterion"],
        judge_verdict=verdicts[c["id"]]["verdict"],
        judge_reason=verdicts[c["id"]]["reason"].strip(),
        judge_model=judge_cfg["model_id"],
    ) for c in prompt["rubric"]]


def main():
    if not RESPONSES.exists():
        raise SystemExit("No results/responses.csv yet - run `python run_models.py` first.")
    judge_cfg = load_config()["judge"]
    prompts = {p["id"]: p for p in load_prompts()}

    responses = pd.read_csv(RESPONSES, keep_default_na=False)
    responses = responses[responses["error"] == ""]

    done = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT, keep_default_na=False)["response_id"])
    todo = [r for r in responses.to_dict("records")
            if r["response_id"] not in done and r["prompt_id"] in prompts]
    print(f"{len(responses)} responses, {len(done)} already judged, {len(todo)} to judge "
          f"(judge: {judge_cfg['model_id']})")
    if not todo:
        return

    new_file = not OUT.exists()
    failures = 0
    with open(OUT, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        with ThreadPoolExecutor(PARALLEL_CALLS) as pool:
            futures = {pool.submit(judge_one, r, prompts[r["prompt_id"]], judge_cfg): r for r in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                r = futures[fut]
                try:
                    rows = fut.result()
                except Exception as e:
                    failures += 1
                    print(f"[{i}/{len(todo)}] {r['prompt_id']:<12} ERROR {type(e).__name__}: {str(e)[:100]}")
                    continue
                writer.writerows(rows)
                f.flush()
                passed = sum(x["judge_verdict"] == "PASS" for x in rows)
                print(f"[{i}/{len(todo)}] {r['prompt_id']:<12} {passed}/{len(rows)} criteria passed")

    print(f"\nSaved to {OUT.relative_to(RESULTS_DIR.parent)}")
    if failures:
        print(f"{failures} response(s) failed to judge - run the script again to retry them.")


if __name__ == "__main__":
    main()
