"""
Step 1: send every prompt in data/prompts.yaml to every model in config.yaml.

Output: results/responses.csv, one row per (prompt, model).

Safe to re-run: rows that already succeeded are skipped, so if it crashes
halfway (network blip, rate limit) just run it again. To regenerate one
model's answers, delete that model's rows from the CSV (or the whole file).

Usage:
    python run_models.py
"""

import csv
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from src.llm import RESULTS_DIR, generate, load_config, load_prompts

OUT = RESULTS_DIR / "responses.csv"
COLUMNS = [
    "response_id",   # random id: lets us grade blind without revealing the model
    "prompt_id", "category", "model", "model_id",
    "response", "stop_reason", "input_tokens", "output_tokens",
    "latency_s", "error", "created_at",
]
PARALLEL_CALLS = 4  # how many API calls run at the same time


def already_done() -> set:
    """(prompt_id, model) pairs that already have a successful response."""
    if not OUT.exists() or OUT.stat().st_size == 0:
        return set()
    with open(OUT, newline="") as f:
        return {(r["prompt_id"], r["model"]) for r in csv.DictReader(f) if not r["error"]}


def drop_failed_rows():
    """Remove earlier failed rows so they can be retried cleanly."""
    if not OUT.exists() or OUT.stat().st_size == 0:
        return
    with open(OUT, newline="") as f:
        rows = [r for r in csv.DictReader(f) if not r["error"]]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def run_one(prompt: dict, model: dict, max_tokens: int) -> dict:
    row = dict(
        response_id=uuid.uuid4().hex[:10],
        prompt_id=prompt["id"],
        category=prompt["category"],
        model=model["name"],
        model_id=model["model_id"],
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        error="",
    )
    try:
        # No system prompt: every model gets the identical bare task.
        out = generate(model["provider"], model["model_id"], prompt["prompt"],
                       effort=model.get("effort"), max_output_tokens=max_tokens)
        out["response"] = out.pop("text")  # generate() calls it "text"; the CSV column is "response"
        row.update(out)
    except Exception as e:  # record the failure and keep going
        row["error"] = f"{type(e).__name__}: {e}"[:500]
    return row


def main():
    cfg = load_config()
    prompts = load_prompts()
    RESULTS_DIR.mkdir(exist_ok=True)
    drop_failed_rows()
    done = already_done()

    jobs = [(p, m) for p in prompts for m in cfg["models"] if (p["id"], m["name"]) not in done]
    total = len(prompts) * len(cfg["models"])
    print(f"{len(prompts)} prompts x {len(cfg['models'])} models = {total} responses "
          f"({total - len(jobs)} already done, {len(jobs)} to run)")
    if not jobs:
        return

    new_file = not OUT.exists() or OUT.stat().st_size == 0
    failures = 0
    with open(OUT, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        with ThreadPoolExecutor(PARALLEL_CALLS) as pool:
            futures = [pool.submit(run_one, p, m, cfg["max_output_tokens"]) for p, m in jobs]
            try:
                for i, fut in enumerate(as_completed(futures), 1):
                    row = fut.result()
                    writer.writerow(row)
                    f.flush()  # save each row immediately
                    if row["error"]:
                        failures += 1
                        status = "ERROR " + row["error"][:80]
                    elif row["stop_reason"] not in ("end_turn", "completed"):
                        status = f"WARNING stop_reason={row['stop_reason']}"
                    else:
                        status = f"ok ({row['output_tokens']} tokens, {row['latency_s']}s)"
                    print(f"[{i}/{len(jobs)}] {row['prompt_id']:<12} {row['model']:<20} {status}")
            except BaseException:
                # On a crash or Ctrl+C, cancel queued calls instead of silently
                # finishing (and paying for) all of them before exiting.
                pool.shutdown(wait=False, cancel_futures=True)
                raise

    print(f"\nSaved to {OUT.relative_to(RESULTS_DIR.parent)}")
    if failures:
        print(f"{failures} call(s) failed - run the script again to retry them.")


if __name__ == "__main__":
    main()
