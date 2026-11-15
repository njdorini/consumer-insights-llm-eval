"""
Step 3: blind human grading in your browser.

Shows one response at a time with its task and rubric. Model names and the
judge's verdicts are hidden so neither can bias you. Responses come in a
shuffled (but fixed) order so models are interleaved.

Output: results/human_scores.csv, one row per (response, criterion).
Every "Save & next" writes to disk, so you can stop and resume any time.

Usage:
    streamlit run grade_app.py
"""

import hashlib
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src.llm import RESULTS_DIR, load_prompts

RESPONSES = RESULTS_DIR / "responses.csv"
OUT = RESULTS_DIR / "human_scores.csv"
COLUMNS = ["response_id", "prompt_id", "category", "criterion_id",
           "human_verdict", "human_note", "graded_at"]

st.set_page_config(page_title="Blind grading", layout="wide")


# ---------- data ----------
@st.cache_data
def load_queue() -> list[dict]:
    df = pd.read_csv(RESPONSES, keep_default_na=False)
    df = df[df["error"] == ""]
    # Only the columns needed to grade: the model name is deliberately dropped.
    rows = df[["response_id", "prompt_id", "category", "response"]].to_dict("records")
    # Fixed pseudo-random order based on the id: same order every session.
    return sorted(rows, key=lambda r: hashlib.md5(r["response_id"].encode()).hexdigest())


def load_scores() -> pd.DataFrame:
    if OUT.exists():
        return pd.read_csv(OUT, keep_default_na=False)
    return pd.DataFrame(columns=COLUMNS)


def save_scores(item: dict, verdicts: dict, note: str):
    scores = load_scores()
    scores = scores[scores["response_id"] != item["response_id"]]  # overwrite if re-graded
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new = pd.DataFrame([dict(
        response_id=item["response_id"], prompt_id=item["prompt_id"],
        category=item["category"], criterion_id=cid, human_verdict=v,
        human_note=note.strip(), graded_at=now,
    ) for cid, v in verdicts.items()])
    pd.concat([scores, new]).to_csv(OUT, index=False)


if not RESPONSES.exists():
    st.error("No results/responses.csv yet. Run `python run_models.py` first.")
    st.stop()

queue = load_queue()
prompts = {p["id"]: p for p in load_prompts()}
scores = load_scores()
graded_ids = set(scores["response_id"])


def next_ungraded(start: int = 0) -> int:
    for offset in range(len(queue)):
        i = (start + offset) % len(queue)
        if queue[i]["response_id"] not in graded_ids:
            return i
    return start


if "idx" not in st.session_state:
    st.session_state.idx = next_ungraded()

# ---------- sidebar: progress + navigation ----------
with st.sidebar:
    n_done = sum(r["response_id"] in graded_ids for r in queue)
    st.metric("Graded", f"{n_done} / {len(queue)}")
    st.progress(n_done / len(queue) if queue else 0)

    c1, c2 = st.columns(2)
    if c1.button("← Prev", width="stretch"):
        st.session_state.idx = (st.session_state.idx - 1) % len(queue)
    if c2.button("Next →", width="stretch"):
        st.session_state.idx = (st.session_state.idx + 1) % len(queue)
    if st.button("Jump to next ungraded", width="stretch"):
        st.session_state.idx = next_ungraded(st.session_state.idx)

    render_md = st.toggle("Render formatting", value=True,
                          help="Off = show raw text exactly as the model wrote it.")
    st.caption("Model names and judge verdicts are hidden while you grade.")

if n_done == len(queue):
    st.success("All responses graded. Run `python analyze.py` for results. "
               "(You can still revisit any response with Prev / Next.)")

# ---------- main: one response ----------
item = queue[st.session_state.idx]
prompt = prompts[item["prompt_id"]]
existing = scores[scores["response_id"] == item["response_id"]]
prev_verdict = dict(zip(existing["criterion_id"], existing["human_verdict"]))
prev_note = existing["human_note"].iloc[0] if len(existing) else ""

status = "✅ graded" if item["response_id"] in graded_ids else "⏳ not graded"
st.subheader(f"Response {st.session_state.idx + 1} of {len(queue)}")
st.caption(f"{item['category'].replace('_', ' ')} · task `{item['prompt_id']}` · {status}")

left, right = st.columns([3, 2], gap="large")

with left:
    with st.expander("Task given to the model", expanded=False):
        st.markdown(prompt["prompt"])
    with st.container(height=650, border=True):
        if render_md:
            st.markdown(item["response"])
        else:
            st.text(item["response"])

with right:
    st.markdown("#### Rubric")
    # A form batches all clicks into one submit, so the page doesn't reload per click.
    with st.form(key=f"form_{item['response_id']}"):
        verdicts = {}
        for c in prompt["rubric"]:
            prev = prev_verdict.get(c["id"])
            choice = st.radio(
                f"**{c['id']}** · {c['criterion']}",
                ["Pass", "Fail"],
                index=None if prev is None else (0 if prev == "PASS" else 1),
                horizontal=True,
                key=f"{item['response_id']}_{c['id']}",
            )
            verdicts[c["id"]] = choice
        note = st.text_input("Note (optional)", value=prev_note,
                             help="Anything worth remembering, e.g. why a borderline call went the way it did.")
        submitted = st.form_submit_button("Save & next", type="primary", width="stretch")

    if submitted:
        unanswered = [cid for cid, v in verdicts.items() if v is None]
        if unanswered:
            st.error(f"Pick Pass or Fail for: {', '.join(unanswered)}")
        else:
            save_scores(item, {cid: v.upper() for cid, v in verdicts.items()}, note)
            graded_ids.add(item["response_id"])
            st.session_state.idx = next_ungraded(st.session_state.idx)
            st.rerun()
