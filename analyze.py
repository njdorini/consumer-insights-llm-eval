"""
Step 4: analyze the results.

Reads responses.csv, judge_scores.csv and (if present) human_scores.csv and
writes:
  results/summary_by_model.csv           pass rates, tokens, latency per model
  results/summary_by_model_category.csv  pass rates per model x category
  results/agreement.csv                  human-vs-judge % agreement + Cohen's kappa
  results/disagreements.csv              every criterion where you and the judge differ
  results/charts/*.png                   3 charts

"Pass rate" = share of rubric criteria passed. Your (human) grades are treated
as ground truth; the judge is what's being checked. Works before you've
graded anything too (judge-only numbers).

Usage:
    python analyze.py
"""

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import cohen_kappa_score

from src.llm import RESULTS_DIR

CHARTS = RESULTS_DIR / "charts"

# ---- chart style: one consistent, quiet look ----
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HUMAN_COLOR, JUDGE_COLOR = "#2a78d6", "#eb6834"  # blue = you, orange = LLM judge
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10.5,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def pct(x):
    return f"{x:.0%}" if pd.notna(x) else "-"


def kappa(a, b):
    """Cohen's kappa; undefined (NaN) if both raters gave only one label."""
    if len(set(a) | set(b)) < 2:
        return float("nan")
    return cohen_kappa_score(a, b)


def load():
    resp = pd.read_csv(RESULTS_DIR / "responses.csv", keep_default_na=False)
    resp = resp[resp["error"] == ""]
    judge = pd.read_csv(RESULTS_DIR / "judge_scores.csv", keep_default_na=False)
    df = judge.copy()
    df["judge_pass"] = (df["judge_verdict"] == "PASS").astype(int)

    human_path = RESULTS_DIR / "human_scores.csv"
    if human_path.exists():
        human = pd.read_csv(human_path, keep_default_na=False)
        df = df.merge(human[["response_id", "criterion_id", "human_verdict", "human_note"]],
                      on=["response_id", "criterion_id"], how="left")
    else:
        df["human_verdict"], df["human_note"] = np.nan, ""
    df["human_verdict"] = df["human_verdict"].replace("", np.nan)
    df["human_pass"] = df["human_verdict"].map({"PASS": 1, "FAIL": 0})
    return resp, df


def summarize(resp, df):
    by_model = df.groupby("model").agg(
        criteria_judged=("judge_pass", "size"),
        judge_pass_rate=("judge_pass", "mean"),
        criteria_human_graded=("human_pass", "count"),
        human_pass_rate=("human_pass", "mean"),
    )
    cost = resp.groupby("model").agg(
        avg_output_tokens=("output_tokens", "mean"), avg_latency_s=("latency_s", "mean"))
    by_model = by_model.join(cost).round(3)
    by_model.to_csv(RESULTS_DIR / "summary_by_model.csv")

    by_cat = df.groupby(["model", "category"]).agg(
        judge_pass_rate=("judge_pass", "mean"),
        human_pass_rate=("human_pass", "mean"),
    ).round(3)
    by_cat.to_csv(RESULTS_DIR / "summary_by_model_category.csv")
    return by_model, by_cat


def agreement(df):
    both = df.dropna(subset=["human_pass"])
    if both.empty:
        return both, None
    rows = []
    for name, g in [("ALL", both)] + list(both.groupby("model")) + list(both.groupby("category")):
        rows.append(dict(
            group=name, n_criteria=len(g),
            pct_agreement=round((g["judge_pass"] == g["human_pass"]).mean(), 3),
            cohens_kappa=round(kappa(g["human_pass"], g["judge_pass"]), 3),
            human_pass_rate=round(g["human_pass"].mean(), 3),
            judge_pass_rate=round(g["judge_pass"].mean(), 3),
        ))
    agree = pd.DataFrame(rows)
    agree.to_csv(RESULTS_DIR / "agreement.csv", index=False)

    dis = both[both["judge_pass"] != both["human_pass"]].copy()
    dis["direction"] = np.where(dis["judge_pass"] == 1,
                                "judge PASS / human FAIL (judge too lenient?)",
                                "judge FAIL / human PASS (judge too strict?)")
    dis = dis[["prompt_id", "category", "model", "criterion_id", "criterion", "direction",
               "judge_verdict", "judge_reason", "human_verdict", "human_note", "response_id"]]
    dis.sort_values(["prompt_id", "criterion_id", "model"]).to_csv(
        RESULTS_DIR / "disagreements.csv", index=False)
    return both, agree


# ----------------------------- charts --------------------------------------
def chart_pass_rate(by_model):
    has_human = by_model["criteria_human_graded"].sum() > 0
    order = by_model.sort_values("human_pass_rate" if has_human else "judge_pass_rate").index
    y = np.arange(len(order))
    h = 0.36 if has_human else 0.55
    fig, ax = plt.subplots(figsize=(8, 1.1 + 0.9 * len(order)))

    series = [("LLM judge", "judge_pass_rate", JUDGE_COLOR)]
    if has_human:
        series = [("Human (you)", "human_pass_rate", HUMAN_COLOR)] + series
    for i, (label, col, color) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * (h + 0.04)
        vals = by_model.loc[order, col]
        ax.barh(y - offset, vals, height=h, color=color, label=label)
        for yy, v in zip(y - offset, vals):
            ax.text(v + 0.01, yy, pct(v), va="center", fontsize=9.5, color=INK)

    ax.set_yticks(y, order)
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Rubric pass rate by model")
    ax.set_xlabel("Share of rubric criteria passed")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(CHARTS / "pass_rate_by_model.png", dpi=200)
    plt.close(fig)


def chart_heatmap(by_cat):
    col = "human_pass_rate" if by_cat["human_pass_rate"].notna().any() else "judge_pass_rate"
    grid = by_cat[col].unstack("category")
    grid.columns = [c.replace("_", " ") for c in grid.columns]
    fig, ax = plt.subplots(figsize=(1.9 * grid.shape[1] + 2.2, 0.75 * grid.shape[0] + 1.6))
    ax.imshow(grid.values, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(grid.values):
        ax.text(j, i, pct(v), ha="center", va="center", fontsize=11,
                color="white" if pd.notna(v) and v >= 0.55 else INK)
    ax.set_xticks(range(grid.shape[1]), grid.columns)
    ax.set_yticks(range(grid.shape[0]), grid.index)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # thin surface-colored gaps between cells
    ax.set_xticks(np.arange(-.5, grid.shape[1]), minor=True)
    ax.set_yticks(np.arange(-.5, grid.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="minor", length=0)
    who = "human grades" if col.startswith("human") else "LLM judge (no human grades yet)"
    ax.set_title(f"Pass rate by model and category ({who})")
    fig.tight_layout()
    fig.savefig(CHARTS / "pass_rate_heatmap.png", dpi=200)
    plt.close(fig)


def chart_agreement(both, agree):
    overall = agree.iloc[0]
    m = pd.crosstab(both["human_verdict"], both["judge_verdict"]).reindex(
        index=["PASS", "FAIL"], columns=["PASS", "FAIL"], fill_value=0)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.imshow(m.values / m.values.sum(), cmap=BLUES, vmin=0, vmax=0.7)
    labels = [["Agree", "Judge too strict"], ["Judge too lenient", "Agree"]]
    for (i, j), v in np.ndenumerate(m.values):
        share = v / m.values.sum()
        color = "white" if share >= 0.4 else INK
        ax.text(j, i - 0.08, f"{v}", ha="center", va="center", fontsize=18, weight="bold", color=color)
        ax.text(j, i + 0.2, labels[i][j], ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks([0, 1], ["PASS", "FAIL"])
    ax.set_yticks([0, 1], ["PASS", "FAIL"])
    ax.set_xlabel("LLM judge")
    ax.set_ylabel("Human (you)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([-.5, .5, 1.5], minor=True)
    ax.set_yticks([-.5, .5, 1.5], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=4)
    ax.tick_params(which="minor", length=0)
    k = overall["cohens_kappa"]
    ax.set_title("Human vs. LLM judge", pad=26)
    ax.text(0, 1.03, f"{pct(overall['pct_agreement'])} agreement · Cohen's κ = "
            f"{'n/a' if pd.isna(k) else f'{k:.2f}'} · n = {int(overall['n_criteria'])} criteria",
            transform=ax.transAxes, fontsize=9.5, color=INK_2)
    fig.tight_layout()
    fig.savefig(CHARTS / "human_vs_judge.png", dpi=200)
    plt.close(fig)


def main():
    if not (RESULTS_DIR / "judge_scores.csv").exists():
        raise SystemExit("No results/judge_scores.csv yet - run `python judge.py` first.")
    CHARTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 140)

    resp, df = load()
    by_model, by_cat = summarize(resp, df)
    both, agree = agreement(df)

    print("\n=== Pass rate by model ===")
    show = by_model.copy()
    for c in ("judge_pass_rate", "human_pass_rate"):
        show[c] = show[c].map(pct)
    print(show.to_string())

    print("\n=== Pass rate by model x category ===")
    print(by_cat.map(pct).to_string())

    chart_pass_rate(by_model)
    chart_heatmap(by_cat)

    if agree is None:
        print("\nNo human grades yet - run `streamlit run grade_app.py` to add them.")
    else:
        print(f"\n=== Human vs judge agreement ({len(both)} criteria graded by both) ===")
        print(agree.to_string(index=False))
        print("  (kappa: <0.2 slight, 0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial, >0.8 near-perfect)")
        n_dis = int((both["judge_pass"] != both["human_pass"]).sum())
        print(f"\n{n_dis} disagreements -> results/disagreements.csv")
        chart_agreement(both, agree)

    print(f"Charts -> {CHARTS.relative_to(RESULTS_DIR.parent)}/")


if __name__ == "__main__":
    main()
