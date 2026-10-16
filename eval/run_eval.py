"""Evaluate routing strategies on the labeled set, then replay a month of traffic against a budget.

    MOCK_MODE=1 python eval/run_eval.py          # no keys; deterministic
    python eval/run_eval.py --limit 60           # live: real answers + LLM-judged quality

Writes results/eval_results.json, results/summary.md and docs/img/dashboard.png.
"""

import argparse
import json
import random
import statistics
import sys
import time
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from router.budget import MemoryLedger  # noqa: E402
from router.config import ROOT, TIER_ORDER, load_config  # noqa: E402
from router.core import Router  # noqa: E402
from router.difficulty import (  # noqa: E402
    LABEL_TO_TIER,
    DifficultyClassifier,
    heuristic_tier,
    llm_judge_label,
    load_dataset,
)
from router.llm import MockLLM, make_llm  # noqa: E402
from router.policy import choose_tier  # noqa: E402

RANK = {t: i for i, t in enumerate(TIER_ORDER)}
QUALITY_SYSTEM = (
    "You compare two answers to the same question. Answer A is a candidate, answer B is a reference from a "
    "stronger model. Reply 'yes' if A is correct and complete enough that a user would be as well served as by B, "
    "otherwise 'no'. Reply with one word."
)


def oof_classifier_tiers(queries, labels, routing, seed=0):
    """Out-of-fold predictions: every query is routed by a model that never saw it."""
    tiers, latencies = [None] * len(queries), []
    for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(queries, labels):
        clf = DifficultyClassifier().fit([queries[i] for i in train], [labels[i] for i in train])
        for i in test:
            t0 = time.perf_counter()
            tiers[i] = choose_tier(clf.predict_proba(queries[i]), routing)
            latencies.append((time.perf_counter() - t0) * 1000)
    return tiers, latencies


def score(tiers, labels, costs):
    n = len(labels)
    need = [RANK[LABEL_TO_TIER[lbl]] for lbl in labels]
    got = [RANK[t] for t in tiers]
    return {
        "mix": {t: sum(x == t for x in tiers) / n for t in TIER_ORDER},
        "tier_accuracy": sum(g == r for g, r in zip(got, need)) / n,
        "under_route_rate": sum(g < r for g, r in zip(got, need)) / n,
        "over_route_rate": sum(g > r for g, r in zip(got, need)) / n,
        "cost_usd": sum(costs),
    }


def simulate_month(cfg, estimate, budget_usd, queries, per_day, seed=7):
    """Replay `per_day` sampled queries per day for 30 days through the real Router (mock LLM)."""
    c = {**cfg, "budget": {**cfg["budget"], "monthly_usd": budget_usd}}
    router = Router(c, MockLLM(), MemoryLedger(), estimate)
    rng = random.Random(seed)
    start = datetime(2026, 11, 1, tzinfo=UTC)
    days = []
    for d in range(30):
        mix = dict.fromkeys(TIER_ORDER, 0)
        degraded = 0
        for i in range(per_day):
            r = router.handle(rng.choice(queries), start + timedelta(days=d, seconds=i))
            mix[r["tier"]] += 1
            degraded += r["degraded"]
        days.append({"day": d + 1, "spent": r["budget"]["spent_usd"], "mix": mix, "degraded": degraded})
    return days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stratified subsample size for live runs (0 = all)")
    ap.add_argument("--per-day", type=int, default=2000)
    args = ap.parse_args()

    cfg = load_config()
    llm = make_llm(cfg["mock"])
    queries, labels = load_dataset(ROOT / "data" / "queries.jsonl")
    if args.limit:
        idx = sorted(random.Random(0).sample(range(len(queries)), args.limit))
        # keep every hard/medium example; trim easy ones
        idx = sorted({i for i, lbl in enumerate(labels) if lbl != "easy"} | set(idx))[: max(args.limit, 50)]
        queries, labels = [queries[i] for i in idx], [labels[i] for i in idx]
    print(f"mode={'MOCK' if cfg['mock'] else 'LIVE'} n={len(queries)}")

    answers, spend = {}, {"llm_usd": 0.0}

    def answer(q, tier_name):
        if (q, tier_name) not in answers:
            tier = cfg["tiers"][tier_name]
            c = llm.complete(tier, q, max_tokens=cfg["max_output_tokens"])
            answers[(q, tier_name)] = (c, tier.cost(c.input_tokens, c.output_tokens))
            spend["llm_usd"] += answers[(q, tier_name)][1]
        return answers[(q, tier_name)]

    strategies = {
        "always_cheap": ["cheap"] * len(queries),
        "always_expert": ["expert"] * len(queries),
        "heuristic": [heuristic_tier(q) for q in queries],
    }
    strategies["classifier"], clf_ms = oof_classifier_tiers(queries, labels, cfg["routing"])
    judge_cost = None
    if not cfg["mock"]:
        judged = [llm_judge_label(llm, cfg["tiers"][cfg["judge_tier"]], q) for q in queries]
        strategies["llm_judge"] = [LABEL_TO_TIER[lbl] for lbl, _ in judged]
        judge_cost = sum(c for _, c in judged)
        spend["llm_usd"] += judge_cost

    results = {}
    for name, tiers in strategies.items():
        costs = [answer(q, t)[1] for q, t in zip(queries, tiers)]
        results[name] = score(tiers, labels, costs)
    expert_cost = results["always_expert"]["cost_usd"]
    for r in results.values():
        r["cost_vs_expert"] = r["cost_usd"] / expert_cost
    if judge_cost is not None:
        results["llm_judge"]["estimator_cost_usd"] = judge_cost

    if not cfg["mock"]:  # quality: does the routed answer serve the user as well as the expert's?
        grader = cfg["tiers"]["mid"]
        verdicts = {}
        for name, tiers in strategies.items():
            ok = []
            for q, t in zip(queries, tiers):
                if t == "expert":
                    ok.append(1)
                    continue
                if (q, t) not in verdicts:
                    a, b = answer(q, t)[0].text, answer(q, "expert")[0].text
                    prompt = f"Question:\n{q}\n\nAnswer A:\n{a}\n\nAnswer B:\n{b}"
                    c = llm.complete(grader, prompt, QUALITY_SYSTEM, 1024)
                    spend["llm_usd"] += grader.cost(c.input_tokens, c.output_tokens)
                    verdicts[(q, t)] = int(c.text.strip().lower().startswith("yes"))
                ok.append(verdicts[(q, t)])
            results[name]["quality_retention"] = sum(ok) / len(ok)

    # Budget month: always mock token counts (a live 60k-request replay would blow the spend cap).
    full_q, full_l = load_dataset(ROOT / "data" / "queries.jsonl")
    clf = DifficultyClassifier().fit(full_q, full_l)
    estimate = lru_cache(maxsize=None)(clf.predict_proba)
    huge = 1e12
    unguarded = simulate_month(cfg, estimate, huge, full_q, args.per_day)
    expert_month = simulate_month(cfg, lambda q: {"hard": 1.0}, huge, full_q, args.per_day)
    budget_usd = round(0.85 * unguarded[-1]["spent"], 2)
    guarded = simulate_month(cfg, estimate, budget_usd, full_q, args.per_day)

    out = {
        "mode": "mock" if cfg["mock"] else "live",
        "n_queries": len(queries),
        "label_counts": {lbl: labels.count(lbl) for lbl in ("easy", "medium", "hard")},
        "strategies": results,
        "classifier_route_ms": {
            "p50": statistics.median(clf_ms),
            "p95": float(np.percentile(clf_ms, 95)),
        },
        "api_spend_usd": 0.0 if cfg["mock"] else spend["llm_usd"],
        "budget_sim": {
            "requests_per_day": args.per_day,
            "budget_usd": budget_usd,
            "always_expert_month_usd": expert_month[-1]["spent"],
            "unguarded_router_month_usd": unguarded[-1]["spent"],
            "guarded_router_month_usd": guarded[-1]["spent"],
            "first_degraded_day": next((d["day"] for d in guarded if d["degraded"]), None),
            "days": {"guarded": guarded, "unguarded": unguarded},
        },
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "eval_results.json").write_text(json.dumps(out, indent=2))
    (ROOT / "results" / "summary.md").write_text(summary_md(out))
    plot(out, queries, labels, strategies["classifier"])
    print(summary_md(out))


def summary_md(out):
    live = out["mode"] == "live"
    head = "| Strategy | Cheap / Mid / Expert | Tier accuracy | Under-routed | Over-routed | Cost vs always-expert |"
    if live:
        head += " Quality retained |"
    rows = [head, "|" + "---|" * (head.count("|") - 1)]
    for name, r in out["strategies"].items():
        m = r["mix"]
        row = (
            f"| {name} | {m['cheap']:.0%} / {m['mid']:.0%} / {m['expert']:.0%} | {r['tier_accuracy']:.1%} | "
            f"{r['under_route_rate']:.1%} | {r['over_route_rate']:.1%} | {r['cost_vs_expert']:.1%} |"
        )
        if live:
            row += f" {r['quality_retention']:.1%} |"
        rows.append(row)
    b = out["budget_sim"]
    rows += [
        "",
        f"Mode: **{out['mode']}**, n={out['n_queries']} {out['label_counts']}. "
        f"Classifier routing overhead: p50 {out['classifier_route_ms']['p50']:.2f} ms, "
        f"p95 {out['classifier_route_ms']['p95']:.2f} ms. Real API spend for this run: ${out['api_spend_usd']:.4f}"
        + (" (mock mode: costs above use list prices x mock token counts)." if out["mode"] == "mock" else "."),
        "",
        f"Budget month ({b['requests_per_day']} req/day x 30 days, mock token counts): "
        f"always-expert ${b['always_expert_month_usd']:.2f}, unguarded router ${b['unguarded_router_month_usd']:.2f}, "
        f"guarded router ${b['guarded_router_month_usd']:.2f} against a ${b['budget_usd']:.2f} budget "
        f"(degradation began on day {b['first_degraded_day']}).",
    ]
    return "\n".join(rows) + "\n"


def plot(out, queries, labels, clf_tiers):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"LLM Model Router — eval dashboard ({out['mode']} mode, n={out['n_queries']})", fontsize=14)

    # 1. Routing matrix: true difficulty x routed tier (classifier, out-of-fold)
    lbls = ["easy", "medium", "hard"]
    m = np.array([[sum(1 for lb, t in zip(labels, clf_tiers) if lb == r and t == c) for c in TIER_ORDER] for r in lbls])
    a = ax[0, 0]
    a.imshow(m / m.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            a.text(j, i, f"{m[i, j]}\n({m[i, j] / m[i].sum():.0%})", ha="center", va="center")
    a.set(
        xticks=range(3),
        xticklabels=TIER_ORDER,
        yticks=range(3),
        yticklabels=lbls,
        xlabel="routed tier",
        ylabel="labeled difficulty",
        title="Classifier routing (5-fold out-of-fold)",
    )

    # 2. Cost vs quality heatmap per strategy
    s = out["strategies"]
    cols = ["cost_vs_expert", "tier_accuracy", "under_route_rate", "over_route_rate"]
    if out["mode"] == "live":
        cols.append("quality_retention")
    names = list(s)
    v = np.array([[s[n][c] for c in cols] for n in names])
    lower_is_better = np.array([c in ("cost_vs_expert", "under_route_rate", "over_route_rate") for c in cols])
    a = ax[0, 1]
    a.imshow(np.where(lower_is_better, 1 - v, v), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(len(names)):
        for j in range(len(cols)):
            a.text(j, i, f"{v[i, j]:.0%}", ha="center", va="center")
    a.set(
        xticks=range(len(cols)),
        xticklabels=[c.replace("_", "\n") for c in cols],
        yticks=range(len(names)),
        yticklabels=names,
        title="Cost vs quality per strategy (green = better)",
    )

    # 3. Budget burn-down
    b = out["budget_sim"]
    days = [d["day"] for d in b["days"]["guarded"]]
    a = ax[1, 0]
    a.plot(days, [b["budget_usd"] - d["spent"] for d in b["days"]["guarded"]], label="router + budget guard", lw=2)
    a.plot(days, [b["budget_usd"] - d["spent"] for d in b["days"]["unguarded"]], "--", label="router, no guard")
    a.plot(days, [b["budget_usd"] * (1 - d / 30) for d in days], ":", color="gray", label="ideal linear burn")
    a.axhline(0, color="red", lw=1)
    a.set(
        xlabel="day of month",
        ylabel="remaining budget (USD)",
        title=f"Budget burn-down (${b['budget_usd']:.2f}/month, {b['requests_per_day']} req/day)",
    )
    a.legend()

    # 4. Daily tier mix under the guard
    a = ax[1, 1]
    g = b["days"]["guarded"]
    bottom = np.zeros(len(g))
    for t in TIER_ORDER:
        vals = np.array([d["mix"][t] for d in g]) / b["requests_per_day"]
        a.bar(days, vals, bottom=bottom, label=t)
        bottom += vals
    a.set(xlabel="day of month", ylabel="share of requests", title="Daily tier mix with budget guard", ylim=(0, 1))
    a.legend(loc="lower left")

    fig.tight_layout()
    (ROOT / "docs" / "img").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "docs" / "img" / "dashboard.png", dpi=110)


if __name__ == "__main__":
    main()
