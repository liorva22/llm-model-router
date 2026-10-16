"""Difficulty estimation: hand-written heuristics, a small classifier on top of them, and an LLM-judge baseline."""

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer

LABELS = ["easy", "medium", "hard"]
LABEL_TO_TIER = {"easy": "cheap", "medium": "mid", "hard": "expert"}

REASONING = re.compile(
    r"\b(prove|proof|derive|design|architect|trade-?offs?|optimi[sz]e|analy[sz]e|compare|critique|justify|"
    r"explain|why|step by step|complexity|algorithm|implement|debug|refactor|strategy|evaluate)\b",
    re.I,
)
EXPERT = re.compile(
    r"\b(prove|proof|derive|linearizable|byzantine|consensus|causal|undecidable|distributed|zero downtime|"
    r"p99|at scale|million|concept drift|failure modes)\b",
    re.I,
)
CODE = re.compile(r"(```|\bdef\b|\bclass\b|\bSQL\b|\bfunction\b|\bscript\b|\bregex|regular expression|\(\))", re.I)
CONSTRAINT = re.compile(r"\b(must|without|ensure|while|every|each|covering|including|discuss)\b", re.I)


def features(q: str) -> list[float]:
    return [
        min(len(q.split()) / 40, 3.0),
        len(REASONING.findall(q)) / 3,
        len(EXPERT.findall(q)) / 2,
        float(bool(CODE.search(q))),
        len(CONSTRAINT.findall(q)) / 3,
        q.count(",") / 5,
    ]


def heuristic_score(q: str) -> float:
    """0..1, hand-weighted. Baseline only: no training data involved."""
    length, reasoning, expert, code, constraint, commas = features(q)
    s = 0.25 * length + 0.2 * reasoning + 0.35 * expert + 0.1 * code + 0.1 * constraint + 0.05 * commas
    return float(min(s, 1.0))


def heuristic_tier(q: str) -> str:
    s = heuristic_score(q)
    return "expert" if s >= 0.6 else "mid" if s >= 0.25 else "cheap"


def _feature_matrix(queries):
    return np.array([features(q) for q in queries])


class DifficultyClassifier:
    """TF-IDF n-grams + the heuristic features -> logistic regression over easy/medium/hard."""

    def __init__(self):
        self.pipe = make_pipeline(
            FeatureUnion(
                [
                    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                    ("heuristics", FunctionTransformer(_feature_matrix)),
                ]
            ),
            LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced"),
        )

    def fit(self, queries, labels):
        self.pipe.fit(queries, labels)
        return self

    def predict_proba(self, query: str) -> dict[str, float]:
        probs = self.pipe.predict_proba([query])[0]
        return {label: float(p) for label, p in zip(self.pipe.classes_, probs)}


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r["query"] for r in rows], [r["label"] for r in rows]


JUDGE_SYSTEM = (
    "Classify how hard it is for an AI assistant to answer the user's request well. "
    "easy = factual lookup, conversion, or a one-line rewrite. "
    "medium = explanation, comparison, short code, or a structured piece of writing. "
    "hard = proofs, system design, multi-step expert analysis. "
    "Reply with exactly one word: easy, medium, or hard."
)


def llm_judge_label(llm, tier, query: str) -> tuple[str, float]:
    """LLM-judge baseline. Returns (label, cost_usd)."""
    c = llm.complete(tier, query, system=JUDGE_SYSTEM, max_tokens=1024)
    word = c.text.strip().lower().split()[0].strip(".") if c.text.strip() else ""
    return (word if word in LABELS else "medium"), tier.cost(c.input_tokens, c.output_tokens)
