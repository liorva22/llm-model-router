import os
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from router.app import create_app
from router.budget import MemoryLedger, PostgresLedger, check_alerts
from router.config import ROOT, load_config
from router.difficulty import DifficultyClassifier, heuristic_score, load_dataset
from router.llm import MockLLM
from router.policy import apply_budget, choose_tier

CFG = load_config()
NOW = datetime(2026, 10, 15, tzinfo=UTC)


@pytest.fixture(scope="module")
def clf():
    return DifficultyClassifier().fit(*load_dataset(ROOT / "data" / "queries.jsonl"))


def test_heuristics_rank_hard_above_easy():
    assert heuristic_score("What is the capital of France?") < heuristic_score(
        "Design a distributed consensus protocol and prove it tolerates one Byzantine fault, analyzing complexity."
    )


def test_classifier_routes_unseen_queries_sensibly(clf):
    assert choose_tier(clf.predict_proba("What is the capital of Peru?"), CFG["routing"]) == "cheap"
    hard = "Prove that the protocol is linearizable and design the failure recovery for a distributed log at scale."
    assert choose_tier(clf.predict_proba(hard), CFG["routing"]) == "expert"


def test_choose_tier_thresholds():
    r = {"expert_threshold": 0.5, "mid_threshold": 0.4}
    assert choose_tier({"easy": 0.9, "medium": 0.05, "hard": 0.05}, r) == "cheap"
    assert choose_tier({"easy": 0.5, "medium": 0.3, "hard": 0.2}, r) == "mid"
    assert choose_tier({"easy": 0.1, "medium": 0.3, "hard": 0.6}, r) == "expert"


def test_budget_degradation():
    b = {"degrade_at": 0.9, "cheap_only_at": 1.0}
    assert apply_budget("expert", 0.5, b) == ("expert", False)
    assert apply_budget("expert", 0.95, b) == ("mid", True)
    assert apply_budget("mid", 0.95, b) == ("mid", False)
    assert apply_budget("mid", 1.0, b) == ("cheap", True)
    assert apply_budget("cheap", 1.2, b) == ("cheap", False)


def test_alerts_fire_once_per_month():
    ledger = MemoryLedger()
    assert check_alerts(ledger, 85, 100, [0.5, 0.8, 1.0], NOW) == [0.5, 0.8]
    assert check_alerts(ledger, 90, 100, [0.5, 0.8, 1.0], NOW) == []
    assert check_alerts(ledger, 101, 100, [0.5, 0.8, 1.0], NOW) == [1.0]
    assert check_alerts(ledger, 60, 100, [0.5], datetime(2026, 11, 1, tzinfo=UTC)) == [0.5]


def test_memory_ledger_is_month_scoped():
    ledger = MemoryLedger()
    ledger.record(datetime(2026, 9, 30, tzinfo=UTC), "cheap", "m", 1, 1, 5.0)
    ledger.record(NOW, "cheap", "m", 1, 1, 2.0)
    assert ledger.month_spend(NOW) == 2.0


def test_api_routes_and_degrades():
    cfg = load_config()
    cfg["budget"] = {**cfg["budget"], "monthly_usd": 0.001}
    client = TestClient(create_app(cfg, MockLLM(), MemoryLedger()))
    hard = "Design a globally distributed rate limiter and prove its consistency guarantees under partitions."

    first = client.post("/v1/chat", json={"query": hard}).json()
    assert first["tier"] == "expert" and not first["degraded"] and first["cost_usd"] > 0

    # The first expert call blows the tiny budget; the next one must degrade to cheap.
    second = client.post("/v1/chat", json={"query": hard}).json()
    assert second["tier"] == "cheap" and second["degraded"] and second["requested_tier"] == "expert"
    assert client.get("/v1/budget").json()["fraction"] > 1
    assert client.post("/v1/chat", json={"query": ""}).status_code == 422


@pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="needs Postgres")
def test_postgres_ledger():
    ledger = PostgresLedger(os.environ["DATABASE_URL"])
    ledger.conn.execute("truncate usage, budget_alerts")
    ledger.record(NOW, "cheap", "m", 10, 10, 1.5)
    ledger.record(datetime(2026, 9, 1, tzinfo=UTC), "cheap", "m", 10, 10, 9.0)
    assert ledger.month_spend(NOW) == pytest.approx(1.5)
    assert ledger.claim_alert("2026-10", 0.5) is True
    assert ledger.claim_alert("2026-10", 0.5) is False
