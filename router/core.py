"""The routing decision + call + accounting. Shared by the API and the budget simulation."""

import os
import time
from datetime import UTC, datetime

from .budget import check_alerts
from .policy import apply_budget, choose_tier


class Router:
    def __init__(self, cfg: dict, llm, ledger, estimate):
        """`estimate(query) -> {"easy": p, "medium": p, "hard": p}`."""
        self.cfg, self.llm, self.ledger, self.estimate = cfg, llm, ledger, estimate
        self.webhook = os.getenv("ALERT_WEBHOOK_URL")

    def handle(self, query: str, now: datetime | None = None) -> dict:
        now = now or datetime.now(UTC)
        budget = self.cfg["budget"]
        limit = budget["monthly_usd"]
        spent = self.ledger.month_spend(now)

        t0 = time.perf_counter()
        probs = self.estimate(query)
        wanted = choose_tier(probs, self.cfg["routing"])
        tier_name, degraded = apply_budget(wanted, spent / limit, budget)
        route_ms = (time.perf_counter() - t0) * 1000

        tier = self.cfg["tiers"][tier_name]
        c = self.llm.complete(tier, query, max_tokens=self.cfg["max_output_tokens"])
        cost = tier.cost(c.input_tokens, c.output_tokens)
        self.ledger.record(now, tier_name, c.model, c.input_tokens, c.output_tokens, cost, degraded)
        spent += cost
        alerts = check_alerts(self.ledger, spent, limit, budget["alert_thresholds"], now, self.webhook)

        return {
            "answer": c.text,
            "refused": c.refused,
            "tier": tier_name,
            "requested_tier": wanted,
            "degraded": degraded,
            "model": c.model,
            "difficulty": probs,
            "input_tokens": c.input_tokens,
            "output_tokens": c.output_tokens,
            "cost_usd": cost,
            "route_ms": route_ms,
            "llm_ms": c.latency_ms,
            "budget": {"spent_usd": spent, "limit_usd": limit, "fraction": spent / limit},
            "alerts_fired": alerts,
        }
