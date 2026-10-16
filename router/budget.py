"""Spend ledger (Postgres in prod, in-memory for tests and simulation) and once-per-month budget alerts."""

import json
import logging
import os
import urllib.request
from collections import defaultdict
from datetime import datetime

log = logging.getLogger("router.budget")

SCHEMA = """
create table if not exists usage (
    id bigserial primary key,
    ts timestamptz not null default now(),
    tier text not null,
    model text not null,
    input_tokens int not null,
    output_tokens int not null,
    cost_usd numeric(12, 6) not null,
    degraded boolean not null default false
);
create index if not exists usage_ts_idx on usage (ts);
create table if not exists budget_alerts (
    month text not null,
    threshold numeric not null,
    fired_at timestamptz not null default now(),
    primary key (month, threshold)
);
"""


def month_key(ts: datetime) -> str:
    return ts.strftime("%Y-%m")


class MemoryLedger:
    def __init__(self):
        self.totals: defaultdict[str, float] = defaultdict(float)
        self.alerts: set[tuple[str, float]] = set()

    def record(self, ts, tier, model, input_tokens, output_tokens, cost, degraded=False):
        self.totals[month_key(ts)] += cost

    def month_spend(self, now: datetime) -> float:
        return self.totals[month_key(now)]

    def claim_alert(self, month: str, threshold: float) -> bool:
        if (month, threshold) in self.alerts:
            return False
        self.alerts.add((month, threshold))
        return True


class PostgresLedger:
    def __init__(self, dsn: str):
        import psycopg

        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute(SCHEMA)

    def record(self, ts, tier, model, input_tokens, output_tokens, cost, degraded=False):
        self.conn.execute(
            "insert into usage (ts, tier, model, input_tokens, output_tokens, cost_usd, degraded)"
            " values (%s, %s, %s, %s, %s, %s, %s)",
            (ts, tier, model, input_tokens, output_tokens, cost, degraded),
        )

    def month_spend(self, now: datetime) -> float:
        row = self.conn.execute(
            "select coalesce(sum(cost_usd), 0) from usage where ts >= date_trunc('month', %s::timestamptz)"
            " and ts < date_trunc('month', %s::timestamptz) + interval '1 month'",
            (now, now),
        ).fetchone()
        return float(row[0])

    def claim_alert(self, month: str, threshold: float) -> bool:
        # The primary key makes this exactly-once across workers.
        cur = self.conn.execute(
            "insert into budget_alerts (month, threshold) values (%s, %s) on conflict do nothing",
            (month, threshold),
        )
        return cur.rowcount == 1


def make_ledger():
    dsn = os.getenv("DATABASE_URL")
    return PostgresLedger(dsn) if dsn else MemoryLedger()


def notify(message: str, webhook_url: str | None = None):
    log.warning(message)
    if not webhook_url:
        return
    req = urllib.request.Request(
        webhook_url, data=json.dumps({"text": message}).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req, timeout=5)
    except OSError as e:
        log.error("alert webhook failed: %s", e)


def check_alerts(ledger, spent: float, limit: float, thresholds, now: datetime, webhook_url=None) -> list[float]:
    fired = []
    for t in sorted(thresholds):
        if spent >= t * limit and ledger.claim_alert(month_key(now), t):
            msg = f"Budget alert: {spent / limit:.0%} of ${limit:.2f} monthly budget used ({month_key(now)})"
            notify(msg, webhook_url)
            fired.append(t)
    return fired
