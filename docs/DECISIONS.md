# Design decisions

| # | Decision | Alternatives | Why |
|---|----------|--------------|-----|
| 1 | **FastAPI** gateway | Next.js API routes | The estimator is scikit-learn, so a Python gateway avoids a second runtime and an RPC hop. |
| 2 | **TF-IDF + 6 heuristic features → logistic regression** | Fine-tuned small transformer; embeddings + kNN; LLM judge per request | Trains in under a second on 200 examples and routes in ~2 ms with no GPU, and its weights can be inspected. An LLM judge on every request adds a network round trip and a cost to every call, the very cost the router exists to save. |
| 3 | **Train at startup, no model artifact** | Pickle a model into the image | 200 rows train in under a second. Skipping the artifact means no binary to version and no pickle-loading risk. Revisit once the dataset gets large. |
| 4 | **Thresholds on P(hard) and P(medium)+P(hard), biased upward** (0.45 / 0.40) | argmax; quantile thresholds forced to exactly 80/15/5 | Under-routing a hard question costs quality, while over-routing an easy one costs only cents. Forcing an exact mix would misroute whenever real traffic drifts from it. The measured mix (75/18/6) already lands near the 80/15/5 target. |
| 5 | **Degrade, don't reject**, at the budget ceiling: expert→mid at 90%, everything→cheap at 100% | Return HTTP 429 at 100% | Users still get an answer. The trade-off is a small overshoot: cheap traffic keeps spending after 100% (+1.5% in the simulation). A hard stop is one extra branch in `apply_budget` if a deployment needs it. |
| 6 | **Exactly-once alerts via a Postgres primary key** `(month, threshold)` | An in-process set; Redis | Exactly-once holds across multiple API workers without adding another service. |
| 7 | **Spend ledger in Postgres (pgvector image)**, in-memory fallback | SQLite; Supabase | Postgres is Supabase-compatible and matches the other projects. The in-memory ledger keeps the tests and the 60k-request simulation fast. |
| 8 | **Model IDs and prices in `config.yaml`** | Constants in code | Prices change, and model IDs get retired. Changing either should be a config edit. |
| 9 | **Server-side refusal fallback** (`fallbacks: "default"`) on Sonnet/Opus tiers | Return refusals to the client | Anthropic recommends it for these models: a declined request is retried server-side instead of failing. |
| 10 | **Out-of-fold (5-fold CV) predictions for the reported classifier numbers** | Train/test on the same 200 rows | In-sample accuracy on 200 rows would be inflated. Every reported prediction comes from a model that never saw that query. |
| 11 | **Mock-mode quality proxy = tier adequacy** (routed tier ≥ labeled tier) | Simulated "quality scores" | A mock LLM has no real quality, so inventing a score would be dishonest. Tier adequacy is measurable without a model. In live mode, a Sonnet grader checks routed answers against Opus answers. |
| 12 | **Effort per tier** (low/medium/high) | Default effort everywhere | Effort is the cheapest quality/cost knob inside a single model, so the cheap tier also gets the cheapest effort. |
