# LLM Model Router: pitch

## The pitch
Most LLM traffic is easy, but teams pay expert-model prices for all of it. This router reads each request, sends it to the cheapest Claude model that can handle it, and enforces a monthly budget that degrades gracefully instead of failing. On a 200-query labeled set it matched the labeled tier 96.5% of the time and under-routed only 2%, at 22.6% of the cost of sending everything to Opus (mock-mode token counts).

## Problem and who it's for
Product teams running LLM features at volume, where a FAQ bot and a "design my architecture" request hit the same endpoint. Sending everything to the best model is simple but expensive. Sending everything to the cheapest model saves money but fails the hard questions (25% of the test set). Finance also wants a number that can't be exceeded.

## How it works
See the diagram in the README. Requests reach a FastAPI gateway. A logistic-regression classifier over TF-IDF n-grams and six hand-written features (length, reasoning verbs, expert vocabulary, code markers, constraints, clause count) outputs P(easy/medium/hard) in about 2 ms. A threshold policy maps that to Haiku 5.5, Sonnet 5.5, or Opus 5.5. A budget guard reads month-to-date spend from Postgres, sends alerts at 50/80/100% exactly once per month, downgrades expert requests at 90%, and goes cheap-only at 100%.

## Stack and design choices
- **Estimator: small linear classifier**, chosen over a per-request LLM judge or a fine-tuned transformer. A judge adds a network call and a fee to every request, which is the very cost the router exists to cut. The classifier routes on CPU in ~2 ms, can be inspected, and trains in under a second. *Trade-off:* it only knows the surface form of a query, so a short but deep question ("Is P = NP?") can slip through as easy.
- **Upward-biased thresholds, not argmax.** Under-routing costs answer quality; over-routing costs cents. *Trade-off:* the mid tier is ~3 points above the 15% target.
- **Degrade instead of reject.** Users keep getting answers. *Trade-off:* spend can overshoot slightly (+1.5% in the simulated month) while cheap-only traffic continues.
- **Postgres ledger with a primary-key-guarded alerts table.** Alerts fire exactly once across workers with no extra infrastructure. *Trade-off:* every request does a sum over the month; at high volume, cache it or keep a running counter.
- **Config-driven model IDs and prices, and a provider protocol** (`complete(tier, prompt)`), so OpenAI or a local model is one class away.

## What I'd do differently
- **Real labels from real traffic.** I wrote and labeled the 200 queries myself, as a single annotator. Production would label sampled traffic using outcome signals (thumbs-down, escalation, regenerate) and track inter-annotator agreement.
- **Live quality numbers.** This run was mock-only because no API key was available. The `--limit 60` live path grades routed answers against Opus answers. Its results belong next to the mock numbers before anyone trusts the cost/quality claim.
- **Cascade, not just pre-routing.** Send the request to the cheap model first, and escalate when its self-check or a verifier fails. This costs latency on hard queries in exchange for robustness to misclassification.
- **Per-tenant budgets** and a hard-stop option. Also price lookup by the model that actually answered (refusal fallbacks can switch models).
- **Calibrated thresholds** chosen by cost-weighted optimization on a validation set instead of by hand.

## Interview talking points
- Why under-routing and over-routing are asymmetric errors, and how that shaped the thresholds.
- Why the reported accuracy is out-of-fold, and why that matters on a 200-row dataset.
- Exactly-once alerting using a database constraint instead of a distributed lock.
- What I refused to measure in mock mode (answer quality) and how the live path measures it.
