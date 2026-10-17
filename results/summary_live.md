| Strategy | Cheap / Mid / Expert | Tier accuracy | Under-routed | Over-routed | Cost vs always-expert | Quality retained |
|---|---|---|---|---|---|---|
| always_cheap | 100% / 0% / 0% | 16.7% | 83.3% | 0.0% | 1.3% | 50.0% |
| always_expert | 0% / 0% / 100% | 25.0% | 0.0% | 75.0% | 100.0% | 100.0% |
| heuristic | 65% / 25% / 10% | 36.7% | 63.3% | 0.0% | 38.1% | 63.3% |
| classifier | 13% / 65% / 22% | 93.3% | 3.3% | 3.3% | 67.3% | 75.0% |
| llm_judge | 17% / 58% / 25% | 100.0% | 0.0% | 0.0% | 70.9% | 76.7% |

Mode: **live**, n=60 {'easy': 10, 'medium': 35, 'hard': 15}. Classifier routing overhead: p50 8.64 ms, p95 13.18 ms. Real API spend for this run: $6.9042.

Budget month (2000 req/day x 30 days, mock token counts): always-expert $198.16, unguarded router $46.06, guarded router $39.75 against a $39.15 budget (degradation began on day 23).
