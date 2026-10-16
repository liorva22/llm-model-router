| Strategy | Cheap / Mid / Expert | Tier accuracy | Under-routed | Over-routed | Cost vs always-expert |
|---|---|---|---|---|---|
| always_cheap | 100% / 0% / 0% | 75.0% | 25.0% | 0.0% | 2.5% |
| always_expert | 0% / 0% / 100% | 7.5% | 0.0% | 92.5% | 100.0% |
| heuristic | 90% / 8% / 3% | 81.0% | 19.0% | 0.0% | 12.0% |
| classifier | 75% / 18% / 6% | 96.5% | 2.0% | 1.5% | 22.6% |

Mode: **mock**, n=200 {'easy': 150, 'medium': 35, 'hard': 15}. Classifier routing overhead: p50 1.87 ms, p95 4.33 ms. Real API spend for this run: $0.0000 (mock mode: costs above use list prices x mock token counts).

Budget month (2000 req/day x 30 days, mock token counts): always-expert $198.16, unguarded router $46.06, guarded router $39.75 against a $39.15 budget (degradation began on day 23).
