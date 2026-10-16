import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TIER_ORDER = ["cheap", "mid", "expert"]


@dataclass(frozen=True)
class Tier:
    name: str
    model: str
    input_per_mtok: float
    output_per_mtok: float
    effort: str
    fallbacks: bool = False

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


def load_config(path: str | None = None) -> dict:
    cfg = yaml.safe_load(Path(path or os.getenv("ROUTER_CONFIG", ROOT / "config.yaml")).read_text())
    cfg["tiers"] = {name: Tier(name=name, **t) for name, t in cfg["tiers"].items()}
    if budget := os.getenv("MONTHLY_BUDGET_USD"):
        cfg["budget"]["monthly_usd"] = float(budget)
    cfg["mock"] = os.getenv("MOCK_MODE", "0") == "1"
    return cfg
