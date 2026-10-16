def choose_tier(probs: dict[str, float], routing: dict) -> str:
    if probs.get("hard", 0) >= routing["expert_threshold"]:
        return "expert"
    if probs.get("medium", 0) + probs.get("hard", 0) >= routing["mid_threshold"]:
        return "mid"
    return "cheap"


def apply_budget(tier: str, spent_fraction: float, budget: dict) -> tuple[str, bool]:
    """Graceful degradation. Returns (tier actually used, whether it was downgraded)."""
    if spent_fraction >= budget["cheap_only_at"]:
        return "cheap", tier != "cheap"
    if spent_fraction >= budget["degrade_at"] and tier == "expert":
        return "mid", True
    return tier, False
