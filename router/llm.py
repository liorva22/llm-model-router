"""LLM providers. Anything with `.complete(tier, prompt, system=None)` works, so OpenAI etc. can be dropped in."""

import time
from dataclasses import dataclass

from .config import Tier


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    refused: bool = False


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class MockLLM:
    """Deterministic fake: no network, no keys. Token counts scale with the prompt so costs are comparable."""

    def complete(self, tier: Tier, prompt: str, system: str | None = None, max_tokens: int = 16000) -> Completion:
        t0 = time.perf_counter()
        in_tok = approx_tokens(prompt) + approx_tokens(system or "")
        out_tok = min(max_tokens, 120 + 3 * in_tok)
        text = f"[mock {tier.model}] response to: {prompt[:80]}"
        return Completion(text, tier.model, in_tok, out_tok, (time.perf_counter() - t0) * 1000)


class AnthropicLLM:
    def __init__(self):
        import anthropic

        self.client = anthropic.Anthropic()

    def complete(self, tier: Tier, prompt: str, system: str | None = None, max_tokens: int = 16000) -> Completion:
        kwargs = dict(
            model=tier.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": tier.effort},
        )
        if system:
            kwargs["system"] = system
        t0 = time.perf_counter()
        if tier.fallbacks:
            resp = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
            )
        else:
            resp = self.client.messages.create(**kwargs)
        latency = (time.perf_counter() - t0) * 1000
        text = "".join(b.text for b in resp.content if b.type == "text")
        # ponytail: cost is billed at the tier's price even if a refusal fallback answered on another model;
        # look up resp.model in a price table if fallbacks become frequent.
        return Completion(
            text, resp.model, resp.usage.input_tokens, resp.usage.output_tokens, latency, resp.stop_reason == "refusal"
        )


def make_llm(mock: bool):
    return MockLLM() if mock else AnthropicLLM()
