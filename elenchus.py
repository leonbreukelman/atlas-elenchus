"""
Elenchus — Deutsch Probe for trading agent recommendations.

The probe does NOT test prediction accuracy. It tests whether the components
of an explanation are load-bearing. Concretely:

    For each reasoning_component in a recommendation:
        1. Replace that component with a plausible alternative
        2. Ask: does the same conclusion (ticker + direction) still follow?
        3. If YES: that component was decorative (easy to vary)
        4. If NO: that component was load-bearing (hard to vary)

    deutsch_score = count(load_bearing) / count(total_components)

A score of 1.0 means every component is necessary — the explanation is
maximally hard to vary. A score of 0.0 means every component could be
swapped out and the conclusion wouldn't change — pure post-hoc rationalization.
"""

import json
import anthropic
from dataclasses import dataclass

from .agent import Recommendation


@dataclass
class ProbeResult:
    """Result of probing a single reasoning component."""
    component_index: int
    original_component: str
    replacement_component: str
    conclusion_survived: bool     # True = component was decorative (bad)
    probe_reasoning: str          # the probe's explanation of its judgment


@dataclass
class ElenchusResult:
    """Full probe result for a single recommendation."""
    recommendation: Recommendation
    probe_results: list[ProbeResult]
    deutsch_score: float          # ratio of load-bearing components
    total_components: int
    load_bearing_count: int

    @property
    def is_hard_to_vary(self) -> bool:
        """Threshold: >0.6 means more components are load-bearing than not."""
        return self.deutsch_score > 0.6


PROBE_SYSTEM_PROMPT = """You are an adversarial epistemology probe. Your job is to test whether
a reasoning component is actually necessary for a conclusion, or whether it's decorative.

You will receive:
- A trading recommendation (ticker, direction, conviction)
- The full list of reasoning components
- ONE component that has been REPLACED with a plausible alternative

Your task: Given the MODIFIED reasoning (with the replacement), does the SAME conclusion
(same ticker, same direction) still logically follow?

You must respond with JSON only:
{
  "conclusion_survives": true/false,
  "replacement_component": "the alternative reasoning you were given",
  "explanation": "why the conclusion does/doesn't survive the swap"
}

Rules:
- Be rigorous. If the replaced component changes the logical basis for the trade, the conclusion does NOT survive.
- A component is load-bearing if removing/replacing it would change what you'd recommend.
- Don't be swayed by surface-level similarity. Test logical dependency.
"""

PERTURBATION_SYSTEM_PROMPT = """You generate plausible alternative reasoning components for trading
recommendations. Given one reasoning component, produce a replacement that:

1. Is about the SAME domain (if original is about volatility, replacement is about volatility)
2. Points in a DIFFERENT analytical direction (if original is bullish on vol, replacement is bearish or neutral)
3. Is equally specific and falsifiable
4. Is realistic — something a competent analyst might actually say

Respond with JSON only:
{
  "replacement": "your alternative reasoning component"
}
"""


class ElenchusProbe:
    """
    Runs the Deutsch Probe on agent recommendations.
    Each recommendation's reasoning components are individually perturbed
    and tested for load-bearing status.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.client = client
        self.model = model

    def probe(self, rec: Recommendation) -> ElenchusResult:
        """
        Probe a single recommendation. For each reasoning component:
        1. Generate a plausible replacement
        2. Test if the conclusion survives the swap
        """
        if not rec.reasoning_components:
            return ElenchusResult(
                recommendation=rec,
                probe_results=[],
                deutsch_score=0.0,
                total_components=0,
                load_bearing_count=0,
            )

        probe_results = []
        load_bearing = 0

        for i, component in enumerate(rec.reasoning_components):
            # Step 1: Generate replacement
            replacement = self._generate_replacement(component, rec)

            # Step 2: Test if conclusion survives the swap
            result = self._test_swap(rec, i, replacement)
            probe_results.append(result)

            if not result.conclusion_survived:
                load_bearing += 1

        total = len(rec.reasoning_components)
        score = load_bearing / total if total > 0 else 0.0

        return ElenchusResult(
            recommendation=rec,
            probe_results=probe_results,
            deutsch_score=score,
            total_components=total,
            load_bearing_count=load_bearing,
        )

    def _generate_replacement(self, component: str, rec: Recommendation) -> str:
        """Generate a plausible alternative for one reasoning component."""
        context = (
            f"Original recommendation: {rec.ticker} {rec.direction} "
            f"(conviction {rec.conviction:.2f})\n\n"
            f"Reasoning component to replace:\n{component}"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=PERTURBATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}],
        )

        try:
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
            data = json.loads(raw.strip())
            return data.get("replacement", component)
        except (json.JSONDecodeError, KeyError):
            return f"[REPLACEMENT FAILED] {component}"

    def _test_swap(
        self,
        rec: Recommendation,
        swap_index: int,
        replacement: str,
    ) -> ProbeResult:
        """Test whether the conclusion survives replacing one component."""
        modified_components = list(rec.reasoning_components)
        original = modified_components[swap_index]
        modified_components[swap_index] = f"[REPLACED] {replacement}"

        context = (
            f"Recommendation being tested: {rec.ticker} {rec.direction} "
            f"(conviction {rec.conviction:.2f})\n\n"
            f"Modified reasoning components:\n"
        )
        for j, comp in enumerate(modified_components):
            marker = " ← THIS ONE WAS REPLACED" if j == swap_index else ""
            context += f"  {j+1}. {comp}{marker}\n"

        context += (
            f"\nOriginal conclusion: {rec.conclusion}\n\n"
            f"Does the same conclusion (same ticker, same direction) still logically follow "
            f"from the MODIFIED reasoning?"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=PROBE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}],
        )

        try:
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
            data = json.loads(raw.strip())
            return ProbeResult(
                component_index=swap_index,
                original_component=original,
                replacement_component=replacement,
                conclusion_survived=bool(data.get("conclusion_survives", True)),
                probe_reasoning=data.get("explanation", ""),
            )
        except (json.JSONDecodeError, KeyError):
            # Parse failure — assume component is decorative (conservative)
            return ProbeResult(
                component_index=swap_index,
                original_component=original,
                replacement_component=replacement,
                conclusion_survived=True,
                probe_reasoning="[PROBE PARSE FAILURE]",
            )

    def probe_batch(self, recs: list[Recommendation]) -> list[ElenchusResult]:
        """Probe all recommendations from a single day."""
        return [self.probe(rec) for rec in recs]
