"""
Autoresearch loop for prompt evolution.
Identifies worst agent by rolling Sharpe, generates a targeted prompt mutation,
evaluates over N trading days, keeps or reverts via git.

This is the ATLAS autoresearch pattern: prompts are the weights,
Sharpe is the loss function, git is the checkpoint mechanism.
"""

import json
import subprocess
from pathlib import Path
from dataclasses import dataclass

from .agent import Agent
from .llm import completion_with_retry


@dataclass
class MutationResult:
    agent_id: str
    sharpe_before: float
    sharpe_after: float
    kept: bool
    mutation_description: str
    prompt_diff_summary: str


MUTATOR_SYSTEM_PROMPT = """You are a prompt engineer specializing in financial analysis agent prompts.
You will receive:
1. An agent's current system prompt
2. Its role (macro, sector, risk, etc.)
3. Its recent performance metrics (rolling Sharpe, hit rate)
4. Specific weaknesses observed

Your job: produce a TARGETED modification to the prompt that addresses the weakness.

Rules:
- Make ONE focused change, not a rewrite
- Preserve the agent's core role and analytical framework
- Add specificity where the prompt is vague
- Add decision criteria where the prompt lacks them
- If the agent is overconfident, add hedging/uncertainty requirements
- If the agent is too conservative, sharpen conviction thresholds

Respond with JSON only:
{
  "mutation_description": "what you changed and why",
  "modified_prompt": "the full modified prompt text"
}
"""


class AutoresearchLoop:
    """
    The prompt mutation engine. Operates on git branches.
    
    Loop:
        1. Identify worst agent by rolling Sharpe
        2. Generate targeted prompt mutation
        3. Run pipeline for eval_window trading days
        4. Compare Sharpe before/after
        5. If improved: git commit (keep)
        6. If worse: git checkout (revert)
    """

    def __init__(
        self,
        repo_dir: Path,
        eval_window: int = 5,
        model: str = "openrouter/qwen/qwen3-235b-a22b",
    ):
        self.repo_dir = repo_dir
        self.eval_window = eval_window
        self.model = model

    def mutate_agent(self, agent: Agent) -> str | None:
        """
        Generate a prompt mutation for the given agent.
        Returns the modified prompt text, or None if mutation failed.
        """
        context = (
            f"Agent ID: {agent.agent_id}\n"
            f"Layer: {agent.layer}\n"
            f"Rolling Sharpe: {agent.rolling_sharpe:.3f}\n"
            f"Darwinian Weight: {agent.darwinian_weight:.3f}\n"
            f"Total recommendations: {agent.total_recommendations}\n"
            f"Correct recommendations: {agent.correct_recommendations}\n"
            f"Hit rate: {agent.correct_recommendations / max(agent.total_recommendations, 1):.1%}\n\n"
            f"Current prompt:\n```\n{agent.prompt}\n```"
        )

        try:
            response = completion_with_retry(
                model=self.model,
                max_tokens=3000,
                messages=[
                    {"role": "system", "content": MUTATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
            )

            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]

            data = json.loads(raw.strip())

            # Defensive parsing — Qwen sometimes returns a list instead of a dict
            if isinstance(data, list):
                data = data[0] if data and isinstance(data[0], dict) else {}
            if isinstance(data, dict):
                result = data.get("modified_prompt")
                if isinstance(result, str) and result.strip():
                    return result
                elif isinstance(result, list):
                    # List of strings — join them
                    return "\n".join(str(r) for r in result) if result else None
                return None
            elif isinstance(data, str) and data.strip():
                # Bare string response — might be the prompt itself
                return data
            return None
        except Exception as e:
            print(f"  Mutation generation failed: {e}")
            return None

    def apply_mutation(self, agent: Agent, new_prompt: str) -> None:
        """Write mutated prompt to disk."""
        agent.prompt = new_prompt

    def revert_mutation(self, agent: Agent) -> None:
        """Git checkout the agent's prompt file to revert."""
        try:
            subprocess.run(
                ["git", "checkout", "--", str(agent.prompt_path)],
                cwd=self.repo_dir,
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"  Git revert failed: {e}")

    def commit_mutation(self, agent: Agent, description: str) -> None:
        """Git commit the successful mutation."""
        try:
            subprocess.run(
                ["git", "add", str(agent.prompt_path)],
                cwd=self.repo_dir,
                capture_output=True,
                check=True,
            )
            msg = f"autoresearch: {agent.agent_id} — {description}"
            subprocess.run(
                ["git", "commit", "-m", msg],
                cwd=self.repo_dir,
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"  Git commit failed: {e}")

    def evaluate(
        self,
        sharpe_before: float,
        sharpe_after: float,
        agent: Agent,
        mutation_description: str,
    ) -> MutationResult:
        """Keep or discard based on Sharpe comparison."""
        kept = sharpe_after > sharpe_before

        if kept:
            self.commit_mutation(agent, mutation_description)
            print(f"  KEEP: {agent.agent_id} Sharpe {sharpe_before:.3f} → {sharpe_after:.3f}")
        else:
            self.revert_mutation(agent)
            print(f"  REVERT: {agent.agent_id} Sharpe {sharpe_before:.3f} → {sharpe_after:.3f}")

        return MutationResult(
            agent_id=agent.agent_id,
            sharpe_before=sharpe_before,
            sharpe_after=sharpe_after,
            kept=kept,
            mutation_description=mutation_description,
            prompt_diff_summary=f"{'KEPT' if kept else 'REVERTED'}",
        )
