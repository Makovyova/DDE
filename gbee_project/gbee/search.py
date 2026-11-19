"""Optional bounded beam refinement; separate from policy-gradient learning.

This is inference with explicit candidate evaluation against training data.
It must not be reported as a formula discovered by an RL rollout.  With an
unlimited proposal count each beam state expands all legal non-STOP edits;
the neural policy then determines only deterministic proposal/tie order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .dag import ExpressionDAG
from .policy import MaskedEditorPolicy
from .rl import HybridEnvironment, ScoredGraph


@dataclass(frozen=True)
class BeamConfig:
    """Explicit inference budget. Zero proposal_limit means all legal edits."""

    width: int = 8
    depth: int = 3
    proposal_limit: int = 0
    target_mse: float = 1e-12

    def __post_init__(self) -> None:
        for name, minimum in (("width", 1), ("depth", 0), ("proposal_limit", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if isinstance(self.target_mse, bool) or not math.isfinite(self.target_mse) or self.target_mse < 0:
            raise ValueError("target_mse must be finite and nonnegative")


@dataclass(frozen=True)
class BeamResult:
    """Best reward plus transparent work counters.

    ``expanded`` counts parent states whose policy was evaluated. ``evaluated``
    counts environment.score calls, including the initial graph, duplicates,
    and environment cache hits. ``depth_reached`` counts completed beam levels.
    Actual SciPy fitting counts are available separately on the environment.
    """

    best: ScoredGraph
    expanded: int
    evaluated: int
    depth_reached: int


def _score(environment: HybridEnvironment, graph: ExpressionDAG) -> ScoredGraph:
    scored = environment.score(graph)
    # Defensive boundary validation: fitting may change constants, never the
    # graph's topology, parameter identities, arities, or operand order.
    ExpressionDAG(scored.graph.nodes, scored.graph.root)
    if scored.graph.signature(False) != graph.signature(False):
        raise RuntimeError("Constant fitting unexpectedly changed graph structure")
    if not math.isfinite(scored.mse) or scored.mse < 0 or not math.isfinite(scored.reward):
        raise FloatingPointError("Beam search received a nonfinite or invalid score")
    return scored


@torch.no_grad()
def beam_refine(
    policy: MaskedEditorPolicy,
    environment: HybridEnvironment,
    initial_graph: ExpressionDAG,
    config: BeamConfig = BeamConfig(),
) -> BeamResult:
    """Search with fixed neural weights, training-only fits, and bounded depth.

    At each level, legal proposals are ordered by their *actual normalized*
    policy log probability. A positive ``proposal_limit`` keeps only the first
    K proposals per parent; zero evaluates all legal non-STOP proposals.
    Fitted candidates and the previous beam are ranked by the same reward as
    REINFORCE. Carrying previous states makes stopping a permitted choice.

    After fitting, equal ``signature(include_constants=False)`` states are
    merged, retaining the representative with the highest reward. This saves
    beam slots but deliberately discards worse local-optimizer basins that
    might become useful after a later edit; it is not a global-optimality claim.
    Equal rewards preserve insertion order, which follows policy proposal order.

    The MSE stopping rule is checked after each complete level on the best
    reward state. No held-out data are accepted, no gradients are accumulated,
    existing parameter gradients are untouched, and every module's prior
    training/evaluation mode is restored even if scoring fails.
    """
    if not isinstance(config, BeamConfig):
        raise TypeError("config must be a BeamConfig")
    if environment.X.shape[1] != policy.action_space.n_variables:
        raise ValueError("Training columns do not match the policy vocabulary")
    policy.action_space.prepare(initial_graph)  # Check vocabulary and budget.
    modes = [(module, module.training) for module in policy.modules()]
    policy.eval()
    try:
        initial = _score(environment, initial_graph)
        beam = [initial]
        best = initial
        expanded, evaluated, depth_reached = 0, 1, 0
        if best.mse <= config.target_mse:
            return BeamResult(best, expanded, evaluated, depth_reached)

        for depth in range(1, config.depth + 1):
            # Insertion-ordered dict gives deterministic tie breaking. The
            # previous beam precedes new candidates, representing STOP choices.
            survivors: dict[tuple, ScoredGraph] = {item.graph.signature(False): item for item in beam}
            for state in beam:
                output = policy(state.graph, environment.X, environment.y)
                expanded += 1
                indices = torch.arange(len(output.candidates.actions), dtype=torch.long, device=policy.device)
                log_probabilities = output.distribution.log_prob(indices).cpu().tolist()
                proposals = [
                    i
                    for i, action in enumerate(output.candidates.actions)
                    if output.candidates.mask[i] and action.kind != "stop"
                ]
                proposals.sort(key=lambda i: (-log_probabilities[i], i))
                if config.proposal_limit:
                    proposals = proposals[: config.proposal_limit]

                for index in proposals:
                    successor = output.candidates.graphs[index]
                    if successor is None:
                        raise RuntimeError("A legal beam proposal has no successor DAG")
                    scored = _score(environment, successor)
                    evaluated += 1
                    signature = scored.graph.signature(False)
                    previous = survivors.get(signature)
                    if previous is None or scored.reward > previous.reward:
                        survivors[signature] = scored
                    if scored.reward > best.reward:
                        best = scored

            # Python's stable sort preserves insertion order for equal reward.
            beam = sorted(survivors.values(), key=lambda item: -item.reward)[: config.width]
            depth_reached = depth
            if best.mse <= config.target_mse:
                break
        return BeamResult(best, expanded, evaluated, depth_reached)
    finally:
        for module, training in modes:
            module.training = training
