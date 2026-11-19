"""Hybrid SciPy environment and terminal-reward REINFORCE training."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from .constants import ConstantFit, ConstantOptimizer, validate_regression_data
from .dag import ExpressionDAG
from .policy import MaskedEditorPolicy


def reward(graph: ExpressionDAG, mse: float, complexity_weight: float = 1e-3) -> float:
    """Source equation (4): 1/(1+MSE) - lambda*(|V|+|E|)."""
    if not math.isfinite(mse) or mse < 0:
        raise ValueError("mse must be finite and nonnegative")
    if not math.isfinite(complexity_weight) or complexity_weight < 0:
        raise ValueError("complexity_weight must be finite and nonnegative")
    return 1.0 / (1.0 + mse) - complexity_weight * graph.complexity


@dataclass(frozen=True)
class ScoredGraph:
    graph: ExpressionDAG
    mse: float
    reward: float
    fit: ConstantFit


class HybridEnvironment:
    """Fit constants only against immutable training data, then compute reward.

    The per-environment bounded cache includes the starting constant values:
    different local-optimizer basins must not accidentally share a cached fit.
    Validation and test sets never enter this object.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        optimizer: ConstantOptimizer | None = None,
        complexity_weight: float = 1e-3,
        cache_size: int = 2048,
    ) -> None:
        matrix, target = validate_regression_data(X, y)
        if not math.isfinite(complexity_weight) or complexity_weight < 0:
            raise ValueError("complexity_weight must be finite and nonnegative")
        if not isinstance(cache_size, int) or isinstance(cache_size, bool) or cache_size < 0:
            raise ValueError("cache_size must be a nonnegative integer")
        self.X, self.y = matrix.copy(), target.copy()
        self.X.setflags(write=False)
        self.y.setflags(write=False)
        self.optimizer = optimizer or ConstantOptimizer()
        self.complexity_weight = complexity_weight
        self.cache_size = cache_size
        self._cache: OrderedDict[ExpressionDAG, ScoredGraph] = OrderedDict()
        self.fit_calls = 0
        self.cache_hits = 0

    def score(self, graph: ExpressionDAG) -> ScoredGraph:
        if graph in self._cache:
            self._cache.move_to_end(graph)
            self.cache_hits += 1
            return self._cache[graph]
        fit = self.optimizer.fit(graph, self.X, self.y)
        self.fit_calls += 1
        scored = ScoredGraph(fit.graph, fit.mse, reward(fit.graph, fit.mse, self.complexity_weight), fit)
        if self.cache_size:
            self._cache[graph] = scored
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return scored

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for array in (self.X, self.y):
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        digest.update(str(self.complexity_weight).encode())
        digest.update(
            json.dumps(
                {key: getattr(self.optimizer, key) for key in ("method", "maxiter", "restarts", "seed", "penalty")},
                sort_keys=True,
            ).encode()
        )
        return digest.hexdigest()


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 120
    episodes_per_batch: int = 16
    horizon: int = 5
    learning_rate: float = 1e-3
    entropy_coefficient: float = 0.01
    baseline_decay: float = 0.9
    max_grad_norm: float = 1.0
    target_mse: float = 1e-12
    seed: int = 7

    def __post_init__(self) -> None:
        for name in ("epochs", "episodes_per_batch", "horizon"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("learning_rate", "max_grad_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("entropy_coefficient", "target_mse"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if not math.isfinite(self.baseline_decay) or not 0 <= self.baseline_decay < 1:
            raise ValueError("baseline_decay must belong to [0,1)")


@dataclass(frozen=True)
class TrainingResult:
    best: ScoredGraph
    history: tuple[dict, ...]
    epochs_completed: int
    episodes: int


class ReinforceTrainer:
    """On-policy REINFORCE, with an action-independent state reward baseline.

    For a trajectory, advantage_t = R(G_T) - R(G_t) - b_previous_batch.
    Subtracting the detached current-state reward and a past-data EMA does not
    change the expected policy gradient. It is also the undiscounted return of
    potential differences R(G_{t+1})-R(G_t). We do NOT discount rewards or add
    intermediate absolute rewards: either would change the requested objective.

    The loss sums decision log probabilities within each episode and averages
    episodes. Optional entropy regularization encourages exploration; set its
    coefficient to zero for the unregularized objective J(theta).
    """

    def __init__(
        self,
        policy: MaskedEditorPolicy,
        environment: HybridEnvironment,
        initial_graph: ExpressionDAG,
        config: TrainConfig | None = None,
    ) -> None:
        self.policy = policy
        self.environment = environment
        self.initial_graph = initial_graph
        self.config = config or TrainConfig()
        if environment.X.shape[1] != policy.action_space.n_variables:
            raise ValueError("training columns do not match policy variables")
        policy.action_space.prepare(initial_graph)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.config.learning_rate)
        self.baseline = 0.0
        self.epochs_completed = 0
        self.episodes = 0
        self.history: list[dict] = []
        self.best = environment.score(initial_graph)
        self.generator = torch.Generator(device=policy.device).manual_seed(self.config.seed)

    def _consider(self, scored: ScoredGraph) -> None:
        if scored.reward > self.best.reward:
            self.best = scored

    def _episode(self) -> tuple[Tensor, Tensor, list[float], float, int]:
        state = self.environment.score(self.initial_graph)
        log_probs: list[Tensor] = []
        entropies: list[Tensor] = []
        state_rewards: list[float] = []
        for _ in range(self.config.horizon):
            output = self.policy(state.graph, self.environment.X, self.environment.y)
            index = torch.multinomial(output.distribution.probs, 1, generator=self.generator).squeeze(0)
            selected = int(index.item())
            log_probs.append(output.distribution.log_prob(index))
            entropies.append(output.distribution.entropy())
            state_rewards.append(state.reward)
            action = output.candidates.actions[selected]
            if action.kind == "stop":
                break
            successor = output.candidates.graphs[selected]
            if successor is None:
                raise RuntimeError("policy sampled an invalid masked action")
            state = self.environment.score(successor)
            self._consider(state)
            if state.mse <= self.config.target_mse:
                break
        improvements = [state.reward - before for before in state_rewards]
        advantages = torch.tensor(
            [value - self.baseline for value in improvements], dtype=log_probs[0].dtype, device=self.policy.device
        )
        loss = -(torch.stack(log_probs) * advantages).sum()
        entropy = torch.stack(entropies).sum()
        self.episodes += 1
        return loss, entropy, improvements, state.reward, len(log_probs)

    def train(self, callback: Callable[[dict], None] | None = None) -> TrainingResult:
        self.policy.train()
        for _ in range(self.config.epochs):
            losses, entropies, improvements, terminal_rewards, lengths = [], [], [], [], []
            # All rollouts in this batch use exactly the same policy weights.
            for _ in range(self.config.episodes_per_batch):
                loss, entropy, delta, terminal_reward, length = self._episode()
                losses.append(loss)
                entropies.append(entropy)
                improvements.extend(delta)
                terminal_rewards.append(terminal_reward)
                lengths.append(length)
            loss = torch.stack(losses).mean() - self.config.entropy_coefficient * torch.stack(entropies).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite policy loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.max_grad_norm, error_if_nonfinite=True
            )
            self.optimizer.step()
            self.baseline = self.config.baseline_decay * self.baseline + (1 - self.config.baseline_decay) * float(
                np.mean(improvements)
            )
            self.epochs_completed += 1
            row = {
                "epoch": self.epochs_completed,
                "episodes": self.episodes,
                "loss": float(loss.detach()),
                "grad_norm": float(grad_norm),
                "mean_terminal_reward": float(np.mean(terminal_rewards)),
                "mean_length": float(np.mean(lengths)),
                "best_reward": self.best.reward,
                "best_train_mse": self.best.mse,
                "best_expression": self.best.graph.to_expression(precision=8),
            }
            self.history.append(row)
            if callback is not None:
                callback(row.copy())
            # This is a practical accuracy stopping rule, not a proof that the
            # globally best reward/shortest equivalent expression was found.
            if self.best.mse <= self.config.target_mse:
                break
        return TrainingResult(self.best, tuple(self.history), self.epochs_completed, self.episodes)

    @torch.no_grad()
    def greedy_rollout(self) -> ScoredGraph:
        """Evaluate learned greedy behavior separately from the best sampled DAG."""
        training = self.policy.training
        self.policy.eval()
        try:
            state = self.environment.score(self.initial_graph)
            for _ in range(self.config.horizon):
                output = self.policy(state.graph, self.environment.X, self.environment.y)
                index = int(output.distribution.probs.argmax().item())
                if output.candidates.actions[index].kind == "stop":
                    break
                successor = output.candidates.graphs[index]
                if successor is None:
                    raise RuntimeError("greedy policy selected an illegal action")
                state = self.environment.score(successor)
                if state.mse <= self.config.target_mse:
                    break
            return state
        finally:
            self.policy.train(training)

    def save_checkpoint(self, path: str | Path) -> None:
        """Atomically save portable state_dicts, RNG state, config, and best DAG."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        state = {
            "schema_version": 2,
            "model": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": asdict(self.config),
            "data_fingerprint": self.environment.fingerprint,
            "initial_graph": self.initial_graph.to_dict(),
            "best_graph": self.best.graph.to_dict(),
            "best_fit": {
                "mse": self.best.mse,
                "initial_mse": self.best.fit.initial_mse,
                "success": self.best.fit.success,
                "message": self.best.fit.message,
                "nfev": self.best.fit.nfev,
            },
            "baseline": self.baseline,
            "epochs_completed": self.epochs_completed,
            "episodes": self.episodes,
            "history": self.history,
            "generator_state": self.generator.get_state(),
            "grammar": {
                "n_variables": self.policy.action_space.n_variables,
                "operators": list(self.policy.action_space.operators),
                "max_nodes": self.policy.action_space.max_nodes,
                "max_constants": self.policy.action_space.max_constants,
            },
        }
        torch.save(state, temporary)
        os.replace(temporary, destination)

    def load_checkpoint(self, path: str | Path) -> None:
        """Resume into the same architecture, grammar, dataset and configuration."""
        state = torch.load(path, map_location=self.policy.device, weights_only=True)
        grammar = {
            "n_variables": self.policy.action_space.n_variables,
            "operators": list(self.policy.action_space.operators),
            "max_nodes": self.policy.action_space.max_nodes,
            "max_constants": self.policy.action_space.max_constants,
        }
        if state.get("schema_version") != 2 or state["data_fingerprint"] != self.environment.fingerprint:
            raise ValueError("checkpoint schema or training data/reward differs")
        if state["grammar"] != grammar or state["initial_graph"] != self.initial_graph.to_dict():
            raise ValueError("checkpoint grammar or initial graph differs")
        if state["config"] != asdict(self.config):
            raise ValueError("checkpoint training configuration differs")
        self.policy.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator_state"].cpu())
        self.baseline = float(state["baseline"])
        self.epochs_completed, self.episodes = int(state["epochs_completed"]), int(state["episodes"])
        self.history = state["history"]
        graph = ExpressionDAG.from_dict(state["best_graph"])
        # Restoring an artifact must not run another local fit or alter the
        # saved incumbent, optimizer basin, metrics, or early-stopping state.
        fit = ConstantFit(graph=graph, **state["best_fit"])
        self.best = ScoredGraph(graph, fit.mse, reward(graph, fit.mse, self.environment.complexity_weight), fit)


def write_json(path: str | Path, value: dict | list) -> None:
    """Write UTF-8 JSON atomically, rejecting nonstandard NaN/Infinity tokens."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
