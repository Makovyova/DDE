"""Small, deterministic checks of policy-gradient and persistence contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gbee.actions import KINDS, ActionSpace
from gbee.constants import ConstantOptimizer
from gbee.dag import ExpressionDAG, Node
from gbee.network import EditorNetwork, graph_to_data
from gbee.policy import MaskedEditorPolicy
from gbee.rl import HybridEnvironment, ReinforceTrainer, TrainConfig, reward


def make_problem() -> tuple[np.ndarray, np.ndarray, ExpressionDAG]:
    rng = np.random.default_rng(41)
    X = rng.uniform(-1.5, 1.5, (24, 2))
    y = 1.7 * X[:, 0] + np.sin(X[:, 1]) + 0.13 * X[:, 0] * X[:, 1]
    graph = ExpressionDAG.build((Node.variable(0), Node.variable(1), Node.operator("+", (0, 1))), 2)
    return X, y, graph


def make_policy() -> MaskedEditorPolicy:
    torch.manual_seed(71)
    return MaskedEditorPolicy(
        EditorNetwork(n_variables=2, hidden_dim=16, num_layers=1),
        ActionSpace(n_variables=2, max_nodes=7, max_constants=2),
    )


def make_trainer(horizon: int = 3) -> ReinforceTrainer:
    X, y, graph = make_problem()
    return ReinforceTrainer(
        make_policy(),
        HybridEnvironment(X, y, ConstantOptimizer(maxiter=15)),
        graph,
        TrainConfig(
            epochs=1,
            episodes_per_batch=4,
            horizon=horizon,
            seed=19,
            entropy_coefficient=0.02,
            target_mse=0.0,
        ),
    )


def test_joint_policy_normalization_and_masked_mutation_formula() -> None:
    X, y, graph = make_problem()
    policy = make_policy()
    output = policy(graph, X, y)
    probabilities = output.distribution.probs
    mask = torch.tensor(output.candidates.mask.copy(), dtype=torch.bool)
    assert probabilities.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert torch.equal(probabilities[~mask], torch.zeros_like(probabilities[~mask]))
    assert torch.isfinite(output.distribution.entropy())

    encoded = policy.network(graph_to_data(graph, X, y, 2))
    available = torch.tensor(
        [
            any(a.kind == kind and output.candidates.mask[i] for i, a in enumerate(output.candidates.actions))
            for kind in KINDS
        ]
    )
    kind_scores = policy.network.kind_logits(encoded.graph_embedding)
    stop_adjustment = torch.zeros_like(kind_scores)
    stop_adjustment[0] = policy.network.stop_logit(encoded.graph_embedding)
    expected_kinds = torch.softmax((kind_scores + stop_adjustment).masked_fill(~available, -torch.inf), dim=0)
    actual_kinds = torch.stack(
        [probabilities[[i for i, a in enumerate(output.candidates.actions) if a.kind == kind]].sum() for kind in KINDS]
    )
    torch.testing.assert_close(actual_kinds, expected_kinds, atol=1e-6, rtol=1e-6)

    # Conditioning on a specific node must recover paper equation (1), even
    # though the actual sampler also chooses a family and a node.
    indices = [
        i
        for i, a in enumerate(output.candidates.actions)
        if a.kind == "mutate" and a.node == graph.root and output.candidates.mask[i]
    ]
    assert len(indices) > 1
    types = [policy.network.vocab.index(output.candidates.actions[i].symbol) for i in indices]
    expected = torch.softmax(policy.network.mutation_logits(encoded.h)[graph.root, types], dim=0)
    actual = probabilities[indices] / probabilities[indices].sum()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_edge_sigmoid_mask_is_exact_and_differentiable() -> None:
    X, y, graph = make_problem()
    network = make_policy().network
    encoded = network(graph_to_data(graph, X, y, 2))
    sources = torch.tensor([0, graph.root])
    destinations = torch.tensor([graph.root, graph.root])
    mask = torch.tensor([True, False])
    probabilities = network.edge_probabilities(encoded.h, sources, destinations, mask)
    expected = torch.sigmoid(network.edge_logits(encoded.h, sources, destinations)[0])
    torch.testing.assert_close(probabilities[0], expected)
    assert probabilities[1].item() == 0.0
    assert torch.equal(
        network.edge_probabilities(encoded.h, sources, destinations, torch.zeros_like(mask)),
        torch.zeros_like(probabilities),
    )
    probabilities.sum().backward()
    grads = [p.grad for p in network.mlp_edge.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads if g is not None) > 0.0


def test_one_step_reinforce_uses_actual_sample_logprob_and_past_baseline() -> None:
    trainer = make_trainer(horizon=1)
    trainer.baseline = 0.23
    state = trainer.environment.score(trainer.initial_graph)
    output = trainer.policy(state.graph, trainer.environment.X, trainer.environment.y)
    generator = torch.Generator().set_state(trainer.generator.get_state())
    index = torch.multinomial(output.distribution.probs, 1, generator=generator).squeeze(0)
    selected = int(index)
    action = output.candidates.actions[selected]
    successor = state if action.kind == "stop" else trainer.environment.score(output.candidates.graphs[selected])
    advantage = successor.reward - state.reward - trainer.baseline
    expected = -output.distribution.log_prob(index) * advantage
    actual, entropy, improvements, terminal_reward, length = trainer._episode()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(entropy, output.distribution.entropy())
    assert improvements == pytest.approx([successor.reward - state.reward])
    assert terminal_reward == pytest.approx(successor.reward)
    assert length == 1
    assert trainer.baseline == 0.23  # Updated only after the complete batch.
    assert torch.equal(trainer.generator.get_state(), generator.get_state())


def test_small_on_policy_update_changes_weights_with_finite_gradients() -> None:
    trainer = make_trainer()
    before = {name: p.detach().clone() for name, p in trainer.policy.named_parameters()}
    versions: list[tuple[int, ...]] = []

    def observe_forward(_module, _inputs):
        versions.append(tuple(p._version for p in trainer.policy.parameters()))

    handle = trainer.policy.register_forward_pre_hook(observe_forward)
    try:
        result = trainer.train()
    finally:
        handle.remove()
    assert result.epochs_completed == 1
    assert result.episodes == 4
    assert versions and all(v == versions[0] for v in versions)
    assert all(np.isfinite(row["loss"]) and np.isfinite(row["grad_norm"]) for row in result.history)
    assert any(not torch.equal(before[name], p) for name, p in trainer.policy.named_parameters())
    for name, parameter in trainer.policy.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
    for head in ("encoder", "mlp_mut", "mlp_edge", "mlp_wrap", "mlp_collapse", "mlp_kind", "mlp_stop"):
        gradients = [
            p.grad
            for name, p in trainer.policy.network.named_parameters()
            if name.startswith(head) and p.grad is not None
        ]
        assert gradients and sum(float(g.abs().sum()) for g in gradients) > 0.0, head


def test_environment_owns_training_data_and_reward_uses_exact_complexity() -> None:
    X, y, graph = make_problem()
    original_X, original_y = X.copy(), y.copy()
    environment = HybridEnvironment(X, y, complexity_weight=0.007)
    fingerprint = environment.fingerprint
    X[:] = 999.0
    y[:] = -999.0
    np.testing.assert_array_equal(environment.X, original_X)
    np.testing.assert_array_equal(environment.y, original_y)
    assert environment.fingerprint == fingerprint
    with pytest.raises(ValueError):
        environment.y[0] = 1.0
    scored = environment.score(graph)
    independently_computed = np.mean((original_X.sum(axis=1) - original_y) ** 2)
    assert scored.mse == pytest.approx(independently_computed)
    assert scored.reward == pytest.approx(1.0 / (1.0 + independently_computed) - 0.007 * 5)
    assert reward(graph, independently_computed, 0.007) == scored.reward


def test_checkpoint_weights_only_roundtrip_and_deterministic_resume(tmp_path: Path) -> None:
    original = make_trainer()
    original.train()
    checkpoint = tmp_path / "checkpoint.pt"
    original.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    assert isinstance(payload["schema_version"], int)

    restored = make_trainer()
    fit_calls_before_restore = restored.environment.fit_calls
    restored.load_checkpoint(checkpoint)
    assert restored.environment.fit_calls == fit_calls_before_restore
    assert restored.baseline == original.baseline
    assert restored.episodes == original.episodes
    assert restored.history == original.history
    for name, parameter in original.policy.state_dict().items():
        torch.testing.assert_close(restored.policy.state_dict()[name], parameter, rtol=0, atol=0)
    assert torch.equal(restored.generator.get_state(), original.generator.get_state())

    result_original = original.train()
    result_restored = restored.train()
    assert result_original.epochs_completed == result_restored.epochs_completed == 2
    assert result_original.episodes == result_restored.episodes == 8
    assert result_original.best.graph.signature() == result_restored.best.graph.signature()
    assert result_original.best.reward == result_restored.best.reward
    assert result_original.history == result_restored.history
    for name, parameter in original.policy.state_dict().items():
        torch.testing.assert_close(restored.policy.state_dict()[name], parameter, rtol=0, atol=0)


@pytest.mark.parametrize("changed", ("targets", "constant_optimizer"))
def test_checkpoint_rejects_changed_training_problem(tmp_path: Path, changed: str) -> None:
    trainer = make_trainer()
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)
    X, y, graph = make_problem()
    if changed == "targets":
        y = y + 0.1
    other = ReinforceTrainer(
        make_policy(),
        HybridEnvironment(X, y, ConstantOptimizer(maxiter=16 if changed == "constant_optimizer" else 15)),
        graph,
        trainer.config,
    )
    with pytest.raises(ValueError, match="data|reward"):
        other.load_checkpoint(checkpoint)
