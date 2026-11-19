"""Bounded, training-only beam inference without oracle graph inputs."""

import numpy as np
import pytest
import torch

from gbee.actions import ActionSpace
from gbee.dag import ExpressionDAG, Node
from gbee.network import EditorNetwork
from gbee.policy import MaskedEditorPolicy
from gbee.rl import HybridEnvironment
from gbee.search import BeamConfig, beam_refine


def problem():
    # The test supplies only observations and an x1 starting expression, never
    # a target expression graph or a list of oracle editing actions.
    X = np.array([[-2.0, 1.0], [-1.0, -2.0], [0.0, 2.0], [1.0, -1.0], [2.0, 0.5]])
    y = X[:, 0] + X[:, 1]
    initial = ExpressionDAG.build([Node.variable(0)], 0)
    torch.manual_seed(21)
    policy = MaskedEditorPolicy(
        EditorNetwork(2, hidden_dim=12, num_layers=1),
        ActionSpace(2, max_nodes=4, max_constants=1, operators=("+",)),
    )
    return X, y, initial, policy, HybridEnvironment(X, y)


@pytest.mark.parametrize("training", [True, False])
def test_one_step_exhaustive_beam_finds_generic_sum_without_changing_weights(training):
    X, y, initial, policy, environment = problem()
    policy.train(training)
    weights = {name: value.clone() for name, value in policy.state_dict().items()}
    result = beam_refine(policy, environment, initial, BeamConfig(width=4, depth=1))
    assert result.best.mse < 1e-20
    np.testing.assert_allclose(result.best.graph.evaluate(X), y, atol=1e-12)
    assert result.expanded == 1 and result.evaluated > 1 and result.depth_reached == 1
    assert policy.training is training
    assert all(parameter.grad is None for parameter in policy.parameters())
    for name, value in policy.state_dict().items():
        assert torch.equal(value, weights[name])
    np.testing.assert_array_equal(environment.X, X)
    np.testing.assert_array_equal(environment.y, y)


def test_zero_depth_returns_scored_initial_expression():
    _, _, initial, policy, environment = problem()
    result = beam_refine(policy, environment, initial, BeamConfig(depth=0))
    assert result.best.graph == initial
    assert (result.expanded, result.evaluated, result.depth_reached) == (0, 1, 0)


def test_proposal_limit_bounds_scoring_and_restores_mixed_module_modes():
    _, _, initial, policy, environment = problem()
    policy.train()
    policy.network.encoder.eval()
    modes = [module.training for module in policy.modules()]
    config = BeamConfig(width=2, depth=2, proposal_limit=1, target_mse=0)
    result = beam_refine(policy, environment, initial, config)
    assert result.expanded <= 1 + config.width
    assert result.evaluated <= 1 + result.expanded * config.proposal_limit
    assert [module.training for module in policy.modules()] == modes


@pytest.mark.parametrize(
    "parameters",
    [
        {"width": 0},
        {"width": True},
        {"depth": -1},
        {"depth": 1.5},
        {"proposal_limit": -1},
        {"target_mse": float("nan")},
        {"target_mse": -1},
    ],
)
def test_invalid_beam_budgets_rejected(parameters):
    with pytest.raises(ValueError):
        BeamConfig(**parameters)
