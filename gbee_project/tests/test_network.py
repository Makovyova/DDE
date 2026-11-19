"""Numerical observations, message direction/position, and neural gradient checks."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from gbee.dag import ExpressionDAG, Node
from gbee.network import EditorNetwork, NodeTypeVocabulary, graph_to_data


@pytest.fixture
def observations():
    rng = np.random.default_rng(17)
    X = rng.uniform(-2.0, 2.0, size=(40, 2))
    return X, 2.5 * X[:, 0] + np.sin(X[:, 1])


@pytest.fixture
def graph():
    return ExpressionDAG.build(
        [
            Node.variable(0),
            Node.variable(1),
            Node.constant(2.5, "c0"),
            Node.operator("*", (0, 2)),
            Node.operator("sin", (1,)),
            Node.operator("+", (3, 4)),
        ],
        5,
    )


@pytest.fixture
def model():
    torch.manual_seed(11)
    return EditorNetwork(2, hidden_dim=24, num_layers=2)


def test_node_vocabulary_has_stable_indices():
    vocabulary = NodeTypeVocabulary(2)
    assert vocabulary.labels == ("+", "*", "-", "/", "sin", "exp", "x1", "x2", "const")
    assert vocabulary.node_type(Node.variable(1)) == 7
    assert vocabulary.node_type(Node.constant(0.0, "a")) == 8
    with pytest.raises(ValueError):
        vocabulary.node_type(Node.variable(2))


def test_observations_are_finite_and_sample_permutation_invariant(graph, observations):
    X, y = observations
    original = graph_to_data(graph, X, y, 2)
    shuffled = graph_to_data(graph, X[::-1].copy(), y[::-1].copy(), 2)
    assert original.node_stats.shape == (len(graph.nodes), 9)
    assert torch.isfinite(original.node_stats).all()
    assert torch.allclose(original.node_stats, shuffled.node_stats, atol=1e-6)
    assert original.edge_index.shape[1] == 2 * len(graph.edges)
    assert original.edge_attr.shape == (2 * len(graph.edges), 2)


def test_all_edit_heads_backpropagate_finite_nonzero_gradients(model, graph, observations):
    X, y = observations
    output = model(graph_to_data(graph, X, y, 2))
    assert output.h.shape == (len(graph.nodes), 24)
    assert output.graph_embedding.shape == (24,)
    sources, targets = torch.tensor([0, 1]), torch.tensor([2, 3])
    slots = torch.tensor([0, 1])
    valid = torch.tensor([True, False])
    probabilities = model.edge_probabilities(output.h, sources, targets, valid)
    assert probabilities[1].item() == 0.0
    loss = model.mutation_logits(output.h).square().mean() + probabilities.sum()
    loss = loss + model.slot_logits(output.h, sources, targets, slots).sum()
    fresh_constants = model.terminal_embeddings(torch.tensor([8, 8]))
    loss = loss + model.wrap_logits(output.h[sources], fresh_constants, slots, slots).sum()
    loss = loss + model.collapse_logits(output.h[sources], output.h[targets]).sum()
    loss = loss + model.kind_logits(output.graph_embedding).square().sum()
    loss = loss + model.stop_logit(output.graph_embedding)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for head in (model.mlp_mut, model.mlp_edge):
        assert sum(parameter.grad.abs().sum().item() for parameter in head.parameters()) > 0


def test_all_masked_sigmoid_is_zero_with_zero_edge_gradients(model, graph, observations):
    X, y = observations
    output = model(graph_to_data(graph, X, y, 2))
    sources, targets = torch.tensor([0, 1]), torch.tensor([2, 3])
    probabilities = model.edge_probabilities(output.h, sources, targets, torch.zeros(2, dtype=torch.bool))
    assert torch.equal(probabilities, torch.zeros(2))
    probabilities.sum().backward()
    for parameter in model.mlp_edge.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0


def test_singleton_and_pyg_batch_equal_separate_encodings(model, graph, observations):
    X, y = observations
    singleton = ExpressionDAG.build([Node.variable(0)], 0)
    a = graph_to_data(graph, X, y, 2)
    b = graph_to_data(singleton, X, y, 2)
    assert b.edge_index.shape == (2, 0)
    assert b.edge_attr.shape == (0, 2)
    separate_a, separate_b = model(a), model(b)
    together = model(Batch.from_data_list([a, b]))
    assert together.graph_embedding.shape == (2, 24)
    assert torch.allclose(together.h[: len(graph.nodes)], separate_a.h, atol=1e-6)
    assert torch.allclose(together.h[len(graph.nodes) :], separate_b.h, atol=1e-6)
    assert torch.allclose(together.graph_embedding[0], separate_a.graph_embedding, atol=1e-6)
    assert torch.allclose(together.graph_embedding[1], separate_b.graph_embedding, atol=1e-6)


def test_invalid_numerical_expression_has_finite_repair_observation(model, observations):
    X, y = observations
    # x1 / (x1 - x1) is a syntactically valid DAG and numerically invalid.
    invalid = ExpressionDAG.build([Node.variable(0), Node.operator("-", (0, 0)), Node.operator("/", (0, 1))], 2)
    data = graph_to_data(invalid, X, y, 2)
    assert torch.equal(data.node_stats[:, 4:8], torch.zeros((3, 4)))
    assert (data.node_stats[:, 8] == 1).all()
    assert torch.isfinite(model(data).h).all()


def test_operand_positions_change_messages_without_changing_connectivity(model, graph, observations):
    X, y = observations
    original = graph_to_data(graph, X, y, 2)
    swapped = original.clone()
    swapped.edge_attr[:, 0] = 1 - swapped.edge_attr[:, 0]
    assert torch.equal(original.edge_index, swapped.edge_index)
    assert not torch.allclose(model(original).h, model(swapped).h)


def test_feature_inputs_reject_missing_variables_and_nonfinite_targets(graph, observations):
    X, y = observations
    with pytest.raises(ValueError):
        graph_to_data(graph, X[:, :1], y, 2)
    y[0] = np.nan
    with pytest.raises(ValueError):
        graph_to_data(graph, X, y, 2)


def test_complex_observations_are_rejected_instead_of_silently_cast(graph, observations):
    X, y = observations
    with pytest.raises(ValueError, match="real"):
        graph_to_data(graph, X.astype(complex) + 1j, y, 2)
    with pytest.raises(ValueError, match="real"):
        graph_to_data(graph, X, y.astype(complex) + 1j, 2)
