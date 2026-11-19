"""Independent numerical and structural checks of the GBEE invariants."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from gbee.constants import ConstantOptimizer, residual_mse
from gbee.dag import Eval, EvaluationError, ExpressionDAG, Node


def target_graph(coefficient: float = 0.25) -> ExpressionDAG:
    """Build c*x1 + sin(x2), using mathematical rather than model code."""
    return ExpressionDAG.build(
        (
            Node.variable(0),
            Node.variable(1),
            Node.constant(coefficient, "coefficient"),
            Node.operator("*", (2, 0)),
            Node.operator("sin", (1,)),
            Node.operator("+", (3, 4)),
        ),
        root=5,
    )


def test_eval_matches_independent_numpy_calculation() -> None:
    rng = np.random.default_rng(17)
    X = rng.uniform(-2.0, 2.0, (31, 2))
    graph = target_graph(2.5)
    expected = 2.5 * X[:, 0] + np.sin(X[:, 1])
    np.testing.assert_allclose(Eval(graph, X), expected, atol=1e-14)
    nodes = graph.evaluate_nodes(X)
    assert nodes.shape == (len(X), len(graph.nodes))
    np.testing.assert_allclose(nodes[:, graph.root], expected)


@pytest.mark.parametrize(
    ("op", "expected"),
    (("-", np.array([5.0, -3.0])), ("/", np.array([3.5, 0.25]))),
)
def test_noncommutative_argument_positions(op: str, expected: np.ndarray) -> None:
    X = np.array([[7.0, 2.0], [1.0, 4.0]])
    graph = ExpressionDAG.build((Node.variable(0), Node.variable(1), Node.operator(op, (0, 1))), 2)
    reverse = ExpressionDAG.build((Node.variable(0), Node.variable(1), Node.operator(op, (1, 0))), 2)
    np.testing.assert_allclose(graph.evaluate(X), expected)
    assert not np.allclose(graph.evaluate(X), reverse.evaluate(X))
    assert sorted(slot for _, target, slot in graph.edges if target == graph.root) == [0, 1]


def test_hash_consing_shares_subexpressions_and_keeps_parallel_operand_slots() -> None:
    graph = ExpressionDAG.build(
        (
            Node.variable(0),
            Node.variable(1),
            Node.operator("+", (0, 1)),
            Node.variable(0),
            Node.operator("+", (3, 1)),
            Node.operator("*", (2, 4)),
        ),
        5,
    )
    assert len(graph.nodes) == 4
    assert graph.nodes[graph.root].args[0] == graph.nodes[graph.root].args[1]
    assert len(graph.edges) == 4  # Two occurrences of the same multiplicand count.
    assert graph.complexity == 8
    X = np.array([[2.0, 3.0], [-4.0, 1.0]])
    np.testing.assert_allclose(graph.evaluate(X), np.array([25.0, 9.0]))


def test_equal_initial_constants_remain_independent_parameters() -> None:
    graph = ExpressionDAG.build(
        (
            Node.variable(0),
            Node.constant(2.0, "slope"),
            Node.constant(2.0, "intercept"),
            Node.operator("*", (0, 1)),
            Node.operator("+", (3, 2)),
        ),
        4,
    )
    assert len(graph.constant_ids) == 2
    X = np.linspace(-2.0, 2.0, 41)[:, None]
    result = ConstantOptimizer(maxiter=120).fit(graph, X, 3.0 * X[:, 0] + 4.0)
    assert result.mse < 1e-10
    assert result.mse <= result.initial_mse
    np.testing.assert_allclose(graph.evaluate(X), 2.0 * X[:, 0] + 2.0)


def test_explicit_shared_constant_is_one_optimized_parameter() -> None:
    graph = ExpressionDAG.build(
        (
            Node.variable(0),
            Node.constant(0.5, "shared"),
            Node.operator("*", (0, 1)),
            Node.operator("+", (2, 1)),
        ),
        3,
    )
    assert len(graph.constant_ids) == 1
    X = np.linspace(-1.5, 2.0, 53)[:, None]
    fit = ConstantOptimizer().fit(graph, X, 2.0 * X[:, 0] + 2.0)
    assert fit.mse < 1e-10


def test_canonical_build_prunes_nodes_unreachable_from_output() -> None:
    graph = ExpressionDAG.build((Node.variable(0), Node.variable(7), Node.operator("sin", (0,))), root=2)
    assert len(graph.nodes) == 2
    assert all(n.kind != "var" or n.var_index == 0 for n in graph.nodes)


def test_strict_graph_rejects_disconnected_node() -> None:
    with pytest.raises(ValueError):
        ExpressionDAG((Node.variable(0), Node.variable(1)), root=0)


@pytest.mark.parametrize(
    ("op", "args"),
    (("+", (0,)), ("sin", (0, 0)), ("exp", ())),
)
def test_exact_operator_arity_is_required(op: str, args: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        ExpressionDAG.build((Node.variable(0), Node.operator(op, args)), root=1)


def test_cycle_is_rejected_before_evaluation() -> None:
    with pytest.raises(ValueError):
        ExpressionDAG.build((Node.operator("sin", (1,)), Node.operator("sin", (0,))), root=1)


def test_json_roundtrip_preserves_topology_parameter_identity_and_values() -> None:
    graph = target_graph(2.5)
    restored = ExpressionDAG.from_dict(json.loads(json.dumps(graph.to_dict())))
    assert restored.signature() == graph.signature()
    assert restored.to_expression() == graph.to_expression()
    assert restored.edges == graph.edges
    X = np.array([[0.3, 0.8], [1.2, -0.7]])
    np.testing.assert_array_equal(restored.evaluate(X), graph.evaluate(X))


def test_with_constants_does_not_mutate_input_graph() -> None:
    original = target_graph(1.0)
    updated = original.with_constants(np.array([2.5]))
    X = np.array([[1.0, 0.0], [2.0, 0.0]])
    np.testing.assert_allclose(original.evaluate(X), np.array([1.0, 2.0]))
    np.testing.assert_allclose(updated.evaluate(X), np.array([2.5, 5.0]))
    assert original.signature(include_constants=False) == updated.signature(include_constants=False)
    with pytest.raises(ValueError):
        original.with_constants(np.array([1.0, 2.0]))


def test_bfgs_fits_target_coefficient_without_changing_topology() -> None:
    rng = np.random.default_rng(23)
    X = rng.uniform(-2.0, 2.0, (80, 2))
    y = 2.5 * X[:, 0] + np.sin(X[:, 1])
    graph = target_graph(-0.5)
    result = ConstantOptimizer(method="BFGS", maxiter=100).fit(graph, X, y)
    assert result.mse < 1e-10
    assert result.initial_mse > 1.0
    assert result.graph.signature(include_constants=False) == graph.signature(include_constants=False)
    X_test = rng.uniform(-2.0, 2.0, (45, 2))
    y_test = 2.5 * X_test[:, 0] + np.sin(X_test[:, 1])
    assert residual_mse(result.graph, X_test, y_test) < 1e-10


def test_optimizer_handles_graph_without_constants() -> None:
    graph = ExpressionDAG.build((Node.variable(0), Node.operator("sin", (0,))), 1)
    X = np.linspace(-1.0, 1.0, 17)[:, None]
    result = ConstantOptimizer().fit(graph, X, np.sin(X[:, 0]))
    assert result.mse == pytest.approx(0.0)
    assert result.graph.signature() == graph.signature()


@pytest.mark.parametrize("op", ("/", "exp"))
def test_invalid_numerical_domains_are_not_silently_clipped(op: str) -> None:
    if op == "/":
        graph = ExpressionDAG.build(
            (Node.variable(0), Node.constant(0.0, "denominator"), Node.operator("/", (0, 1))), 2
        )
        X = np.array([[1.0]])
    else:
        graph = ExpressionDAG.build((Node.variable(0), Node.operator("exp", (0,))), 1)
        X = np.array([[1000.0]])
    with pytest.raises(EvaluationError):
        graph.evaluate(X)


def test_cycle_creating_rewire_is_masked_and_transactionally_rejected() -> None:
    from gbee.actions import Action, ActionSpace, InvalidAction

    graph = target_graph()
    space = ActionSpace(n_variables=2)
    sin_id = next(i for i, node in enumerate(graph.nodes) if node.op == "sin")
    cycle = Action(kind="rewire", node=sin_id, source=graph.root, slot=0)
    snapshot = graph.to_dict()
    assert not space.mask(graph, (cycle,))[0]
    with pytest.raises(InvalidAction):
        space.apply(graph, cycle)
    assert graph.to_dict() == snapshot


def test_mutation_cannot_change_number_of_operand_edges() -> None:
    from gbee.actions import Action, ActionSpace, InvalidAction

    graph = target_graph()
    space = ActionSpace(n_variables=2)
    sin_id = next(i for i, node in enumerate(graph.nodes) if node.op == "sin")
    invalid = Action(kind="mutate", node=sin_id, symbol="+")
    assert not space.mask(graph, (invalid,))[0]
    with pytest.raises(InvalidAction):
        space.apply(graph, invalid)


def test_atomic_wrap_creates_complete_operator_and_respects_operand_position() -> None:
    from gbee.actions import Action, ActionSpace

    graph = ExpressionDAG.build((Node.variable(0),), 0)
    space = ActionSpace(n_variables=2)
    wrapped = space.apply(graph, Action(kind="wrap", node=0, symbol="-", terminal="x2", slot=1))
    X = np.array([[7.0, 2.0], [-3.0, 5.0]])
    np.testing.assert_allclose(wrapped.evaluate(X), X[:, 1] - X[:, 0])
    assert len(wrapped.nodes) == 3
    assert len(wrapped.nodes[wrapped.root].args) == 2
    assert len(graph.nodes) == 1
    ExpressionDAG(wrapped.nodes, wrapped.root)  # Strict validation, including reachability.


def test_single_terminal_is_valid_and_stop_is_always_available() -> None:
    from gbee.actions import ActionSpace

    graph = ExpressionDAG.build((Node.variable(0),), 0)
    space = ActionSpace(n_variables=1, max_nodes=1)
    prepared = space.prepare(graph)
    valid_stop = [i for i, action in enumerate(prepared.actions) if action.kind == "stop" and prepared.mask[i]]
    assert valid_stop
    assert space.apply(graph, prepared.actions[valid_stop[0]]).signature() == graph.signature()
    assert np.asarray(space.raw_edge_mask(graph)).shape == (1, 1, 2)
    assert not np.any(space.raw_edge_mask(graph))


def test_rewiring_prunes_old_unreferenced_subexpression() -> None:
    from gbee.actions import Action, ActionSpace

    graph = ExpressionDAG.build(
        (
            Node.variable(0),
            Node.variable(1),
            Node.operator("sin", (0,)),
            Node.operator("+", (2, 1)),
        ),
        3,
    )
    x2 = next(i for i, n in enumerate(graph.nodes) if n.kind == "var" and n.var_index == 1)
    sin_id = next(i for i, n in enumerate(graph.nodes) if n.op == "sin")
    slot = graph.nodes[graph.root].args.index(sin_id)
    edited = ActionSpace(n_variables=2).apply(graph, Action(kind="rewire", node=graph.root, source=x2, slot=slot))
    assert len(edited.nodes) == 2  # x2 and +; both x1 and sin have become unreachable.
    assert len(edited.edges) == 2  # Duplicate operand occurrences remain explicit.
    X = np.array([[1.0, 3.0], [9.0, -2.0]])
    np.testing.assert_allclose(edited.evaluate(X), 2.0 * X[:, 1])


def test_masked_logits_give_zero_probability_and_safe_gradients() -> None:
    from gbee.actions import mask_logits

    logits = torch.tensor([0.2, 0.7, -0.4], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([True, False, True])
    masked = mask_logits(logits, mask)
    assert torch.isneginf(masked[1])
    assert torch.softmax(masked, dim=0)[1].item() == 0.0
    assert torch.sigmoid(masked)[1].item() == 0.0
    loss = -torch.log_softmax(masked, dim=0)[0]
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[1].item() == 0.0
    assert logits.grad[[0, 2]].abs().sum().item() > 0.0
    with pytest.raises(ValueError):
        mask_logits(torch.zeros(2), torch.zeros(2, dtype=torch.bool))


def test_invalid_constant_fit_cannot_worsen_penalty_or_claim_convergence() -> None:
    graph = ExpressionDAG.build((Node.constant(230.3, "exponent"), Node.operator("exp", (0,))), 1)
    optimizer = ConstantOptimizer(maxiter=1, restarts=2, seed=0)
    # A restart may briefly enter the finite domain with enormous residuals;
    # that must not replace the better baseline penalty or claim convergence.
    with np.errstate(over="ignore", invalid="ignore"):
        result = optimizer.fit(graph, np.zeros((3, 1)), np.ones(3))
    assert result.mse == result.initial_mse == optimizer.penalty
    assert result.success is False
    assert result.graph.signature() == graph.signature()


@pytest.mark.parametrize(
    "payload",
    (
        {"kind": "mutate", "node": 0, "symbol": "x2", "source": 1},
        {"kind": "rewire", "node": 2, "source": 0, "slot": 1, "symbol": "+"},
        {"kind": "collapse", "node": 2, "source": 0, "slot": 1},
        {"kind": "stop", "source": -1.0},
        {"kind": "wrap", "node": 0, "symbol": "sin", "source": -1.0},
    ),
)
def test_manual_actions_reject_irrelevant_fields_and_noninteger_sentinels(payload: dict) -> None:
    from gbee.actions import Action, ActionSpace, InvalidAction

    graph = ExpressionDAG.build((Node.variable(0), Node.variable(1), Node.operator("+", (0, 1))), 2)
    with pytest.raises(InvalidAction):
        ActionSpace(n_variables=2).apply(graph, Action(**payload))


def test_restricted_grammar_rejects_existing_unsupported_operator() -> None:
    from gbee.actions import ActionSpace, InvalidAction

    graph = ExpressionDAG.build((Node.variable(0), Node.operator("exp", (0,))), 1)
    with pytest.raises(InvalidAction):
        ActionSpace(n_variables=1, operators=("+", "*", "sin")).prepare(graph)
