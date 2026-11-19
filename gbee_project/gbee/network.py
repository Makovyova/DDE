"""Position-aware PyG encoder and neural heads for a graph equation editor.

The expression DAG contains argument-to-operator edges.  Reverse edges are
created only in the encoder's :class:`~torch_geometric.data.Data` object, so
operators can communicate context to their arguments without changing the DAG.
All numerical observations are detached NumPy statistics of the training data;
SciPy's constant optimisation is deliberately outside the policy gradient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool

from .dag import EvaluationError, ExpressionDAG, Node

OPERATOR_LABELS = ("+", "*", "-", "/", "sin", "exp")
ACTION_KINDS = ("stop", "mutate", "rewire", "wrap", "collapse")
NODE_STAT_DIM = 9


@dataclass(frozen=True)
class NodeTypeVocabulary:
    """Stable type indices shared by the encoder and mutation/growth heads."""

    n_variables: int

    def __post_init__(self) -> None:
        if isinstance(self.n_variables, bool) or not isinstance(self.n_variables, int):
            raise TypeError("n_variables must be an integer")
        if self.n_variables < 1:
            raise ValueError("n_variables must be positive")

    @property
    def labels(self) -> tuple[str, ...]:
        return OPERATOR_LABELS + tuple(f"x{i + 1}" for i in range(self.n_variables)) + ("const",)

    @property
    def size(self) -> int:
        return len(OPERATOR_LABELS) + self.n_variables + 1

    def index(self, label: str) -> int:
        """Return a type id, rejecting an operator or variable outside the grammar."""
        try:
            return self.labels.index(label)
        except ValueError as exc:
            raise ValueError(f"Unknown node type {label!r}") from exc

    def node_type(self, node: Node) -> int:
        if node.kind == "op":
            return self.index(node.op)
        if node.kind == "var":
            if node.var_index is None or not 0 <= node.var_index < self.n_variables:
                raise ValueError("Graph variable is outside the configured vocabulary")
            return len(OPERATOR_LABELS) + node.var_index
        if node.kind == "const":
            return self.size - 1
        raise ValueError(f"Unknown node kind {node.kind!r}")


def _training_arrays(X: ArrayLike, y: ArrayLike, n_variables: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if np.iscomplexobj(X) or np.iscomplexobj(y):
        raise ValueError("Training data must be real")
    features = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("X must have shape (n_samples, n_variables), n_samples > 0")
    if features.shape[1] != n_variables:
        raise ValueError("X column count does not match n_variables")
    if target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 1 or target.shape[0] != features.shape[0]:
        raise ValueError("y must have shape (n_samples,) or (n_samples, 1)")
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("Training data must be finite")
    return features, target


def _correlations(columns: NDArray[np.float64], reference: NDArray[np.float64]) -> NDArray[np.float64]:
    """Correlations for already bounded columns; constant signals map to zero."""
    centered = columns - columns.mean(axis=0, keepdims=True)
    ref = reference - reference.mean()
    denominator = np.linalg.norm(centered, axis=0) * np.linalg.norm(ref)
    result = np.zeros(columns.shape[1], dtype=np.float64)
    np.divide(centered.T @ ref, denominator, out=result, where=denominator > 1e-14)
    return np.clip(result, -1.0, 1.0)


def graph_to_data(graph: ExpressionDAG, X: ArrayLike, y: ArrayLike, n_variables: int) -> Data:
    """Encode a graph and training observations as one CPU PyG ``Data`` object.

    The nine scalar features are, in order: output indicator, arity / 2,
    consumer count / node count, bounded constant value, bounded activation
    mean, bounded activation standard deviation, correlation with ``y``,
    correlation with the current residual, and numerical-evaluation failure.

    Mean and standard deviation are relative to the target magnitude.  Feature
    bounding applies only to neural observations; it never changes ``Eval``.
    Correlation and moments are sample-permutation invariant.  A numerically
    invalid but structurally valid expression receives finite zero activation
    features and an explicit failure indicator, allowing the policy to repair it.
    """
    vocab = NodeTypeVocabulary(n_variables)
    features, target = _training_arrays(X, y, n_variables)
    count = len(graph.nodes)
    node_type = [vocab.node_type(node) for node in graph.nodes]
    stats = np.zeros((count, NODE_STAT_DIM), dtype=np.float64)
    stats[graph.root, 0] = 1.0
    for i, node in enumerate(graph.nodes):
        stats[i, 1] = len(node.args) / 2.0
        if node.kind == "const":
            stats[i, 3] = math.tanh(math.copysign(math.log1p(abs(node.value)), node.value) / 5.0)

    connections: list[tuple[int, int]] = []
    attributes: list[tuple[int, int]] = []
    for source, destination, slot in graph.edges:
        stats[source, 2] += 1.0 / count
        connections.extend(((source, destination), (destination, source)))
        attributes.extend(((slot, 0), (slot, 1)))

    try:
        values = graph.evaluate_nodes(features)
        if values.shape != (features.shape[0], count) or not np.isfinite(values).all():
            raise EvaluationError("Invalid node activation matrix")
    except EvaluationError:
        stats[:, 8] = 1.0
    else:
        # Scaling before reductions avoids overflow even with very large finite
        # inputs.  Log ratios avoid overflowing activation_scale / target_scale.
        scales = np.maximum(np.max(np.abs(values), axis=0), 1e-100)
        normalized = values / scales[None, :]
        target_scale = max(float(np.max(np.abs(target))), 1e-100)
        relative_scale = np.exp(np.clip(np.log(scales) - math.log(target_scale), -20.0, 20.0))
        stats[:, 4] = np.tanh(normalized.mean(axis=0) * relative_scale)
        stats[:, 5] = np.tanh(normalized.std(axis=0) * relative_scale)
        stats[:, 6] = _correlations(normalized, target / target_scale)
        common_scale = max(target_scale, float(scales[graph.root]))
        residual = target / common_scale - values[:, graph.root] / common_scale
        stats[:, 7] = _correlations(normalized, residual)

    # A final finite assertion prevents a future feature change from silently
    # introducing NaN logits.  Casting is safe because every feature is bounded.
    if not np.isfinite(stats).all():
        raise FloatingPointError("Nonfinite graph observation features")
    edge_index = torch.tensor(connections, dtype=torch.long).t().contiguous()
    if not connections:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(attributes, dtype=torch.long).reshape(-1, 2)
    return Data(
        node_type=torch.tensor(node_type, dtype=torch.long),
        node_stats=torch.tensor(stats, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        root_index=torch.tensor([graph.root], dtype=torch.long),
        num_nodes=count,
    )


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))


class PositionAwareConv(MessagePassing):
    """Sum messages using source, destination, operand position and direction."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__(aggr="sum", node_dim=0)
        edge_dim = max(4, hidden_dim // 4)
        self.position_embedding = nn.Embedding(2, edge_dim)
        self.direction_embedding = nn.Embedding(2, edge_dim)
        self.message_mlp = _mlp(2 * hidden_dim + 2 * edge_dim, hidden_dim, hidden_dim)
        self.update_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        aggregated = self.propagate(edge_index, x=h, edge_attr=edge_attr, size=(h.size(0), h.size(0)))
        return self.norm(h + self.update_cell(aggregated, h))

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        positions = self.position_embedding(edge_attr[:, 0])
        directions = self.direction_embedding(edge_attr[:, 1])
        return self.message_mlp(torch.cat((x_j, x_i, positions, directions), dim=-1))


class EquationMPNN(nn.Module):
    """Category embeddings plus positional message passing, returning ``h_v``."""

    def __init__(self, n_variables: int, hidden_dim: int = 64, num_layers: int = 3) -> None:
        super().__init__()
        self.vocab = NodeTypeVocabulary(n_variables)
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim < 4:
            raise ValueError("hidden_dim must be an integer >= 4")
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be a positive integer")
        self.hidden_dim = hidden_dim
        self.type_embedding = nn.Embedding(self.vocab.size, hidden_dim)
        self.stat_projection = _mlp(NODE_STAT_DIM, hidden_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(PositionAwareConv(hidden_dim) for _ in range(num_layers))

    def forward(self, data: Data) -> Tensor:
        h = self.input_norm(self.type_embedding(data.node_type) + self.stat_projection(data.node_stats))
        for layer in self.layers:
            h = layer(h, data.edge_index, data.edge_attr)
        return h


@dataclass(frozen=True)
class NetworkOutput:
    h: Tensor
    graph_embedding: Tensor


class EditorNetwork(nn.Module):
    """MPNN and edit heads; the caller masks and normalises the action policy.

    ``mutation_logits`` implements the pre-softmax MLP in source equation (1).
    ``edge_logits`` implements the pre-sigmoid MLP in equation (2) without an
    implicit positional correction.  ``slot_logits`` is a separate extension
    for choosing one argument occurrence in an atomic edge replacement.

    Growth, collapse, edit-kind and STOP heads complete the operational policy;
    their architecture is an implementation choice, not specified by the paper.
    No dropout is used, so an action's policy log probability is reproducible
    between sampling and recomputation within a policy-gradient update.
    """

    def __init__(self, n_variables: int, hidden_dim: int = 64, num_layers: int = 3) -> None:
        super().__init__()
        self.vocab = NodeTypeVocabulary(n_variables)
        self.hidden_dim = hidden_dim
        self.encoder = EquationMPNN(n_variables, hidden_dim, num_layers)
        self.pool_mlp = _mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.mlp_mut = _mlp(hidden_dim, hidden_dim, self.vocab.size)
        self.mlp_edge = _mlp(2 * hidden_dim, hidden_dim, 1)
        self.slot_embedding = nn.Embedding(2, hidden_dim)
        self.mlp_slot = _mlp(3 * hidden_dim, hidden_dim, 1)
        self.mlp_wrap = _mlp(4 * hidden_dim, hidden_dim, 1)
        self.mlp_collapse = _mlp(2 * hidden_dim, hidden_dim, 1)
        self.mlp_kind = _mlp(hidden_dim, hidden_dim, len(ACTION_KINDS))
        self.mlp_stop = _mlp(hidden_dim, hidden_dim, 1)

    def forward(self, data: Data) -> NetworkOutput:
        """Return node embeddings and one pooled vector per input graph.

        A single graph has ``graph_embedding.shape == (hidden_dim,)``; a PyG
        batch with multiple graphs returns ``(num_graphs, hidden_dim)``.  PyG
        increments ``root_index`` automatically when constructing a Batch.
        """
        h = self.encoder(data)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        pooled = global_mean_pool(h, batch)
        roots = h[data.root_index.reshape(-1)]
        if roots.shape != pooled.shape:
            raise ValueError("Exactly one root_index is required per graph")
        embedding = self.pool_mlp(torch.cat((pooled, roots), dim=-1))
        if embedding.size(0) == 1:
            embedding = embedding[0]
        return NetworkOutput(h=h, graph_embedding=embedding)

    def mutation_logits(self, h: Tensor) -> Tensor:
        return self.mlp_mut(h)

    def edge_logits(self, h: Tensor, sources: Tensor, targets: Tensor) -> Tensor:
        """Unmasked pair score ``MLPedge(h_source || h_target)``."""
        self._validate_pairs(h, sources, targets)
        return self.mlp_edge(torch.cat((h[sources], h[targets]), dim=-1)).squeeze(-1)

    def edge_probabilities(self, h: Tensor, sources: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        """Source equation (2) with illegal logits set to ``-inf`` first.

        These are independent *scores*, not a joint multi-edge action policy.
        Apply one atomic edit and recompute legality before sampling another.
        An entirely masked batch correctly returns all zeros.
        """
        logits = self.edge_logits(h, sources, targets)
        if mask.dtype != torch.bool or mask.shape != logits.shape or mask.device != logits.device:
            raise ValueError("mask must be boolean and match the logits' shape and device")
        return torch.sigmoid(logits.masked_fill(~mask, -torch.inf))

    def slot_logits(self, h: Tensor, sources: Tensor, targets: Tensor, slots: Tensor) -> Tensor:
        self._validate_pairs(h, sources, targets)
        self._validate_ids(slots, sources.shape, h.device, 2, "slots")
        operands = torch.cat((h[sources], h[targets], self.slot_embedding(slots)), dim=-1)
        return self.mlp_slot(operands).squeeze(-1)

    def terminal_embeddings(self, type_ids: Tensor) -> Tensor:
        """Embeddings for prospective fresh variable or constant operands."""
        self._validate_ids(
            type_ids,
            type_ids.shape,
            self.encoder.type_embedding.weight.device,
            self.vocab.size,
            "type_ids",
        )
        if torch.any(type_ids < len(OPERATOR_LABELS)):
            raise ValueError("terminal_embeddings accepts only variable/constant types")
        return self.encoder.type_embedding(type_ids)

    def wrap_logits(self, h_target: Tensor, h_operand: Tensor, op_ids: Tensor, slots: Tensor) -> Tensor:
        """Score insertion of a fully populated operator above an existing node.

        ``slots`` is the position occupied by the existing target.  Unary
        insertions use slot zero and a zero vector for ``h_operand``.
        """
        if h_target.ndim != 2 or h_target.shape != h_operand.shape:
            raise ValueError("h_target and h_operand must have matching (A, hidden_dim) shapes")
        expected = h_target.shape[:-1]
        self._validate_ids(op_ids, expected, h_target.device, len(OPERATOR_LABELS), "op_ids")
        self._validate_ids(slots, expected, h_target.device, 2, "slots")
        inputs = torch.cat(
            (h_target, h_operand, self.encoder.type_embedding(op_ids), self.slot_embedding(slots)), dim=-1
        )
        return self.mlp_wrap(inputs).squeeze(-1)

    def collapse_logits(self, h_target: Tensor, h_child: Tensor) -> Tensor:
        if h_target.shape != h_child.shape:
            raise ValueError("h_target and h_child must have identical shapes")
        return self.mlp_collapse(torch.cat((h_target, h_child), dim=-1)).squeeze(-1)

    def kind_logits(self, graph_embedding: Tensor) -> Tensor:
        return self.mlp_kind(graph_embedding)

    def stop_logit(self, graph_embedding: Tensor) -> Tensor:
        return self.mlp_stop(graph_embedding).squeeze(-1)

    @staticmethod
    def _validate_ids(ids: Tensor, shape: torch.Size, device: torch.device, upper: int, name: str) -> None:
        if ids.dtype != torch.long or ids.shape != shape or ids.device != device:
            raise ValueError(f"{name} must be torch.long with the matching shape and device")
        if torch.any(ids < 0) or torch.any(ids >= upper):
            raise ValueError(f"{name} contains an index outside [0, {upper})")

    @classmethod
    def _validate_pairs(cls, h: Tensor, sources: Tensor, targets: Tensor) -> None:
        if h.ndim != 2 or sources.ndim != 1:
            raise ValueError("Expected h with shape (V,H) and one-dimensional source indices")
        cls._validate_ids(sources, sources.shape, h.device, h.size(0), "sources")
        cls._validate_ids(targets, sources.shape, h.device, h.size(0), "targets")
