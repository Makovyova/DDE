"""Immutable, shared expression DAGs with exact operator semantics.

Edges are argument -> operator and carry an argument position.  Consequently
``x*x`` has one variable node but two distinct operand edges.  Hash-consing
merges identical operators/variables; constants are merged only when their
explicit parameter names agree, never merely because their values agree.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

OPS_ARITY: dict[str, int] = {"+": 2, "*": 2, "-": 2, "/": 2, "sin": 1, "exp": 1}
MAX_ABS_VALUE = 1e100
FloatArray = NDArray[np.float64]


class EvaluationError(ArithmeticError):
    """A structurally valid equation is undefined or numerically unusable."""


def _index(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}")
    return int(value)


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class Node:
    """One expression node; ``args[k]`` is the source of operand edge ``k``."""

    kind: str
    op: str | None = None
    args: tuple[int, ...] = ()
    var_index: int | None = None
    value: float = 0.0
    parameter: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"op", "var", "const"}:
            raise ValueError(f"Unknown node kind {self.kind!r}")
        try:
            args = tuple(_index(i, "argument index") for i in self.args)
        except TypeError as exc:
            raise ValueError("args must be a sequence of node indices") from exc
        object.__setattr__(self, "args", args)
        object.__setattr__(self, "value", _real(self.value, "node value"))
        if self.kind == "op":
            if self.op not in OPS_ARITY or len(args) != OPS_ARITY[self.op]:
                raise ValueError(f"Operator {self.op!r} has incorrect arity {len(args)}")
            if self.var_index is not None or self.parameter is not None:
                raise ValueError("Operator nodes cannot contain variable/parameter metadata")
        elif self.kind == "var":
            object.__setattr__(self, "var_index", _index(self.var_index, "variable index"))
            if self.op is not None or args or self.parameter is not None:
                raise ValueError("Variable nodes cannot have operands or operator/parameter metadata")
        else:
            if self.op is not None or args or self.var_index is not None:
                raise ValueError("Constant nodes cannot have operands or operator/variable metadata")
            if not isinstance(self.parameter, str) or not self.parameter:
                raise ValueError("Every trainable constant needs a nonempty parameter identity")

    @classmethod
    def variable(cls, index: int) -> Node:
        return cls(kind="var", var_index=index)

    @classmethod
    def constant(cls, value: float, parameter: str) -> Node:
        return cls(kind="const", value=value, parameter=parameter)

    @classmethod
    def operator(cls, op: str, args: Sequence[int]) -> Node:
        return cls(kind="op", op=op, args=tuple(args))


def _matrix(X: ArrayLike) -> FloatArray:
    if np.iscomplexobj(X):
        raise ValueError("X must contain real values")
    result = np.asarray(X, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0:
        raise ValueError("X must have shape (n_samples, n_features), with n_samples > 0")
    if not np.isfinite(result).all():
        raise ValueError("X must contain only finite values")
    return result


@dataclass(frozen=True, slots=True)
class ExpressionDAG:
    """Complete expression whose nodes are indexed in topological order.

    Direct construction is strict.  Use :meth:`build` after an atomic rewrite:
    it prunes nodes no longer contributing to the root, checks for cycles, and
    performs structural sharing before publishing the new immutable graph.
    A single variable/constant root is a valid equation, including zero edges.
    """

    nodes: tuple[Node, ...]
    root: int

    def __post_init__(self) -> None:
        nodes = tuple(self.nodes)
        if not nodes or any(not isinstance(n, Node) for n in nodes):
            raise ValueError("nodes must be a nonempty sequence of Node objects")
        object.__setattr__(self, "nodes", nodes)
        root = _index(self.root, "root")
        object.__setattr__(self, "root", root)
        if root >= len(nodes):
            raise ValueError("root refers to a missing node")
        for node_id, node in enumerate(nodes):
            if any(arg >= node_id for arg in node.args):
                raise ValueError("Nodes must be in strict topological order; cycles/forward edges are invalid")
        reachable: set[int] = set()
        stack = [root]
        while stack:
            current = stack.pop()
            if current not in reachable:
                reachable.add(current)
                stack.extend(nodes[current].args)
        if len(reachable) != len(nodes):
            raise ValueError("Every node must contribute to the root; dangling nodes are invalid")
        parameters: set[str] = set()
        for node in nodes:
            if node.kind == "const":
                assert node.parameter is not None
                if node.parameter in parameters:
                    raise ValueError("Duplicate parameter nodes must be shared; construct with build()")
                parameters.add(node.parameter)

    @classmethod
    def build(cls, nodes: Sequence[Node], root: int) -> ExpressionDAG:
        """Commit a complete rewrite, pruning unreachable nodes and hash-consing.

        Only reachable nodes are part of the resulting expression.  Their
        cycles, missing references, conflicting constant identities and arity
        errors are rejected.  Operand order is preserved even for commutative
        operations; no algebraic simplification is implied by hash-consing.
        """
        source = tuple(nodes)
        root = _index(root, "root")
        if not source or root >= len(source):
            raise ValueError("root refers to a missing node")
        if any(not isinstance(node, Node) for node in source):
            raise ValueError("nodes must contain only Node objects")
        # Iterative DFS avoids Python recursion limits for deeply nested input.
        state: dict[int, int] = {root: 1}
        stack: list[tuple[int, int]] = [(root, 0)]
        order: list[int] = []
        while stack:
            current, next_arg = stack[-1]
            operands = source[current].args
            if next_arg == len(operands):
                stack.pop()
                state[current] = 2
                order.append(current)
                continue
            child = operands[next_arg]
            stack[-1] = (current, next_arg + 1)
            if child >= len(source):
                raise ValueError(f"Node {current} references missing node {child}")
            if state.get(child) == 1:
                raise ValueError("Rewrite would create a directed cycle")
            if state.get(child, 0) == 0:
                state[child] = 1
                stack.append((child, 0))

        remap: dict[int, int] = {}
        intern: dict[tuple[Any, ...], int] = {}
        canonical: list[Node] = []
        for old_id in order:
            node = source[old_id]
            if node.kind == "op":
                node = replace(node, args=tuple(remap[arg] for arg in node.args))
                key: tuple[Any, ...] = ("op", node.op, node.args)
            elif node.kind == "var":
                key = ("var", node.var_index)
            else:
                key = ("const", node.parameter)
            if key in intern:
                existing_id = intern[key]
                if node.kind == "const" and canonical[existing_id].value != node.value:
                    raise ValueError(f"Conflicting values for shared constant {node.parameter!r}")
                remap[old_id] = existing_id
            else:
                remap[old_id] = len(canonical)
                intern[key] = len(canonical)
                canonical.append(node)
        return cls(tuple(canonical), remap[root])

    @property
    def edges(self) -> tuple[tuple[int, int, int], ...]:
        """Return ``(source, target, argument_position)`` for each occurrence."""
        return tuple(
            (child, parent, position)
            for parent, node in enumerate(self.nodes)
            for position, child in enumerate(node.args)
        )

    @property
    def constant_ids(self) -> tuple[int, ...]:
        """Order used by the external optimizer and ``constants`` override."""
        return tuple(i for i, node in enumerate(self.nodes) if node.kind == "const")

    @property
    def complexity(self) -> int:
        return len(self.nodes) + sum(len(node.args) for node in self.nodes)

    def _constant_vector(self, constants: ArrayLike | None) -> FloatArray:
        ids = self.constant_ids
        if constants is None:
            return np.asarray([self.nodes[i].value for i in ids], dtype=np.float64)
        if np.iscomplexobj(constants):
            raise ValueError("Constant overrides must be real")
        result = np.asarray(constants, dtype=np.float64)
        if result.shape != (len(ids),) or not np.isfinite(result).all():
            raise ValueError(f"constants must be a finite vector of shape ({len(ids)},)")
        return result

    def evaluate_nodes(self, X: ArrayLike, constants: ArrayLike | None = None) -> FloatArray:
        """Evaluate every node once; return shape ``(n_samples, n_nodes)``.

        Division is ordinary division and exp is ordinary exp.  Undefined,
        overflowing or extremely large intermediates raise EvaluationError;
        no protected operator silently changes the represented equation.
        """
        matrix = _matrix(X)
        values = np.empty((len(matrix), len(self.nodes)), dtype=np.float64)
        vector = self._constant_vector(constants)
        constant_index = 0
        for node_id, node in enumerate(self.nodes):
            try:
                with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
                    if node.kind == "var":
                        assert node.var_index is not None
                        if node.var_index >= matrix.shape[1]:
                            raise ValueError(f"X has no column for variable x{node.var_index + 1}")
                        result = matrix[:, node.var_index]
                    elif node.kind == "const":
                        result = vector[constant_index]
                        constant_index += 1
                    else:
                        a = values[:, node.args[0]]
                        if node.op == "sin":
                            result = np.sin(a)
                        elif node.op == "exp":
                            result = np.exp(a)
                        else:
                            b = values[:, node.args[1]]
                            if node.op == "+":
                                result = a + b
                            elif node.op == "*":
                                result = a * b
                            elif node.op == "-":
                                result = a - b
                            else:
                                result = a / b
                    if not np.isfinite(result).all() or np.any(np.abs(result) > MAX_ABS_VALUE):
                        raise EvaluationError(f"Nonfinite/extreme intermediate at node {node_id}")
                    values[:, node_id] = result
            except FloatingPointError as exc:
                raise EvaluationError(f"Numerical domain error at node {node_id} ({node.op or node.kind})") from exc
        return values

    def evaluate(self, X: ArrayLike, constants: ArrayLike | None = None) -> FloatArray:
        return self.evaluate_nodes(X, constants)[:, self.root]

    def with_constants(self, values: ArrayLike) -> ExpressionDAG:
        vector = self._constant_vector(values)
        updated = list(self.nodes)
        for node_id, value in zip(self.constant_ids, vector, strict=True):
            updated[node_id] = replace(updated[node_id], value=float(value))
        return ExpressionDAG(tuple(updated), self.root)

    def to_expression(self, precision: int = 6) -> str:
        """Human-readable infix expression; parameter sharing remains in JSON."""
        if isinstance(precision, bool) or not isinstance(precision, Integral) or precision < 1:
            raise ValueError("precision must be a positive integer")
        expressions: list[str] = []
        for node in self.nodes:
            if node.kind == "var":
                expressions.append(f"x{node.var_index + 1}")  # type: ignore[operator]
            elif node.kind == "const":
                expressions.append(format(node.value, f".{precision}g"))
            elif len(node.args) == 1:
                expressions.append(f"{node.op}({expressions[node.args[0]]})")
            else:
                a, b = (expressions[i] for i in node.args)
                expressions.append(f"({a} {node.op} {b})")
        return expressions[self.root]

    def signature(self, include_constants: bool = True) -> tuple[Any, ...]:
        """Hashable state identity, retaining independent parameter identities."""
        return (
            self.root,
            tuple(
                (
                    node.kind,
                    node.op,
                    node.args,
                    node.var_index,
                    node.parameter,
                    node.value if include_constants else None,
                )
                for node in self.nodes
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "root": self.root,
            "nodes": [
                {
                    "kind": n.kind,
                    "op": n.op,
                    "args": list(n.args),
                    "var_index": n.var_index,
                    "value": n.value,
                    "parameter": n.parameter,
                }
                for n in self.nodes
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExpressionDAG:
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported expression DAG schema version")
        if set(payload) != {"schema_version", "root", "nodes"}:
            raise ValueError("Malformed expression DAG payload")
        return cls(tuple(Node(**node) for node in payload["nodes"]), payload["root"])


def Eval(graph: ExpressionDAG, X: ArrayLike) -> FloatArray:
    """Paper-compatible spelling for evaluating the graph's scalar output."""
    return graph.evaluate(X)
