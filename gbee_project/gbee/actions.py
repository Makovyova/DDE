"""Transactional, grammar-constrained edits of a complete expression DAG.

Every observable state is evaluable *syntactically*. A binary operator always
has exactly two argument occurrences. Consequently a standalone ADD_EDGE is
never legal in a complete fixed-arity DAG: use REWIRE (remove + add, atomically)
or WRAP (create a fully specified operator and reconnect its consumers).
Numerical validity, such as division by zero, is handled by the evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .dag import OPS_ARITY, ExpressionDAG, Node

KINDS = ("stop", "mutate", "rewire", "wrap", "collapse")


class InvalidAction(ValueError):
    """An edit would violate the grammar, graph invariants, or search budget."""


@dataclass(frozen=True, slots=True)
class Action:
    """One atomic edit. Node identifiers refer to the current graph only.

    ``rewire``: replace ``node.args[slot]`` by ``source``.
    ``wrap``: insert ``symbol`` above ``node``; for a binary operator the old
    node occupies ``slot`` and the other operand is ``source`` or ``terminal``.
    ``collapse``: replace all uses of ``node`` with its direct child ``source``.
    ``mutate``: replace the semantic type by ``symbol`` without changing arity.
    """

    kind: str
    node: int = -1
    symbol: str | None = None
    source: int = -1
    slot: int = 0
    terminal: str | None = None


@dataclass(frozen=True)
class ActionCandidates:
    """A state-specific action catalogue and validated, uncommitted successors."""

    actions: tuple[Action, ...]
    mask: np.ndarray
    graphs: tuple[ExpressionDAG | None, ...]


def mask_logits(logits: Tensor, mask: Tensor) -> Tensor:
    """Set illegal logits to -inf *before* normalization.

    Reject an empty support instead of letting softmax create NaNs. Use this
    for categorical choices; an all-false binary edge mask is valid and handled
    separately by ``EditorNetwork.edge_probabilities``.
    """
    if logits.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("logits and a boolean mask must have identical shapes")
    if logits.device != mask.device:
        raise ValueError("logits and mask must be on the same device")
    if logits.ndim != 1 or not bool(mask.any()):
        raise ValueError("categorical logits must be 1-D with nonempty support")
    if not bool(torch.isfinite(logits[mask]).all()):
        raise ValueError("valid action logits must be finite")
    return logits.masked_fill(~mask, -torch.inf)


class ActionSpace:
    """Enumerate and validate edits, recalculating legality at every step.

    Exhaustive transaction validation is deliberately used as the reference
    implementation. It is appropriate for small symbolic DAGs; for very large
    graphs replace catalogue enumeration with conditional masked sampling.
    """

    def __init__(
        self,
        n_variables: int,
        max_nodes: int = 12,
        max_constants: int = 4,
        operators: Sequence[str] = tuple(OPS_ARITY),
    ) -> None:
        for name, value, lower in (
            ("n_variables", n_variables, 1),
            ("max_nodes", max_nodes, 1),
            ("max_constants", max_constants, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        if not operators or len(set(operators)) != len(operators):
            raise ValueError("operators must be nonempty and unique")
        if not set(operators).issubset(OPS_ARITY):
            raise ValueError("unsupported operator")
        self.n_variables = n_variables
        self.max_nodes = max_nodes
        self.max_constants = max_constants
        self.operators = tuple(operators)
        self.terminals = tuple(f"x{i + 1}" for i in range(n_variables)) + ("const",)
        self.types = self.operators + self.terminals

    @staticmethod
    def _fresh_parameter(nodes: Sequence[Node]) -> str:
        used = {node.parameter for node in nodes if node.kind == "const"}
        index = 0
        while f"c{index}" in used:
            index += 1
        return f"c{index}"

    def _terminal_node(self, label: str, nodes: Sequence[Node]) -> Node:
        if label == "const":
            return Node.constant(1.0, self._fresh_parameter(nodes))
        if label not in self.terminals:
            raise InvalidAction(f"unknown terminal: {label}")
        return Node.variable(int(label[1:]) - 1)

    @staticmethod
    def _check_node(graph: ExpressionDAG, index: int) -> None:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(graph.nodes):
            raise InvalidAction(f"invalid node index: {index!r}")

    def _check_budget(self, graph: ExpressionDAG) -> None:
        if len(graph.nodes) > self.max_nodes:
            raise InvalidAction("node budget exceeded")
        if len(graph.constant_ids) > self.max_constants:
            raise InvalidAction("constant budget exceeded")
        if any(node.kind == "op" and node.op not in self.operators for node in graph.nodes):
            raise InvalidAction("operator is outside the configured vocabulary")
        if any(node.kind == "var" and node.var_index >= self.n_variables for node in graph.nodes):
            raise InvalidAction("variable is outside the configured vocabulary")

    @staticmethod
    def _depends_on(graph: ExpressionDAG, node: int, dependency: int) -> bool:
        stack, seen = [node], set()
        while stack:
            current = stack.pop()
            if current == dependency:
                return True
            if current not in seen:
                seen.add(current)
                stack.extend(graph.nodes[current].args)
        return False

    def apply(self, graph: ExpressionDAG, action: Action) -> ExpressionDAG:
        """Return a new normalized DAG or raise; never partially mutate input."""
        self._check_budget(graph)
        if not isinstance(action, Action):
            raise InvalidAction("expected an Action instance")
        for name in ("node", "source", "slot"):
            value = getattr(action, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise InvalidAction(f"{name} must be an integer")
        if action.symbol is not None and not isinstance(action.symbol, str):
            raise InvalidAction("symbol must be a string or None")
        if action.terminal is not None and not isinstance(action.terminal, str):
            raise InvalidAction("terminal must be a string or None")
        if action.kind not in KINDS:
            raise InvalidAction(f"unknown edit kind: {action.kind}")
        if action.kind == "stop":
            if action != Action("stop"):
                raise InvalidAction("STOP must not carry edit arguments")
            return graph
        if action.kind == "mutate" and (action.source != -1 or action.slot != 0 or action.terminal is not None):
            raise InvalidAction("mutation carries irrelevant edit arguments")
        if action.kind in ("rewire", "collapse") and (action.symbol is not None or action.terminal is not None):
            raise InvalidAction("edge replacement/collapse carries irrelevant arguments")
        if action.kind == "collapse" and action.slot != 0:
            raise InvalidAction("collapse does not select an argument position")
        self._check_node(graph, action.node)
        old = graph.nodes[action.node]
        nodes = list(graph.nodes)
        root = graph.root

        if action.kind == "mutate":
            if action.symbol not in self.types:
                raise InvalidAction("unknown mutation type")
            if action.symbol in self.operators:
                if old.kind != "op" or len(old.args) != OPS_ARITY[action.symbol]:
                    raise InvalidAction("mutation must preserve arity and connectivity")
                if old.op == action.symbol:
                    raise InvalidAction("identity mutation")
                nodes[action.node] = Node.operator(action.symbol, old.args)
            else:
                if old.args:
                    raise InvalidAction("an operator cannot become a terminal without pruning")
                new = self._terminal_node(action.symbol, nodes)
                if (old.kind == "const" and new.kind == "const") or (
                    old.kind == new.kind == "var" and old.var_index == new.var_index
                ):
                    raise InvalidAction("identity terminal mutation")
                nodes[action.node] = new

        elif action.kind == "rewire":
            self._check_node(graph, action.source)
            if (
                isinstance(action.slot, bool)
                or not isinstance(action.slot, int)
                or old.kind != "op"
                or not 0 <= action.slot < len(old.args)
            ):
                raise InvalidAction("rewiring requires an existing argument slot")
            if old.args[action.slot] == action.source:
                raise InvalidAction("identity edge replacement")
            if self._depends_on(graph, action.source, action.node):
                raise InvalidAction("replacement edge would create a cycle")
            args = list(old.args)
            args[action.slot] = action.source
            nodes[action.node] = replace(old, args=tuple(args))

        elif action.kind == "wrap":
            if action.symbol not in self.operators:
                raise InvalidAction("unknown wrapper operator")
            arity = OPS_ARITY[action.symbol]
            if isinstance(action.slot, bool) or not isinstance(action.slot, int) or not 0 <= action.slot < arity:
                raise InvalidAction("invalid operand position")
            if arity == 1:
                if action.source != -1 or action.terminal is not None:
                    raise InvalidAction("unary wrappers accept exactly one operand")
                args = (action.node,)
            else:
                has_source = action.source != -1
                if has_source == (action.terminal is not None):
                    raise InvalidAction("choose exactly one second operand")
                if has_source:
                    self._check_node(graph, action.source)
                    other = action.source
                else:
                    other = len(nodes)
                    nodes.append(self._terminal_node(action.terminal, nodes))
                args = (action.node, other) if action.slot == 0 else (other, action.node)
            wrapper = len(nodes)
            # Reconnect only preexisting consumers, never the new wrapper's args.
            for index, node in enumerate(graph.nodes):
                if action.node in node.args:
                    nodes[index] = replace(node, args=tuple(wrapper if a == action.node else a for a in node.args))
            nodes.append(Node.operator(action.symbol, args))
            if root == action.node:
                root = wrapper

        elif action.kind == "collapse":
            self._check_node(graph, action.source)
            if action.source not in old.args:
                raise InvalidAction("collapse must choose a direct operand")
            for index, node in enumerate(graph.nodes):
                if action.node in node.args:
                    nodes[index] = replace(
                        node, args=tuple(action.source if a == action.node else a for a in node.args)
                    )
            if root == action.node:
                root = action.source

        try:
            # Removes newly unreachable operands, merges identical subexpressions,
            # validates cycles and arities, and reindexes nodes topologically.
            result = ExpressionDAG.build(nodes, root)
        except ValueError as exc:
            raise InvalidAction(str(exc)) from exc
        self._check_budget(result)
        return result

    def enumerate(self, graph: ExpressionDAG) -> tuple[Action, ...]:
        """Include structurally invalid proposals so masks are explicit."""
        self._check_budget(graph)
        actions = [Action("stop")]
        existing_variables = {f"x{node.var_index + 1}" for node in graph.nodes if node.kind == "var"}
        fresh_terminals = tuple(t for t in self.terminals if t not in existing_variables)
        for index, node in enumerate(graph.nodes):
            actions.extend(Action("mutate", node=index, symbol=symbol) for symbol in self.types)
            if node.kind == "op":
                for slot in range(len(node.args)):
                    actions.extend(
                        Action("rewire", node=index, source=source, slot=slot) for source in range(len(graph.nodes))
                    )
                actions.extend(Action("collapse", node=index, source=child) for child in dict.fromkeys(node.args))
            for symbol in self.operators:
                if OPS_ARITY[symbol] == 1:
                    actions.append(Action("wrap", node=index, symbol=symbol))
                    continue
                # Commutative duplicates are omitted, but positional edges stay.
                positions = (0,) if symbol in ("+", "*") else (0, 1)
                for slot in positions:
                    actions.extend(
                        Action("wrap", node=index, symbol=symbol, source=source, slot=slot)
                        for source in range(len(graph.nodes))
                    )
                    actions.extend(
                        Action("wrap", node=index, symbol=symbol, terminal=terminal, slot=slot)
                        for terminal in fresh_terminals
                    )
        return tuple(actions)

    def prepare(self, graph: ExpressionDAG) -> ActionCandidates:
        actions = self.enumerate(graph)
        successors: list[ExpressionDAG | None] = []
        for action in actions:
            try:
                successors.append(self.apply(graph, action))
            except InvalidAction:
                successors.append(None)
        mask = np.array([successor is not None for successor in successors], dtype=bool)
        mask.setflags(write=False)
        return ActionCandidates(actions, mask, tuple(successors))

    def mask(self, graph: ExpressionDAG, actions: Sequence[Action]) -> np.ndarray:
        result = []
        for action in actions:
            try:
                self.apply(graph, action)
                result.append(True)
            except InvalidAction:
                result.append(False)
        return np.asarray(result, dtype=bool)

    def raw_edge_mask(self, graph: ExpressionDAG) -> np.ndarray:
        """No empty slots exist in a valid complete graph; no raw additions."""
        self._check_budget(graph)
        return np.zeros((len(graph.nodes), len(graph.nodes), 2), dtype=bool)
