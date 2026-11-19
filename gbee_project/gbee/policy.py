"""A normalized policy over one legal atomic edit at a time."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from .actions import KINDS, ActionCandidates, ActionSpace, mask_logits
from .dag import ExpressionDAG
from .network import EditorNetwork, graph_to_data


@dataclass(frozen=True)
class PolicyOutput:
    distribution: Categorical
    candidates: ActionCandidates


class MaskedEditorPolicy(nn.Module):
    """Choose edit kind, then one masked proposal within that kind.

    Kind normalization prevents a family with O(V^2) proposals from drowning
    out smaller families merely by counting actions. Within mutation proposals,
    conditioning on the chosen node gives exactly the masked softmax of
    MLPmut. Edge replacements use log(sigmoid(MLPedge)) plus a slot score.
    The final categorical log probability is the *actual* sampling probability,
    including both levels of the hierarchy, as required by REINFORCE.
    """

    def __init__(self, network: EditorNetwork, action_space: ActionSpace, cache_size: int = 64) -> None:
        super().__init__()
        if network.vocab.n_variables != action_space.n_variables:
            raise ValueError("network and action vocabulary disagree")
        if not isinstance(cache_size, int) or isinstance(cache_size, bool) or cache_size < 0:
            raise ValueError("cache_size must be a nonnegative integer")
        self.network = network
        self.action_space = action_space
        self.cache_size = cache_size
        self._candidates: OrderedDict[ExpressionDAG, ActionCandidates] = OrderedDict()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def prepare(self, graph: ExpressionDAG) -> ActionCandidates:
        cached = self._candidates.get(graph)
        if cached is not None:
            self._candidates.move_to_end(graph)
            return cached
        candidates = self.action_space.prepare(graph)
        if self.cache_size:
            self._candidates[graph] = candidates
            if len(self._candidates) > self.cache_size:
                self._candidates.popitem(last=False)
        return candidates

    def forward(self, graph: ExpressionDAG, X: np.ndarray, y: np.ndarray) -> PolicyOutput:
        candidates = self.prepare(graph)
        data = graph_to_data(graph, X, y, self.action_space.n_variables).to(self.device)
        encoded = self.network(data)
        h, pooled = encoded.h, encoded.graph_embedding
        actions = candidates.actions
        # copy=True avoids exposing a read-only NumPy mask to writable tensors.
        legal = torch.tensor(candidates.mask.copy(), dtype=torch.bool, device=h.device)
        raw = h.new_full((len(actions),), -torch.inf)
        kind_index = torch.tensor([KINDS.index(a.kind) for a in actions], device=h.device)
        mutation = self.network.mutation_logits(h)

        for kind_id, kind in enumerate(KINDS):
            indices = (kind_index == kind_id).nonzero(as_tuple=False).flatten()
            if not len(indices):
                continue
            proposals = [actions[i] for i in indices.tolist()]
            if kind == "stop":
                scores = self.network.stop_logit(pooled).reshape(1)
            else:
                destinations = torch.tensor([a.node for a in proposals], dtype=torch.long, device=h.device)
                if kind == "mutate":
                    types = torch.tensor([self.network.vocab.index(a.symbol) for a in proposals], device=h.device)
                    scores = mutation[destinations, types]
                elif kind in ("rewire", "collapse"):
                    sources = torch.tensor([a.source for a in proposals], dtype=torch.long, device=h.device)
                    if kind == "collapse":
                        scores = self.network.collapse_logits(h[destinations], h[sources])
                    else:
                        slots = torch.tensor([a.slot for a in proposals], dtype=torch.long, device=h.device)
                        pair = self.network.edge_logits(h, sources, destinations)
                        # The grammar mask precedes sigmoid, including cycles.
                        pair = pair.masked_fill(~legal[indices], -torch.inf)
                        scores = F.logsigmoid(pair) + self.network.slot_logits(h, sources, destinations, slots)
                else:
                    operands = h.new_zeros((len(proposals), h.shape[1]))
                    existing = [j for j, a in enumerate(proposals) if a.source >= 0]
                    fresh = [j for j, a in enumerate(proposals) if a.terminal is not None]
                    if existing:
                        rows = torch.tensor(existing, device=h.device)
                        sources = torch.tensor([proposals[j].source for j in existing], device=h.device)
                        operands = operands.index_copy(0, rows, h[sources])
                    if fresh:
                        rows = torch.tensor(fresh, device=h.device)
                        types = torch.tensor(
                            [self.network.vocab.index(proposals[j].terminal) for j in fresh], device=h.device
                        )
                        operands = operands.index_copy(0, rows, self.network.terminal_embeddings(types))
                    op_ids = torch.tensor([self.network.vocab.index(a.symbol) for a in proposals], device=h.device)
                    slots = torch.tensor([a.slot for a in proposals], device=h.device)
                    scores = self.network.wrap_logits(h[destinations], operands, op_ids, slots)
            raw = raw.index_copy(0, indices, scores)

        available = torch.stack([legal[kind_index == k].any() for k in range(len(KINDS))])
        family_scores = self.network.kind_logits(pooled)
        # STOP is a singleton family, so its proposal score would otherwise
        # cancel in the within-family softmax. Include it in family selection.
        stop_adjustment = torch.stack([raw[0], *[h.new_zeros(()) for _ in KINDS[1:]]])
        family_log_prob = F.log_softmax(mask_logits(family_scores + stop_adjustment, available), dim=0)
        joint = h.new_full(raw.shape, -torch.inf)
        for kind_id in range(len(KINDS)):
            if not bool(available[kind_id]):
                continue
            indices = (kind_index == kind_id).nonzero(as_tuple=False).flatten()
            conditional = F.log_softmax(mask_logits(raw[indices], legal[indices]), dim=0)
            joint = joint.index_copy(0, indices, conditional + family_log_prob[kind_id])
        distribution = Categorical(logits=mask_logits(joint, legal))
        return PolicyOutput(distribution, candidates)
