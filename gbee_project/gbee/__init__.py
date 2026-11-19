"""Graph-Based Equation Editor: constrained symbolic regression."""

from .actions import Action, ActionSpace, InvalidAction
from .constants import ConstantOptimizer, residual_mse
from .dag import Eval, EvaluationError, ExpressionDAG, Node
from .network import EditorNetwork, EquationMPNN
from .policy import MaskedEditorPolicy
from .rl import HybridEnvironment, ReinforceTrainer, TrainConfig, reward

__all__ = [
    "Action",
    "ActionSpace",
    "ConstantOptimizer",
    "EditorNetwork",
    "EquationMPNN",
    "Eval",
    "EvaluationError",
    "ExpressionDAG",
    "HybridEnvironment",
    "InvalidAction",
    "MaskedEditorPolicy",
    "Node",
    "ReinforceTrainer",
    "TrainConfig",
    "residual_mse",
    "reward",
]

__version__ = "0.1.0"
