"""Search y = 2.5*x1 + sin(x2), starting from the single variable x1.

Run ``python -m examples.demo --epochs 40 --seed 7`` from the project root.
The target expression appears only in data generation and a separately named
constant-optimizer sanity check. No target graph, oracle actions or test labels
are provided to the policy, action masks, training rewards or selection logic.
"""

from __future__ import annotations

import argparse
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy
import torch
import torch_geometric
from gbee.search import BeamConfig, beam_refine

from gbee import (
    ActionSpace,
    ConstantOptimizer,
    EditorNetwork,
    ExpressionDAG,
    HybridEnvironment,
    MaskedEditorPolicy,
    Node,
    ReinforceTrainer,
    TrainConfig,
    residual_mse,
)
from gbee.rl import write_json


def synthetic_data(seed: int = 7, n_train: int = 128, n_test: int = 1024) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    train_X = rng.uniform(-2.0, 2.0, size=(n_train, 2))
    test_X = rng.uniform(-2.0, 2.0, size=(n_test, 2))

    def target(X: np.ndarray) -> np.ndarray:
        return 2.5 * X[:, 0] + np.sin(X[:, 1])

    return train_X, target(train_X), test_X, target(test_X)


def constant_optimizer_sanity_check(X: np.ndarray, y: np.ndarray) -> dict:
    """Known topology ONLY: verifies coefficient fitting, not structure search."""
    graph = ExpressionDAG.build(
        (
            Node.variable(0),
            Node.variable(1),
            Node.constant(1.0, "c0"),
            Node.operator("*", (0, 2)),
            Node.operator("sin", (1,)),
            Node.operator("+", (3, 4)),
        ),
        5,
    )
    fit = ConstantOptimizer().fit(graph, X, y)
    return {
        "purpose": "known-topology constant-fitting check, excluded from RL",
        "expression": fit.graph.to_expression(12),
        "mse": fit.mse,
        "nfev": fit.nfev,
        "success": fit.success,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beam-width", type=int, default=8, help="Optional inference beam; 0 disables it")
    parser.add_argument("--beam-depth", type=int, default=3)
    parser.add_argument("--beam-proposals", type=int, default=0, help="Top policy proposals per state; 0 evaluates all")
    parser.add_argument("--operators", nargs="+", default=["+", "*", "sin"], choices=["+", "*", "-", "/", "sin", "exp"])
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.beam_width < 0:
        parser.error("--beam-width must be nonnegative")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_X, train_y, test_X, test_y = synthetic_data(args.seed)
    initial = ExpressionDAG.build((Node.variable(0),), 0)
    config = TrainConfig(epochs=args.epochs, episodes_per_batch=args.batch_size, horizon=args.horizon, seed=args.seed)
    actions = ActionSpace(n_variables=2, max_nodes=9, max_constants=3, operators=tuple(args.operators))
    policy = MaskedEditorPolicy(EditorNetwork(2, args.hidden_dim, args.layers), actions).to(args.device)
    environment = HybridEnvironment(train_X, train_y, ConstantOptimizer(maxiter=40), complexity_weight=1e-3)
    trainer = ReinforceTrainer(policy, environment, initial, config)
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    def progress(row: dict) -> None:
        if row["epoch"] == 1 or row["epoch"] % 5 == 0 or row["best_train_mse"] <= config.target_mse:
            print(
                f"epoch={row['epoch']:3d} episodes={row['episodes']:4d} best_mse={row['best_train_mse']:.6g} "
                f"mean_reward={row['mean_terminal_reward']:.5f} equation={row['best_expression']}",
                flush=True,
            )

    print("Initial equation: x1; training operators:", ", ".join(args.operators), flush=True)
    result = trainer.train(progress)
    greedy = trainer.greedy_rollout()
    best = result.best
    beam_result = None
    beam_config = None
    if args.beam_width:
        beam_config = BeamConfig(
            width=args.beam_width,
            depth=args.beam_depth,
            proposal_limit=args.beam_proposals,
            target_mse=config.target_mse,
        )
        print("Starting separate beam refinement from x1 (training rewards only).", flush=True)
        beam_result = beam_refine(policy, environment, initial, beam_config)
    selected = best
    selected_stage = "reinforce_archive"
    if beam_result is not None and beam_result.best.reward > best.reward:
        selected, selected_stage = beam_result.best, "beam_refinement"

    def summary(scored) -> dict:
        # Held-out evaluation happens after all training/search choices, with
        # frozen constants. It never influences selection or graph expansion.
        return {
            "expression": scored.graph.to_expression(12),
            "train_mse": scored.mse,
            "test_mse": residual_mse(scored.graph, test_X, test_y),
            "reward": scored.reward,
            "complexity": scored.graph.complexity,
            "graph": scored.graph.to_dict(),
        }

    report = {
        "seed": args.seed,
        "initial_expression": initial.to_expression(),
        "operators": args.operators,
        "config": asdict(config),
        "epochs_completed": result.epochs_completed,
        "episodes": result.episodes,
        "best_sampled": summary(best),
        "greedy_policy": summary(greedy),
        "beam_refinement": None
        if beam_result is None
        else {
            **summary(beam_result.best),
            "config": asdict(beam_config),
            "expanded": beam_result.expanded,
            "evaluated": beam_result.evaluated,
            "depth_reached": beam_result.depth_reached,
            "note": "Separate bounded beam inference, not an RL recovery claim. With proposal_limit=0 all legal proposals are evaluated.",
        },
        "final_selected": {
            **summary(selected),
            "stage": selected_stage,
            "selection_criterion": "maximum training reward; no test-data selection",
        },
        "constant_optimizer_sanity_check": constant_optimizer_sanity_check(train_X, train_y),
        "fit_calls": environment.fit_calls,
        "fit_cache_hits": environment.cache_hits,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
        },
    }
    write_json(args.output / "result.json", report)
    write_json(args.output / "history.json", list(result.history))
    write_json(args.output / "best_graph.json", selected.graph.to_dict())
    write_json(args.output / "rl_best_graph.json", best.graph.to_dict())
    trainer.save_checkpoint(args.output / "checkpoint.pt")
    print(f"RL best sampled: {best.graph.to_expression(12)}; train MSE: {best.mse:.8g}")
    print(f"Greedy policy: {greedy.graph.to_expression(12)}; train MSE: {greedy.mse:.8g}")
    print(
        f"Selected ({selected_stage}): {selected.graph.to_expression(12)}\nTrain MSE: {selected.mse:.8g}\n"
        f"Test MSE:  {report['final_selected']['test_mse']:.8g}"
    )
    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
