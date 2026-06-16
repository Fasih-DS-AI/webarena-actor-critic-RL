"""Evaluate all agents on the train and held-out task suites.

Produces the final comparison the proposal's evaluation section calls for:
the learned Actor-Critic policies (PPO, A2C, Actor-only ablation) against the
non-learning baselines (Random, Heuristic). Reports success rate, steps to
completion (efficiency) and partial-progress, on both the training tasks and
the held-out compositional-generalisation tasks.

Usage:
    python -m webarena_ac.evaluate
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import torch

from .agents import GreedyAgent, RandomAgent
from .env import EVAL_TASKS, TRAIN_TASKS
from .models import Actor
from .utils import (evaluate_policy, get_device, load_config, make_actor_act_fn,
                    make_env, set_seed, tune_cpu_threads)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
LEARNED = ["actor_only", "a2c", "ppo"]
PRETTY = {"ppo": "PPO Actor-Critic", "a2c": "A2C Actor-Critic",
          "actor_only": "Actor-only (no Critic)"}


def load_actor(algo: str, obs_dim: int, n_actions: int, hidden, device) -> Actor:
    actor = Actor(obs_dim, n_actions, hidden).to(device)
    path = os.path.join(RESULTS_DIR, f"{algo}_actor.pt")
    actor.load_state_dict(torch.load(path, map_location=device))
    actor.eval()
    return actor


def main(config_path: str = None) -> Dict:
    config_path = config_path or os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = load_config(config_path)
    set_seed(cfg["seed"] + 999)
    device = get_device(cfg["device"])
    if device.type == "cpu":
        tune_cpu_threads()
    n_eval = cfg["eval"]["n_eval_episodes"]
    hidden = tuple(cfg["network"]["hidden_sizes"])

    splits = {"train": TRAIN_TASKS, "heldout": EVAL_TASKS}
    summary: Dict[str, Dict[str, Dict]] = {}

    # Scripted baselines. mistake_prob is tuned so the static heuristic lands
    # near WebArena's ~15% baseline: a fixed per-step grounding error compounds
    # over the long task horizon, exactly the failure mode the proposal targets.
    scripted = {"Random": RandomAgent(seed=1),
                "Heuristic": GreedyAgent(mistake_prob=0.92, seed=1)}
    for name, agent in scripted.items():
        summary[name] = {}
        for split, tasks in splits.items():
            env = make_env(cfg, tasks, seed=cfg["seed"] + 555)
            summary[name][split] = evaluate_policy(
                env, lambda o, m, ag=agent: ag.act(o, m), n_eval)

    # Learned policies.
    probe = make_env(cfg, TRAIN_TASKS, seed=0)
    obs_dim, n_actions = probe.observation_dim, probe.num_actions
    for algo in LEARNED:
        path = os.path.join(RESULTS_DIR, f"{algo}_actor.pt")
        if not os.path.exists(path):
            print(f"!! skipping {algo}: checkpoint not found ({path}). Train it first.")
            continue
        actor = load_actor(algo, obs_dim, n_actions, hidden, device)
        act_fn = make_actor_act_fn(actor, device)
        summary[PRETTY[algo]] = {}
        for split, tasks in splits.items():
            env = make_env(cfg, tasks, seed=cfg["seed"] + 555)
            summary[PRETTY[algo]][split] = evaluate_policy(env, act_fn, n_eval)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _print_table(summary)
    return summary


def _print_table(summary: Dict) -> None:
    print("\n================ FINAL EVALUATION ================")
    header = f"{'Agent':<26} {'Split':<9} {'SuccRate':>9} {'Steps(succ)':>12} {'Progress':>9}"
    print(header)
    print("-" * len(header))
    order = ["Random", "Heuristic", "Actor-only (no Critic)",
             "A2C Actor-Critic", "PPO Actor-Critic"]
    for name in order:
        if name not in summary:
            continue
        for split in ("train", "heldout"):
            m = summary[name][split]
            steps = m["mean_steps_success"]
            steps_s = f"{steps:.2f}" if steps == steps else "  n/a"  # nan check
            print(f"{name:<26} {split:<9} {m['success_rate']*100:>8.1f}% "
                  f"{steps_s:>12} {m['mean_progress']*100:>8.1f}%")
    print("==================================================\n")


if __name__ == "__main__":
    main()
