"""Generate all figures for the report from saved metrics.

Reads ``results/<algo>_metrics.json`` (learning curves) and
``results/eval_summary.json`` (final comparison) and writes PNGs to results/:
    - learning_curves.png      held-out success rate vs env steps (3 algos)
    - ablation_critic.png      PPO vs Actor-only (isolates the Critic)
    - final_success_rate.png   final held-out success rate, all agents
    - steps_to_completion.png  efficiency (steps on solved tasks), all agents
    - ppo_train_vs_heldout.png generalisation gap for the main method

Usage:
    python -m webarena_ac.plot_results
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

PRETTY = {"ppo": "PPO Actor-Critic", "a2c": "A2C Actor-Critic",
          "actor_only": "Actor-only (no Critic)"}
COLORS = {"ppo": "#1f3864", "a2c": "#2e75b6", "actor_only": "#c00000"}


def _load(name: str) -> Optional[Dict]:
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        print(f"!! missing {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_learning_curves() -> None:
    plt.figure(figsize=(7, 4.5))
    for algo in ("ppo", "a2c", "actor_only"):
        m = _load(f"{algo}_metrics.json")
        if not m:
            continue
        h = m["history"]
        plt.plot(h["env_steps"], [s * 100 for s in h["heldout_sr"]],
                 label=PRETTY[algo], color=COLORS[algo], linewidth=2)
    plt.xlabel("Environment steps")
    plt.ylabel("Held-out success rate (%)")
    plt.title("Learning curves: held-out task success rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "learning_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


def plot_ablation() -> None:
    plt.figure(figsize=(7, 4.5))
    for algo in ("ppo", "actor_only"):
        m = _load(f"{algo}_metrics.json")
        if not m:
            continue
        h = m["history"]
        plt.plot(h["env_steps"], [s * 100 for s in h["heldout_sr"]],
                 label=PRETTY[algo], color=COLORS[algo], linewidth=2)
    plt.xlabel("Environment steps")
    plt.ylabel("Held-out success rate (%)")
    plt.title("Ablation: contribution of the Critic (PPO-AC vs Actor-only)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "ablation_critic.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


def plot_ppo_train_vs_heldout() -> None:
    m = _load("ppo_metrics.json")
    if not m:
        return
    h = m["history"]
    plt.figure(figsize=(7, 4.5))
    plt.plot(h["env_steps"], [s * 100 for s in h["train_sr"]],
             label="Train tasks", color="#1f3864", linewidth=2)
    plt.plot(h["env_steps"], [s * 100 for s in h["heldout_sr"]],
             label="Held-out tasks", color="#2e9e5b", linewidth=2, linestyle="--")
    plt.xlabel("Environment steps")
    plt.ylabel("Success rate (%)")
    plt.title("PPO Actor-Critic: train vs held-out (generalisation)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "ppo_train_vs_heldout.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


_BAR_ORDER = ["Random", "Heuristic", "Actor-only (no Critic)",
              "A2C Actor-Critic", "PPO Actor-Critic"]
_BAR_COLORS = ["#999999", "#c0a000", "#c00000", "#2e75b6", "#1f3864"]


def plot_final_bars() -> None:
    summary = _load("eval_summary.json")
    if not summary:
        return
    names = [n for n in _BAR_ORDER if n in summary]
    colors = [_BAR_COLORS[_BAR_ORDER.index(n)] for n in names]

    # Success rate (held-out).
    sr = [summary[n]["heldout"]["success_rate"] * 100 for n in names]
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(names, sr, color=colors)
    for b, v in zip(bars, sr):
        plt.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%",
                 ha="center", va="bottom", fontsize=9)
    plt.ylabel("Held-out success rate (%)")
    plt.title("Final success rate on held-out tasks")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 105)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "final_success_rate.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")

    # Steps to completion (efficiency) on solved held-out episodes.
    steps = [summary[n]["heldout"]["mean_steps_success"] for n in names]
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(names, steps, color=colors)
    for b, v in zip(bars, steps):
        if v == v:  # not nan
            plt.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.1f}",
                     ha="center", va="bottom", fontsize=9)
    plt.ylabel("Avg steps to completion (solved episodes)")
    plt.title("Efficiency on held-out tasks (lower is better)")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "steps_to_completion.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_learning_curves()
    plot_ablation()
    plot_ppo_train_vs_heldout()
    plot_final_bars()
    print("All figures written to", RESULTS_DIR)


if __name__ == "__main__":
    main()
