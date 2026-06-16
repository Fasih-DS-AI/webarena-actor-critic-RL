"""Render the Actor-Critic / environment architecture diagram used in the report.

Produces results/architecture.png: the closed RL loop (Environment -> state ->
Actor + Critic -> action -> reward), annotated with the advantage/PPO update.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def _box(ax, xy, w, h, text, fc, tc="white"):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.5, edgecolor="#333333", facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=10, color=tc, weight="bold", wrap=True)


def _arrow(ax, p1, p2, text="", color="#333333", off=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.6, color=color,
                                 connectionstyle="arc3,rad=%.2f" % off))
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.12, text, ha="center", va="center", fontsize=9,
                color=color, style="italic")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    _box(ax, (0.4, 2.0), 2.2, 1.1, "Environment\n(MiniWebArena)\nDOM + reward", "#1f3864")
    _box(ax, (3.8, 3.1), 2.4, 1.1, "Actor  pi(a|s)\nmasked policy (MLP)", "#2e75b6")
    _box(ax, (3.8, 0.7), 2.4, 1.1, "Critic  V(s)\nvalue estimator (MLP)", "#c0392b")
    _box(ax, (7.4, 2.0), 2.2, 1.1, "PPO update\nGAE advantage\nA = r+gammaV(s')-V(s)", "#2e7d32")

    # state to actor & critic
    _arrow(ax, (2.6, 2.8), (3.8, 3.4), "state s")
    _arrow(ax, (2.6, 2.3), (3.8, 1.3), "state s")
    # actor to env (action)
    _arrow(ax, (3.9, 3.3), (2.6, 2.7), "action a", color="#2e75b6", off=0.25)
    # env reward to PPO
    _arrow(ax, (2.6, 2.05), (7.4, 2.2), "reward r  (+1 / -0.01 / -0.5)", color="#1f3864", off=-0.35)
    # critic value to PPO
    _arrow(ax, (6.2, 1.25), (7.4, 2.2), "V(s)", color="#c0392b")
    # actor logprob to PPO
    _arrow(ax, (6.2, 3.5), (7.6, 3.0), "log pi(a|s)", color="#2e75b6")
    # PPO back to actor & critic (gradient updates)
    _arrow(ax, (8.4, 3.1), (6.2, 3.9), "policy grad", color="#2e7d32", off=0.3)
    _arrow(ax, (8.4, 2.0), (6.2, 0.9), "TD target", color="#2e7d32", off=-0.3)

    ax.set_title("Actor-Critic (PPO) framework for autonomous web navigation",
                 fontsize=12, weight="bold")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "architecture.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
