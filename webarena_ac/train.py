"""Training entry point.

Implements the proposal's training pipeline (section 2.6):
    1. Rollout collection   — deploy the Actor, collect N transitions.
    2. Critic update        — regress V(s) onto bootstrapped returns.
    3. Advantage estimation — GAE using the updated Critic.
    4. Actor update         — PPO clipped objective (K epochs) or A2C.
    5. Iterate              — until the env-step budget is exhausted.

Usage:
    python -m webarena_ac.train --algo ppo
    python -m webarena_ac.train --algo a2c
    python -m webarena_ac.train --algo actor_only

The three algorithms share identical networks, env, seed and budget, so their
learning curves are directly comparable (this is the experiment that validates
the proposal's central claim).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import torch

from .algo import PPOTrainer
from .algo.a2c import A2CConfig, A2CTrainer
from .algo.ppo import PPOConfig
from .algo.rollout import collect_rollouts
from .env import EVAL_TASKS, TRAIN_TASKS
from .models import Actor, Critic
from .utils import (evaluate_policy, get_device, load_config, make_actor_act_fn,
                    make_env, set_seed, tune_cpu_threads)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

ALGOS = ("ppo", "a2c", "actor_only")


# An effectively-infinite clip range turns PPO's clipped objective into the
# plain (unclipped) multi-epoch policy gradient -- i.e. advantage actor-critic.
_NO_CLIP = 1e6


def build_trainer(algo: str, actor: Actor, critic: Critic, cfg: Dict,
                  device: torch.device):
    """Controlled ablation: all three learners share PPO's multi-epoch update
    machinery (so they are compared on equal footing at the same env-step
    budget) and differ by exactly ONE controlled variable:

        ppo        : clipping ON  + learned Critic baseline   (full method)
        a2c        : clipping OFF + learned Critic baseline    (isolates the clip)
        actor_only : clipping ON  + NO Critic baseline         (isolates the Critic)

    For ``actor_only`` the rollouts are collected with use_critic=False, so the
    stored values are zero and GAE advantages reduce to (normalised) discounted
    returns -- a pure policy gradient with no learned baseline.
    """
    use_critic = (algo != "actor_only")
    clip = _NO_CLIP if algo == "a2c" else cfg["ppo"]["clip_eps"]
    pcfg = PPOConfig(
        gamma=cfg["gamma"], gae_lambda=cfg["gae_lambda"],
        clip_eps=clip, epochs=cfg["ppo"]["epochs"],
        minibatch_size=cfg["ppo"]["minibatch_size"],
        actor_lr=cfg["actor_lr"], critic_lr=cfg["critic_lr"],
        value_coef=cfg["value_coef"], entropy_coef=cfg["entropy_coef"],
        max_grad_norm=cfg["max_grad_norm"], normalize_adv=cfg["normalize_adv"],
    )
    return PPOTrainer(actor, critic, pcfg, device), use_critic


def train(algo: str, config_path: str, seed_override=None) -> Dict:
    cfg = load_config(config_path)
    seed = seed_override if seed_override is not None else cfg["seed"]
    set_seed(seed)
    device = get_device(cfg["device"])
    if device.type == "cpu":
        tune_cpu_threads()
    print(f"=== Training algo={algo}  seed={seed}  device={device} ===")

    train_env = make_env(cfg, TRAIN_TASKS, seed=seed)
    train_eval_env = make_env(cfg, TRAIN_TASKS, seed=seed + 100)
    heldout_env = make_env(cfg, EVAL_TASKS, seed=seed + 200)

    obs_dim = train_env.observation_dim
    n_actions = train_env.num_actions
    hidden = tuple(cfg["network"]["hidden_sizes"])
    actor = Actor(obs_dim, n_actions, hidden).to(device)
    critic = Critic(obs_dim, hidden).to(device)

    trainer, use_critic = build_trainer(algo, actor, critic, cfg, device)

    n_steps = cfg["train"]["n_steps"]
    total_steps = cfg["train"]["total_env_steps"]
    n_updates = max(1, total_steps // n_steps)
    eval_every = cfg["train"]["eval_every_updates"]
    eval_eps = cfg["train"]["eval_episodes"]

    history = {"env_steps": [], "train_sr": [], "heldout_sr": [],
               "rollout_sr": [], "mean_return": [], "entropy": [],
               "value_loss": [], "policy_loss": []}
    best_heldout = -1.0
    best_state = None
    env_steps = 0
    t0 = time.time()

    act_fn = make_actor_act_fn(actor, device)

    for update in range(1, n_updates + 1):
        buf = collect_rollouts(train_env, actor, critic, n_steps, device,
                               use_critic=use_critic)
        env_steps += len(buf)
        stats = trainer.update(buf)

        rollout_sr = (sum(buf.ep_successes) / len(buf.ep_successes)
                      if buf.ep_successes else 0.0)

        if update % eval_every == 0 or update == n_updates:
            train_m = evaluate_policy(train_eval_env, act_fn, eval_eps)
            held_m = evaluate_policy(heldout_env, act_fn, eval_eps)
            history["env_steps"].append(env_steps)
            history["train_sr"].append(train_m["success_rate"])
            history["heldout_sr"].append(held_m["success_rate"])
            history["rollout_sr"].append(rollout_sr)
            history["mean_return"].append(train_m["mean_return"])
            history["entropy"].append(stats["entropy"])
            history["value_loss"].append(stats["value_loss"])
            history["policy_loss"].append(stats["policy_loss"])
            print(f"[{algo}] upd {update:3d}/{n_updates} steps {env_steps:>7d} | "
                  f"rollout_sr {rollout_sr:.2f} | train_sr {train_m['success_rate']:.3f} | "
                  f"held_sr {held_m['success_rate']:.3f} | ent {stats['entropy']:.3f} | "
                  f"vloss {stats['value_loss']:.3f}")
            if held_m["success_rate"] >= best_heldout:
                best_heldout = held_m["success_rate"]
                best_state = {k: v.detach().cpu().clone()
                              for k, v in actor.state_dict().items()}

    elapsed = time.time() - t0
    print(f"[{algo}] done in {elapsed:.1f}s | best held-out SR {best_heldout:.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Save best-checkpoint actor for later evaluation.
    if best_state is not None:
        torch.save(best_state, os.path.join(RESULTS_DIR, f"{algo}_actor.pt"))
    result = {
        "algo": algo, "seed": seed, "device": str(device),
        "total_env_steps": env_steps, "n_updates": n_updates,
        "elapsed_sec": elapsed, "best_heldout_sr": best_heldout,
        "config": {k: cfg[k] for k in ("gamma", "gae_lambda", "actor_lr",
                                       "critic_lr", "entropy_coef")},
        "history": history,
    }
    with open(os.path.join(RESULTS_DIR, f"{algo}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    p = argparse.ArgumentParser(description="Train MiniWebArena agents")
    p.add_argument("--algo", choices=ALGOS + ("all",), default="ppo")
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.algo == "all":
        for a in ALGOS:
            train(a, args.config, args.seed)
    else:
        train(args.algo, args.config, args.seed)


if __name__ == "__main__":
    main()
