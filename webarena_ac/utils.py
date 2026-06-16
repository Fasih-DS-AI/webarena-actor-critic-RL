"""Shared utilities: seeding, device selection, config loading, evaluation."""

from __future__ import annotations

import random
from typing import Callable, Dict, List

import numpy as np
import torch
import yaml

from .env import MiniWebArenaEnv, Task


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tune_cpu_threads() -> None:
    """Use a single CPU thread.

    The networks are tiny (128x128 MLPs) and rollouts run a *sequential* stream
    of batch-size-1 forward passes. Multi-threaded BLAS on such small tensors is
    pure synchronisation overhead and is dramatically *slower* than one thread.
    """
    torch.set_num_threads(1)
    # set_num_interop_threads can only be called once, before any parallel work
    # has started; guard it so repeated calls (e.g. `--algo all`) don't raise.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def get_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(cfg: Dict, tasks: List[Task], seed: int) -> MiniWebArenaEnv:
    e = cfg["env"]
    return MiniWebArenaEnv(
        tasks=tasks,
        max_steps=e["max_steps"],
        step_budget_factor=e["step_budget_factor"],
        trap_prob=e["trap_prob"],
        n_distractors=e["n_distractors"],
        seed=seed,
    )


def evaluate_policy(
    env: MiniWebArenaEnv,
    act_fn: Callable[[np.ndarray, np.ndarray], int],
    n_episodes: int,
) -> Dict[str, float]:
    """Run ``n_episodes`` greedy episodes and return aggregate metrics.

    ``act_fn`` maps (obs, action_mask) -> action index. Works uniformly for the
    learned policies (greedy decode) and the scripted baseline agents.
    """
    successes, returns, lengths, progress = [], [], [], []
    success_lengths = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_ret, ep_len = 0.0, 0
        while not done:
            a = act_fn(obs, info["action_mask"])
            obs, r, term, trunc, info = env.step(a)
            ep_ret += r
            ep_len += 1
            done = term or trunc
        succ = float(info["success"])
        successes.append(succ)
        returns.append(ep_ret)
        lengths.append(ep_len)
        progress.append(float(info["progress"]))
        if succ:
            success_lengths.append(ep_len)
    return {
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "mean_steps_success": float(np.mean(success_lengths)) if success_lengths else float("nan"),
        "mean_progress": float(np.mean(progress)),
        "n_episodes": n_episodes,
    }


def make_actor_act_fn(actor, device: torch.device) -> Callable[[np.ndarray, np.ndarray], int]:
    """Wrap a torch Actor into a greedy (obs, mask) -> action callable."""
    def act_fn(obs: np.ndarray, mask: np.ndarray) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        action, _ = actor.act(obs_t, mask_t, greedy=True)
        return int(action.item())
    return act_fn
