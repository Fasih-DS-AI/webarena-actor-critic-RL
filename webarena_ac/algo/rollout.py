"""Rollout collection and storage.

Implements step 1 of the proposal's training pipeline (section 2.6): deploy the
current Actor policy in the environment to collect a batch of on-policy
transitions, recording the full trajectory (s, a, r, log-prob, value, done) for
each timestep so the Critic update and advantage computation can follow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

from ..env import MiniWebArenaEnv
from ..models import Actor, Critic


@dataclass
class RolloutBuffer:
    """Flat storage for a batch of on-policy transitions."""

    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)        # episode terminated (true end)
    timeouts: List[bool] = field(default_factory=list)     # truncated by step budget
    next_values: List[float] = field(default_factory=list)  # V(s') for bootstrapping

    # Episode-level diagnostics (filled at episode end).
    ep_returns: List[float] = field(default_factory=list)
    ep_successes: List[float] = field(default_factory=list)
    ep_lengths: List[int] = field(default_factory=list)
    ep_progress: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.obs)

    def as_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "obs": torch.as_tensor(np.array(self.obs), dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.actions, dtype=torch.long, device=device),
            "masks": torch.as_tensor(np.array(self.masks), dtype=torch.bool, device=device),
            "rewards": torch.as_tensor(self.rewards, dtype=torch.float32, device=device),
            "values": torch.as_tensor(self.values, dtype=torch.float32, device=device),
            "log_probs": torch.as_tensor(self.log_probs, dtype=torch.float32, device=device),
            "dones": torch.as_tensor(self.dones, dtype=torch.float32, device=device),
            "timeouts": torch.as_tensor(self.timeouts, dtype=torch.float32, device=device),
            "next_values": torch.as_tensor(self.next_values, dtype=torch.float32, device=device),
        }


def collect_rollouts(
    env: MiniWebArenaEnv,
    actor: Actor,
    critic: Critic,
    n_steps: int,
    device: torch.device,
    use_critic: bool = True,
) -> RolloutBuffer:
    """Collect ``n_steps`` transitions of on-policy experience.

    When ``use_critic`` is False (the Actor-only ablation) the recorded values
    are zero, so advantages reduce to discounted returns with no learned
    baseline — isolating the Critic's variance-reduction contribution.
    """
    buf = RolloutBuffer()
    actor.eval()
    critic.eval()

    obs, info = env.reset()
    ep_return, ep_len = 0.0, 0

    for _ in range(n_steps):
        mask = info["action_mask"]
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

        action, log_prob = actor.act(obs_t, mask_t, greedy=False)
        value = critic(obs_t).item() if use_critic else 0.0

        a = int(action.item())
        next_obs, reward, terminated, truncated, next_info = env.step(a)

        # Bootstrap value of the next state (0 if the episode truly terminated;
        # for a timeout we still bootstrap, since the task isn't impossible).
        if use_critic and not terminated:
            next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            next_value = critic(next_obs_t).item()
        else:
            next_value = 0.0

        buf.obs.append(obs)
        buf.actions.append(a)
        buf.masks.append(mask)
        buf.rewards.append(reward)
        buf.values.append(value)
        buf.log_probs.append(float(log_prob.item()))
        buf.dones.append(bool(terminated))
        buf.timeouts.append(bool(truncated))
        buf.next_values.append(next_value)

        ep_return += reward
        ep_len += 1

        if terminated or truncated:
            buf.ep_returns.append(ep_return)
            buf.ep_successes.append(float(next_info["success"]))
            buf.ep_lengths.append(ep_len)
            buf.ep_progress.append(float(next_info["progress"]))
            obs, info = env.reset()
            ep_return, ep_len = 0.0, 0
        else:
            obs, info = next_obs, next_info

    return buf
