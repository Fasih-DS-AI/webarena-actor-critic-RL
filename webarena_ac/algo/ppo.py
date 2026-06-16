"""Proximal Policy Optimization (PPO) trainer — the proposal's main method.

Implements the clipped surrogate objective from section 2.4:

    L_CLIP = E[ min( r_t * A_t,  clip(r_t, 1-eps, 1+eps) * A_t ) ]

where r_t = pi(a|s) / pi_old(a|s) is the probability ratio. The Critic is
trained jointly by regressing V(s) onto the GAE returns (the bootstrapped TD
target of section 2.3), with an entropy bonus to maintain exploration. PPO's
clipping prevents destructively large policy updates, which the proposal notes
is empirically more stable than vanilla A2C.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from ..models import Actor, Critic
from .gae import compute_gae
from .rollout import RolloutBuffer


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 4              # K: optimisation epochs per rollout
    minibatch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalize_adv: bool = True


class PPOTrainer:
    def __init__(self, actor: Actor, critic: Critic, cfg: PPOConfig,
                 device: torch.device):
        self.actor = actor
        self.critic = critic
        self.cfg = cfg
        self.device = device
        self.actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    def update(self, buf: RolloutBuffer) -> Dict[str, float]:
        cfg = self.cfg
        data = buf.as_tensors(self.device)

        # --- Advantage computation (uses the Critic just collected with). ---
        advantages, returns = compute_gae(
            data["rewards"], data["values"], data["next_values"],
            data["dones"], data["timeouts"], cfg.gamma, cfg.gae_lambda,
        )
        if cfg.normalize_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs, actions, masks = data["obs"], data["actions"], data["masks"]
        old_log_probs = data["log_probs"]

        n = len(buf)
        idx = np.arange(n)
        self.actor.train()
        self.critic.train()

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_frac": 0.0}
        n_updates = 0

        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, cfg.minibatch_size):
                mb = idx[start:start + cfg.minibatch_size]
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=self.device)

                new_log_probs, entropy = self.actor.evaluate_actions(
                    obs[mb_t], masks[mb_t], actions[mb_t]
                )
                ratio = torch.exp(new_log_probs - old_log_probs[mb_t])
                mb_adv = advantages[mb_t]

                # Clipped surrogate (PPO L_CLIP).
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                entropy_loss = -entropy.mean()

                # Critic regression onto GAE returns.
                values_pred = self.critic(obs[mb_t])
                value_loss = nn.functional.mse_loss(values_pred, returns[mb_t])

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss = (policy_loss
                        + cfg.entropy_coef * entropy_loss
                        + cfg.value_coef * value_loss)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()

                with torch.no_grad():
                    approx_kl = (old_log_probs[mb_t] - new_log_probs).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item()
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.mean().item()
                stats["approx_kl"] += approx_kl
                stats["clip_frac"] += clip_frac
                n_updates += 1

        for k in stats:
            stats[k] /= max(1, n_updates)
        return stats
