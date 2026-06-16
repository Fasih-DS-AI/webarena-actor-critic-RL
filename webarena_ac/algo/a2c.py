"""Advantage Actor-Critic (A2C) trainer + the Actor-only ablation.

A2C is the vanilla synchronous actor-critic the proposal contrasts PPO against
(section 2.4: "PPO has been empirically shown to be more stable than vanilla
A2C"). Unlike PPO it does a single gradient pass over each rollout with no
probability-ratio clipping:

    L_actor  = -E[ log pi(a|s) * A_t ]            (policy gradient)
    L_critic =  E[ (V(s) - return_t)^2 ]          (value regression)

Setting ``use_critic=False`` removes the learned value baseline entirely, so
advantages collapse to discounted Monte-Carlo returns (REINFORCE). This is the
"Actor without Critic" ablation from the proposal's evaluation (section 3),
isolating the Critic's variance-reduction contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from ..models import Actor, Critic
from .gae import compute_gae
from .rollout import RolloutBuffer


@dataclass
class A2CConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalize_adv: bool = True
    use_critic: bool = True


class A2CTrainer:
    def __init__(self, actor: Actor, critic: Critic, cfg: A2CConfig,
                 device: torch.device):
        self.actor = actor
        self.critic = critic
        self.cfg = cfg
        self.device = device
        self.actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = (torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
                           if cfg.use_critic else None)

    def update(self, buf: RolloutBuffer) -> Dict[str, float]:
        cfg = self.cfg
        data = buf.as_tensors(self.device)

        advantages, returns = compute_gae(
            data["rewards"], data["values"], data["next_values"],
            data["dones"], data["timeouts"], cfg.gamma, cfg.gae_lambda,
        )
        if cfg.normalize_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.actor.train()
        log_probs, entropy = self.actor.evaluate_actions(
            data["obs"], data["masks"], data["actions"]
        )
        policy_loss = -(log_probs * advantages).mean()
        entropy_loss = -entropy.mean()

        self.actor_opt.zero_grad()
        actor_total = policy_loss + cfg.entropy_coef * entropy_loss

        value_loss_val = 0.0
        if cfg.use_critic:
            self.critic.train()
            values_pred = self.critic(data["obs"])
            value_loss = nn.functional.mse_loss(values_pred, returns)
            value_loss_val = value_loss.item()
            self.critic_opt.zero_grad()
            (actor_total + cfg.value_coef * value_loss).backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
            self.actor_opt.step()
            self.critic_opt.step()
        else:
            actor_total.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
            self.actor_opt.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss_val,
            "entropy": entropy.mean().item(),
            "approx_kl": 0.0,
            "clip_frac": 0.0,
        }
