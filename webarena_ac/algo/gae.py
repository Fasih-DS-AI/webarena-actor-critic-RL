"""Generalised Advantage Estimation (GAE).

Implements the advantage computation from the proposal (sections 2.4 & 2.6).
The one-step TD residual is

    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)

and GAE exponentially weights multi-step residuals with factor (gamma * lambda):

    A_t = sum_{l>=0} (gamma * lambda)^l * delta_{t+l}

The lambda knob trades bias for variance (lambda=1 -> Monte-Carlo returns,
lambda=0 -> one-step TD). Returns targets for the Critic are ``A_t + V(s_t)``.

Note on truncation vs termination: when an episode is *truncated* by the step
budget (rather than the task genuinely ending), we still bootstrap from
V(s_{t+1}); we only zero the bootstrap on true termination. We also stop the
advantage recursion at *every* episode boundary (done or timeout) so advantages
never leak across episodes within the flat buffer.
"""

from __future__ import annotations

from typing import Tuple

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    timeouts: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (advantages, returns), each shape (T,)."""
    T = rewards.shape[0]
    advantages = torch.zeros(T, dtype=torch.float32, device=rewards.device)
    last_adv = 0.0

    for t in reversed(range(T)):
        non_terminal = 1.0 - dones[t]              # zero bootstrap on true termination
        boundary = max(dones[t].item(), timeouts[t].item())  # episode ended either way
        delta = rewards[t] + gamma * next_values[t] * non_terminal - values[t]
        last_adv = delta + gamma * lam * (1.0 - boundary) * last_adv
        advantages[t] = last_adv

    returns = advantages + values
    return advantages, returns
