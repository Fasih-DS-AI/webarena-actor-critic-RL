"""MiniWebArena: a tractable, fully-runnable web-navigation MDP.

This environment is a faithful, lightweight stand-in for the WebArena benchmark
(Zhou et al., 2023). It preserves the properties that make WebArena a
reinforcement-learning problem — long-horizon sequential decisions, grounded
element interactions, sparse task-completion reward with shaping, dead-end/error
states requiring recovery — while being small enough to train on CPU/a 6 GB GPU
in minutes.

MDP definition (matching the proposal):
    State (s):   the current page's element slots + required subgoal + site +
                 progress + step fraction (see ``observation.encode``).
    Action (a):  select one of A_MAX element slots (click/activate). An action
                 mask prevents selecting empty slots — the agent is *grounded*
                 to elements that actually exist, exactly as the proposal
                 specifies ("no hallucinated interactions").
    Reward (r):  +1.0 on verified task completion,
                 -0.01 per step (efficiency penalty),
                 -0.5 on entering a dead-end / error page.
    Discount:    gamma = 0.99 (set by the training algorithm, not the env).

Episodes terminate on task completion (success) and truncate at ``max_steps``.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np

from .observation import A_MAX, NUM_ACTIONS, OBS_DIM, Page, encode
from .tasks import (ALL_TASKS, DISTRACTOR_INTENTS, RECOVERY_INTENT, TRAP_INTENTS,
                    Task, TRAIN_TASKS)

# Reward constants (from the proposal, section 2.1).
REWARD_SUCCESS = 1.0
REWARD_STEP = -0.01
REWARD_DEAD_END = -0.5


class MiniWebArenaEnv:
    """A Gym-style environment over a distribution of web-navigation tasks.

    Each ``reset`` samples a task from ``tasks`` and the agent must satisfy its
    ordered subgoals by activating the correct element on each page.
    """

    def __init__(
        self,
        tasks: Optional[List[Task]] = None,
        max_steps: int = 40,
        step_budget_factor: float = 3.0,
        trap_prob: float = 0.35,
        n_distractors: int = 6,
        seed: int = 0,
    ) -> None:
        self.tasks: List[Task] = list(tasks) if tasks is not None else list(TRAIN_TASKS)
        self.max_steps = max_steps          # hard cap on episode length
        # Per-episode budget = factor * task.length (capped at max_steps). A
        # tight budget means an aimlessly-wandering policy runs out of time and
        # fails -- this is what separates a learned policy from random.
        self.step_budget_factor = step_budget_factor
        self.trap_prob = trap_prob          # chance a trap element appears on a page
        self.n_distractors = n_distractors  # distractor elements per page
        self.rng = np.random.default_rng(seed)
        self._episode_max_steps = max_steps

        self.observation_dim = OBS_DIM
        self.num_actions = NUM_ACTIONS

        # Episode state.
        self.task: Optional[Task] = None
        self.subgoal_idx = 0
        self.step_count = 0
        self.in_dead_end = False
        self._page: Optional[Page] = None
        self._page_slot_kinds: List[Optional[str]] = []

    # ------------------------------------------------------------------
    # Core gym API
    # ------------------------------------------------------------------
    def reset(self, task: Optional[Task] = None) -> Tuple[np.ndarray, Dict]:
        """Start a new episode. Optionally force a specific task."""
        if task is None:
            task = self.tasks[self.rng.integers(len(self.tasks))]
        self.task = task
        self.subgoal_idx = 0
        self.step_count = 0
        self.in_dead_end = False
        self._episode_max_steps = min(
            self.max_steps, int(round(self.step_budget_factor * task.length))
        )
        self._render_page()
        return self._obs(), self._info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Apply an element-activation action.

        Returns (obs, reward, terminated, truncated, info), following the
        Gymnasium 5-tuple convention.
        """
        assert self.task is not None, "Call reset() before step()."
        self.step_count += 1
        reward = REWARD_STEP
        terminated = False

        intent = self._page.slot_intents[action] if 0 <= action < A_MAX else None

        if self.in_dead_end:
            # Only the recovery ("back") element makes progress out of a dead end.
            if intent == RECOVERY_INTENT:
                self.in_dead_end = False
                self._render_page()
            # any other action: wasted step, still stuck (re-render dead end)
        else:
            required = self._required_intent()
            if intent == required:
                # Correct grounded action -> advance subgoal.
                self.subgoal_idx += 1
                if self.subgoal_idx >= self.task.length:
                    reward += REWARD_SUCCESS
                    terminated = True
                else:
                    self._render_page()
            elif intent in TRAP_INTENTS:
                reward += REWARD_DEAD_END
                self.in_dead_end = True
                self._render_dead_end()
            else:
                # Wrong-but-harmless distractor (or empty slot): wasted step.
                self._render_page()

        truncated = (not terminated) and (self.step_count >= self._episode_max_steps)
        return self._obs(), reward, terminated, truncated, self._info()

    # ------------------------------------------------------------------
    # Page rendering
    # ------------------------------------------------------------------
    def _required_intent(self) -> str:
        # Clamp on the terminal step: when all subgoals are done the returned
        # observation is never used for action selection, but must still encode.
        idx = min(self.subgoal_idx, self.task.length - 1)
        return self.task.subgoals[idx]

    def _render_page(self) -> None:
        """Build a page containing the correct element plus distractors/traps."""
        required = self._required_intent()
        elements: List[str] = [required]

        # Distractors (never equal to the required intent).
        pool = [d for d in DISTRACTOR_INTENTS]
        # Also use *other* goal intents from the task as plausible distractors.
        other_goals = [g for g in self.task.subgoals if g != required]
        pool = pool + other_goals
        self.rng.shuffle(pool)
        n_dist = min(self.n_distractors, len(pool), A_MAX - 1)
        elements += pool[:n_dist]

        # Optionally inject a trap.
        if self.rng.random() < self.trap_prob and len(elements) < A_MAX:
            elements.append(TRAP_INTENTS[self.rng.integers(len(TRAP_INTENTS))])

        elements = elements[:A_MAX]
        self.rng.shuffle(elements)

        slots: List[Optional[str]] = list(elements)
        slots += [None] * (A_MAX - len(slots))
        self._page = Page(slot_intents=slots)

    def _render_dead_end(self) -> None:
        """Dead-end/error page: only the recovery element is actionable."""
        slots: List[Optional[str]] = [None] * A_MAX
        back_slot = int(self.rng.integers(A_MAX))
        slots[back_slot] = RECOVERY_INTENT
        self._page = Page(slot_intents=slots)

    # ------------------------------------------------------------------
    # Observation / info helpers
    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        required = RECOVERY_INTENT if self.in_dead_end else self._required_intent()
        progress = self.subgoal_idx / self.task.length
        step_frac = self.step_count / max(1, self._episode_max_steps)
        return encode(self._page, required, self.task.site, progress, step_frac,
                      self.in_dead_end)

    def action_mask(self) -> np.ndarray:
        return self._page.action_mask()

    def _info(self) -> Dict:
        return {
            "task": self.task.name,
            "site": self.task.site,
            "subgoal_idx": self.subgoal_idx,
            "num_subgoals": self.task.length,
            "progress": self.subgoal_idx / self.task.length,
            "in_dead_end": self.in_dead_end,
            "action_mask": self.action_mask(),
            "success": self.subgoal_idx >= self.task.length,
        }

    # ------------------------------------------------------------------
    # Oracle (for the greedy baseline + sanity checks)
    # ------------------------------------------------------------------
    def optimal_action(self) -> int:
        """Return the index of the correct element on the current page."""
        target = RECOVERY_INTENT if self.in_dead_end else self._required_intent()
        for i, intent in enumerate(self._page.slot_intents):
            if intent == target:
                return i
        return 0  # should not happen


# ----------------------------------------------------------------------
# Self-test CLI
# ----------------------------------------------------------------------
def _selftest() -> None:
    print(f"OBS_DIM={OBS_DIM}  NUM_ACTIONS={NUM_ACTIONS}  A_MAX={A_MAX}")
    env = MiniWebArenaEnv(tasks=ALL_TASKS, seed=123)

    # 1) Random rollout: should mostly fail / truncate.
    rng = np.random.default_rng(0)
    succ = 0
    n = 200
    for _ in range(n):
        obs, info = env.reset()
        done = False
        while not done:
            mask = info["action_mask"]
            valid = np.flatnonzero(mask)
            a = int(rng.choice(valid)) if len(valid) else 0
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
        succ += int(info["success"])
    print(f"[random]  success rate over {n} eps: {succ / n:.3f}")

    # 2) Oracle rollout: should always succeed, with bounded steps.
    succ = 0
    total_steps = 0
    for _ in range(n):
        obs, info = env.reset()
        done = False
        steps = 0
        while not done:
            a = env.optimal_action()
            obs, r, term, trunc, info = env.step(a)
            steps += 1
            done = term or trunc
        succ += int(info["success"])
        total_steps += steps
    print(f"[oracle]  success rate: {succ / n:.3f}  avg steps: {total_steps / n:.2f}")

    # 3) Reward-sign / shape checks.
    obs, info = env.reset(task=ALL_TASKS[0])
    obs, r, term, trunc, info = env.step(env.optimal_action())
    assert obs.shape == (OBS_DIM,), f"bad obs shape {obs.shape}"
    assert abs(r - REWARD_STEP) < 1e-6, f"correct non-final step should cost {REWARD_STEP}, got {r}"
    assert info["subgoal_idx"] == 1, "correct action should advance one subgoal"
    print("[checks]  obs shape:", obs.shape, "| reward on correct step:", round(r, 3),
          "| subgoal advanced ->", info["subgoal_idx"])
    print("Self-test OK.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniWebArena environment")
    parser.add_argument("--selftest", action="store_true", help="run sanity checks")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
    else:
        parser.print_help()
