# MiniWebArena Actor-Critic

A faithful, fully-runnable demonstrator of the **Actor-Critic (PPO + GAE)**
framework proposed in *"Applying Actor-Critic Reinforcement Learning to
Autonomous Web Navigation"* (based on WebArena, Zhou et al., 2023).

The original WebArena LLM agents reach only ~14.9% task success because of
long-horizon planning failures, poor error recovery, reward sparsity, and a
**static** (non-learning) policy. This project implements the proposed fix —
a learning Actor-Critic agent with a Critic-shaped reward signal — and shows
empirically that it beats non-learning baselines, with an ablation isolating
the Critic's contribution.

Because the full WebArena stack (four Dockerised sites + GPT-4-scale LLM Actor
fine-tuned with PPO+LoRA) needs far more than a 6 GB GPU, this repo implements
the **exact RL algorithm** on a tractable, WebArena-faithful navigation MDP
(`MiniWebArena`) that trains in minutes on CPU and produces real results. The
LLM-Actor version is the production scale-up (see the report's Future Work).

## What is faithful to WebArena
- **Sequential, long-horizon tasks** across 4 simulated sites (shop / forum /
  gitlab / cms), each requiring an ordered sequence of grounded UI interactions.
- **Grounded action space** (`click/type/...` over real page elements) with an
  action mask — the agent can never "hallucinate" a non-existent element.
- **Shaped reward** exactly per the proposal: `+1` task completion, `-0.01` per
  step, `-0.5` on a dead-end/error page; discount `gamma = 0.99`.
- **Functional-evaluator-style** success checks (ordered subgoal completion).
- **Held-out tasks** that recombine seen intents into novel sequences, testing
  compositional generalisation rather than memorisation.

## Algorithm
- **Actor** `pi(a|s)`: masked-categorical MLP policy.
- **Critic** `V(s)`: separate MLP value estimator (lightweight, per proposal §2.3).
- **PPO** clipped surrogate `L_CLIP = E[min(r_t·A_t, clip(r_t,1±eps)·A_t)]`.
- **GAE** advantages; Critic trained on bootstrapped TD returns.
- Training loop = proposal §2.6 (rollout → critic update → advantage → actor update → iterate).

## Setup
```bash
pip install -r requirements.txt
# torch from the right index (CPU is fastest at this network size):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Reproduce every result
```bash
# 0) sanity-check the environment
python -m webarena_ac.env.mini_webarena --selftest

# 1) train all three learning algorithms (PPO, A2C, Actor-only ablation)
python -m webarena_ac.train --algo all          # or: --algo ppo

# 2) evaluate all agents (learned + Random/Heuristic baselines)
python -m webarena_ac.evaluate

# 3) generate all figures into results/
python -m webarena_ac.plot_results
```
Everything is seeded from `config.yaml` (`seed: 42`) for reproducibility.

## Layout
```
webarena_ac/
  env/        MiniWebArena environment, tasks, observation encoder
  models/     Actor (policy), Critic (value), shared MLP torso
  algo/       rollout buffer, GAE, PPO, A2C (+ Actor-only ablation)
  agents/     Random and Heuristic baselines
  train.py    training pipeline (proposal §2.6)
  evaluate.py final evaluation on train + held-out task suites
  plot_results.py  figure generation
  config.yaml hyperparameters (gamma, eps, lr, GAE lambda, budget, seed)
results/      metrics json, checkpoints, and figures (generated)
```

## Expected outcome
PPO Actor-Critic > A2C ≳ Actor-only(no Critic) > Heuristic > Random on held-out
success rate — validating the proposal's central claim that a learned, Critic-
guided policy overcomes the failure modes of the static LLM baseline.
