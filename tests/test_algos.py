import math
import random

import torch

from algos import Reinforce, TRAINERS
from env import ToolEnv
from policy import Policy, reference_copy


def build(trainer_cls, seed=0, **kwargs):
    torch.manual_seed(seed)
    policy = Policy(d_model=32, heads=2, layers=1)
    ref = reference_copy(policy)
    trainer = trainer_cls(policy, ref, **kwargs)
    env = ToolEnv(start_range=(0, 6), target_range=(0, 12), max_steps=5, seed=seed)
    rng = random.Random(seed)
    return trainer, env, rng, policy


def test_trainer_registry():
    assert "reinforce" in TRAINERS


def test_step_returns_finite_metrics():
    trainer, env, rng, _ = build(Reinforce)
    metrics, episodes = trainer.step(env, rng)
    for key in ("reward", "entropy", "kl", "success", "loss"):
        assert key in metrics and math.isfinite(metrics[key])
    assert len(episodes) > 0


def test_step_updates_parameters():
    trainer, env, rng, policy = build(Reinforce)
    before = [p.clone() for p in policy.parameters()]
    trainer.step(env, rng)
    after = list(policy.parameters())
    assert any(not torch.allclose(a, b) for a, b in zip(after, before))


def improvement(cls, iters, window=8, **kwargs):
    trainer, env, rng, _ = build(cls, **kwargs)
    rewards = [trainer.step(env, rng)[0]["reward"] for _ in range(iters)]
    early = sum(rewards[:window]) / window
    late = sum(rewards[-window:]) / window
    return early, late


def test_reinforce_learns():
    early, late = improvement(Reinforce, 45)
    assert late > early + 0.03
