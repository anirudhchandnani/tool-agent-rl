import random

import torch

from env import ToolEnv
from interp import (train_probe, probe_decodability, policy_difference,
                    causality_check, decision_dataset)
from policy import Policy, reference_copy


def small_env(seed=0):
    return ToolEnv(start_range=(0, 6), target_range=(0, 12), max_steps=5, seed=seed)


def test_probe_separates_linearly_separable_data():
    torch.manual_seed(0)
    positive = torch.randn(80, 16) + 2.0
    negative = torch.randn(80, 16) - 2.0
    states = torch.cat([positive, negative])
    labels = torch.cat([torch.ones(80), torch.zeros(80)])
    result = train_probe(states, labels)
    assert result["accuracy"] > 0.8
    assert result["direction"].shape == (16,)
    assert abs(result["direction"].norm().item() - 1.0) < 1e-4


def test_decision_dataset_shapes():
    policy = Policy(d_model=32, heads=2, layers=1)
    states, labels = decision_dataset(policy, small_env(), random.Random(0), episodes=20)
    assert states.shape[0] == 20 and labels.shape[0] == 20
    assert set(labels.tolist()).issubset({0.0, 1.0})


def test_probe_decodability_in_unit_interval():
    policy = Policy(d_model=32, heads=2, layers=1)
    result = probe_decodability(policy, small_env(), random.Random(0), episodes=40)
    assert 0.0 <= result["accuracy"] <= 1.0


def test_policy_difference_is_nonnegative():
    policy = Policy(d_model=32, heads=2, layers=1)
    ref = reference_copy(policy)
    diff = policy_difference(ref, policy, small_env(), random.Random(0), episodes=20)
    assert diff >= -1e-6
    assert diff < 1e-4


def test_causality_check_reports_effect():
    policy = Policy(d_model=32, heads=2, layers=1)
    direction = torch.randn(32)
    direction = direction / direction.norm()
    result = causality_check(policy, small_env(), random.Random(0), direction, tasks=30, alpha=4.0)
    for key in ("base", "boosted", "suppressed", "effect", "causal"):
        assert key in result
    assert isinstance(result["causal"], bool)
    assert -1.0 <= result["effect"] <= 1.0
