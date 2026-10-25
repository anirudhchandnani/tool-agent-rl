import numpy as np
import torch
from torch.distributions import Categorical, kl_divergence

from env import sample_reachable
from rollout import rollout, evaluate


def decision_dataset(policy, env, rng, episodes=200):
    states, labels = [], []
    for _ in range(episodes):
        start, target = sample_reachable(rng, env.start_range, env.max_steps)
        episode = rollout(policy, env, start, target)
        states.append(episode["hidden"][0])
        labels.append(1.0 if episode["success"] else 0.0)
    return torch.stack(states), torch.tensor(labels)


def train_probe(states, labels, steps=300, lr=0.05, split=0.7):
    n = states.shape[0]
    order = torch.randperm(n)
    cut = int(n * split)
    train_idx, test_idx = order[:cut], order[cut:]
    mean = states.mean(0, keepdim=True)
    std = states.std(0, keepdim=True) + 1e-6
    x = (states - mean) / std
    weight = torch.zeros(states.shape[1], requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([weight, bias], lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        opt.zero_grad()
        logits = x[train_idx] @ weight + bias
        loss = loss_fn(logits, labels[train_idx])
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = (x[test_idx] @ weight + bias) > 0
        accuracy = (preds.float() == labels[test_idx]).float().mean().item() if len(test_idx) else float("nan")
    direction = (weight.detach() / std.squeeze(0))
    direction = direction / (direction.norm() + 1e-8)
    return {"direction": direction, "accuracy": accuracy, "baseline": labels.mean().item()}


def probe_decodability(policy, env, rng, episodes=200):
    states, labels = decision_dataset(policy, env, rng, episodes)
    return train_probe(states, labels)


def policy_difference(base, rl, env, rng, episodes=120):
    diffs = []
    for _ in range(episodes):
        start, target = sample_reachable(rng, env.start_range, env.max_steps)
        episode = rollout(rl, env, start, target)
        _, _, _, rl_logits, _ = evaluate(rl, episode)
        _, _, _, base_logits, _ = evaluate(base, episode)
        divergence = kl_divergence(Categorical(logits=rl_logits), Categorical(logits=base_logits))
        diffs.append(divergence.mean().item())
    return float(np.mean(diffs))


def steered_success(policy, env, tasks, direction, alpha):
    hits = 0
    for start, target in tasks:
        steer = None if alpha == 0 else (direction, alpha)
        episode = rollout(policy, env, start, target, steer=steer)
        hits += int(episode["success"])
    return hits / len(tasks)


def causality_check(policy, env, rng, direction, tasks=150, alpha=4.0):
    pairs = [sample_reachable(rng, env.start_range, env.max_steps) for _ in range(tasks)]
    base = steered_success(policy, env, pairs, direction, 0.0)
    boosted = steered_success(policy, env, pairs, direction, alpha)
    suppressed = steered_success(policy, env, pairs, direction, -alpha)
    effect = boosted - suppressed
    return {
        "base": base,
        "boosted": boosted,
        "suppressed": suppressed,
        "effect": effect,
        "causal": abs(effect) > 0.15,
    }
