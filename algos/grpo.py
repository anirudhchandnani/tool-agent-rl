import numpy as np
import torch

from env import sample_reachable
from rollout import rollout, evaluate, reference_kl


class GRPO:
    name = "grpo"

    def __init__(self, policy, ref, lr=1e-3, groups=6, group_size=6, epochs=2,
                 clip=0.2, kl_coef=0.02, entropy_coef=0.05):
        self.policy = policy
        self.ref = ref
        self.groups = groups
        self.group_size = group_size
        self.epochs = epochs
        self.clip = clip
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)

    def collect(self, env, rng):
        episodes = []
        for _ in range(self.groups):
            start, target = sample_reachable(rng, env.start_range, env.max_steps)
            group = [rollout(self.policy, env, start, target) for _ in range(self.group_size)]
            rewards = torch.tensor([e["return"] for e in group])
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            for episode, advantage in zip(group, advantages):
                episode["adv"] = advantage
            episodes.extend(group)
        return episodes

    def step(self, env, rng):
        episodes = self.collect(env, rng)
        returns = torch.tensor([e["return"] for e in episodes])
        last = {}
        for _ in range(self.epochs):
            self.opt.zero_grad()
            losses, entropies, kls = [], [], []
            for episode in episodes:
                logprobs, entropy, _, logits, _ = evaluate(self.policy, episode)
                advantage = episode["adv"]
                ratio = (logprobs - episode["old_logprobs"]).exp()
                clipped = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
                pg = -torch.min(ratio * advantage, clipped * advantage).mean()
                kl = reference_kl(self.ref, episode, logits)
                losses.append(pg + self.kl_coef * kl - self.entropy_coef * entropy.mean())
                entropies.append(entropy.mean())
                kls.append(kl)
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()
            last = {
                "entropy": torch.stack(entropies).mean().item(),
                "kl": torch.stack(kls).mean().item(),
                "loss": loss.item(),
            }

        metrics = {
            "reward": returns.mean().item(),
            "success": float(np.mean([e["success"] for e in episodes])),
            **last,
        }
        return metrics, episodes
