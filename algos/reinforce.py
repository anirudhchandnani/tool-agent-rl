import numpy as np
import torch

from rollout import rollout, evaluate, reference_kl


class Reinforce:
    name = "reinforce"

    def __init__(self, policy, ref, lr=3e-3, batch=24, kl_coef=0.02, entropy_coef=0.01):
        self.policy = policy
        self.ref = ref
        self.batch = batch
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)

    def collect(self, env, rng):
        episodes = []
        for _ in range(self.batch):
            start = rng.randint(*env.start_range)
            target = rng.randint(*env.target_range)
            episodes.append(rollout(self.policy, env, start, target))
        return episodes

    def step(self, env, rng):
        episodes = self.collect(env, rng)
        returns = torch.tensor([e["return"] for e in episodes])
        advantages = returns / (returns.std() + 1e-8)

        self.opt.zero_grad()
        losses, entropies, kls = [], [], []
        for episode, advantage in zip(episodes, advantages):
            logprobs, entropy, _, logits, _ = evaluate(self.policy, episode)
            kl = reference_kl(self.ref, episode, logits)
            pg = -advantage.detach() * logprobs.sum()
            losses.append(pg - self.entropy_coef * entropy.mean() + self.kl_coef * kl)
            entropies.append(entropy.mean())
            kls.append(kl)
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()

        metrics = {
            "reward": returns.mean().item(),
            "entropy": torch.stack(entropies).mean().item(),
            "kl": torch.stack(kls).mean().item(),
            "success": float(np.mean([e["success"] for e in episodes])),
            "loss": loss.item(),
        }
        return metrics, episodes
