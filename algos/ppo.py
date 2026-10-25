import numpy as np
import torch

from env import sample_reachable
from rollout import rollout, evaluate, reference_kl, returns_to_go


class PPO:
    name = "ppo"

    def __init__(self, policy, ref, lr=1e-3, batch=24, epochs=4, clip=0.2,
                 vf_coef=0.5, kl_coef=0.02, entropy_coef=0.05):
        self.policy = policy
        self.ref = ref
        self.batch = batch
        self.epochs = epochs
        self.clip = clip
        self.vf_coef = vf_coef
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)

    def collect(self, env, rng):
        episodes = []
        for _ in range(self.batch):
            start, target = sample_reachable(rng, env.start_range, env.max_steps)
            episode = rollout(self.policy, env, start, target)
            episode["rtg"] = returns_to_go(episode["rewards"])
            episodes.append(episode)
        return episodes

    def step(self, env, rng):
        episodes = self.collect(env, rng)
        returns = torch.tensor([e["return"] for e in episodes])
        last = {}
        for _ in range(self.epochs):
            self.opt.zero_grad()
            logp_parts, ent_parts, val_parts, rtg_parts, old_parts, kls = [], [], [], [], [], []
            for episode in episodes:
                logprobs, entropy, values, logits, _ = evaluate(self.policy, episode)
                logp_parts.append(logprobs)
                ent_parts.append(entropy)
                val_parts.append(values)
                rtg_parts.append(episode["rtg"])
                old_parts.append(episode["old_logprobs"])
                kls.append(reference_kl(self.ref, episode, logits))
            logp = torch.cat(logp_parts)
            entropy = torch.cat(ent_parts)
            values = torch.cat(val_parts)
            rtg = torch.cat(rtg_parts)
            old = torch.cat(old_parts)
            advantage = (rtg - values).detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            ratio = (logp - old).exp()
            clipped = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
            pg = -torch.min(ratio * advantage, clipped * advantage).mean()
            vf = ((values - rtg) ** 2).mean()
            kl = torch.stack(kls).mean()
            loss = pg + self.vf_coef * vf + self.kl_coef * kl - self.entropy_coef * entropy.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()
            last = {"entropy": entropy.mean().item(), "kl": kl.item(), "loss": loss.item()}

        metrics = {
            "reward": returns.mean().item(),
            "success": float(np.mean([e["success"] for e in episodes])),
            **last,
        }
        return metrics, episodes
