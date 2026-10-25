# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.2", "numpy>=1.26"]
# ///
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from algos import TRAINERS
from env import ToolEnv, TOOLS, sample_reachable
from interp import probe_decodability, policy_difference, causality_check
from policy import Policy, reference_copy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_episodes(policy, env, rng, count):
    from rollout import rollout
    episodes = []
    for _ in range(count):
        start, target = sample_reachable(rng, env.start_range, env.max_steps)
        episode = rollout(policy, env, start, target, greedy=True)
        steps = []
        value = start
        for action, reward in zip(episode["actions"], episode["rewards"]):
            tool = TOOLS[action]
            steps.append({"tool": tool, "reward": round(reward, 3), "observation": None})
            value = episode_value(value, tool)
            steps[-1]["observation"] = value
        episodes.append({
            "start": start, "target": target,
            "success": episode["success"], "return": round(episode["return"], 3),
            "steps": steps,
        })
    return episodes


def episode_value(value, tool):
    from env import apply_tool
    return apply_tool(value, tool)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=list(TRAINERS), required=True)
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-every", type=int, default=10)
    parser.add_argument("--out", type=str, default="runs")
    args = parser.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)
    env = ToolEnv(seed=args.seed)
    policy = Policy()
    base = reference_copy(policy)
    ref = reference_copy(policy)
    trainer = TRAINERS[args.algo](policy, ref)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = Path(args.out) / f"{args.algo}-{stamp}.jsonl"
    log = path.open("w", encoding="utf-8")

    def write(record):
        log.write(json.dumps(record) + "\n")
        log.flush()

    write({"type": "config", "algo": args.algo, "iters": args.iters, "seed": args.seed})
    for it in range(args.iters):
        metrics, _ = trainer.step(env, rng)
        write({"type": "metrics", "iter": it, **{k: round(v, 4) for k, v in metrics.items()}})
        if it % args.probe_every == 0 or it == args.iters - 1:
            probe = probe_decodability(policy, env, rng, episodes=160)
            write({"type": "probe", "iter": it,
                   "decodability": round(probe["accuracy"], 4),
                   "label_rate": round(probe["baseline"], 4)})
        print(f"{args.algo} iter {it} reward {metrics['reward']:.3f} "
              f"entropy {metrics['entropy']:.3f} success {metrics['success']:.2f}")

    final_probe = probe_decodability(policy, env, rng, episodes=240)
    diff = policy_difference(base, policy, env, rng)
    causal = causality_check(policy, env, rng, final_probe["direction"])
    write({"type": "interp",
           "final_decodability": round(final_probe["accuracy"], 4),
           "policy_diff": round(diff, 4),
           "causal_base": round(causal["base"], 4),
           "causal_boosted": round(causal["boosted"], 4),
           "causal_suppressed": round(causal["suppressed"], 4),
           "causal_effect": round(causal["effect"], 4),
           "causal_direction": bool(causal["causal"])})
    for episode in sample_episodes(policy, env, rng, 5):
        write({"type": "episode", **episode})
    log.close()
    print(f"wrote {path}")
    print(f"probe causal effect {causal['effect']:.3f} causal {causal['causal']}")


if __name__ == "__main__":
    main()
