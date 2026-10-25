import torch
from torch.distributions import Categorical, kl_divergence

from env import ToolEnv, sample_reachable
from policy import BOS, value_token, target_token, action_token


def decision_positions(num_actions):
    return [2 + 2 * i for i in range(num_actions)]


@torch.no_grad()
def rollout(policy, env, start, target, greedy=False, steer=None, device="cpu"):
    env.reset(start, target)
    tokens = [BOS, value_token(env.value), target_token(env.target)]
    actions, old_logprobs, rewards = [], [], []
    hidden_states = []
    info = {"hit": False}
    done = False
    while not done:
        seq = torch.tensor([tokens], device=device)
        last = policy.hidden(seq)[0, -1]
        hidden_states.append(last.clone())
        probe = last if steer is None else last + steer[1] * steer[0]
        logits = policy.action_logits(probe)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits) if greedy else dist.sample()
        old_logprobs.append(dist.log_prob(action))
        a = int(action)
        (value, _), reward, done, info = env.step(a)
        actions.append(a)
        rewards.append(reward)
        tokens.append(action_token(a))
        tokens.append(value_token(value))
    return {
        "tokens": tokens,
        "actions": actions,
        "rewards": rewards,
        "return": float(sum(rewards)),
        "success": bool(info["hit"]),
        "old_logprobs": torch.stack(old_logprobs),
        "hidden": torch.stack(hidden_states),
        "start": start,
        "target": target,
    }


def evaluate(policy, episode):
    seq = torch.tensor([episode["tokens"]])
    states = policy.hidden(seq)[0, decision_positions(len(episode["actions"]))]
    logits = policy.action_logits(states)
    dist = Categorical(logits=logits)
    actions = torch.tensor(episode["actions"])
    return dist.log_prob(actions), dist.entropy(), policy.state_value(states), logits, states


def reference_kl(ref, episode, logits):
    with torch.no_grad():
        ref_seq = torch.tensor([episode["tokens"]])
        ref_states = ref.hidden(ref_seq)[0, decision_positions(len(episode["actions"]))]
        ref_logits = ref.action_logits(ref_states)
    return kl_divergence(Categorical(logits=logits), Categorical(logits=ref_logits)).mean()


def returns_to_go(rewards):
    out = []
    running = 0.0
    for r in reversed(rewards):
        running = r + running
        out.append(running)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32)


def sample_batch(policy, env, rng, size, greedy=False):
    episodes = []
    for _ in range(size):
        start, target = sample_reachable(rng, env.start_range, env.max_steps)
        episodes.append(rollout(policy, env, start, target, greedy=greedy))
    return episodes
