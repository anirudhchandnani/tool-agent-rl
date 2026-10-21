import random

TOOLS = ["add2", "mul3", "sub1", "stop"]
NUM_ACTIONS = len(TOOLS)

VALUE_MIN = -10
VALUE_MAX = 60


def clip_value(v):
    return max(VALUE_MIN, min(VALUE_MAX, v))


def apply_tool(value, tool):
    if tool == "add2":
        return clip_value(value + 2)
    if tool == "mul3":
        return clip_value(value * 3)
    if tool == "sub1":
        return clip_value(value - 1)
    return value


class ToolEnv:
    def __init__(self, max_steps=6, start_range=(0, 12), target_range=(0, 30),
                 step_penalty=0.02, overshoot_penalty=0.05, seed=None):
        self.max_steps = max_steps
        self.start_range = start_range
        self.target_range = target_range
        self.step_penalty = step_penalty
        self.overshoot_penalty = overshoot_penalty
        self.rng = random.Random(seed)
        self.reset()

    def sample_task(self):
        start = self.rng.randint(*self.start_range)
        target = self.rng.randint(*self.target_range)
        return start, target

    def reset(self, start=None, target=None):
        if start is None or target is None:
            start, target = self.sample_task()
        self.start = clip_value(start)
        self.target = clip_value(target)
        self.value = self.start
        self.steps = 0
        self.done = False
        self.hit = False
        return self.value, self.target

    def step(self, action):
        if self.done:
            raise RuntimeError("step called on a finished episode")
        tool = TOOLS[action]
        before = abs(self.value - self.target)
        self.value = apply_tool(self.value, tool)
        self.steps += 1
        reward = -self.step_penalty
        after = abs(self.value - self.target)
        if tool != "stop" and after > before:
            reward -= self.overshoot_penalty
        self.hit = self.value == self.target
        if tool == "stop" or self.steps >= self.max_steps:
            self.done = True
            if self.hit:
                reward += 1.0
        return (self.value, self.target), reward, self.done, {"hit": self.hit, "tool": tool}
