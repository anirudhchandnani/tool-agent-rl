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


def sample_reachable(rng, start_range, max_steps):
    start = rng.randint(*start_range)
    value = start
    for _ in range(rng.randint(1, max_steps - 1)):
        value = apply_tool(value, rng.choice(TOOLS[:-1]))
    return start, value


class ToolEnv:
    def __init__(self, max_steps=6, start_range=(0, 12), target_range=(0, 30),
                 step_penalty=0.02, shaping_coef=0.05, seed=None):
        self.max_steps = max_steps
        self.start_range = start_range
        self.target_range = target_range
        self.step_penalty = step_penalty
        self.shaping_coef = shaping_coef
        self.rng = random.Random(seed)
        self.reset()

    def sample_task(self):
        return sample_reachable(self.rng, self.start_range, self.max_steps)

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
        after = abs(self.value - self.target)
        reward = -self.step_penalty + self.shaping_coef * (before - after)
        self.hit = self.value == self.target
        if tool == "stop" or self.steps >= self.max_steps:
            self.done = True
            if self.hit:
                reward += 1.0
        return (self.value, self.target), reward, self.done, {"hit": self.hit, "tool": tool}
