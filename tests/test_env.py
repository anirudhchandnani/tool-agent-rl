import random

from env import ToolEnv, apply_tool, clip_value, sample_reachable, TOOLS, VALUE_MIN, VALUE_MAX


def test_apply_tool_operations():
    assert apply_tool(4, "add2") == 6
    assert apply_tool(4, "mul3") == 12
    assert apply_tool(4, "sub1") == 3
    assert apply_tool(4, "stop") == 4


def test_clip_bounds():
    assert clip_value(1000) == VALUE_MAX
    assert clip_value(-1000) == VALUE_MIN
    assert apply_tool(VALUE_MAX, "mul3") == VALUE_MAX


def test_reset_is_deterministic_with_seed():
    a = ToolEnv(seed=3).reset()
    b = ToolEnv(seed=3).reset()
    assert a == b


def test_reset_with_explicit_task():
    env = ToolEnv()
    value, target = env.reset(start=5, target=11)
    assert value == 5 and target == 11


def test_hitting_target_rewards_positive():
    env = ToolEnv()
    env.reset(start=9, target=11)
    (value, _), reward, done, info = env.step(TOOLS.index("add2"))
    assert value == 11
    (_, _), reward, done, info = env.step(TOOLS.index("stop"))
    assert done and info["hit"]
    assert reward > 0.9


def test_step_penalty_applies():
    env = ToolEnv(step_penalty=0.02)
    env.reset(start=0, target=30)
    _, reward, _, _ = env.step(TOOLS.index("stop"))
    assert reward < 0


def test_progress_rewards_positive():
    env = ToolEnv(step_penalty=0.02, shaping_coef=0.05)
    env.reset(start=0, target=30)
    _, reward, _, _ = env.step(TOOLS.index("add2"))
    assert reward > 0


def test_moving_away_is_penalised():
    env = ToolEnv(step_penalty=0.0, shaping_coef=0.1)
    env.reset(start=10, target=11)
    _, reward, _, _ = env.step(TOOLS.index("mul3"))
    assert reward < 0


def test_sample_reachable_within_bounds():
    start, target = sample_reachable(random.Random(0), (0, 12), 6)
    assert VALUE_MIN <= target <= VALUE_MAX


def test_episode_ends_at_max_steps():
    env = ToolEnv(max_steps=3)
    env.reset(start=0, target=59)
    done = False
    count = 0
    while not done:
        _, _, done, _ = env.step(TOOLS.index("add2"))
        count += 1
    assert count == 3 and env.done


def test_stop_ends_episode():
    env = ToolEnv()
    env.reset(start=1, target=20)
    _, _, done, _ = env.step(TOOLS.index("stop"))
    assert done
