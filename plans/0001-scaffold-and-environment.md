# Scaffold and environment

## Goal
Set up the project and the task the agent will learn to solve.

## What this adds
- Project metadata in pyproject.toml so everything runs through uv.
- The tool environment in env.py: an agent turns a start integer into a target
  integer by calling tools from a fixed set (add 2, mul 3, sub 1, stop).
- Reward shaping: plus one for hitting the target, a small penalty per step, and
  a penalty for moving away from the target.
- Value clipping so observations stay in a bounded range.

## Tests
- Tool math and clipping.
- Reward for a successful episode, step penalty, and overshoot penalty.
- Episodes end on stop and at the step limit.

## Notes
Episodes are short so training finishes on a CPU in a few minutes.
