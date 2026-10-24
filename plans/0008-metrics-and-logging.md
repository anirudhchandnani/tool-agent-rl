# Metrics and jsonl logging

## Goal
Make every run produce a record the UI can read back.

## What this adds
- train.py writes one jsonl file per run into runs/.
- Line types:
  - config: the algorithm, iteration count, and seed.
  - metrics: reward, KL, entropy, success, and loss per iteration.
  - probe: decodability of success at a set interval.
  - interp: final decodability, base versus trained policy difference, and the
    full result of the causality check.
  - episode: a few greedy rollouts with each tool call, observation, and reward
    for the episode player.

## Notes
Every value is rounded so the files stay small and easy to read.
