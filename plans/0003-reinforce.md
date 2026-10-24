# REINFORCE with a baseline

## Goal
Add the first and simplest trainer.

## What this adds
- algos/reinforce.py: policy-gradient training with a baseline subtracted from
  returns to lower variance, an entropy bonus, and a KL penalty to the frozen
  reference.
- train.py: a command line entry point that picks an algorithm, runs it, and
  prints progress.

## How it works
Sample a batch of episodes, turn returns into advantages, and push up the log
probability of actions that did better than the batch baseline.

## Tests
- A training step returns finite metrics and updates parameters.
- Reward goes up over a short run.
