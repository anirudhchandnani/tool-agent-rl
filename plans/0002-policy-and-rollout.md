# Policy and rollout

## Goal
Add the policy network and the code that runs episodes with it.

## What this adds
- policy.py: a small from-scratch autoregressive transformer over a stream of
  observation and action tokens. It has an action head and a value head, plus a
  helper to make a frozen reference copy.
- A token scheme that maps values, targets, and actions to distinct ids so a
  larger pretrained model could replace it later.
- rollout.py: sampling of episodes, a teacher-forced evaluate pass that returns
  log probabilities, entropy, values, and decision-step hidden states, a KL
  helper against the reference, and returns-to-go.

## Tests
- Hidden and head shapes.
- The causal mask so a later token cannot change an earlier position.
- The reference copy is frozen and matches the policy at creation.

## Notes
The decision-step hidden states are kept so the interpretability work can probe
them later.
