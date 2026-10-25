# Stable learning: reachable tasks and shaped reward

## Problem
Training collapsed. Entropy fell to near zero within about five iterations while
the success rate stayed near zero, so the policy committed to a do-nothing
strategy before it learned anything. Two causes: the only positive reward was the
plus one for landing exactly on the target, which random exploration almost never
hit, and the exploration pressure was too weak against too large a step size.

## Changes
- Generate every target by applying a random sequence of tool moves to the start,
  so each task is reachable within the step budget.
- Replace the sparse exact-hit reward with a dense shaping reward each step,
  proportional to how much closer the move gets to the target, kept alongside the
  terminal bonus for an exact hit and the small per-step penalty.
- Lower the learning rate and raise the entropy bonus across all three trainers so
  entropy declines gradually instead of collapsing.
- Raise the default iteration count.

## Result
Entropy declines gradually, reward and success rise over training, and the
learning tests pass for all three algorithms.
