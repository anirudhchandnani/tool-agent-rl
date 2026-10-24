# Interpretability

## Goal
Look inside the policy while it trains, and be honest about what the signals mean.

## What this adds
- interp.py:
  - A linear probe trained on the decision-step hidden state to predict whether
    an episode will succeed, with accuracy tracked over training.
  - A base versus trained policy difference measured as the divergence of their
    action distributions on the same states.
  - A causality check: steer the hidden state along the probe direction, both
    positively and negatively, and measure the change in actual success rate.

## Honesty
Entropy collapse and a rising probe accuracy only show the direction is readable,
not that the policy uses it. The steering test reports the real effect size and
calls the direction causal only if moving along it actually changes behavior.

## Tests
- The probe separates clearly separable data.
- Decodability stays in the unit interval.
- The policy difference is near zero for identical policies.
- The causality check returns a signed effect and a boolean.
