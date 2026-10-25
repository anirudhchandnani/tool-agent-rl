# Streamlit UI

## Goal
Make the runs explorable in a browser.

## What this adds
- app.py built with Streamlit and Plotly. It reads the jsonl run logs and shows:
  - Training curves for reward, KL, and entropy that animate as a run plays.
  - An episode player that steps through a rollout, showing each tool call, the
    observation, and the reward, with a gauge of the current value against the
    target.
  - An algorithm comparison view across REINFORCE, PPO, and GRPO.
  - An interpretability view that animates entropy collapse and probe
    decodability and reports the steering result.

## Run
uv run streamlit run app.py
