# tool-agent-rl

Three policy-gradient algorithms trained from scratch on the same multi-step
tool-using task, with a small transformer policy, a set of interpretability
checks that stay honest about what they prove, and a Streamlit app for exploring
every run.

The point of the project is not to hit state of the art on a benchmark. It is to
build REINFORCE, PPO, and GRPO side by side on one task so the differences
between them are visible in code and in the training curves, and to show a
disciplined way of looking inside the policy as it learns.

## The task

The environment lives in `env.py`. An agent starts from an integer and has to
reach a target integer by calling tools:

- `add2` adds 2
- `mul3` multiplies by 3
- `sub1` subtracts 1
- `stop` ends the episode

Each episode lasts at most six steps. All values are clipped to a bounded range
so observations stay finite. The reward is shaped so the right behavior is to
reach the target quickly and then stop:

- plus 1.0 for finishing exactly on the target
- a small penalty of 0.02 per step so dawdling costs something
- a penalty of 0.05 for any non-stop move that increases the distance to the
  target, which discourages overshooting

Because start and target are sampled fresh every episode, the policy has to learn
a general strategy, not a single memorized sequence.

## The policy

`policy.py` defines a small autoregressive transformer that reads a stream of
tokens. Values, targets, and actions each map to their own block of token ids, so
the model sees a single flat sequence of observations and tool calls and predicts
the next tool. It has two heads: an action head over the four tools and a value
head used as a learned baseline by PPO. A causal mask keeps each position from
attending to the future, and a helper produces a frozen reference copy of the
policy for the KL penalty described below.

The token scheme is deliberately generic so the hand-rolled transformer could be
swapped for a larger pretrained model without changing the training code.

`rollout.py` runs episodes with the policy and also provides a teacher-forced
evaluation pass that returns log probabilities, entropy, state values, and the
hidden state at each decision step. Those decision-step hidden states are kept so
the interpretability code can probe them later.

## The three algorithms

All three share the same policy, the same rollout code, an entropy bonus to keep
exploration alive, and a KL penalty against the frozen reference so the policy
does not drift too far from where it started.

### REINFORCE (`algos/reinforce.py`)

The simplest trainer. Sample a batch of episodes, turn each episode return into
an advantage by subtracting the batch mean and dividing by the batch standard
deviation, and push up the log probability of actions from episodes that did
better than the batch average. The batch mean is the baseline that cuts variance.
REINFORCE uses each batch for a single update and then throws it away.

### PPO (`algos/ppo.py`)

A more stable trainer that reuses each batch for several update epochs. It learns
a value function with the value head and uses it as the baseline, computing the
advantage as return-to-go minus the predicted value. The clipped surrogate
objective caps how far the probability ratio can move in one update, which is
what makes reusing the same batch safe. A value-function loss trains the critic
alongside the policy.

### GRPO (`algos/grpo.py`)

A critic-free trainer. For each start and target it samples a group of
trajectories and turns each trajectory return into a group-relative advantage by
centering and scaling within the group. The group mean plays the role PPO gives
to the value head, so there is no critic to train and no value loss to balance.
It keeps the clipped ratio and the KL penalty.

The short version of the differences: REINFORCE is one update per batch with a
batch-mean baseline. PPO reuses the batch under a clipped objective and learns a
value function for the baseline. GRPO reuses the batch under the same clipped
objective but replaces the learned value function with the mean return of a group
of samples for the same task.

## The advantage-normalization fix

An early version of REINFORCE divided returns by their standard deviation without
subtracting the batch mean first. That leaves a large positive offset on every
advantage, so almost every action gets pushed up and the baseline does nothing to
reduce variance. The fix subtracts the batch mean before dividing, so the
advantage is centered on the baseline. See `plans/0006-fix-advantage-normalization.md`.

## Interpretability

`interp.py` looks inside the policy while it trains and is careful about what each
signal actually shows.

- Entropy over training. Tracked per iteration. A falling entropy shows the
  policy is committing to a strategy, nothing more.
- A success probe. A linear probe is trained on the decision-step hidden state to
  predict whether the episode will succeed. Rising probe accuracy shows that
  whether the policy will succeed becomes linearly readable from its hidden
  state.
- A base versus trained difference. The divergence between the action
  distributions of the untrained reference policy and the trained policy on the
  same states, which measures how much behavior actually changed.
- A causality check. This is the honest part. A readable direction is not the
  same as a used direction. The check steers the hidden state along the probe
  direction, both positively and negatively, and measures the change in the real
  success rate. It reports the signed effect size and only calls the direction
  causal if moving along it actually changes behavior, rather than assuming that
  a decodable feature is one the policy relies on.

## Running it

Everything runs through `uv`, which reads `pyproject.toml` and `uv.lock`.

Train one algorithm:

```
uv run train.py --algo reinforce
uv run train.py --algo ppo
uv run train.py --algo grpo
```

Useful flags: `--iters` for the number of iterations, `--seed` for the random
seed, `--probe-every` for how often the success probe is trained, and `--out` for
the output directory. Each run writes one JSON Lines file into `runs/` with one
record per line: a config line, a metrics line per iteration, periodic probe
lines, a final interpretability line with the full causality result, and a few
sample greedy episodes.

Run the tests:

```
uv run pytest -q
```

Explore the runs in the browser:

```
uv run streamlit run app.py
```

The app (`app.py`) reads the run logs and shows animated training curves for
reward, KL, and entropy, an episode player that steps through a rollout with a
gauge of the current value against the target, a comparison of reward across the
three algorithms, and an interpretability view that animates entropy and probe
decodability and reports the steering result.

## Results and notes

The results are reproducible by running the commands above; this repository ships
the code and the plans rather than a set of baked-in numbers. What the code
measures on each run:

- reward, success rate, KL to the reference, and entropy per iteration
- success-probe decodability at a set interval
- the base versus trained policy difference
- the causality check: success rate with no steering, with the probe direction
  boosted, and with it suppressed, plus the signed effect and the causal verdict

A few things worth knowing before running:

- It is built to finish on a CPU. Episodes are short and the policy is small, so a
  full run takes minutes rather than hours. The learning tests run real short
  training loops, so the test suite is slower than a typical unit-test suite.
- `runs/` is gitignored. Train first, then open the app.
- Seeds are fixed through `--seed`, so a given command reproduces the same run.

## Layout

```
env.py           the tool environment and reward
policy.py        the transformer policy, token scheme, reference copy
rollout.py       episode sampling, evaluation pass, KL, returns-to-go
algos/           reinforce.py, ppo.py, grpo.py and the trainer registry
interp.py        probe, policy difference, causality check
train.py         CLI entry point and JSON Lines logging
app.py           Streamlit and Plotly run explorer
tests/           tests for the environment, policy, algorithms, and interp
plans/           one short plan per unit of work
```
