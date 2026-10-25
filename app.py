# /// script
# requires-python = ">=3.10"
# dependencies = ["streamlit>=1.37", "plotly>=5.22", "numpy>=1.26"]
# ///
import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

RUNS = Path("runs")


def load_run(path):
    records = {"metrics": [], "probe": [], "episode": [], "interp": None, "config": None}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        kind = record["type"]
        if kind == "interp":
            records["interp"] = record
        elif kind == "config":
            records["config"] = record
        elif kind in records:
            records[kind].append(record)
    return records


def list_runs():
    return sorted(RUNS.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def animated_curve(metrics, key, title, color):
    xs = [m["iter"] for m in metrics]
    ys = [m.get(key) for m in metrics]
    frames = [
        go.Frame(data=[go.Scatter(x=xs[: i + 1], y=ys[: i + 1], mode="lines+markers",
                                  line=dict(color=color))], name=str(i))
        for i in range(len(xs))
    ]
    figure = go.Figure(
        data=[go.Scatter(x=xs[:1], y=ys[:1], mode="lines+markers", line=dict(color=color))],
        frames=frames,
    )
    figure.update_layout(
        title=title, height=320, margin=dict(l=40, r=20, t=40, b=30),
        xaxis_title="iteration", yaxis_title=key,
        updatemenus=[dict(type="buttons", showactive=False, x=0, y=1.18,
                          buttons=[dict(label="play", method="animate",
                                        args=[None, dict(frame=dict(duration=120, redraw=True),
                                                         fromcurrent=True)])])],
    )
    return figure


def training_view(run):
    metrics = run["metrics"]
    if not metrics:
        st.info("no metrics in this run")
        return
    columns = st.columns(3)
    specs = [("reward", "reward", "#2f6df0"), ("kl", "kl to reference", "#e06c2b"),
             ("entropy", "policy entropy", "#2fae7a")]
    for column, (key, title, color) in zip(columns, specs):
        with column:
            st.plotly_chart(animated_curve(metrics, key, title, color), use_container_width=True)
    last = metrics[-1]
    st.metric("final reward", f"{last['reward']:.3f}")
    st.metric("final success rate", f"{last.get('success', 0):.2f}")


def episode_view(run):
    episodes = run["episode"]
    if not episodes:
        st.info("no sample episodes in this run")
        return
    choice = st.selectbox("episode", range(len(episodes)),
                          format_func=lambda i: f"start {episodes[i]['start']} -> target {episodes[i]['target']}")
    episode = episodes[choice]
    st.write(f"target {episode['target']}, success {episode['success']}, return {episode['return']}")
    steps = episode["steps"]
    step = st.slider("step", 0, len(steps), 0)
    value = episode["start"]
    rows = [{"step": 0, "tool": "start", "observation": value, "reward": 0.0}]
    for i, s in enumerate(steps[:step]):
        rows.append({"step": i + 1, "tool": s["tool"], "observation": s["observation"], "reward": s["reward"]})
    st.table(rows)
    reached = rows[-1]["observation"]
    progress = go.Figure(go.Indicator(
        mode="gauge+number", value=reached,
        gauge=dict(axis=dict(range=[-10, 60]), bar=dict(color="#2f6df0"),
                   threshold=dict(line=dict(color="#e0462b", width=4), value=episode["target"])),
        title=dict(text="current value vs target")))
    progress.update_layout(height=260, margin=dict(l=30, r=30, t=40, b=10))
    st.plotly_chart(progress, use_container_width=True)


def comparison_view(runs):
    figure = go.Figure()
    palette = {"reinforce": "#2f6df0", "ppo": "#e06c2b", "grpo": "#2fae7a"}
    for path in runs:
        run = load_run(path)
        if not run["metrics"] or not run["config"]:
            continue
        algo = run["config"]["algo"]
        xs = [m["iter"] for m in run["metrics"]]
        ys = [m["reward"] for m in run["metrics"]]
        figure.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"{algo} ({path.stem})",
                                    line=dict(color=palette.get(algo))))
    figure.update_layout(title="reward by algorithm", height=420, xaxis_title="iteration",
                         yaxis_title="reward")
    st.plotly_chart(figure, use_container_width=True)


def interp_view(run):
    probes = run["probe"]
    metrics = run["metrics"]
    if probes:
        xs = [p["iter"] for p in probes]
        ys = [p["decodability"] for p in probes]
        frames = [go.Frame(data=[go.Scatter(x=xs[: i + 1], y=ys[: i + 1], mode="lines+markers",
                                            line=dict(color="#7b3ff0"))], name=str(i))
                  for i in range(len(xs))]
        figure = go.Figure(data=[go.Scatter(x=xs[:1], y=ys[:1], mode="lines+markers",
                                            line=dict(color="#7b3ff0"))], frames=frames)
        figure.update_layout(title="probe decodability of success", height=320,
                             xaxis_title="iteration", yaxis_title="accuracy",
                             updatemenus=[dict(type="buttons", x=0, y=1.18,
                                               buttons=[dict(label="play", method="animate",
                                                             args=[None, dict(frame=dict(duration=200, redraw=True),
                                                                              fromcurrent=True)])])])
        st.plotly_chart(figure, use_container_width=True)
    if metrics:
        st.plotly_chart(animated_curve(metrics, "entropy", "entropy collapse", "#2fae7a"),
                        use_container_width=True)
    interp = run["interp"]
    if interp:
        st.subheader("probe causality check")
        columns = st.columns(3)
        columns[0].metric("suppress direction", f"{interp['causal_suppressed']:.2f}")
        columns[1].metric("no steering", f"{interp['causal_base']:.2f}")
        columns[2].metric("boost direction", f"{interp['causal_boosted']:.2f}")
        verdict = "causal" if interp["causal_direction"] else "not causal (correlational)"
        st.write(f"effect of steering on success rate: {interp['causal_effect']:.3f} -> {verdict}")
        st.write(f"base vs trained policy divergence: {interp['policy_diff']:.3f}")


def main():
    st.set_page_config(page_title="tool-agent-rl", layout="wide")
    st.title("tool-agent-rl")
    runs = list_runs()
    if not runs:
        st.warning("no runs found. train first with: uv run train.py --algo reinforce")
        return
    tabs = st.tabs(["training", "episode player", "algorithm comparison", "interpretability"])
    with tabs[2]:
        comparison_view(runs)
    selected = st.sidebar.selectbox("run", runs, format_func=lambda p: p.name)
    run = load_run(selected)
    with tabs[0]:
        training_view(run)
    with tabs[1]:
        episode_view(run)
    with tabs[3]:
        interp_view(run)


if __name__ == "__main__":
    main()
