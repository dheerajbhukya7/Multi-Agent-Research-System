"""
Streamlit UI for the multi-agent research system.

Run from the project folder (M:\\Machine Learning\\Multiagentsystem):
    streamlit run app.py
"""

from __future__ import annotations

import os
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import streamlit as st
from dotenv import dotenv_values, load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

ENV_PATH = Path(__file__).with_name(".env")


# --------------------------------------------------------------------------- #
# Lazy import: importing `pipeline` builds models/clients, so only do it when
# the user actually runs something. That way the UI still loads if a key is
# missing and we can show a readable error instead of a crashed page.
# --------------------------------------------------------------------------- #

def load_pipeline():
    from pipeline import PIPELINE_STEPS, run_research_pipeline, state_to_markdown
    return run_research_pipeline, state_to_markdown, PIPELINE_STEPS


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

st.session_state.setdefault("result", None)
st.session_state.setdefault("history", [])
st.session_state.setdefault("error", None)
st.session_state.setdefault("topic_input", "")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return (slug or "report")[:60]


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("⚙️ Settings")

    scrape_chars = st.slider(
        "Search context passed to the Reader agent",
        min_value=400,
        max_value=4000,
        value=800,
        step=200,
        help="How many characters of the search output the reader sees when "
             "choosing a URL to scrape. Larger = better choices, more tokens.",
    )

    st.divider()
    st.subheader("🔑 Environment")

    if ENV_PATH.exists():
        env_keys = [k for k in dotenv_values(ENV_PATH) if k]
        if env_keys:
            for key in sorted(env_keys):
                value = os.getenv(key, "")
                if value:
                    st.write(f"✅ `{key}` · `{value[:4]}…{value[-3:] if len(value) > 8 else ''}`")
                else:
                    st.write(f"⚠️ `{key}` · empty")
        else:
            st.warning("`.env` found but no keys parsed.")
    else:
        st.error("No `.env` file found next to `app.py`.")

    st.divider()
    st.subheader("🕘 History")

    if st.session_state.history:
        for i, run in enumerate(reversed(st.session_state.history)):
            idx = len(st.session_state.history) - 1 - i
            label = f"{run['topic'][:34]} · {run['time']}"
            if st.button(label, key=f"hist_{idx}", use_container_width=True):
                st.session_state.result = run["state"]
                st.session_state.error = None
                st.rerun()
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.session_state.result = None
            st.rerun()
    else:
        st.caption("Completed runs will appear here.")


# --------------------------------------------------------------------------- #
# Header + input
# --------------------------------------------------------------------------- #

st.title("🔬 Multi-Agent Research System")
st.caption("Search → Read → Write → Critique. Four agents, one report.")

with st.form("research_form"):
    topic = st.text_input(
        "Research topic",
        placeholder="e.g. Recent advances in retrieval-augmented generation",
        value=st.session_state.topic_input,
    )
    submitted = st.form_submit_button("🚀 Run research", type="primary")

example_cols = st.columns(3)
EXAMPLES = [
    "State of small language models in 2026",
    "Solid-state battery commercialisation",
    "AI regulation in the European Union",
]
for col, example in zip(example_cols, EXAMPLES):
    if col.button(example, use_container_width=True):
        st.session_state.topic_input = example
        st.rerun()


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

def execute(topic: str, scrape_chars: int) -> None:
    run_research_pipeline, _, steps = load_pipeline()

    progress = st.progress(0.0, text="Starting…")
    total_steps = len(steps)

    with st.status(f"Researching “{topic}”", expanded=True) as status:
        def handle(event: Dict[str, Any]) -> None:
            etype = event["type"]
            if etype == "step_start":
                status.update(label=f"Step {event['step']}/{total_steps} — {event['label']}")
                st.write(f"**{event['label']}** — {event['desc']}…")
                progress.progress(
                    (event["step"] - 1) / total_steps,
                    text=f"{event['label']} running…",
                )
            elif etype == "step_end":
                st.write(f"✅ {event['label']} finished in {event['elapsed']:.1f}s")
                preview = (event["content"] or "")[:600]
                with st.expander(f"Preview · {event['label']}", expanded=False):
                    st.markdown(preview + ("…" if len(event["content"]) > 600 else ""))
                progress.progress(
                    event["step"] / total_steps,
                    text=f"{event['step']}/{total_steps} steps done",
                )
            elif etype == "step_error":
                st.write(f"❌ {event['label']}: {event['error']}")

        try:
            state = run_research_pipeline(
                topic,
                on_event=handle,
                scrape_context_chars=scrape_chars,
            )
        except Exception:
            status.update(label="Pipeline failed", state="error")
            progress.empty()
            st.session_state.error = traceback.format_exc()
            st.session_state.result = None
            return

        status.update(
            label=f"Done in {state['timings']['total']:.1f}s",
            state="complete",
            expanded=False,
        )

    progress.empty()
    st.session_state.result = state
    st.session_state.error = None
    st.session_state.history.append({
        "topic": topic,
        "time": datetime.now().strftime("%H:%M"),
        "state": state,
    })


if submitted:
    if not topic.strip():
        st.warning("Enter a topic first.")
    else:
        st.session_state.topic_input = topic
        execute(topic.strip(), scrape_chars)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

if st.session_state.error:
    st.error("The pipeline raised an exception.")
    with st.expander("Traceback", expanded=True):
        st.code(st.session_state.error, language="text")

result = st.session_state.result

if result:
    _, state_to_markdown, _ = load_pipeline()

    st.divider()
    st.subheader(f"Results — {result['topic']}")

    timings = result.get("timings", {})
    metric_cols = st.columns(5)
    metric_cols[0].metric("Total", f"{timings.get('total', 0):.1f}s")
    metric_cols[1].metric("Search", f"{timings.get('search_results', 0):.1f}s")
    metric_cols[2].metric("Read", f"{timings.get('scraped_content', 0):.1f}s")
    metric_cols[3].metric("Write", f"{timings.get('report', 0):.1f}s")
    metric_cols[4].metric("Critique", f"{timings.get('feedback', 0):.1f}s")

    report_tab, critic_tab, search_tab, scrape_tab = st.tabs(
        ["📝 Report", "🧐 Critic Review", "🔎 Search Results", "📄 Scraped Content"]
    )

    with report_tab:
        st.markdown(result.get("report", "_No report generated._"))

    with critic_tab:
        st.markdown(result.get("feedback", "_No feedback generated._"))

    with search_tab:
        st.markdown(result.get("search_results", "_No search results._"))

    with scrape_tab:
        st.markdown(result.get("scraped_content", "_No scraped content._"))

    st.divider()
    slug = slugify(result["topic"])
    dl_cols = st.columns(2)
    dl_cols[0].download_button(
        "⬇️ Download report (.md)",
        data=result.get("report", ""),
        file_name=f"{slug}-report.md",
        mime="text/markdown",
        use_container_width=True,
    )
    dl_cols[1].download_button(
        "⬇️ Download full bundle (.md)",
        data=state_to_markdown(result),
        file_name=f"{slug}-full.md",
        mime="text/markdown",
        use_container_width=True,
    )
else:
    if not st.session_state.error:
        st.info("Enter a topic above and hit **Run research** to start the agents.")
