"""
Multi-agent research pipeline.

The pipeline is UI-agnostic: instead of printing directly, it emits events to an
optional `on_event` callback. The CLI at the bottom of this file prints those
events; `app.py` (Streamlit) renders them as live progress.

Usage (terminal):
    python pipeline.py

Usage (library):
    from pipeline import run_research_pipeline
    state = run_research_pipeline("quantum error correction", on_event=my_handler)
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from agents import (
    build_reader_agent,
    build_search_agent,
    writer_chain,
    critic_chain,
)

# --------------------------------------------------------------------------- #
# Step metadata (shared with the UI so both stay in sync)
# --------------------------------------------------------------------------- #

PIPELINE_STEPS: List[Dict[str, Any]] = [
    {
        "n": 1,
        "key": "search_results",
        "label": "Search Agent",
        "desc": "Finding recent, reliable sources on the web",
        "icon": "🔎",
    },
    {
        "n": 2,
        "key": "scraped_content",
        "label": "Reader Agent",
        "desc": "Scraping the most relevant source for depth",
        "icon": "📄",
    },
    {
        "n": 3,
        "key": "report",
        "label": "Writer",
        "desc": "Synthesising research into a structured report",
        "icon": "✍️",
    },
    {
        "n": 4,
        "key": "feedback",
        "label": "Critic",
        "desc": "Reviewing the report for gaps and weaknesses",
        "icon": "🧐",
    },
]

STEP_BY_KEY = {s["key"]: s for s in PIPELINE_STEPS}

EventHandler = Optional[Callable[[Dict[str, Any]], None]]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def as_text(value: Any) -> str:
    """Normalise LangChain output (str / AIMessage / list of parts) to plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value

    content = getattr(value, "content", None)
    if content is None and isinstance(value, dict):
        content = value.get("content") or value.get("output") or value.get("text")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        return "\n".join(p for p in parts if p)

    return str(value)


def last_message_text(agent_result: Any) -> str:
    """Pull the final message out of a LangGraph agent result."""
    if isinstance(agent_result, dict) and "messages" in agent_result:
        messages = agent_result["messages"]
        if messages:
            return as_text(messages[-1])
    return as_text(agent_result)


def _emit(on_event: EventHandler, **payload: Any) -> None:
    if on_event is not None:
        on_event(payload)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def run_research_pipeline(
    topic: str,
    on_event: EventHandler = None,
    scrape_context_chars: int = 800,
) -> Dict[str, Any]:
    """
    Run the full research pipeline for `topic`.

    Emits these events to `on_event`:
        {"type": "pipeline_start", "topic": str}
        {"type": "step_start",  "step": int, "key": str, "label": str, "desc": str}
        {"type": "step_end",    "step": int, "key": str, "label": str,
                                "content": str, "elapsed": float}
        {"type": "step_error",  "step": int, "key": str, "label": str, "error": str}
        {"type": "pipeline_end", "elapsed": float}

    Returns the state dict with keys: topic, search_results, scraped_content,
    report, feedback, timings.
    """
    topic = topic.strip()
    if not topic:
        raise ValueError("Topic must not be empty.")

    state: Dict[str, Any] = {"topic": topic, "timings": {}}
    pipeline_started = time.perf_counter()
    _emit(on_event, type="pipeline_start", topic=topic)

    def run_step(key: str, fn: Callable[[], str]) -> str:
        meta = STEP_BY_KEY[key]
        _emit(
            on_event,
            type="step_start",
            step=meta["n"],
            key=key,
            label=meta["label"],
            desc=meta["desc"],
        )
        started = time.perf_counter()
        try:
            content = fn()
        except Exception as exc:  # surface which step broke, then bubble up
            _emit(
                on_event,
                type="step_error",
                step=meta["n"],
                key=key,
                label=meta["label"],
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        elapsed = time.perf_counter() - started

        state[key] = content
        state["timings"][key] = elapsed
        _emit(
            on_event,
            type="step_end",
            step=meta["n"],
            key=key,
            label=meta["label"],
            content=content,
            elapsed=elapsed,
        )
        return content

    # ----- Step 1: search ---------------------------------------------------
    def _search() -> str:
        search_agent = build_search_agent()
        result = search_agent.invoke({
            "messages": [(
                "user",
                f"Find recent, reliable and detailed information about: {topic}",
            )]
        })
        return last_message_text(result)

    run_step("search_results", _search)

    # ----- Step 2: read / scrape -------------------------------------------
    def _read() -> str:
        reader_agent = build_reader_agent()
        result = reader_agent.invoke({
            "messages": [(
                "user",
                f"Based on the following search results about '{topic}', "
                f"pick the most relevant URL and scrape it for deeper content.\n\n"
                f"Search Results:\n{state['search_results'][:scrape_context_chars]}",
            )]
        })
        return last_message_text(result)

    run_step("scraped_content", _read)

    # ----- Step 3: write ----------------------------------------------------
    def _write() -> str:
        research_combined = (
            f"SEARCH RESULTS:\n{state['search_results']}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
        )
        return as_text(writer_chain.invoke({
            "topic": topic,
            "research": research_combined,
        }))

    run_step("report", _write)

    # ----- Step 4: critique -------------------------------------------------
    def _critique() -> str:
        return as_text(critic_chain.invoke({"report": state["report"]}))

    run_step("feedback", _critique)

    total = time.perf_counter() - pipeline_started
    state["timings"]["total"] = total
    _emit(on_event, type="pipeline_end", elapsed=total)
    return state


# --------------------------------------------------------------------------- #
# Export helper (used by the UI's download buttons)
# --------------------------------------------------------------------------- #

def state_to_markdown(state: Dict[str, Any]) -> str:
    """Bundle the whole run into a single markdown document."""
    lines = [
        f"# Research Report: {state.get('topic', 'Untitled')}",
        "",
        "## Final Report",
        "",
        state.get("report", "_not generated_"),
        "",
        "## Critic Review",
        "",
        state.get("feedback", "_not generated_"),
        "",
        "## Appendix A — Search Results",
        "",
        state.get("search_results", "_not generated_"),
        "",
        "## Appendix B — Scraped Content",
        "",
        state.get("scraped_content", "_not generated_"),
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _cli_printer(event: Dict[str, Any]) -> None:
    etype = event["type"]
    bar = "=" * 60

    if etype == "step_start":
        print(f"\n{bar}")
        print(f"Step {event['step']} — {event['label']}: {event['desc']}...")
        print(bar)
    elif etype == "step_end":
        print(f"\n{event['content']}")
        print(f"\n[done in {event['elapsed']:.1f}s]")
    elif etype == "step_error":
        print(f"\n!! Step {event['step']} ({event['label']}) failed: {event['error']}")
    elif etype == "pipeline_end":
        print(f"\n{bar}")
        print(f"Pipeline finished in {event['elapsed']:.1f}s")
        print(bar)


if __name__ == "__main__":
    user_topic = input("\nEnter a research topic: ")
    run_research_pipeline(user_topic, on_event=_cli_printer)
