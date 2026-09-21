# 🔬 Multi-Agent Research System

An autonomous research pipeline powered by four cooperating AI agents — each 
with a single responsibility — orchestrated end-to-end with a live Streamlit UI.

## How it works

1. **🔎 Search Agent** — queries the web (via Tavily) for recent, reliable 
   sources on the given topic.
2. **📄 Reader Agent** — picks the most relevant result and scrapes it for 
   deeper, structured content.
3. **✍️ Writer** — synthesizes the search results + scraped content into a 
   coherent, well-structured report.
4. **🧐 Critic** — reviews the report for gaps, weak claims, and missing 
   context, and returns actionable feedback.

Each step streams live into the UI with progress, timing, and content previews 
— no waiting on a black-box terminal run.

## Tech stack

- **Orchestration:** LangGraph / LangChain agent framework
- **LLMs:** Groq + Mistral
- **Search:** Tavily API
- **UI:** Streamlit
- **Language:** Python 3.13

## Features

- 🔄 Real-time step-by-step progress (search → read → write → critique)
- ⏱️ Per-agent timing metrics
- 📑 Tabbed results view (Report / Critic Review / Raw Search / Scraped Content)
- 💾 One-click Markdown export (single report or full research bundle)
- 🕘 Session history to revisit past runs
- ✅ Built-in environment/API key health check

## Setup

\`\`\`bash
git clone <repo-url>
cd Multiagentsystem
pip install -r requirements.txt
\`\`\`

Create a `.env` file with:
\`\`\`
GROQ_API_KEY=your_key
MISTRAL_API_KEY=your_key
TAVILY_API_KEY=your_key
\`\`\`

## Run

**CLI:**
\`\`\`bash
python pipeline.py
\`\`\`

**Web UI:**
\`\`\`bash
streamlit run app.py
\`\`\`

## Architecture

\`\`\`
Topic Input
    │
    ▼
Search Agent (Tavily) ──▶ Reader Agent (scrape) ──▶ Writer Chain ──▶ Critic Chain
    │                          │                        │               │
    └──────────────────────────┴────────────────────────┴───────────────┘
                                    │
                            Streamlit UI (live progress + report tabs)
\`\`\`

## Roadmap

- [ ] Multi-source scraping (not just the single top result)
- [ ] Citation tracking in the final report
- [ ] Configurable LLM provider per agent
- [ ] Async parallel search + scrape for speed
