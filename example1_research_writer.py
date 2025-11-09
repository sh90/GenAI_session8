# session8_research_writer_graph.py
from __future__ import annotations
import os, json
from typing import TypedDict, List, Literal, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
UA = os.getenv("USER_AGENT", "GenAI-Session8/1.0 (+contact: you@example.com)")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment/.env")
if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY is not set in environment/.env")

# ---------- LLMs ----------
LLM_PLAN   = ChatOpenAI(model="gpt-4o-mini", temperature=0)
LLM_WRITE  = ChatOpenAI(model="gpt-4o",      temperature=0)
LLM_REVIEW = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------- Tools ----------
tavily = TavilySearchAPIWrapper()  # uses TAVILY_API_KEY
splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=120)

# ---------- State ----------
class RWState(TypedDict, total=False):
    query: str
    urls: List[str]
    chunks: List[str]
    draft: str
    feedback: str
    decision: Literal["ok", "revise"]
    error: Optional[str]

# ---------- helpers ----------
def _safe_json_list(text: str, fallback: List[str]) -> List[str]:
    try:
        v = json.loads(text)
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            return v
    except Exception:
        pass
    return fallback

# ---------- Nodes ----------
def plan(state: RWState) -> RWState:
    """Turn the user question into 3–5 focused search queries and fetch top URLs."""
    q = state["query"]
    prompt = f"""Break the question into 3–5 focused web queries.
Question: {q}
Return as a JSON list of strings only."""
    res = LLM_PLAN.invoke(prompt)
    subqueries = _safe_json_list(res.content, [q])

    urls: List[str] = []
    for sq in subqueries[:5]:
        try:
            hits = tavily.results(sq, max_results=3) or []
            for h in hits:
                url = h.get("url") or h.get("link")
                if url and url not in urls:
                    urls.append(url)
        except Exception:
            # skip this subquery on any search error
            continue
    return {"urls": urls[:8]}

def fetch(state: RWState) -> RWState:
    """Load and chunk pages."""
    urls = state.get("urls", [])
    texts: List[str] = []
    for u in urls[:8]:
        try:
            docs = WebBaseLoader(u, requests_kwargs={"headers": {"User-Agent": UA}}).load()
            text = docs[0].page_content if docs else ""
            if text:
                for d in splitter.create_documents([text]):
                    if d.page_content:
                        texts.append(d.page_content)
        except Exception:
            continue
    return {"chunks": texts[:12]}

def write(state: RWState) -> RWState:
    """Write a short answer using the gathered chunks with inline citations."""
    q = state["query"]
    ch = state.get("chunks", [])
    ctx = "\n\n".join([f"[{i+1}] {c[:900]}" for i, c in enumerate(ch)])
    prompt = f"""Answer the question using ONLY the evidence below.
Question: {q}

Evidence:
{ctx}

Write 8–12 sentences with 3–5 bullet takeaways.
Cite inline as [n]. At end, add "Sources: [1] ... [2] ...".
Keep it tight and factual."""
    res = LLM_WRITE.invoke(prompt)
    return {"draft": res.content}

def review(state: RWState) -> RWState:
    """Quality review with pass/fail decision."""
    draft = state.get("draft", "")
    prompt = (
        'You are a strict editor. Check: accuracy (use only provided citations), clarity, and length.\n'
        'If acceptable, respond with JSON: {"decision":"ok","feedback":"..."}.\n'
        'If not, respond with JSON: {"decision":"revise","feedback":"what to fix ..."}.\n'
        f"Draft:\n{draft}"
    )
    res = LLM_REVIEW.invoke(prompt)
    try:
        payload = json.loads(res.content)
        decision = payload.get("decision", "ok")
        feedback = payload.get("feedback", "Looks good.")
    except Exception:
        decision, feedback = "ok", "Looks good."
    return {"decision": decision, "feedback": feedback}

def rewrite(state: RWState) -> RWState:
    """Revise draft according to feedback."""
    draft = state.get("draft", "")
    fb    = state.get("feedback", "")
    prompt = f"""Revise the draft per feedback (preserve citations).
Feedback: {fb}
Draft:
{draft}
"""
    res = LLM_WRITE.invoke(prompt)
    return {"draft": res.content}

def finalize(_: RWState) -> RWState:
    """No-op node; could persist/publish."""
    return {}

# ---------- Router ----------
def route_after_review(state: RWState) -> Literal["finalize", "rewrite"]:
    return "finalize" if state.get("decision") == "ok" else "rewrite"

# ---------- Graph ----------
graph = StateGraph(RWState)
graph.add_node("plan",    plan)
graph.add_node("fetch",   fetch)
graph.add_node("write",   write)
graph.add_node("review",  review)
graph.add_node("rewrite", rewrite)
graph.add_node("finalize", finalize)

graph.set_entry_point("plan")
graph.add_edge("plan",   "fetch")
graph.add_edge("fetch",  "write")
graph.add_edge("write",  "review")
graph.add_conditional_edges("review", route_after_review, {
    "finalize": "finalize",
    "rewrite":  "rewrite",
})
graph.add_edge("rewrite", "review")
graph.add_edge("finalize", END)

# ---------- Run with persistent checkpoint (keep context OPEN) ----------
if __name__ == "__main__":
    user_query = "Best practices to evaluate RAG systems in production and open-source tools supporting them."
    start_state: RWState = {"query": user_query}
    config = {"configurable": {"thread_id": "sess8-demo1"}}

    # Keep compile + stream INSIDE the context so the DB stays open
    with SqliteSaver.from_conn_string("session8_runs.sqlite") as checkpointer:
        app = graph.compile(checkpointer=checkpointer)

        for ev in app.stream(start_state, config=config):
            # each `ev` is a dict {node_name: state_delta}
            print(ev)

        print("\n--- FINAL ---")
        final_state = app.get_state(config).values
        print(final_state.get("draft", "<no draft>"))
