# demo4_support_ticket_orchestrator.py
from __future__ import annotations
import os, json, time
from typing import TypedDict, List, Dict, Literal, Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or ""
UA = os.getenv("USER_AGENT", "GenAI-Session8/1.0 (+contact: you@example.com)")
CHECKPOINT_DB = os.getenv("TICKET_DB", "session8_tickets.sqlite")
TIMEOUT_SEC = float(os.getenv("KB_TIMEOUT_SEC", "12"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY not set")

LLM_TRIAGE  = ChatOpenAI(model=os.getenv("LLM_TRIAGE_MODEL","gpt-4o-mini"), temperature=0)
LLM_WRITE   = ChatOpenAI(model=os.getenv("LLM_WRITE_MODEL","gpt-4o"),      temperature=0)
LLM_REVIEW  = ChatOpenAI(model=os.getenv("LLM_REVIEW_MODEL","gpt-4o-mini"),temperature=0)

tavily = TavilySearchAPIWrapper()
splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)

class TicketState(TypedDict, total=False):
    ticket: Dict
    severity: Literal["low","medium","high"]
    queries: List[str]
    kb_chunks: List[str]
    kb_ok: bool
    draft_fix: str
    review_decision: Literal["close","escalate"]
    error: Optional[str]

def _safe_json_list(text: str, fallback: List[str]) -> List[str]:
    try:
        v = json.loads(text)
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            return v
    except Exception:
        pass
    return fallback

def triage(state: TicketState) -> TicketState:
    tk = state["ticket"]
    prompt = f"""Classify ticket severity (low/medium/high) and propose 2–4 focused web queries to retrieve docs.
Return JSON: {{"severity":"low|medium|high","queries":["..."]}}
Ticket: {json.dumps(tk, ensure_ascii=False)}"""
    res = LLM_TRIAGE.invoke(prompt)
    parsed = {}
    try: parsed = json.loads(res.content)
    except Exception: pass
    severity = parsed.get("severity","medium")
    queries  = parsed.get("queries") or [f"{tk.get('product','')} {tk.get('issue','')} troubleshooting"]
    return {"severity": severity, "queries": queries[:4]}

def kb_search(state: TicketState) -> TicketState:
    queries = state.get("queries", [])
    chunks: List[str] = []
    ok = False
    started = time.time()
    for q in queries:
        # time budget check
        if time.time() - started > TIMEOUT_SEC:
            return {"kb_ok": False, "kb_chunks": chunks, "error": "kb timeout"}
        try:
            hits = tavily.results(q, max_results=3) or []
            for h in hits:
                url = h.get("url") or h.get("link")
                if not url:
                    continue
                docs = WebBaseLoader(url, requests_kwargs={"headers": {"User-Agent": UA, "timeout": str(TIMEOUT_SEC)}}).load()
                if docs:
                    ok = True
                    for d in splitter.create_documents([docs[0].page_content]):
                        if d.page_content:
                            chunks.append(d.page_content)
                if len(chunks) >= 12:
                    break
        except Exception:
            continue
    return {"kb_chunks": chunks[:12], "kb_ok": ok}

def propose_fix(state: TicketState) -> TicketState:
    tk = state["ticket"]; ch = state.get("kb_chunks", [])
    ctx = "\n\n".join([c[:900] for c in ch])
    prompt = f"""Using ONLY the knowledge base context, propose safe, actionable steps to resolve the ticket.
Ticket: {json.dumps(tk, ensure_ascii=False)}

KB Context:
{ctx}

Write 6–10 numbered steps. If context insufficient, say so and suggest what to collect next."""
    res = LLM_WRITE.invoke(prompt)
    return {"draft_fix": res.content}

def review(state: TicketState) -> TicketState:
    sev = state.get("severity", "medium")
    okkb = state.get("kb_ok", False)
    draft = state.get("draft_fix","")
    prompt = f"""Decide to "close" or "escalate".
Rules: if severity is high OR kb_ok is false OR draft lacks steps → escalate.
Else close.
Return JSON: {{"decision":"close|escalate"}}.
Severity={sev}, kb_ok={okkb}
Draft:\n{draft}"""
    res = LLM_REVIEW.invoke(prompt)
    decision = "escalate"
    try:
        decision = json.loads(res.content).get("decision","escalate")
    except Exception:
        pass
    return {"review_decision": decision}

def route(state: TicketState) -> TicketState:
    return {}

def finalize(state: TicketState, *, thread_id: str) -> TicketState:
    out = {
        "thread_id": thread_id,
        "ticket": state.get("ticket"),
        "severity": state.get("severity"),
        "kb_ok": state.get("kb_ok"),
        "decision": state.get("review_decision"),
        "draft_fix": state.get("draft_fix"),
        "error": state.get("error"),
    }
    fname = f"ticket_out_{thread_id}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved: {fname}")
    return {}

def route_after_review(state: TicketState) -> Literal["close","escalate"]:
    return "close" if state.get("review_decision") == "close" else "escalate"

graph = StateGraph(TicketState)
graph.add_node("triage",     triage)
graph.add_node("kb_search",  kb_search)
graph.add_node("propose_fix",propose_fix)
graph.add_node("review",     review)
graph.add_node("close",      lambda s: s)
graph.add_node("escalate",   lambda s: s)
graph.add_node("finalize",   lambda s, config: finalize(s, thread_id=config['configurable']['thread_id']))

graph.set_entry_point("triage")
graph.add_edge("triage", "kb_search")
graph.add_edge("kb_search", "propose_fix")
graph.add_edge("propose_fix","review")
graph.add_conditional_edges("review", route_after_review, {
    "close": "close",
    "escalate": "escalate",
})
graph.add_edge("close",    "finalize")
graph.add_edge("escalate", "finalize")
graph.add_edge("finalize", END)

def run_one(ticket: Dict, thread_id: str):
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as ckpt:
        app = graph.compile(checkpointer=ckpt)
        cfg = {"configurable": {"thread_id": thread_id}}
        start = {"ticket": ticket}
        for ev in app.stream(start, config=cfg):
            for k in ev.keys(): print(f"[{k}] ✓")
        print("DONE.")

if __name__ == "__main__":
    tickets = [
        {"id":"T-1001","product":"WidgetX","issue":"API timeout on /v1/charge","env":"prod","impact":"payments failing"},
        {"id":"T-1002","product":"WidgetX","issue":"UI shows stale price after rule change","env":"staging","impact":"incorrect price"},
    ]
    for i, tk in enumerate(tickets, start=1):
        run_one(tk, thread_id=f"ticket-{i}")
