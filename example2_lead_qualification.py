# session8_demo2_lead_qual_router.py
from __future__ import annotations
import os, json
from typing import TypedDict, List, Literal, Optional, Dict

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# ──────────────────────────────────────────────────────────────────────────────
# Env / Config
# ──────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
UA = os.getenv("USER_AGENT", "GenAI-Session8/1.0 (+contact: you@example.com)")
CHECKPOINT_DB = os.getenv("LEAD_GRAPH_DB", "session8_leads.sqlite")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment/.env")
if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY is not set in environment/.env")

# Scoring thresholds & limits
AUTO_CUTOFF   = int(os.getenv("AUTO_CUTOFF", "70"))
REJECT_CUTOFF = int(os.getenv("REJECT_CUTOFF", "40"))
MAX_RETRIES   = int(os.getenv("MAX_RETRIES", "1"))
MAX_URLS      = int(os.getenv("MAX_URLS", "6"))
MAX_CHUNKS    = int(os.getenv("MAX_CHUNKS", "12"))

# LLMs
LLM_SCORE = ChatOpenAI(model=os.getenv("LLM_SCORE_MODEL", "gpt-4o-mini"), temperature=0)
LLM_WRITE = ChatOpenAI(model=os.getenv("LLM_WRITE_MODEL", "gpt-4o-mini"), temperature=0)

# Tools
tavily = TavilySearchAPIWrapper()  # uses TAVILY_API_KEY
splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)

# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────
class LeadState(TypedDict, total=False):
    company: str
    role: Optional[str]
    domain: Optional[str]
    website: Optional[str]
    icp: Optional[str]
    notes: Optional[str]

    urls: List[str]
    sources: List[Dict[str, str]]  # {"title":..., "url":...}
    chunks: List[str]

    score: int
    reasons: List[str]
    risks: List[str]
    confidence: float
    decision: Literal["auto_qualify", "research_more", "reject"]
    retry_count: int

    summary: Optional[str]
    error: Optional[str]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_urls(items: List[Dict]) -> List[str]:
    out = []
    for h in items or []:
        u = h.get("url") or h.get("link")
        if u and u not in out:
            out.append(u)
    return out

def _search_urls_for_company(company: str, domain: Optional[str]) -> List[str]:
    urls: List[str] = []
    if domain:
        urls += [f"https://{domain}", f"https://{domain}/about"]
    for q in [
        f"{company} official website",
        f"{company} about page",
        f"{company} platform overview",
        f"{company} customers case studies",
    ]:
        try:
            hits = tavily.results(q, max_results=5) or []
            urls.extend(_normalize_urls(hits))
        except Exception:
            continue
    # dedupe and cap
    seen = set()
    dedup = []
    for u in urls:
        if u not in seen:
            dedup.append(u)
            seen.add(u)
    return dedup[:MAX_URLS]

def _parse_int(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


# ──────────────────────────────────────────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────────────────────────────────────────
def enrich(state: LeadState) -> LeadState:
    """Collect candidate URLs and fetch/chunk pages."""
    company = state["company"]
    domain  = state.get("domain")
    website = state.get("website")

    urls: List[str] = []
    if website:
        urls += [website, website.rstrip("/") + "/about"]
    urls += _search_urls_for_company(company, domain)
    urls = urls[:MAX_URLS]

    chunks: List[str] = []
    sources: List[Dict[str, str]] = []
    for u in urls:
        try:
            docs = WebBaseLoader(u, requests_kwargs={"headers": {"User-Agent": UA}}).load()
            if not docs:
                continue
            text = docs[0].page_content
            title = docs[0].metadata.get("title", "") or company
            sources.append({"title": title, "url": u})
            for d in splitter.create_documents([text]):
                if d.page_content and len(chunks) < MAX_CHUNKS:
                    chunks.append(d.page_content)
        except Exception:
            continue

    return {
        "urls": urls,
        "sources": sources,
        "chunks": chunks,
        "retry_count": state.get("retry_count", 0),
    }

def score(state: LeadState) -> LeadState:
    """LLM scoring against ICP."""
    company = state["company"]
    icp = state.get("icp") or (
        "B2B SaaS/Tech/E-commerce; 100–2000 employees; cloud-first; operates in US/EU/India; "
        "likely to buy data, analytics, or automation tooling."
    )
    chunks = state.get("chunks", [])
    ctx = "\n\n".join([f"[{i+1}] {c[:900]}" for i, c in enumerate(chunks)])
    prompt = f"""You are scoring company fit to an Ideal Customer Profile (ICP).

ICP:
{icp}

Evidence (cite ONLY from the provided context as [n]):
{ctx}

Return ONLY JSON:
{{
  "score": 0-100,
  "reasons": ["..."],
  "risks": ["..."],
  "confidence": 0.0-1.0
}}

Company: {company}
"""
    res = LLM_SCORE.invoke(prompt)
    try:
        payload = json.loads(res.content)
    except Exception:
        # Safe fallback if parsing fails
        text = (" ".join(chunks)[:2000]).lower()
        base = 55
        if any(k in text for k in ["cloud", "saas", "api", "integration"]):
            base += 10
        payload = {"score": base, "reasons": ["Heuristic fallback"], "risks": [], "confidence": 0.55}

    sc = _parse_int(payload.get("score", 0))
    reasons = payload.get("reasons", [])
    risks = payload.get("risks", [])
    conf = float(payload.get("confidence", 0.6))

    if sc >= AUTO_CUTOFF:
        decision: Literal["auto_qualify","research_more","reject"] = "auto_qualify"
    elif sc < REJECT_CUTOFF:
        decision = "reject"
    else:
        decision = "research_more"

    return {"score": sc, "reasons": reasons, "risks": risks, "confidence": conf, "decision": decision}

def research_more(state: LeadState) -> LeadState:
    """Do one more search pass if borderline and retries remain."""
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return {}
    company = state["company"]
    urls = list(state.get("urls", []))
    sources = list(state.get("sources", []))
    chunks = list(state.get("chunks", []))

    extra_qs = [
        f"{company} funding",
        f"{company} revenue",
        f"{company} customers",
        f"{company} integrations",
    ]
    for q in extra_qs:
        try:
            hits = tavily.results(q, max_results=3) or []
            for u in _normalize_urls(hits):
                if u in urls or len(urls) >= MAX_URLS:
                    continue
                urls.append(u)
                try:
                    docs = WebBaseLoader(u, requests_kwargs={"headers": {"User-Agent": UA}}).load()
                    if not docs:
                        continue
                    title = docs[0].metadata.get("title", "") or company
                    sources.append({"title": title, "url": u})
                    for d in splitter.create_documents([docs[0].page_content]):
                        if d.page_content and len(chunks) < MAX_CHUNKS:
                            chunks.append(d.page_content)
                except Exception:
                    continue
        except Exception:
            continue

    return {
        "urls": urls[:MAX_URLS],
        "sources": sources[:MAX_URLS],
        "chunks": chunks[:MAX_CHUNKS],
        "retry_count": state.get("retry_count", 0) + 1,
        "decision": "research_more",
    }

def auto_qualify(state: LeadState) -> LeadState:
    """Compose an SDR-ready handoff summary."""
    company = state["company"]
    score = state.get("score", 0)
    reasons = state.get("reasons", [])
    risks = state.get("risks", [])
    srcs = state.get("sources", [])
    src_text = "\n".join([f"- {s.get('title','')}: {s.get('url','')}" for s in srcs[:5]])
    prompt = f"""Write a crisp SDR handoff summary (120–170 words).
Include: what they do, why they fit the ICP, top 2 hooks, and 1 risk to probe.
Score: {score}/100
Reasons: {reasons}
Risks: {risks}
Sources:
{src_text}

Return plain text."""
    res = LLM_WRITE.invoke(prompt)
    return {"summary": res.content}

def reject(state: LeadState) -> LeadState:
    return {"summary": f"Rejected: score={state.get('score')}, reasons={state.get('reasons')}, risks={state.get('risks')}"}


# ──────────────────────────────────────────────────────────────────────────────
# Routers
# ──────────────────────────────────────────────────────────────────────────────
def route_after_score(state: LeadState) -> Literal["auto_qualify", "research_more", "reject"]:
    return state.get("decision", "reject")

def route_after_research(state: LeadState) -> Literal["score", "reject"]:
    # After research_more, either score again or give up
    if state.get("retry_count", 0) >= MAX_RETRIES and state.get("score", 0) < AUTO_CUTOFF:
        return "reject"
    return "score"


# ──────────────────────────────────────────────────────────────────────────────
# Graph
# ──────────────────────────────────────────────────────────────────────────────
graph = StateGraph(LeadState)
graph.add_node("enrich",        enrich)
graph.add_node("score",         score)
graph.add_node("research_more", research_more)
graph.add_node("auto_qualify",  auto_qualify)
graph.add_node("reject",        reject)

graph.set_entry_point("enrich")
graph.add_edge("enrich", "score")
graph.add_conditional_edges("score", route_after_score, {
    "auto_qualify": "auto_qualify",
    "research_more": "research_more",
    "reject": "reject",
})
graph.add_conditional_edges("research_more", route_after_research, {
    "score": "score",
    "reject": "reject",
})
graph.add_edge("auto_qualify", END)
graph.add_edge("reject", END)


# ──────────────────────────────────────────────────────────────────────────────
# Run (compile + stream INSIDE the SqliteSaver context)
# ──────────────────────────────────────────────────────────────────────────────
def run_one(app, prospect: Dict, thread_id: str):
    start: LeadState = {
        "company": prospect["company"],
        "role": prospect.get("role"),
        "domain": prospect.get("domain"),
        "website": prospect.get("website"),
        "icp": prospect.get("icp"),
        "notes": prospect.get("notes"),
        "retry_count": 0,
    }
    cfg = {"configurable": {"thread_id": thread_id}}

    print(f"\n=== RUN {thread_id} :: {prospect['company']} ===")
    for ev in app.stream(start, config=cfg):
        for node in ev.keys():
            print(f"[{node}] ✓")

    final = app.get_state(cfg).values
    print("\n--- RESULT ---")
    print("Decision   :", final.get("decision"))
    print("Score      :", final.get("score"))
    print("Reasons    :", final.get("reasons"))
    print("Risks      :", final.get("risks"))
    print("Retries    :", final.get("retry_count"))
    print("Summary    :\n", (final.get("summary") or "")[:600])
    print("Sources    :")
    for s in final.get("sources", [])[:5]:
        print("  -", s.get("title",""), "=>", s.get("url",""))

if __name__ == "__main__":
    # Load prospects list from LEADS_JSON (if provided) or use defaults
    leads_path = os.getenv("LEADS_JSON")
    if leads_path and os.path.exists(leads_path):
        with open(leads_path, "r", encoding="utf-8") as f:
            prospects = json.load(f)
    else:
        prospects = [
            {"company": "Freshworks", "domain": "freshworks.com", "role": "Head of Sales"},
            {"company": "Zoho",       "domain": "zoho.com",       "role": "CIO"},
            {"company": "Shiprocket", "domain": "shiprocket.in",  "role": "VP Growth"},
        ]

    # Compile + run inside the checkpointer context (prevents the error you saw)
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        app = graph.compile(checkpointer=checkpointer)
        for i, p in enumerate(prospects, start=1):
            run_one(app, p, thread_id=f"lead-{i}")
