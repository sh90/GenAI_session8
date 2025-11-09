# demo5_news_monitor_alerts.py
from __future__ import annotations
import os, json, hashlib
from typing import TypedDict, List, Dict, Literal, Optional

from dotenv import load_dotenv
load_dotenv()

import feedparser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

UA = os.getenv("USER_AGENT", "GenAI-Session8/1.0 (+contact: you@example.com)")
CHECKPOINT_DB = os.getenv("NEWS_DB", "session8_news.sqlite")
ALERTS_JSONL = os.getenv("ALERTS_JSONL", "news_alerts.jsonl")

FEEDS = json.loads(os.getenv("FEEDS_JSON", '["https://feeds.bbci.co.uk/news/rss.xml"]'))
WATCHLIST = set(json.loads(os.getenv("WATCHLIST_JSON",'["ai","pricing","ecommerce","inflation","cybersecurity"]')))
MAX_ITEMS = int(os.getenv("MAX_ITEMS","20"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS","10"))

LLM_TOPIC = ChatOpenAI(model=os.getenv("LLM_TOPIC_MODEL","gpt-4o-mini"), temperature=0)
splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)

class NewsState(TypedDict, total=False):
    items: List[Dict]
    dedup: List[Dict]
    chunks: List[Dict]  # [{"url":..., "text":...}]
    topics: List[Dict]  # [{"url":..., "title":"...", "topic":"...", "reason":"..."}]
    alerts: List[Dict]
    error: Optional[str]

def _hash_title(t: str) -> str:
    return hashlib.sha1((t or "").strip().lower().encode()).hexdigest()

def fetch_rss(_: NewsState) -> NewsState:
    all_items: List[Dict] = []
    for url in FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:MAX_ITEMS]:
                all_items.append({
                    "title": e.get("title","").strip(),
                    "link":  e.get("link") or e.get("id",""),
                    "published": e.get("published",""),
                })
        except Exception:
            continue
    # dedup by title hash
    seen, de = set(), []
    for it in all_items:
        h = _hash_title(it["title"])
        if h in seen:
            continue
        seen.add(h); de.append(it)
    return {"items": all_items, "dedup": de[:MAX_ITEMS]}

def extract_fulltext(state: NewsState) -> NewsState:
    dedup = state.get("dedup", [])
    out: List[Dict] = []
    for it in dedup:
        url = it.get("link","")
        if not url:
            continue
        try:
            docs = WebBaseLoader(url, requests_kwargs={"headers": {"User-Agent": UA}}).load()
            text = docs[0].page_content if docs else ""
            if not text:
                continue
            for d in splitter.create_documents([text]):
                out.append({"url": url, "text": d.page_content[:1500], "title": it.get("title",""), "published": it.get("published","")})
                if len(out) >= MAX_CHUNKS:
                    break
        except Exception:
            continue
        if len(out) >= MAX_CHUNKS:
            break
    return {"chunks": out}

def classify(state: NewsState) -> NewsState:
    ch = state.get("chunks", [])
    topics: List[Dict] = []
    for c in ch:
        prompt = f"""Classify the main topic of this article into one of:
["ai","pricing","ecommerce","cybersecurity","finance","politics","health","sports","world","other"].
Return JSON: {{"topic":"...","reason":"..."}}
Title: {c.get('title')}
Text: {c.get('text')}"""
        res = LLM_TOPIC.invoke(prompt)
        parsed = {"topic":"other","reason":""}
        try: parsed = json.loads(res.content)
        except Exception: pass
        topics.append({"url": c["url"], "title": c.get("title",""), "topic": parsed.get("topic","other"),
                       "reason": parsed.get("reason",""), "published": c.get("published","")})
    return {"topics": topics}

def route(_: NewsState) -> NewsState:
    return {}

def alert_or_skip(state: NewsState) -> NewsState:
    topics = state.get("topics", [])
    alerts: List[Dict] = []
    for t in topics:
        if t.get("topic","").lower() in WATCHLIST:
            alerts.append(t)
    # write alerts
    if alerts:
        with open(ALERTS_JSONL, "a", encoding="utf-8") as f:
            for a in alerts:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
        print(f"Appended {len(alerts)} alerts → {ALERTS_JSONL}")
    return {"alerts": alerts}

def finalize(state: NewsState) -> NewsState:
    out = {
        "processed": len(state.get("dedup", [])),
        "classified": len(state.get("topics", [])),
        "alerts": len(state.get("alerts", [])),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return {}

def route_after_classify(state: NewsState) -> Literal["alert","skip"]:
    any_watch = any((t.get("topic","").lower() in WATCHLIST) for t in state.get("topics", []))
    return "alert" if any_watch else "skip"

graph = StateGraph(NewsState)
graph.add_node("fetch_rss",        fetch_rss)
graph.add_node("extract_fulltext", extract_fulltext)
graph.add_node("classify",         classify)
graph.add_node("alert",            alert_or_skip)
graph.add_node("skip",             lambda s: s)
graph.add_node("finalize",         finalize)

graph.set_entry_point("fetch_rss")
graph.add_edge("fetch_rss", "extract_fulltext")
graph.add_edge("extract_fulltext", "classify")
graph.add_conditional_edges("classify", route_after_classify, {
    "alert": "alert",
    "skip":  "skip",
})
graph.add_edge("alert", "finalize")
graph.add_edge("skip",  "finalize")
graph.add_edge("finalize", END)

def run(thread_id: str):
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as ckpt:
        app = graph.compile(checkpointer=ckpt)
        cfg = {"configurable": {"thread_id": thread_id}}
        for ev in app.stream({}, config=cfg):
            for k in ev.keys(): print(f"[{k}] ✓")
        print("DONE.")

if __name__ == "__main__":
    run(thread_id=os.getenv("NEWS_THREAD_ID","news-run-1"))
