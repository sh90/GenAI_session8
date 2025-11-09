# demo3_expense_approval_flow.py
from __future__ import annotations
import os, csv, json, time, math
from typing import TypedDict, List, Dict, Optional, Literal
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import requests

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# ────────── ENV / CONFIG ──────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

UA = os.getenv("USER_AGENT", "GenAI-Session8/1.0 (+contact: you@example.com)")
CSV_PATH = os.getenv("EXPENSES_CSV", "expenses.csv")   # date,amount,currency,category,desc
CHECKPOINT_DB = os.getenv("EXP_APPROVAL_DB", "session8_expenses.sqlite")
DECISION_FILE = os.getenv("MANAGER_DECISIONS_JSON", "manager_decisions.json")  # {"thread-id-1":"approved"}

LLM_SUMMARY = ChatOpenAI(model=os.getenv("LLM_SUMMARY_MODEL","gpt-4o-mini"), temperature=0)

USD = "USD"

# ────────── STATE ──────────
class ExpState(TypedDict, total=False):
    rows_raw: List[Dict]
    rows_usd: List[Dict]
    flagged: List[Dict]
    decision: Literal["approve", "manager", "approved", "rejected"]
    awaiting: Optional[str]
    manager_notes_md: Optional[str]
    error: Optional[str]

# ────────── Helpers ──────────
def _retry_request(url: str, headers: Dict[str,str], tries=3, backoff=1.5, timeout=12) -> requests.Response:
    err = None
    for i in range(tries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            err = e
            time.sleep(backoff**(i+1))
    raise RuntimeError(f"Request failed after retries: {url} :: {err}")

def frankfurter_rate(date_iso: str, base: str, to: str = "USD") -> float:
    """
    Reliable FX from Frankfurter: https://www.frankfurter.app/docs/
    GET /YYYY-MM-DD?from=EUR&to=USD  => {"amount":1,"base":"EUR","date":"2020-04-04","rates":{"USD":1.08}}
    """
    if base.upper() == to.upper():
        return 1.0
    url = f"https://api.frankfurter.app/{date_iso}?from={base.upper()}&to={to.upper()}"
    r = _retry_request(url, headers={"User-Agent": UA})
    data = r.json()
    rate = float(data["rates"][to.upper()])
    return rate

def _load_decision(thread_id: str) -> Optional[str]:
    if not os.path.exists(DECISION_FILE):
        return None
    try:
        with open(DECISION_FILE, "r", encoding="utf-8") as f:
            m = json.load(f)
        d = m.get(thread_id)
        if d in ("approved","rejected"):
            return d
    except Exception:
        return None
    return None

def _llm_manager_notes(flagged: List[Dict]) -> str:
    if not flagged:
        return ""
    prompt = (
        "You are a finance controller. Summarize the following flagged expenses in ~120 words. "
        "Highlight reasons they were flagged and propose a recommendation (approve/reject/clarify). "
        "Return a crisp Markdown note.\n\n"
        + json.dumps(flagged, ensure_ascii=False)
    )
    res = LLM_SUMMARY.invoke(prompt)
    return res.content

# ────────── Nodes ──────────
def parse_csv(_: ExpState) -> ExpState:
    rows: List[Dict] = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            rows.append(rec)
    return {"rows_raw": rows}

def fx_convert(state: ExpState) -> ExpState:
    rows = state.get("rows_raw", [])
    out: List[Dict] = []
    for r in rows:
        date_iso = datetime.fromisoformat(r["date"]).date().isoformat()
        amt = float(r["amount"])
        cur = r["currency"].upper()
        rate = frankfurter_rate(date_iso, cur, USD)
        usd = round(amt * rate, 2)
        r2 = dict(r)
        r2["date"] = date_iso
        r2["currency"] = cur
        r2["usd"] = usd
        r2["fx_rate_to_usd"] = rate
        out.append(r2)
    return {"rows_usd": out}

def auto_rules(state: ExpState) -> ExpState:
    rows = state.get("rows_usd", [])
    flagged: List[Dict] = []
    for r in rows:
        flag = "ok"
        usd = float(r["usd"])
        cat = (r.get("category") or "").lower()
        if usd > 5000:
            flag = "manager_review"
        elif usd > 1000 and cat in {"meals","misc"}:
            flag = "manager_review"
        r2 = dict(r); r2["flag"] = flag
        flagged.append(r2)
    return {"flagged": flagged}

def route(state: ExpState) -> ExpState:
    flagged = state.get("flagged", [])
    needs_manager = any(x.get("flag") == "manager_review" for x in flagged)
    return {"decision": "manager" if needs_manager else "approve"}

def manager_gate(state: ExpState, *, thread_id: str) -> ExpState:
    """
    If manager decision exists in DECISION_FILE for this thread, use it.
    Otherwise, set awaiting marker + provide manager_notes_md and stop via router.
    """
    current = _load_decision(thread_id)
    if current in ("approved","rejected"):
        return {"decision": current, "awaiting": None}

    # build summary note once per pause
    notes = state.get("manager_notes_md") or _llm_manager_notes(
        [x for x in state.get("flagged", []) if x.get("flag")=="manager_review"][:8]
    )
    return {"awaiting": "manager", "manager_notes_md": notes}

def approve(_: ExpState) -> ExpState:
    return {"decision": "approved"}

def finalize(state: ExpState, *, thread_id: str) -> ExpState:
    out = {
        "thread_id": thread_id,
        "decision": state.get("decision"),
        "awaiting": state.get("awaiting"),
        "manager_notes_md": state.get("manager_notes_md"),
        "flagged": state.get("flagged", []),
    }
    fname = f"expense_review_{thread_id}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved: {fname}")
    if out.get("awaiting") == "manager":
        print(f"\nPAUSED: add a decision for '{thread_id}' in {DECISION_FILE} as 'approved' or 'rejected' and re-run.")
    return {}

# ────────── Routers ──────────
def route_after_route(state: ExpState) -> Literal["approve","manager_gate"]:
    return "manager_gate" if state.get("decision") == "manager" else "approve"

def route_after_manager(state: ExpState) -> Literal["approve","finalize","await_external"]:
    d = state.get("decision")
    if d in ("approved","rejected"):
        return "approve" if d == "approved" else "finalize"
    # no decision yet → pause
    return "await_external"

# ────────── Graph ──────────
graph = StateGraph(ExpState)
graph.add_node("parse_csv", parse_csv)
graph.add_node("fx_convert", fx_convert)
graph.add_node("auto_rules", auto_rules)
graph.add_node("route",     route)
graph.add_node("manager_gate", lambda s, config: manager_gate(s, thread_id=config['configurable']['thread_id']))
graph.add_node("approve",   approve)
graph.add_node("finalize",  lambda s, config: finalize(s, thread_id=config['configurable']['thread_id']))
graph.add_node("await_external", lambda s: s)  # sink that ends and waits

graph.set_entry_point("parse_csv")
graph.add_edge("parse_csv", "fx_convert")
graph.add_edge("fx_convert","auto_rules")
graph.add_edge("auto_rules","route")
graph.add_conditional_edges("route", route_after_route, {
    "approve": "approve",
    "manager_gate": "manager_gate",
})
graph.add_conditional_edges("manager_gate", route_after_manager, {
    "approve": "approve",
    "finalize": "finalize",
    "await_external": "await_external",
})
graph.add_edge("approve",  "finalize")
graph.add_edge("await_external", END)
graph.add_edge("finalize", END)

# ────────── Run ──────────
def run(thread_id: str):
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as ckpt:
        app = graph.compile(checkpointer=ckpt)
        cfg = {"configurable": {"thread_id": thread_id}}
        for ev in app.stream({}, config=cfg):
            for k in ev.keys():
                print(f"[{k}] ✓")
        print("\nDONE (or PAUSED).")

if __name__ == "__main__":
    run(thread_id=os.getenv("THREAD_ID","exp-run-1"))
