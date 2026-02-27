from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta
from typing import Any, AsyncIterator, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from google import genai
from google.genai import types as gtypes
from mcp import ClientSession
from mcp.client.sse import sse_client
from pathlib import Path
from fastapi.responses import FileResponse

try:
    from mcp.client.streamable_http import streamable_http_client
except ImportError:
    try:
        from mcp.client.streamable_http import streamablehttp_client as streamable_http_client
    except ImportError:
        streamable_http_client = None  # type: ignore


def _is_sse_url(url: str) -> bool:
    u = (url or "").lower().strip()
    if "/mcp/message" in u:
        return False
    if any(seg in u for seg in ("/sse", "/events", "/event-stream")):
        return True
    if u.rstrip("/").endswith(("/mcp", "/rpc", "/jsonrpc")):
        return False
    return False

from zoho_agent import (
    AgentState,
    AuditLog,
    AUDIT_LOG_PATH,
    GEMINI_TIMEOUT,
    MAX_AUTOPAGINATE_PAGES,
    MAX_REPLAN_ATTEMPTS,
    MODEL,
    PLANNER_MODEL,
    TOOL_TIMEOUT,
    _estimate_tokens,
    _extract_records,
    _has_more_pages,
    _parse_mcp_result,
    _safe_result_str,
    _stream_collect_json,
    add_memory,
    build_tool_catalog,
    execute_plan,
    gemini_plan,
    is_number_string,
    is_risky,
    log,
    normalize_plan,
    validate_env,
    _build_summarize_system_with_date,
)

load_dotenv("api.env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SESSION_TTL_MINUTES: int = int(os.getenv("SESSION_TTL_MINUTES", "60"))
MAX_SESSIONS: int        = int(os.getenv("MAX_SESSIONS", "500"))
CORS_ORIGINS: list[str]  = os.getenv("CORS_ORIGINS", "*").split(",")

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message:    str           = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = Field(None, description="Omit to start a new session")
    mcp_url:    Optional[str] = Field(None, description="MCP server URL — required when starting a new session")
    org_id:     Optional[str] = Field(None, description="Zoho Organisation ID — required when starting a new session")

class ConfirmRequest(BaseModel):
    session_id: str
    answer:     bool  = Field(..., description="true = YES, false = NO")
    mcp_url:    Optional[str] = Field(None, description="MCP server URL (used only if session needs re-init)")

class ChatResponse(BaseModel):
    session_id:   str
    type:         str
    reply:        Optional[str] = None
    structured:   Optional[dict] = None
    confirm_text: Optional[str] = None
    tools_used:   list[str]     = []
    elapsed_ms:   int           = 0

class SessionInfo(BaseModel):
    session_id:      str
    state:           dict
    memory_entries:  int
    memory_tokens:   int
    last_active:     str
    pending_confirm: bool
    mcp_url:         str

# ---------------------------------------------------------------------------
# Session — owns its own MCP connection + toolbox
# ---------------------------------------------------------------------------
@dataclass
class AgentSession:
    session_id:      str
    state:           AgentState
    mcp_url:         str
    mcp_session:     ClientSession
    toolbox:         dict
    memory:          list[dict]     = field(default_factory=list)
    pending_confirm: Optional[dict] = None
    replan_attempts: int            = 0
    last_active:     float          = field(default_factory=time.monotonic)
    _http_client:    Any            = field(default=None, repr=False)
    _cm_http:        Any            = field(default=None, repr=False)
    _cm_stream:      Any            = field(default=None, repr=False)
    _cm_mcp:         Any            = field(default=None, repr=False)

    def touch(self) -> None:
        self.last_active = time.monotonic()

    def is_expired(self) -> bool:
        return (time.monotonic() - self.last_active) > SESSION_TTL_MINUTES * 60

    async def close(self) -> None:
        for cm in (self._cm_mcp, self._cm_stream, self._cm_http):
            try:
                if cm is not None:
                    await cm.__aexit__(None, None, None)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------
async def create_agent_session(
    session_id: str,
    mcp_url: str,
    gemini: genai.Client,
    org_id: Optional[str] = None,
) -> AgentSession:
    zoho_bearer = os.getenv("ZOHO_MCP_BEARER")
    headers     = {"Authorization": f"Bearer {zoho_bearer}"} if zoho_bearer else {}

    use_sse = _is_sse_url(mcp_url)
    log.info("mcp_transport_selected", extra={
        "sid": session_id, "transport": "sse" if use_sse else "streamable_http", "url": mcp_url
    })

    if use_sse:
        stream_cm = sse_client(mcp_url, headers=headers)
        read, write = await stream_cm.__aenter__()
        mcp_cm = ClientSession(read, write)
        mcp_session: ClientSession = await mcp_cm.__aenter__()
        await mcp_session.initialize()

        toolbox = build_tool_catalog(await mcp_session.list_tools())
        log.info("session_mcp_connected", extra={"sid": session_id, "tools": len(toolbox), "url": mcp_url, "transport": "sse"})

        return AgentSession(
            session_id   = session_id,
            state        = AgentState(organization_id=org_id or None),
            mcp_url      = mcp_url,
            mcp_session  = mcp_session,
            toolbox      = toolbox,
            _http_client = None,
            _cm_http     = None,
            _cm_stream   = stream_cm,
            _cm_mcp      = mcp_cm,
        )
    else:
        if streamable_http_client is None:
            raise RuntimeError("streamable_http transport not available in this mcp version")

        http_client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(TOOL_TIMEOUT + 10),
        )
        await http_client.__aenter__()

        stream_cm = streamable_http_client(mcp_url, http_client=http_client)
        read, write, _ = await stream_cm.__aenter__()
        mcp_cm = ClientSession(read, write)
        mcp_session = await mcp_cm.__aenter__()
        await mcp_session.initialize()

        toolbox = build_tool_catalog(await mcp_session.list_tools())
        log.info("session_mcp_connected", extra={"sid": session_id, "tools": len(toolbox), "url": mcp_url, "transport": "streamable_http"})

        return AgentSession(
            session_id   = session_id,
            state        = AgentState(organization_id=org_id or None),
            mcp_url      = mcp_url,
            mcp_session  = mcp_session,
            toolbox      = toolbox,
            _http_client = http_client,
            _cm_http     = http_client,
            _cm_stream   = stream_cm,
            _cm_mcp      = mcp_cm,
        )


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------
class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, AgentSession] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        session_id: Optional[str],
        mcp_url: Optional[str],
        gemini: genai.Client,
        org_id: Optional[str] = None,
    ) -> AgentSession:
        async with self._lock:
            self._evict_expired_sync()

            if session_id and session_id in self._sessions:
                s = self._sessions[session_id]
                s.touch()
                return s

            if not mcp_url:
                raise HTTPException(400, "mcp_url is required to start a new session.")
            if not org_id:
                raise HTTPException(400, "org_id is required to start a new session.")

            if len(self._sessions) >= MAX_SESSIONS:
                raise HTTPException(503, "Too many active sessions — try again later.")

            sid = session_id or uuid.uuid4().hex
            try:
                sess = await create_agent_session(sid, mcp_url, gemini, org_id=org_id)
            except Exception as exc:
                import traceback as _tb
                tb = _tb.format_exc()
                log.error("session_create_failed", extra={
                    "sid": sid, "url": mcp_url,
                    "error": str(exc), "error_type": type(exc).__name__,
                    "traceback": tb,
                })
                msg = str(exc)
                if "connection" in msg.lower() or "connect" in msg.lower():
                    detail = f"Could not connect to MCP server at {mcp_url} — is the server running and reachable?"
                elif "401" in msg or "403" in msg or "unauthorized" in msg.lower() or "forbidden" in msg.lower():
                    detail = f"MCP server rejected the connection (auth error). Check your bearer token or API key."
                elif "404" in msg:
                    detail = f"MCP server URL not found (404). Double-check the URL: {mcp_url}"
                elif "timeout" in msg.lower():
                    detail = f"MCP server connection timed out. The server may be overloaded or the URL is wrong."
                else:
                    detail = f"MCP connection failed ({type(exc).__name__}): {msg}"
                raise HTTPException(502, detail=detail)
            self._sessions[sid] = sess
            log.info("session_created", extra={"sid": sid, "org_id": org_id})
            return sess

    async def get(self, session_id: str) -> Optional[AgentSession]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def delete(self, session_id: str) -> bool:
        async with self._lock:
            sess = self._sessions.pop(session_id, None)
            if sess:
                await sess.close()
                return True
            return False

    def _evict_expired_sync(self) -> None:
        dead = [k for k, s in self._sessions.items() if s.is_expired()]
        for k in dead:
            sess = self._sessions.pop(k)
            asyncio.create_task(sess.close())
            log.info("session_evicted", extra={"sid": k})

    async def close_all(self) -> None:
        async with self._lock:
            for sess in list(self._sessions.values()):
                await sess.close()
            self._sessions.clear()


# ---------------------------------------------------------------------------
# App-level singletons
# ---------------------------------------------------------------------------
@dataclass
class _App:
    gemini:   genai.Client
    audit:    AuditLog
    sessions: SessionStore


_state: Optional[_App] = None


def _app() -> _App:
    if _state is None:
        raise RuntimeError("Server not initialized")
    return _state


# ---------------------------------------------------------------------------
# Core summarizer — used by /chat endpoint
# ---------------------------------------------------------------------------
async def api_summarize(
    gemini: genai.Client,
    tool: str,
    args: dict,
    result: Any,
    cid: str = "",
    user_question: str = "",
) -> dict:
    is_action = any(w in tool.lower() for w in
                    ("create","update","delete","void","send","submit","mark"))
    _, all_records = _extract_records(result) if isinstance(result, dict) else ("", [])
    filtered = _python_filter_records(all_records, user_question)
    query_class = _classify_query(user_question)

    if is_action:
        return await _gemini_summarize_single(gemini, tool, user_question, result, cid)
    elif query_class == "python_math":
        return _python_aggregate(user_question, tool, [result])
    elif query_class == "gemini_analysis" or len(filtered) == 1:
        target = filtered[0] if len(filtered) == 1 else {"records": filtered[:500], "total_records": len(filtered)}
        return await _gemini_summarize_single(gemini, tool, user_question, target, cid)
    elif len(filtered) == 0 and all_records:
        return _python_no_records(user_question, tool)
    elif len(filtered) > 1:
        s = _build_table_structured(result, tool, records_override=filtered)
        s.pop("col_keys", None)
        return s
    else:
        return await _gemini_summarize_single(gemini, tool, user_question, result, cid)


# ── Per-entity preferred column lists ────────────────────────────────────────
_ENTITY_COLS: dict[str, list[str]] = {
    "invoice":       ["invoice_number","customer_name","date","due_date","total","balance","status","due_days"],
    "bill":          ["bill_number","vendor_name","date","due_date","total","balance","status","due_days"],
    "contact":       ["contact_name","company_name","email","phone","contact_type","outstanding_receivable_amount","outstanding_payable_amount","status"],
    "item":          ["name","sku","rate","purchase_rate","stock_on_hand","unit","item_type","status"],
    "expense":       ["date","account_name","vendor_name","total","is_billable","status","reference_number"],
    "payment":       ["date","customer_name","payment_number","amount","payment_mode","reference_number","description"],
    "creditnote":    ["creditnote_number","customer_name","date","total","balance","status"],
    "estimate":      ["estimate_number","customer_name","date","expiry_date","total","status"],
    "salesorder":    ["salesorder_number","customer_name","date","shipment_date","total","status"],
    "purchaseorder": ["purchaseorder_number","vendor_name","date","delivery_date","total","status"],
    "transaction":   ["date","transaction_type","description","debit_amount","credit_amount","running_balance"],
    "journal":       ["journal_date","journal_number","reference_number","notes","total","status"],
    "account":       ["account_name","account_type","account_code","current_balance","description"],
    "tax":           ["tax_name","tax_percentage","tax_type","is_default_tax","status"],
}

_SKIP_FIELDS: frozenset[str] = frozenset({
    "organization_id","created_time","last_modified_time","created_by","last_modified_by",
    "template_id","template_name","color_code","zcrm_potential_id","zcrm_potential_name",
    "is_viewed_by_client","client_viewed_time","ach_payment_initiated","tax_rounding",
    "is_emailed","is_inclusive_tax","salesorder_id","purchaseorder_id","estimate_id",
    "page_context","__paginate__",
})

_AMOUNT_FIELDS: tuple[str, ...] = (
    "balance","total","amount","balance_due","amount_applied",
    "outstanding_receivable_amount","outstanding_payable_amount",
    "debit_amount","credit_amount",
)


def _detect_cols(records: list, tool_name: str) -> list[str]:
    tl = tool_name.lower()
    preferred: list[str] = []
    for cat, cols in _ENTITY_COLS.items():
        if cat in tl:
            preferred = cols
            break
    col_freq: dict[str, int] = {}
    for rec in records[:50]:
        if isinstance(rec, dict):
            for k in rec:
                if k not in _SKIP_FIELDS:
                    col_freq[k] = col_freq.get(k, 0) + 1
    result_cols = [c for c in preferred if col_freq.get(c, 0) > 0]
    if len(result_cols) < 4:
        extras = [c for c, _ in sorted(col_freq.items(), key=lambda x: -x[1])
                  if c not in result_cols]
        result_cols = (result_cols + extras)[:8]
    return result_cols[:8]


def _compute_totals(records: list) -> dict[str, float]:
    totals: dict[str, float] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        for f in _AMOUNT_FIELDS:
            if f in rec:
                try:
                    totals[f] = totals.get(f, 0.0) + float(rec[f])
                except (TypeError, ValueError):
                    pass
    return totals


def _records_to_rows(records: list, tool_name: str = "") -> dict:
    cols = _detect_cols(records, tool_name)
    rows = [[str(rec.get(c, "")) for c in cols] for rec in records if isinstance(rec, dict)]
    return {"col_keys": cols, "columns": [c.replace("_", " ").title() for c in cols], "rows": rows}


def _build_table_structured(result: Any, tool_name: str,
                             more_pages: bool = False,
                             records_override: Optional[list] = None) -> dict:
    if records_override is not None:
        records = records_override
    else:
        _, records = _extract_records(result) if isinstance(result, dict) else ("", [])

    if not records:
        return {"format": "status", "ok": False, "headline": "No records found.",
                "detail": f"Tool: {tool_name}"}

    count    = len(records)
    totals   = _compute_totals(records)
    col_info = _records_to_rows(records, tool_name)
    title    = tool_name.replace("ZohoBooks_", "").replace("_", " ").title()
    primary  = next((f for f in _AMOUNT_FIELDS if f in totals), None)
    footer   = f"Showing {count} records"
    if primary:
        footer += f"  ·  Total {primary.replace('_', ' ').title()}: ₹{totals[primary]:,.2f}"
    if more_pages:
        footer += "  ·  Loading more…"
    structured: dict = {
        "format":   "table",
        "title":    title,
        "columns":  col_info["columns"],
        "col_keys": col_info["col_keys"],
        "rows":     col_info["rows"],
        "footer":   footer,
    }
    if more_pages:
        structured["__more_pages__"] = True
    return structured


def structured_to_markdown(data: dict) -> str:
    fmt = data.get("format", "status")

    if fmt == "answer":
        question  = data.get("question", "")
        answer    = data.get("answer", "Done.")
        breakdown = data.get("breakdown") or []
        note      = data.get("note", "")
        lines = []
        if question:
            lines.append(f"_{question}_\n")
        lines.append(f"**{answer}**")
        if breakdown:
            lines.append("")
            for label, value in breakdown:
                lines.append(f"- **{label}:** {value}")
        if note:
            lines.append(f"\n_{note}_")
        return "\n".join(lines)

    if fmt == "table":
        title  = data.get("title", "")
        cols   = data.get("columns") or []
        rows   = data.get("rows") or []
        footer = data.get("footer", "")
        lines: list[str] = []
        if title:
            lines.append(f"### {title}\n")
        if cols:
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        if footer:
            lines.append(f"\n_{footer}_")
        return "\n".join(lines)

    if fmt == "panel":
        title  = data.get("title", "Details")
        fields = data.get("fields") or []
        note   = data.get("note", "")
        lines  = [f"### {title}\n"]
        for label, value in fields:
            lines.append(f"**{label}:** {value}")
        if note:
            lines.append(f"\n_{note}_")
        return "\n".join(lines)

    ok       = data.get("ok", True)
    headline = data.get("headline", "Done.")
    detail   = data.get("detail", "")
    icon     = "✅" if ok else "⚠️"
    out      = f"{icon} {headline}"
    if detail:
        out += f"\n\n{detail}"
    return out


# ---------------------------------------------------------------------------
# Core turn logic (/chat non-streaming)
# ---------------------------------------------------------------------------
async def _run_plan(sess: AgentSession, plan: dict, cid: str) -> dict:
    a     = _app()
    ptype = plan.get("type")

    if ptype == "ask":
        sess.state.update_from_dict(plan.get("save") or {})
        text = plan.get("text", "I need more details.")
        return {"type": "reply", "reply": text,
                "structured": {"format": "status", "ok": True, "headline": text},
                "tools_used": []}

    if ptype == "confirm":
        sess.pending_confirm = {"on_yes": plan.get("on_yes"), "on_no": plan.get("on_no")}
        return {"type": "confirm", "confirm_text": plan.get("text", "Confirm?"),
                "reply": None, "structured": None, "tools_used": []}

    if ptype != "plan":
        return {"type": "error",
                "reply": "Could not determine an action. Please rephrase.",
                "structured": {"format": "status", "ok": False, "headline": "Unknown plan type"},
                "tools_used": []}

    steps: list[dict] = plan.get("steps") or []
    if not steps:
        return {"type": "error", "reply": "Plan had no steps. Please rephrase.",
                "structured": {"format": "status", "ok": False, "headline": "Empty plan"},
                "tools_used": []}

    risky = [s for s in steps if is_risky(s.get("tool", ""))]
    if risky:
        names = ", ".join(s["tool"] for s in risky)
        sess.pending_confirm = {
            "on_yes": plan,
            "on_no":  {"type": "ask", "text": "Cancelled. What would you like instead?"},
        }
        a.audit.write("risky_confirm_requested", tools=names, cid=cid)
        return {"type": "confirm",
                "confirm_text": f"This involves sensitive operations: {names}. Confirm to proceed?",
                "reply": None, "structured": None, "tools_used": []}

    ok, last_result, all_step_results = await execute_plan(
        sess.mcp_session, sess.toolbox, sess.state,
        sess.memory, a.audit, a.gemini, steps, cid,
    )
    tools_used = [s.get("tool", "") for s in steps]

    if ok and last_result is not None:
        last_tool  = steps[-1].get("tool", "")
        user_q     = steps[-1].get("note", "")
        structured = await api_summarize(a.gemini, last_tool, steps[-1].get("args", {}), last_result, cid,
                                              user_question=user_q)
        sess.state.clear_workflow()
        sess.replan_attempts = 0
        a.audit.write("turn_success", tool=last_tool, cid=cid)
        return {"type": "reply", "reply": structured_to_markdown(structured),
                "structured": structured, "tools_used": tools_used}

    if sess.replan_attempts >= MAX_REPLAN_ATTEMPTS:
        sess.replan_attempts = 0
        return {"type": "error",
                "reply": "I've tried multiple times but couldn't complete this. Please rephrase.",
                "structured": {"format": "status", "ok": False, "headline": "Max replan attempts reached"},
                "tools_used": tools_used}

    sess.replan_attempts += 1
    replan_raw = await gemini_plan(
        a.gemini, sess.toolbox, sess.state, sess.memory,
        "Fix the plan using MEMORY errors. Ask only for genuinely missing required fields.", cid,
    )
    return await _run_plan(sess, normalize_plan(replan_raw), cid)


async def process_turn(sess: AgentSession, message: str, cid: str) -> dict:
    a = _app()

    if is_number_string(message) and not sess.state.organization_id:
        sess.state.organization_id = message
        a.audit.write("org_id_set", org_id=message, cid=cid)
        return {"type": "reply",
                "reply": f"Organization ID saved: `{message}`.",
                "structured": {"format": "status", "ok": True, "headline": f"Org ID saved: {message}"},
                "tools_used": []}

    raw  = await gemini_plan(a.gemini, sess.toolbox, sess.state, sess.memory, message, cid)
    plan = normalize_plan(raw)
    sess.replan_attempts = 0
    return await _run_plan(sess, plan, cid)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _state
    validate_env()

    gemini   = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    audit    = AuditLog(AUDIT_LOG_PATH)
    sessions = SessionStore()

    log.info("server_ready", extra={"model": MODEL})
    _state = _App(gemini=gemini, audit=audit, sessions=sessions)

    async def _reap() -> None:
        while True:
            await asyncio.sleep(300)
            async with sessions._lock:
                sessions._evict_expired_sync()

    reaper = asyncio.create_task(_reap())
    yield
    reaper.cancel()
    await sessions.close_all()
    audit.close()
    _state = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Zoho Books Agent API",
    version="3.0.0",
    description="AI-powered Zoho Books assistant (Gemini + MCP, per-session MCP URL + Org ID)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    a = _app()
    return {
        "status":          "ok",
        "model":           MODEL,
        "active_sessions": len(a.sessions._sessions),
        "ts":              datetime.now(timezone.utc).isoformat(),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    t0   = time.monotonic()
    a    = _app()
    sess = await a.sessions.get_or_create(
        req.session_id, req.mcp_url, a.gemini, org_id=req.org_id
    )
    cid  = uuid.uuid4().hex[:10]
    a.audit.write("user_message", msg=req.message, cid=cid, sid=sess.session_id)

    result = await process_turn(sess, req.message, cid)
    sess.touch()

    return ChatResponse(
        session_id   = sess.session_id,
        type         = result["type"],
        reply        = result.get("reply"),
        structured   = result.get("structured"),
        confirm_text = result.get("confirm_text"),
        tools_used   = result.get("tools_used", []),
        elapsed_ms   = int((time.monotonic() - t0) * 1000),
    )


@app.post("/confirm", response_model=ChatResponse)
async def confirm(req: ConfirmRequest) -> ChatResponse:
    t0   = time.monotonic()
    a    = _app()
    cid  = uuid.uuid4().hex[:10]
    sess = await a.sessions.get(req.session_id)

    if not sess:
        raise HTTPException(404, f"Session '{req.session_id}' not found or expired.")
    if not sess.pending_confirm:
        raise HTTPException(400, "No pending confirmation for this session.")

    if not req.answer:
        no_obj = sess.pending_confirm.get("on_no") or {}
        sess.pending_confirm = None
        a.audit.write("confirm_rejected", cid=cid, sid=sess.session_id)
        text = no_obj.get("text", "Cancelled.")
        return ChatResponse(
            session_id=sess.session_id, type="reply", reply=text,
            structured={"format": "status", "ok": True, "headline": text},
            tools_used=[], elapsed_ms=int((time.monotonic() - t0) * 1000),
        )

    plan = sess.pending_confirm["on_yes"]
    sess.pending_confirm = None
    a.audit.write("confirm_accepted", cid=cid, sid=sess.session_id)

    result = await _run_plan(sess, normalize_plan(plan), cid)
    sess.touch()

    return ChatResponse(
        session_id   = sess.session_id,
        type         = result["type"],
        reply        = result.get("reply"),
        structured   = result.get("structured"),
        confirm_text = result.get("confirm_text"),
        tools_used   = result.get("tools_used", []),
        elapsed_ms   = int((time.monotonic() - t0) * 1000),
    )


# ---------------------------------------------------------------------------
# Python-side filter — name / date / status
# ---------------------------------------------------------------------------
_MONTH_MAP: dict[str, str] = {
    "january":"01","february":"02","march":"03","april":"04",
    "may":"05","june":"06","july":"07","august":"08",
    "september":"09","october":"10","november":"11","december":"12",
    "jan":"01","feb":"02","mar":"03","apr":"04","jun":"06",
    "jul":"07","aug":"08","sep":"09","oct":"10","nov":"11","dec":"12",
}

_NAME_FIELDS    = ("customer_name", "vendor_name", "contact_name", "display_name", "name")
_DATE_FIELDS    = ("date", "invoice_date", "bill_date", "expense_date", "transaction_date")
_BALANCE_FIELDS = ("balance", "balance_due", "amount_due")

_QUERY_STOPWORDS: frozenset[str] = frozenset({
    "give","me","the","of","for","in","list","show","get","all","tell",
    "what","how","much","many","find","fetch","display","my","our",
    "filter","and","from","to","with","that","by","on","a","an","is","are",
    "invoices","invoice","bills","bill","payments","payment","expenses",
    "expense","contacts","contact","items","item","records","record",
    "total","profit","revenue","income","net","earnings","loss","amount",
    "balance","outstanding","overdue","paid","unpaid","pending","due",
    "receive","pay","collect","owed","receivable","payable",
    "january","february","march","april","may","june","july","august",
    "september","october","november","december","jan","feb","mar","apr",
    "jun","jul","aug","sep","oct","nov","dec","month","year","today",
    "this","last","recent","latest","current",
    "2023","2024","2025","2026","2027","2028",
    "fonly","okay","please","just","only","will","i","you","your",
    "top","bottom","best","worst","highest","lowest","compare","trend",
    "ranking","ranked","growth","analysis","breakdown","summarize",
})


def _python_filter_records(records: list, user_question: str) -> list:
    if not records:
        return records

    q = user_question.lower()
    today = date.today()
    filters_applied = False

    # ── Status filter ──────────────────────────────────────────────────────
    status_filter: Optional[str] = None
    if any(w in q for w in ("unpaid", "outstanding", "pending", "to collect",
                             "to receive", "not paid")):
        status_filter = "unpaid"
        filters_applied = True
    elif "overdue" in q:
        status_filter = "overdue"
        filters_applied = True
    elif "paid" in q and "unpaid" not in q and "not paid" not in q:
        status_filter = "paid"
        filters_applied = True
    elif "draft" in q:
        status_filter = "draft"
        filters_applied = True

    # ── Date filter ────────────────────────────────────────────────────────
    date_prefix: Optional[str] = None
    date_month_only: Optional[str] = None

    MONTH_PAT = (r'\b(january|february|march|april|may|june|july|august|september'
                 r'|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep'
                 r'|oct|nov|dec)')

    m = re.search(MONTH_PAT + r'\s+(\d{4})\b', q)
    if m:
        date_prefix     = f"{m.group(2)}-{_MONTH_MAP[m.group(1)]}"
        filters_applied = True

    if not date_prefix:
        m = re.search(r'\b(20\d{2})-(0[1-9]|1[0-2])\b', q)
        if m:
            date_prefix = f"{m.group(1)}-{m.group(2)}"
            filters_applied = True

    if not date_prefix:
        m = re.search(r'\bin\s+(20\d{2})\b', q)
        if m:
            date_prefix = m.group(1)
            filters_applied = True

    if not date_prefix:
        m = re.search(r'\b(20\d{2})\b', q)
        if m:
            date_prefix = m.group(1)
            filters_applied = True

    if not date_prefix and "this month" in q:
        date_prefix     = today.strftime("%Y-%m")
        filters_applied = True

    if not date_prefix and "last month" in q:
        last_month_end  = today.replace(day=1) - timedelta(days=1)
        date_prefix     = last_month_end.strftime("%Y-%m")
        filters_applied = True

    if not date_prefix:
        m = re.search(MONTH_PAT + r'\b', q)
        if m:
            mm              = _MONTH_MAP[m.group(1)]
            date_prefix     = f"{today.year}-{mm}"
            date_month_only = mm
            filters_applied = True

    # ── Name filter ────────────────────────────────────────────────────────
    name_filter: Optional[str] = None
    words = re.findall(r"[a-z]+", q)
    candidate_words = [w for w in words if w not in _QUERY_STOPWORDS and len(w) > 2]

    if candidate_words:
        best_name = ""
        for length in range(min(6, len(candidate_words)), 0, -1):
            for start in range(len(candidate_words) - length + 1):
                chunk = " ".join(candidate_words[start:start + length])
                for rec in records[:100]:
                    if not isinstance(rec, dict):
                        continue
                    for nf in _NAME_FIELDS:
                        val = str(rec.get(nf, "")).lower()
                        if chunk in val and len(chunk) > len(best_name):
                            best_name = chunk
                if best_name:
                    break
            if best_name:
                break

        if best_name:
            name_filter = best_name
            filters_applied = True

    if not filters_applied:
        return records

    result: list = []
    for rec in records:
        if not isinstance(rec, dict):
            continue

        if name_filter:
            matched = any(
                name_filter in str(rec.get(nf, "")).lower()
                for nf in _NAME_FIELDS
            )
            if not matched:
                continue

        if date_prefix:
            matched = False
            for df in _DATE_FIELDS:
                val = str(rec.get(df, ""))
                if val.startswith(date_prefix):
                    matched = True
                    break
            if not matched and date_month_only:
                prev_year = today.year - 1
                alt_prefix = f"{prev_year}-{date_month_only}"
                for df in _DATE_FIELDS:
                    val = str(rec.get(df, ""))
                    if val.startswith(alt_prefix):
                        matched = True
                        break
            if not matched:
                continue

        if status_filter:
            rec_status = str(rec.get("status", "")).lower()
            if status_filter == "unpaid":
                balance_val = 0.0
                for bf in _BALANCE_FIELDS:
                    try:
                        balance_val = float(rec.get(bf, 0) or 0)
                        break
                    except (TypeError, ValueError):
                        pass
                if rec_status in ("paid", "void") or balance_val <= 0:
                    continue
            elif status_filter == "overdue":
                due_str = str(rec.get("due_date", ""))
                is_overdue = rec_status == "overdue"
                if due_str and not is_overdue:
                    try:
                        is_overdue = date.fromisoformat(due_str) < today
                    except ValueError:
                        pass
                if not is_overdue:
                    continue
            elif status_filter == "paid":
                if rec_status != "paid":
                    continue
            elif status_filter == "draft":
                if rec_status != "draft":
                    continue

        result.append(rec)

    log.info("python_filter_applied", extra={
        "input": len(records), "output": len(result),
        "name": name_filter, "date": date_prefix, "status": status_filter,
    })
    return result


# ---------------------------------------------------------------------------
# Python-native aggregation helpers
# ---------------------------------------------------------------------------
def _fmt_inr(amount: float) -> str:
    negative = amount < 0
    amount   = abs(amount)
    s        = f"{amount:.2f}"
    ip, dp   = s.split(".")
    if len(ip) <= 3:
        fmt = ip
    else:
        last3 = ip[-3:]
        rest  = ip[:-3]
        grps  = []
        while rest:
            grps.append(rest[-2:])
            rest = rest[:-2]
        fmt = ",".join(reversed(grps)) + "," + last3
    r = f"₹{fmt}.{dp}"
    return f"-{r}" if negative else r


def _get_amount(rec: dict, prefer_total: bool) -> float:
    if prefer_total:
        fields = ("total", "sub_total", "bcy_total", "amount", "balance", "balance_due")
    else:
        fields = ("balance", "balance_due", "amount_due", "total", "amount", "sub_total")
    for f in fields:
        v = rec.get(f)
        if v not in (None, "", 0, "0", "0.0"):
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


def _entity_name(rec: dict) -> str:
    for f in ("customer_name", "vendor_name", "contact_name", "display_name", "name"):
        if rec.get(f):
            return str(rec[f])
    for f in ("invoice_number", "bill_number", "number", "reference_number"):
        if rec.get(f):
            return str(rec[f])
    return "(unknown)"


def _python_no_records(user_q: str, tool_name: str) -> dict:
    entity = (tool_name
              .replace("ZohoBooks_list_", "")
              .replace("ZohoBooks_get_", "")
              .replace("_", " "))
    return {
        "format":    "answer",
        "question":  user_q,
        "answer":    f"No {entity} records found matching your query.",
        "breakdown": [],
        "note":      f"Filters applied from: \"{user_q}\"",
    }


def _python_aggregate(user_q: str, tool_name: str,
                      all_step_results: list) -> dict:
    q = user_q.lower()

    wants_profit  = any(w in q for w in ("profit", "net profit", "net income",
                                          "net revenue", "earnings", "p&l",
                                          "income minus", "revenue minus",
                                          "minus expense", "minus bill"))
    wants_balance = any(w in q for w in ("outstanding", "owed", "receivable",
                                          "payable", "balance", "unpaid",
                                          "pending", "to receive", "to pay",
                                          "to collect", "due"))
    wants_count   = any(w in q for w in ("how many", "count", "number of")) \
                    and not any(w in q for w in ("total", "amount", "sum", "how much"))
    wants_avg     = "average" in q
    prefer_total  = not wants_balance

    invoice_recs: list[dict] = []
    bill_recs:    list[dict] = []
    other_recs:   list[dict] = []

    for result in all_step_results:
        if not isinstance(result, dict):
            continue
        _, recs = _extract_records(result)
        if not recs:
            continue
        filtered = _python_filter_records(recs, user_q)
        if not filtered:
            continue

        first = filtered[0]
        if "invoice_number" in first or "invoice_id" in first:
            invoice_recs.extend(filtered)
        elif "bill_number" in first or "bill_id" in first:
            bill_recs.extend(filtered)
        else:
            other_recs.extend(filtered)

    all_recs = invoice_recs + bill_recs + other_recs
    n = len(all_recs)

    if wants_profit:
        inv_total  = sum(_get_amount(r, prefer_total=True) for r in invoice_recs)
        bill_total = sum(_get_amount(r, prefer_total=True) for r in bill_recs)
        profit     = inv_total - bill_total
        label      = "Profit" if profit >= 0 else "Loss"

        date_note = ""
        month_words = ("january","february","march","april","may","june","july",
                       "august","september","october","november","december",
                       "jan","feb","mar","apr","jun","jul","aug","sep","oct","nov","dec")
        if any(w in q for w in month_words) or re.search(r'\b20\d{2}\b', q):
            date_note = " · date filter applied"

        return {
            "format":    "answer",
            "question":  user_q,
            "answer":    f"{_fmt_inr(profit)} ({label})",
            "breakdown": [
                [f"Revenue — invoices ({len(invoice_recs)} records)", _fmt_inr(inv_total)],
                [f"Expenses — bills ({len(bill_recs)} records)",      _fmt_inr(bill_total)],
                [label,                                                _fmt_inr(profit)],
            ],
            "note": f"Profit = Revenue − Expenses{date_note}",
        }

    if invoice_recs and not bill_recs:
        calc_recs = invoice_recs
    elif bill_recs and not invoice_recs:
        calc_recs = bill_recs
    else:
        calc_recs = all_recs

    if not calc_recs:
        return _python_no_records(user_q, tool_name)

    n = len(calc_recs)
    total_amount = sum(_get_amount(r, prefer_total=prefer_total) for r in calc_recs)

    if wants_count:
        answer_text = f"{n} records"
    elif wants_avg:
        avg = total_amount / n if n else 0
        answer_text = f"{_fmt_inr(avg)} average across {n} records"
    else:
        answer_text = f"{_fmt_inr(total_amount)} across {n} records"

    entity_totals: dict[str, float] = {}
    entity_counts: dict[str, int]   = {}
    for rec in calc_recs:
        name = _entity_name(rec)
        amt  = _get_amount(rec, prefer_total=prefer_total)
        entity_totals[name] = entity_totals.get(name, 0.0) + amt
        entity_counts[name] = entity_counts.get(name, 0)  + 1

    breakdown = []
    for name, amt in sorted(entity_totals.items(), key=lambda x: -x[1])[:15]:
        cnt   = entity_counts[name]
        label = f"{name} ({cnt} records)" if cnt > 1 else name
        breakdown.append([label, str(cnt) if wants_count else _fmt_inr(amt)])

    note_parts = []
    month_words = ("january","february","march","april","may","june","july",
                   "august","september","october","november","december",
                   "jan","feb","mar","apr","jun","jul","aug","sep","oct","nov","dec")
    if any(w in q for w in month_words) or re.search(r'\b20\d{2}\b', q):
        note_parts.append("date filter applied")
    if any(w in q for w in ("overdue","unpaid","outstanding","paid","draft")):
        note_parts.append("status filter applied")
    note_parts.append(f"computed from {n} matching records")

    return {
        "format":    "answer",
        "question":  user_q,
        "answer":    answer_text,
        "breakdown": breakdown,
        "note":      " · ".join(note_parts),
    }


_VALID_FORMATS = frozenset({"table", "answer", "panel", "status"})


def _sanitize_structured(data: dict, user_q: str) -> dict:
    fmt = data.get("format", "")
    if fmt in _VALID_FORMATS:
        return data

    log.warning("gemini_invalid_format", extra={"format": fmt, "question": user_q[:80]})

    rescued_breakdown: list = []
    data_list = data.get("data") or data.get("items") or data.get("rows") or []
    if isinstance(data_list, list):
        for item in data_list[:30]:
            if isinstance(item, dict):
                label = item.get("label") or item.get("name") or item.get("month") or ""
                value = item.get("value") or item.get("total") or item.get("amount") or ""
                if label:
                    rescued_breakdown.append([str(label), str(value)])

    title   = data.get("title", user_q)
    summary = data.get("summary") or data.get("answer") or data.get("headline") or title

    return {
        "format":    "answer",
        "question":  user_q,
        "answer":    str(summary),
        "breakdown": rescued_breakdown,
        "note":      f"(converted from unsupported format '{fmt}')",
    }


async def _gemini_summarize_single(gemini_client, tool_name: str,
                                    user_q: str, result: Any, cid: str) -> dict:
    import re as _re

    today      = date.today()
    system_str = _build_summarize_system_with_date()

    prompt = (
        f"TODAY: {today.isoformat()}\n"
        f"USER_QUESTION: {user_q or '(not specified)'}\n"
        f"TOOL: {tool_name}\n"
        f"RESULT: {_safe_result_str(result)}"
    )
    buf = ""
    try:
        async def _collect():
            nonlocal buf
            async for chunk in await gemini_client.aio.models.generate_content_stream(
                model=MODEL,
                contents=[gtypes.Content(role="user", parts=[gtypes.Part(text=prompt)])],
                config=gtypes.GenerateContentConfig(
                    system_instruction=system_str,
                    temperature=0.1,  # LOW = consistent, no hallucinations
                    response_mime_type="application/json",
                    max_output_tokens=4096,
                ),
            ):
                buf += chunk.text or ""
        await asyncio.wait_for(_collect(), timeout=GEMINI_TIMEOUT)
    except asyncio.TimeoutError:
        return {"format": "status", "ok": True,
                "headline": "Response timed out.", "detail": str(result)[:300]}
    except Exception as exc:
        log.warning("gemini_single_error", extra={"cid": cid, "error": str(exc)})
        return {"format": "status", "ok": True, "headline": "Done.", "detail": ""}

    cleaned = _re.sub(r"^```(?:json)?\s*|\s*```$", "", buf.strip(), flags=_re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
        return _sanitize_structured(parsed, user_q)
    except Exception:
        return {"format": "answer", "question": user_q,
                "answer": cleaned[:400] if cleaned else "No result.",
                "breakdown": [], "note": ""}


# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------
_PYTHON_MATH_PATTERNS = (
    r'\btotal\b', r'\bsum\b', r'\bhow much\b', r'\bamount\b',
    r'\bhow many\b', r'\bcount\b', r'\bnumber of\b',
    r'\baverage\b', r'\bmean\b', r'\bavg\b',
    r'\bprofit\b', r'\bnet income\b', r'\bnet profit\b', r'\bearnings\b',
    r'\boutstanding\b', r'\breceivable\b', r'\bpayable\b',
    r'\bbalance\b', r'\bowed\b',
)

_GEMINI_ANALYSIS_PATTERNS = (
    r'\btop\s+\d+\b',     r'\bbottom\s+\d+\b',
    r'\branking\b',        r'\branked\b',
    r'\bbest\b',           r'\bworst\b',
    r'\bhighest\b',        r'\blowest\b',
    r'\blargest\b',        r'\bsmallest\b',  r'\bbiggest\b',
    r'\bwho\b',            r'\bwhich\b',
    r'\bmost\b',           r'\bleast\b',     r'\bfrequent\b',
    r'\bcompare\b',        r'\bcomparison\b', r'\btrend\b',
    r'\bgrowth\b',         r'\bincrease\b',  r'\bdecrease\b',
    r'\banalysis\b',       r'\banalyze\b',   r'\binsight\b',
    r'\bperformance\b',    r'\bbreakdown\b', r'\bsummarize\b',
    r'\bby value\b',       r'\bby revenue\b', r'\bby volume\b',
    r'\bby amount\b',      r'\bby total\b',  r'\bby customer\b',
)

def _classify_query(user_question: str) -> str:
    q = user_question.lower()
    if any(re.search(p, q) for p in _GEMINI_ANALYSIS_PATTERNS):
        return "gemini_analysis"
    if any(re.search(p, q) for p in _PYTHON_MATH_PATTERNS):
        return "python_math"
    return "python_table"


def _user_wants_aggregation(user_question: str) -> bool:
    return _classify_query(user_question) == "python_math"


# ---------------------------------------------------------------------------
# Streaming endpoint
# ---------------------------------------------------------------------------
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    a   = _app()
    cid = uuid.uuid4().hex[:10]

    try:
        sess = await a.sessions.get_or_create(
            req.session_id, req.mcp_url, a.gemini, org_id=req.org_id
        )
    except HTTPException as exc:
        _err_detail = exc.detail
        async def _err_stream() -> AsyncIterator[str]:
            yield "data: " + json.dumps({"type": "error", "message": _err_detail}) + "\n\n"
        return StreamingResponse(_err_stream(), media_type="text/event-stream")
    except Exception as exc:
        _err_msg = f"Unexpected error: {exc}"
        async def _err_stream2() -> AsyncIterator[str]:
            yield "data: " + json.dumps({"type": "error", "message": _err_msg}) + "\n\n"
        return StreamingResponse(_err_stream2(), media_type="text/event-stream")

    a.audit.write("user_message_stream", msg=req.message, cid=cid, sid=sess.session_id)

    async def _stream() -> AsyncIterator[str]:
        def sse(data: dict) -> str:
            return f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

        try:
            if is_number_string(req.message) and not sess.state.organization_id:
                sess.state.organization_id = req.message
                yield sse({"type": "token", "token": f"Organization ID saved: {req.message}."})
                yield sse({"type": "done",
                           "structured": {"format": "status", "ok": True, "headline": "Org ID saved"},
                           "session_id": sess.session_id, "tools_used": []})
                return

            t_plan_start = time.monotonic()
            raw   = await gemini_plan(a.gemini, sess.toolbox, sess.state, sess.memory, req.message, cid)
            plan  = normalize_plan(raw)
            ptype = plan.get("type")
            sess.replan_attempts = 0
            log.info("plan_latency", extra={"cid": cid, "ms": int((time.monotonic()-t_plan_start)*1000), "type": ptype})

            if ptype == "ask":
                sess.state.update_from_dict(plan.get("save") or {})
                text = plan.get("text", "I need more details.")
                yield sse({"type": "token", "token": text})
                yield sse({"type": "done",
                           "structured": {"format": "status", "ok": True, "headline": text},
                           "session_id": sess.session_id, "tools_used": []})
                return

            if ptype == "confirm":
                sess.pending_confirm = {"on_yes": plan.get("on_yes"), "on_no": plan.get("on_no")}
                yield sse({"type": "confirm", "confirm_text": plan.get("text", "Confirm?"),
                           "session_id": sess.session_id})
                return

            if ptype != "plan":
                yield sse({"type": "error", "message": "Could not determine an action."})
                return

            steps = plan.get("steps") or []
            risky = [s for s in steps if is_risky(s.get("tool", ""))]
            if risky:
                names = ", ".join(s["tool"] for s in risky)
                sess.pending_confirm = {"on_yes": plan, "on_no": {"type": "ask", "text": "Cancelled."}}
                yield sse({"type": "confirm",
                           "confirm_text": f"Sensitive operations: {names}. Confirm to proceed?",
                           "session_id": sess.session_id})
                return

            t_exec_start = time.monotonic()
            ok, last_result, all_step_results = await execute_plan(
                sess.mcp_session, sess.toolbox, sess.state,
                sess.memory, a.audit, a.gemini, steps, cid,
            )
            tools_used = [s.get("tool", "") for s in steps]
            log.info("exec_latency", extra={"cid": cid, "ms": int((time.monotonic()-t_exec_start)*1000), "tools": tools_used})

            if not ok:
                if isinstance(last_result, dict) and last_result.get("__execution_error__"):
                    err_msg = last_result.get("message", "Execution failed.")
                    display = f"Could not complete: {err_msg}"
                else:
                    display = "Execution failed. Please try rephrasing your request."
                yield sse({"type": "error", "message": display,
                           "session_id": sess.session_id})
                return

            last_step = steps[-1]
            last_tool = last_step.get("tool", "")
            user_q    = last_step.get("note", "") or req.message

            # Extract pagination metadata BEFORE routing
            paginate_meta = None
            if isinstance(last_result, dict):
                paginate_meta = last_result.pop("__paginate__", None)

            # ── 4-tier routing ────────────────────────────────────
            t_summ_start  = time.monotonic()
            query_class   = _classify_query(user_q)
            is_action_tool = any(w in last_tool.lower() for w in
                                 ("create","update","delete","void","send","submit","mark"))

            list_key, all_records = _extract_records(last_result) if isinstance(last_result, dict) else ("", [])
            filtered_records = _python_filter_records(all_records, user_q)
            _page1_col_keys: list[str] = []

            if is_action_tool:
                # Tier 1: Action — Gemini writes natural status message
                structured = await _gemini_summarize_single(
                    a.gemini, last_tool, user_q, last_result, cid
                )

            elif query_class == "python_math":
                # Tier 2: Math — Python computes exact totals/counts/profit
                structured = _python_aggregate(user_q, last_tool, all_step_results)

            elif query_class == "gemini_analysis" or (
                len(filtered_records) == 1 and not is_action_tool
            ):
                # Tier 3: Analytical — Gemini with full filtered dataset
                if len(filtered_records) == 0:
                    structured = _python_no_records(user_q, last_tool)
                else:
                    _KEY_FIELDS = frozenset({
                        "invoice_number","bill_number","customer_name","vendor_name",
                        "contact_name","name","date","due_date","total","balance",
                        "balance_due","amount","status","invoice_id","bill_id",
                        "contact_id","currency_code","sub_total",
                    })
                    compact = [
                        {k: v for k, v in r.items() if k in _KEY_FIELDS}
                        for r in filtered_records if isinstance(r, dict)
                    ]
                    gemini_payload = {
                        "records": compact[:500],
                        "total_records": len(filtered_records),
                    }
                    structured = await _gemini_summarize_single(
                        a.gemini, last_tool, user_q, gemini_payload, cid
                    )

            else:
                # Tier 4: Plain list — Python table (fast, no AI)
                if len(filtered_records) == 0:
                    structured = _python_no_records(user_q, last_tool)
                else:
                    structured = _build_table_structured(
                        last_result, last_tool,
                        more_pages=bool(paginate_meta),
                        records_override=filtered_records,
                    )
                    _page1_col_keys = structured.get("col_keys") or []
                    structured.pop("col_keys", None)

            log.info("summarize_latency", extra={
                "cid": cid, "ms": int((time.monotonic()-t_summ_start)*1000),
                "filtered_count": len(filtered_records),
                "query_class": query_class,
            })

            # Emit tokens
            reply_md = structured_to_markdown(structured)
            CHUNK = 80
            for i in range(0, len(reply_md), CHUNK):
                yield sse({"type": "token", "token": reply_md[i:i + CHUNK]})
                if i % (CHUNK * 8) == 0:
                    await asyncio.sleep(0)

            sess.state.clear_workflow()
            sess.touch()

            client_structured = {k: v for k, v in structured.items() if k != "col_keys"}
            yield sse({"type": "done", "structured": client_structured,
                       "session_id": sess.session_id, "tools_used": tools_used})

            # Background pagination
            if paginate_meta:
                pg_tool  = paginate_meta["tool"]
                pg_args  = dict(paginate_meta["args"])
                col_keys = _page1_col_keys or _detect_cols(filtered_records, pg_tool)
                page     = 2
                total_fetched = len(filtered_records)

                while page <= MAX_AUTOPAGINATE_PAGES:
                    pg_args["page"] = page
                    log.info("bg_paginate_fetch", extra={"tool": pg_tool, "page": page, "cid": cid})
                    try:
                        raw_page    = await asyncio.wait_for(
                            sess.mcp_session.call_tool(pg_tool, pg_args),
                            timeout=TOOL_TIMEOUT,
                        )
                        page_result = _parse_mcp_result(raw_page)
                    except Exception as exc:
                        log.warning("bg_paginate_error", extra={"page": page, "error": str(exc), "cid": cid})
                        break

                    _, page_records = _extract_records(page_result)
                    if not page_records:
                        break

                    page_filtered = _python_filter_records(page_records, user_q)
                    total_fetched += len(page_filtered)
                    has_more       = _has_more_pages(page_result)

                    rows = [[str(r.get(c, "")) for c in col_keys]
                            for r in page_filtered if isinstance(r, dict)]

                    if rows:
                        yield sse({
                            "type":         "table_append",
                            "rows":         rows,
                            "total_so_far": total_fetched,
                            "has_more":     has_more,
                            "page":         page,
                        })
                    await asyncio.sleep(0)

                    if not has_more:
                        break
                    page += 1

                yield sse({"type": "table_complete", "total": total_fetched})

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            import traceback as _tb
            log.error("stream_error", extra={
                "cid":        cid,
                "error":      str(exc),
                "error_type": type(exc).__name__,
                "traceback":  _tb.format_exc(),
            })
            yield f"data: {json.dumps({'type': 'error', 'message': f'{type(exc).__name__}: {exc}'})}\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


@app.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str) -> SessionInfo:
    sess = await _app().sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found or expired.")
    return SessionInfo(
        session_id      = sess.session_id,
        state           = sess.state.to_dict(),
        memory_entries  = len(sess.memory),
        memory_tokens   = sum(_estimate_tokens(m) for m in sess.memory),
        last_active     = datetime.fromtimestamp(sess.last_active, tz=timezone.utc).isoformat(),
        pending_confirm = sess.pending_confirm is not None,
        mcp_url         = sess.mcp_url,
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str) -> dict:
    if not await _app().sessions.delete(session_id):
        raise HTTPException(404, "Session not found.")
    return {"deleted": session_id}


BASE_DIR = Path(__file__).resolve().parent


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(BASE_DIR / "favicon.ico", media_type="image/x-icon")
