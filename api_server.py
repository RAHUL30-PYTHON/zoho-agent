from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
from pathlib import Path
from fastapi.responses import FileResponse

try:
    from mcp.client.streamable_http import streamable_http_client
except ImportError:
    from mcp.client.streamable_http import streamablehttp_client as streamable_http_client

from zoho_agent import (
    AgentState,
    AuditLog,
    AUDIT_LOG_PATH,
    GEMINI_TIMEOUT,
    MAX_AUTOPAGINATE_PAGES,
    MAX_REPLAN_ATTEMPTS,
    MODEL,
    PLANNER_MODEL,
    SUMMARIZE_SYSTEM,
    TOOL_TIMEOUT,
    _estimate_tokens,
    _extract_records,
    _has_more_pages,
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
# Session factory — org_id now comes from the caller, not env
# ---------------------------------------------------------------------------
async def create_agent_session(
    session_id: str,
    mcp_url: str,
    gemini: genai.Client,
    org_id: Optional[str] = None,
) -> AgentSession:
    zoho_bearer = os.getenv("ZOHO_MCP_BEARER")
    headers     = {"Authorization": f"Bearer {zoho_bearer}"} if zoho_bearer else {}

    http_client = httpx.AsyncClient(
        headers=headers,
        timeout=httpx.Timeout(TOOL_TIMEOUT + 10),
    )
    await http_client.__aenter__()

    stream_cm = streamable_http_client(mcp_url, http_client=http_client)
    read, write, _ = await stream_cm.__aenter__()

    mcp_cm = ClientSession(read, write)
    mcp_session: ClientSession = await mcp_cm.__aenter__()
    await mcp_session.initialize()

    toolbox = build_tool_catalog(await mcp_session.list_tools())
    log.info("session_mcp_connected", extra={"sid": session_id, "tools": len(toolbox), "url": mcp_url})

    return AgentSession(
        session_id   = session_id,
        # org_id from request takes priority; no hardcoded default
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

            # Return existing session (mcp_url / org_id ignored — already set)
            if session_id and session_id in self._sessions:
                s = self._sessions[session_id]
                s.touch()
                return s

            # New session — mcp_url is required
            if not mcp_url:
                raise HTTPException(400, "mcp_url is required to start a new session.")
            if not org_id:
                raise HTTPException(400, "org_id is required to start a new session.")

            if len(self._sessions) >= MAX_SESSIONS:
                raise HTTPException(503, "Too many active sessions — try again later.")

            sid = session_id or uuid.uuid4().hex
            sess = await create_agent_session(sid, mcp_url, gemini, org_id=org_id)
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
# Summarizer
# ---------------------------------------------------------------------------
async def api_summarize(
    gemini: genai.Client,
    tool: str,
    args: dict,
    result: Any,
    cid: str = "",
    user_question: str = "",
) -> dict:
    prompt = (
        f"USER_QUESTION: {user_question or '(not specified — use best judgement on format)'}\n"
        f"TOOL: {tool}\n"
        f"ARGS: {json.dumps(args, default=str)}\n"
        f"RESULT: {json.dumps(result, default=str)}"
    )
    raw     = await _stream_collect_json(gemini, prompt, SUMMARIZE_SYSTEM, cid)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return {"format": "status", "ok": True, "headline": cleaned or "Done.", "detail": ""}


def _count_records(result: Any) -> int:
    """Count total records in the primary list of a Zoho result."""
    _, records = _extract_records(result) if isinstance(result, dict) else ("", [])
    return len(records)


def _records_to_rows(records: list) -> dict:
    """Convert a list of Zoho record dicts into {columns, rows} for SSE emission."""
    PREFERRED = ["invoice_number", "bill_number", "contact_name", "customer_name",
                 "vendor_name", "date", "due_date", "total", "balance", "status",
                 "payment_expected_date", "due_days"]
    SKIP = {"organization_id", "created_time", "last_modified_time",
            "template_id", "color_code", "zcrm_potential_id"}

    col_freq: dict[str, int] = {}
    for rec in records[:50]:
        if isinstance(rec, dict):
            for k in rec:
                col_freq[k] = col_freq.get(k, 0) + 1

    cols = [c for c in PREFERRED if col_freq.get(c, 0) > 0]
    if not cols:
        cols = [c for c, _ in sorted(col_freq.items(), key=lambda x: -x[1])
                if c not in SKIP][:8]

    rows = []
    for rec in records:
        if isinstance(rec, dict):
            rows.append([str(rec.get(c, "")) for c in cols])

    return {
        "columns": [c.replace("_", " ").title() for c in cols],
        "rows":    rows,
    }


def _fast_fallback_summary(result: Any, tool_name: str) -> str:
    """
    Pure-Python fallback summary when Gemini times out on a large result.
    Extracts the biggest record list, computes totals/counts, returns
    structured JSON matching the summarizer output format.
    """
    import json as _json

    # Find the largest list inside the result
    records: list = []
    if isinstance(result, dict):
        for k, v in result.items():
            if isinstance(v, list) and len(v) > len(records):
                records = v

    if not records:
        return _json.dumps({
            "format": "status", "ok": True,
            "headline": f"{tool_name} completed — {len(records)} records returned.",
        })

    count = len(records)

    # Try to find amount/balance fields and sum them
    AMOUNT_FIELDS = ("balance", "total", "amount", "balance_due",
                     "outstanding_receivable_amount", "outstanding_payable_amount")
    totals: dict[str, float] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        for field in AMOUNT_FIELDS:
            if field in rec:
                try:
                    totals[field] = totals.get(field, 0.0) + float(rec[field])
                except (TypeError, ValueError):
                    pass

    # Build headline
    primary_field = next((f for f in AMOUNT_FIELDS if f in totals), None)
    if primary_field:
        total_val = totals[primary_field]
        headline  = f"{count} records · Total {primary_field.replace('_',' ')}: ₹{total_val:,.2f}"
    else:
        headline = f"{count} records returned by {tool_name}"

    # Build a slim table from the first 100 records
    # Identify columns present in most records
    col_freq: dict[str, int] = {}
    for rec in records[:100]:
        if isinstance(rec, dict):
            for k in rec:
                col_freq[k] = col_freq.get(k, 0) + 1

    # Prefer useful display columns, skip internal IDs and noise
    PREFERRED = ["invoice_number", "bill_number", "contact_name", "customer_name",
                 "vendor_name", "date", "due_date", "total", "balance", "status",
                 "payment_expected_date", "due_days"]
    SKIP      = {"organization_id", "created_time", "last_modified_time",
                 "template_id", "color_code", "zcrm_potential_id"}

    cols = [c for c in PREFERRED if col_freq.get(c, 0) > 0]
    if not cols:
        cols = [c for c in sorted(col_freq, key=lambda k: -col_freq[k])
                if c not in SKIP][:8]

    rows = []
    for rec in records[:100]:
        if isinstance(rec, dict):
            rows.append([str(rec.get(c, "")) for c in cols])

    footer = f"Showing {min(count,100)} of {count} records"
    if primary_field:
        footer += f" · Total {primary_field.replace('_',' ')}: ₹{totals[primary_field]:,.2f}"

    return _json.dumps({
        "format":  "table",
        "title":   tool_name.replace("ZohoBooks_", "").replace("_", " ").title(),
        "columns": [c.replace("_", " ").title() for c in cols],
        "rows":    rows,
        "footer":  footer,
    })


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
# Core turn logic
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

    ok, last_result = await execute_plan(
        sess.mcp_session, sess.toolbox, sess.state,
        sess.memory, a.audit, a.gemini, steps, cid,
    )
    tools_used = [s.get("tool", "") for s in steps]

    if ok and last_result is not None:
        last_tool  = steps[-1].get("tool", "")
        structured = await api_summarize(a.gemini, last_tool, steps[-1].get("args", {}), last_result, cid,
                                              user_question=steps[-1].get("note", ""))
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
    version="2.1.0",
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


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    a    = _app()
    sess = await a.sessions.get_or_create(
        req.session_id, req.mcp_url, a.gemini, org_id=req.org_id
    )
    cid  = uuid.uuid4().hex[:10]
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
            ok, last_result = await execute_plan(
                sess.mcp_session, sess.toolbox, sess.state,
                sess.memory, a.audit, a.gemini, steps, cid,
            )
            tools_used = [s.get("tool", "") for s in steps]
            log.info("exec_latency", extra={"cid": cid, "ms": int((time.monotonic()-t_exec_start)*1000), "tools": tools_used})

            if not ok or last_result is None:
                yield sse({"type": "error", "message": "Execution failed.",
                           "session_id": sess.session_id})
                return

            last_step = steps[-1]
            last_tool = last_step.get("tool", "")
            user_q    = last_step.get("note", "")

            # ── Progressive pagination ─────────────────────────────────────
            # If execute_step flagged that more pages exist, we:
            # 1. Summarise page 1 immediately and emit it to the client
            # 2. Start a background loop fetching remaining pages
            # 3. Each new page emits table_append SSE events so the frontend
            #    can extend the live table while the user is reading
            paginate_meta = None
            if isinstance(last_result, dict):
                paginate_meta = last_result.pop("__paginate__", None)

            # Helper: run Gemini summarizer and yield structured + tokens
            async def _summarise_and_emit(result: Any, is_page1: bool) -> Optional[dict]:
                nonlocal full_json
                result_str   = _safe_result_str(result)
                prompt_inner = (
                    f"USER_QUESTION: {user_q or '(not specified — use best judgement on format)'}\n"
                    f"TOOL: {last_tool}\n"
                    f"ARGS: {json.dumps(last_step.get('args', {}), default=str)}\n"
                    f"RESULT: {result_str}"
                )
                timeout_inner = max(GEMINI_TIMEOUT, min(180.0, len(result_str) / 2000))
                buf = ""

                async def _collect() -> None:
                    nonlocal buf
                    async for chunk in await a.gemini.aio.models.generate_content_stream(
                        model=MODEL,
                        contents=[gtypes.Content(role="user", parts=[gtypes.Part(text=prompt_inner)])],
                        config=gtypes.GenerateContentConfig(
                            system_instruction=SUMMARIZE_SYSTEM,
                            temperature=0.1,
                            response_mime_type="application/json",
                            max_output_tokens=65536,
                        ),
                    ):
                        buf += chunk.text or ""

                try:
                    await asyncio.wait_for(_collect(), timeout=timeout_inner)
                except asyncio.TimeoutError:
                    log.warning("summarize_timeout", extra={"cid": cid, "tool": last_tool})
                    if not (buf.strip().endswith("}") or buf.strip().endswith("]")):
                        buf = _fast_fallback_summary(result, last_tool)

                cleaned_inner = re.sub(r"^```(?:json)?\s*|\s*```$", "", buf.strip(), flags=re.MULTILINE).strip()
                try:
                    structured_inner = json.loads(cleaned_inner)
                except Exception:
                    structured_inner = {"format": "status", "ok": True, "headline": cleaned_inner or "Done."}

                if is_page1 and paginate_meta:
                    # Mark this as the first page of a progressive table —
                    # the frontend will show a loading indicator
                    structured_inner["__more_pages__"] = True

                return structured_inner

            t_summ_start = time.monotonic()
            full_json    = ""
            structured   = await _summarise_and_emit(last_result, is_page1=bool(paginate_meta))
            log.info("summarize_latency", extra={"cid": cid, "ms": int((time.monotonic()-t_summ_start)*1000)})

            # Emit page 1 tokens immediately
            reply_md = structured_to_markdown(structured)
            CHUNK = 80
            for i in range(0, len(reply_md), CHUNK):
                yield sse({"type": "token", "token": reply_md[i:i + CHUNK]})
                if i % (CHUNK * 8) == 0:
                    await asyncio.sleep(0)

            sess.state.clear_workflow()
            sess.touch()
            yield sse({"type": "done", "structured": structured,
                       "session_id": sess.session_id, "tools_used": tools_used})

            # ── Background pagination ──────────────────────────────────────
            # After sending done, keep the SSE connection open and stream
            # additional pages silently. Each page emits table_append.
            if paginate_meta:
                pg_tool = paginate_meta["tool"]
                pg_args = dict(paginate_meta["args"])
                page    = 2
                total_fetched = _count_records(last_result)

                while page <= MAX_AUTOPAGINATE_PAGES:
                    pg_args["page"] = page
                    log.info("bg_paginate_fetch", extra={"tool": pg_tool, "page": page, "cid": cid})
                    try:
                        raw_page = await asyncio.wait_for(
                            sess.mcp_session.call_tool(pg_tool, pg_args),
                            timeout=TOOL_TIMEOUT,
                        )
                        page_result = raw_page.model_dump() if hasattr(raw_page, "model_dump") else raw_page
                    except Exception as exc:
                        log.warning("bg_paginate_error", extra={"page": page, "error": str(exc), "cid": cid})
                        break

                    _, page_records = _extract_records(page_result)
                    if not page_records:
                        break

                    total_fetched += len(page_records)
                    has_more = _has_more_pages(page_result)

                    # Build slim rows using the same column logic as fast_fallback
                    rows = _records_to_rows(page_records)
                    yield sse({
                        "type":          "table_append",
                        "rows":          rows["rows"],
                        "columns":       rows["columns"],
                        "total_so_far":  total_fetched,
                        "has_more":      has_more,
                        "page":          page,
                    })
                    await asyncio.sleep(0)

                    if not has_more:
                        break
                    page += 1

                # Signal that all pages have been fetched
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


