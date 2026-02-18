"""
zoho_agent.py — Production-ready Zoho Books MCP Agent
======================================================
Features:
  - Structured logging (JSON) with configurable level
  - Retry with exponential back-off on transient MCP/Gemini failures
  - Request-scoped correlation IDs for full traceability
  - Circuit-breaker per tool (auto-disables flapping tools)
  - Token-budget-aware memory: evicts by size, not just count
  - Parallel safe-step execution (non-dependent read steps run concurrently)
  - Graceful shutdown on SIGINT / SIGTERM
  - Session-level audit log (JSONL file)
  - Rich console output with colours (degrades gracefully without `rich`)
  - Environment validation at startup
  - Full type annotations throughout
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import sys
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Rich console — degrades gracefully if not installed
# ---------------------------------------------------------------------------
RICH_AVAILABLE = False
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    from rich.columns import Columns
    from rich.markup import escape as rich_escape
    _console = Console(highlight=False)
    RICH_AVAILABLE = True
    def _print(msg: str, style: str = "") -> None:
        _console.print(msg, style=style)
    def _print_raw(obj: Any) -> None:
        _console.print(obj)
except ImportError:
    def _print(msg: str, style: str = "") -> None:  # type: ignore[misc]
        print(msg)
    def _print_raw(obj: Any) -> None:  # type: ignore[misc]
        print(obj)

# ---------------------------------------------------------------------------
# Environment & constants
# ---------------------------------------------------------------------------
load_dotenv("api.env")

MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
MAX_MEMORY_TOKENS: int = int(os.getenv("MAX_MEMORY_TOKENS", "20000"))
MAX_MEMORY_ENTRIES: int = int(os.getenv("MAX_MEMORY_ENTRIES", "30"))
MAX_TOOL_PROPS: int = int(os.getenv("MAX_TOOL_PROPS", "60"))
MAX_REPLAN_ATTEMPTS: int = int(os.getenv("MAX_REPLAN_ATTEMPTS", "3"))
AUDIT_LOG_PATH: str = os.getenv("AUDIT_LOG_PATH", "audit.jsonl")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "WARNING")
TOOL_TIMEOUT: float = float(os.getenv("TOOL_TIMEOUT_SECONDS", "30"))
GEMINI_TIMEOUT: float = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "45"))
CIRCUIT_BREAK_THRESHOLD: int = int(os.getenv("CIRCUIT_BREAK_THRESHOLD", "3"))
CIRCUIT_BREAK_RESET_S: int = int(os.getenv("CIRCUIT_BREAK_RESET_SECONDS", "120"))

REQUIRED_ENV_VARS = ["GEMINI_API_KEY"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
            **getattr(record, "extra", {}),
        })

def _setup_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    logger = logging.getLogger("zoho_agent")
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    logger.addHandler(handler)
    logger.propagate = False
    return logger

log = _setup_logging()

# ---------------------------------------------------------------------------
# Audit logger
# ---------------------------------------------------------------------------
class AuditLog:
    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._fh = self._path.open("a", encoding="utf-8")

    def write(self, event: str, **kwargs: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **kwargs,
        }
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

# ---------------------------------------------------------------------------
# Risky tool classification
# ---------------------------------------------------------------------------
RISKY_EXACT: frozenset[str] = frozenset({
    "create_vendor_payment",
    "refund_excess_vendor_payment",
    "refund_vendor_credit",
    "create_customer_payment_refund",
    "delete_contact",
    "delete_invoice",
    "delete_bill",
    "delete_item",
    "delete_expense",
    "delete_customer_payment",
    "delete_vendor_payment",
    "delete_account",
    "void_invoice",
    "void_bill",
    "void_salesorder",
})
RISKY_SUBSTR: frozenset[str] = frozenset({
    "delete", "void", "write_off", "refund",
})

def is_risky(tool_name: str) -> bool:
    ln = tool_name.lower()
    return ln in RISKY_EXACT or any(s in ln for s in RISKY_SUBSTR)

def is_read_only(tool_name: str) -> bool:
    ln = tool_name.lower()
    return ln.startswith(("list_", "get_", "search_")) or "report" in ln

# ---------------------------------------------------------------------------
# Circuit breaker (per tool)
# ---------------------------------------------------------------------------
@dataclass
class CircuitBreaker:
    failures: int = 0
    opened_at: Optional[float] = None

    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.monotonic() - self.opened_at > CIRCUIT_BREAK_RESET_S:
            self.failures = 0
            self.opened_at = None
            return False
        return True

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= CIRCUIT_BREAK_THRESHOLD:
            self.opened_at = time.monotonic()
            log.warning("circuit_breaker_opened", extra={"failures": self.failures})

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

_circuit_breakers: dict[str, CircuitBreaker] = defaultdict(CircuitBreaker)

# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------
async def retry_async(
    coro_fn,
    *args,
    retries: int = 3,
    base_delay: float = 1.0,
    retryable_exceptions: tuple = (httpx.TransportError, asyncio.TimeoutError),
    label: str = "op",
    **kwargs,
) -> Any:
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(retries):
        try:
            return await coro_fn(*args, **kwargs)
        except retryable_exceptions as exc:
            last_exc = exc
            wait = base_delay * (2 ** attempt)
            log.warning("retry", extra={"label": label, "attempt": attempt + 1, "wait_s": wait, "error": str(exc)})
            await asyncio.sleep(wait)
        except Exception as exc:
            raise  # non-retryable
    raise last_exc

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ToolMeta:
    name: str
    desc: str
    required: list[str]
    allowed: list[str]
    body_key: Optional[str]
    body_required: list[str]
    schema_compact: dict
    read_only: bool = False

    def to_planner_dict(self) -> dict:
        return {
            "desc": self.desc,
            "required": self.required,
            "allowed": self.allowed[:40],
            "schema": self.schema_compact,
            "read_only": self.read_only,
        }


@dataclass
class AgentState:
    organization_id: Optional[str] = None
    pending_intent: Optional[str] = None
    pending_question: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {}
        if self.organization_id:
            d["organization_id"] = self.organization_id
        if self.pending_intent:
            d["pending_intent"] = self.pending_intent
        if self.pending_question:
            d["pending_question"] = self.pending_question
        d.update(self.extra)
        return d

    def update_from_dict(self, data: dict) -> None:
        for k, v in data.items():
            if v in (None, ""):
                continue
            if k == "organization_id":
                self.organization_id = str(v)
            elif k == "pending_intent":
                self.pending_intent = str(v)
            elif k == "pending_question":
                self.pending_question = str(v)
            else:
                self.extra[k] = v

    def clear_workflow(self) -> None:
        self.pending_intent = None
        self.pending_question = None


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------
def compact_schema(schema: dict, max_props: int = MAX_TOOL_PROPS) -> dict:
    if not schema:
        return {}
    props = schema.get("properties") or {}
    req = schema.get("required") or []
    keys = list(props.keys())[:max_props]
    overflow = len(props) - len(keys)
    slim = {
        k: {"type": props[k].get("type")} if isinstance(props[k], dict) else {}
        for k in keys
    }
    out: dict = {
        "type": schema.get("type"),
        "required": [k for k in req if k in slim],
        "properties": slim,
    }
    if overflow > 0:
        out["_truncated"] = overflow
    return out


def build_tool_catalog(list_tools_result: Any) -> dict[str, ToolMeta]:
    catalog: dict[str, ToolMeta] = {}
    for t in list_tools_result.tools:
        schema = t.inputSchema or {}
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        allowed = sorted(props.keys())

        body_key: Optional[str] = None
        body_required: list[str] = []
        for key in ("body", "JSONString"):
            if key in props and isinstance(props[key], dict):
                body_key = key
                body_required = props[key].get("required") or []
                break

        catalog[t.name] = ToolMeta(
            name=t.name,
            desc=(t.description or "").strip()[:160],
            required=[k for k in required if k in props],
            allowed=allowed,
            body_key=body_key,
            body_required=body_required,
            schema_compact=compact_schema(schema),
            read_only=is_read_only(t.name),
        )
    log.info("catalog_built", extra={"tool_count": len(catalog)})
    return catalog


def validate_args(meta: ToolMeta, args: dict) -> tuple[list[str], list[str]]:
    """Returns (unknown_keys, missing_required_keys)."""
    allowed_set = set(meta.allowed)
    unknown = [k for k in args if k not in allowed_set]
    missing: list[str] = []

    for k in meta.required:
        if k not in args or args.get(k) in (None, "", []):
            missing.append(k)

    if meta.body_key and meta.body_key in args and meta.body_required:
        raw = args[meta.body_key]
        try:
            body = raw if isinstance(raw, dict) else json.loads(raw or "{}")
        except Exception:
            body = {}
        for bk in meta.body_required:
            if bk not in body or body.get(bk) in (None, "", []):
                missing.append(f"{meta.body_key}.{bk}")

    return unknown, missing


# ---------------------------------------------------------------------------
# Memory — token-budget-aware ring buffer
# ---------------------------------------------------------------------------
def _estimate_tokens(obj: Any) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(json.dumps(obj, default=str)) // 4)


def add_memory(memory: list[dict], entry: dict) -> None:
    # Trim oversized result payloads
    if "result" in entry and isinstance(entry["result"], dict):
        content = entry["result"].get("content")
        if isinstance(content, list) and len(content) > 1:
            entry = {**entry, "result": {**entry["result"], "content": content[:2]}}

    memory.append(entry)

    # Evict oldest until within budget
    while len(memory) > MAX_MEMORY_ENTRIES:
        memory.pop(0)
    while (
        len(memory) > 2
        and sum(_estimate_tokens(m) for m in memory) > MAX_MEMORY_TOKENS
    ):
        memory.pop(0)


# ---------------------------------------------------------------------------
# Planner system prompt
# ---------------------------------------------------------------------------
PLANNER_SYSTEM = """
You are a Zoho Books automation agent. You decide which MCP tools to call and with what arguments.

INPUT (JSON):
  TOOLBOX  — tool_name → {desc, required, allowed, schema, read_only}
  STATE    — persisted values (organization_id, pending_intent, etc.)
  MEMORY   — recent tool call history, results, and errors
  USER     — the user's latest message

RULES:
1. NEVER invent or guess IDs. Look them up with list/get tools first if not in STATE or MEMORY.
2. Only use argument keys present in a tool's "allowed" list.
3. If STATE.organization_id is missing, ask the user for it (numeric Zoho org id) before calling tools that require it.
4. For body/JSONString fields, pass valid JSON constructed from context.
5. When collecting missing fields for a multi-step operation, do NOT call any tool — only ask.
6. When MEMORY contains a schema_error, fix the tool call using only allowed keys.
7. If a required entity (contact, item, etc.) is missing, ask the user to confirm creation.
8. For independent read-only steps (read_only: true), they may run in parallel — group them into a single plan step array.
9. Continue the active workflow until complete; never ask "What would you like to do?" mid-workflow.
10. Mark pending_intent in save{} when starting a multi-step workflow so context survives turns.

OUTPUT — one of exactly three forms (valid JSON object only, no prose):

A) Ask user for missing info:
{"type":"ask","text":"<question>","save":{"key":"value"}}

B) Execute tool steps:
{"type":"plan","steps":[{"tool":"ToolName","args":{...},"note":"optional","parallel":false},...]}
   parallel:true on a step means it can run concurrently with adjacent parallel:true steps.

C) Confirm before risky / irreversible action:
{"type":"confirm","text":"<description + impact>","on_yes":{"type":"plan","steps":[...]},"on_no":{"type":"ask","text":"..."}}
""".strip()


def _build_planner_payload(
    toolbox: dict[str, ToolMeta],
    state: AgentState,
    memory: list[dict],
    user_msg: str,
) -> str:
    return json.dumps(
        {
            "TOOLBOX": {n: m.to_planner_dict() for n, m in toolbox.items()},
            "STATE": state.to_dict(),
            "MEMORY": memory[-10:],
            "USER": user_msg,
        },
        ensure_ascii=False,
        default=str,
    )


async def gemini_plan(
    client: genai.Client,
    toolbox: dict[str, ToolMeta],
    state: AgentState,
    memory: list[dict],
    user_msg: str,
    correlation_id: str = "",
) -> dict:
    payload = _build_planner_payload(toolbox, state, memory, user_msg)
    log.debug("planner_request", extra={"cid": correlation_id, "payload_chars": len(payload)})

    async def _call() -> dict:
        resp = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=MODEL,
                contents=[types.Content(role="user", parts=[types.Part(text=payload)])],
                config=types.GenerateContentConfig(
                    system_instruction=PLANNER_SYSTEM,
                    temperature=0.4,
                    response_mime_type="application/json",
                ),
            ),
            timeout=GEMINI_TIMEOUT,
        )
        return json.loads(resp.text or "{}")

    try:
        result = await retry_async(_call, retries=3, base_delay=1.5, label="gemini_plan")
        log.debug("planner_response", extra={"cid": correlation_id, "type": result.get("type")})
        return result
    except Exception as exc:
        log.error("planner_failed", extra={"cid": correlation_id, "error": str(exc)})
        return {"type": "ask", "text": "I ran into a problem thinking through that. Please rephrase."}


# ---------------------------------------------------------------------------
# Structured rich renderer
# ---------------------------------------------------------------------------

SUMMARIZE_SYSTEM = """
You are a Zoho Books assistant. Given a tool call result, produce a clean terminal-friendly response.

Decide the best format based on the data:
- LIST of records (bills, invoices, contacts, items, payments, expenses, etc.) → use format "table"
- SINGLE record details (one bill, one invoice, one contact) → use format "panel"
- SIMPLE confirmation / status (created, deleted, sent, voided, etc.) → use format "status"
- FINANCIAL SUMMARY (totals, reports, P&L, aged payables/receivables) → use format "table"
- ERROR or warning → use format "status"

Return ONLY valid JSON in this exact shape:

For "table":
{
  "format": "table",
  "title": "string",
  "columns": ["Col1", "Col2", ...],
  "rows": [["val1", "val2", ...], ...],
  "footer": "optional string shown below table"
}

For "panel":
{
  "format": "panel",
  "title": "string",
  "fields": [["Label", "Value"], ...],
  "note": "optional string"
}

For "status":
{
  "format": "status",
  "ok": true,
  "headline": "short one-line message",
  "detail": "optional extra line"
}

Rules:
- Currency values: always include symbol (Rs., $, etc.)
- Dates: format as DD MMM YYYY
- Status values: capitalise (Open, Paid, Overdue, Draft, Void)
- Overdue or unpaid items: append a warning indicator, e.g. "Rs.50.00  Due Today"
- Never use markdown bold (**), asterisks (*), or backticks anywhere in values
- IDs: keep as-is, do not shorten
- Empty result sets: use format "status" with ok=false
""".strip()


def _render_structured(data: dict) -> str:
    """Render a structured Gemini response using rich. Returns headline string."""
    fmt = data.get("format", "status")

    if fmt == "table" and RICH_AVAILABLE:
        cols = data.get("columns") or []
        rows = data.get("rows") or []
        title = data.get("title", "")
        footer = data.get("footer", "")

        tbl = Table(
            title=title,
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            border_style="bright_black",
            pad_edge=True,
            expand=False,
            show_lines=False,
        )
        for col in cols:
            numeric = any(k in col.lower() for k in ("amount", "total", "balance", "qty", "price", "rate", "tax"))
            tbl.add_column(col, justify="right" if numeric else "left", no_wrap=True)

        for row in rows:
            styled_cells = []
            for cell in row:
                s = str(cell)
                sl = s.lower()
                if any(w in sl for w in ("overdue", "due today", "unpaid")):
                    styled_cells.append(f"[bold red]{rich_escape(s)}[/bold red]")
                elif sl in ("paid", "completed", "accepted", "closed"):
                    styled_cells.append(f"[green]{rich_escape(s)}[/green]")
                elif sl in ("open", "pending", "draft", "sent"):
                    styled_cells.append(f"[yellow]{rich_escape(s)}[/yellow]")
                elif sl in ("void", "voided", "cancelled", "canceled"):
                    styled_cells.append(f"[dim]{rich_escape(s)}[/dim]")
                else:
                    styled_cells.append(rich_escape(s))
            tbl.add_row(*styled_cells)

        _print_raw(tbl)
        if footer:
            _print(f"  [dim]{footer}[/dim]")
        _print("")
        return title

    elif fmt == "panel" and RICH_AVAILABLE:
        title = data.get("title", "Details")
        fields = data.get("fields") or []
        note = data.get("note", "")

        lines = []
        max_label = max((len(str(f[0])) for f in fields), default=10)
        for label, value in fields:
            v = str(value)
            vl = v.lower()
            if any(w in vl for w in ("overdue", "due today", "unpaid")):
                v_styled = f"[bold red]{rich_escape(v)}[/bold red]"
            elif vl in ("paid", "completed", "accepted"):
                v_styled = f"[green]{rich_escape(v)}[/green]"
            elif vl in ("open", "pending", "draft", "sent"):
                v_styled = f"[yellow]{rich_escape(v)}[/yellow]"
            elif vl in ("void", "voided"):
                v_styled = f"[dim]{rich_escape(v)}[/dim]"
            else:
                v_styled = rich_escape(v)
            padded_label = str(label).ljust(max_label)
            lines.append(f"  [bold]{rich_escape(padded_label)}[/bold]  {v_styled}")

        body = "\n".join(lines)
        if note:
            body += f"\n\n  [dim italic]{rich_escape(note)}[/dim italic]"

        _print_raw(Panel(
            body,
            title=f"[bold cyan]{rich_escape(title)}[/bold cyan]",
            border_style="bright_black",
            padding=(0, 1),
        ))
        _print("")
        return title

    else:
        # status format (or rich unavailable fallback)
        ok = data.get("ok", True)
        headline = str(data.get("headline", "Done."))
        detail = str(data.get("detail", ""))
        if RICH_AVAILABLE:
            icon = "[bold green]✓[/bold green]" if ok else "[bold yellow]![/bold yellow]"
            _print(f"\n{icon}  {rich_escape(headline)}")
            if detail:
                _print(f"   [dim]{rich_escape(detail)}[/dim]")
            _print("")
        else:
            print(f"\n{'✓' if ok else '!'} {headline}")
            if detail:
                print(f"  {detail}")
            print()
        return headline


async def _stream_collect_json(client: genai.Client, prompt: str, system: str, cid: str) -> str:
    """Stream JSON tokens from Gemini, show a subtle indicator, return full text."""
    full = ""
    started = False
    try:
        async with asyncio.timeout(GEMINI_TIMEOUT):
            async for chunk in await client.aio.models.generate_content_stream(
                model=MODEL,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            ):
                token = chunk.text or ""
                if not token:
                    continue
                if not started:
                    started = True
                full += token
        return full
    except Exception as exc:
        log.warning("stream_collect_failed", extra={"cid": cid, "error": str(exc)})
        resp = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=MODEL,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            ),
            timeout=GEMINI_TIMEOUT,
        )
        return resp.text or "{}"


async def gemini_summarize(
    client: genai.Client,
    tool: str,
    args: dict,
    result: Any,
    state: AgentState,
    correlation_id: str = "",
) -> str:
    """
    Ask Gemini (streaming) to choose the best display format for the result,
    then render it using rich tables / panels / status blocks.
    """
    prompt = (
        f"TOOL: {tool}\n"
        f"ARGS: {json.dumps(args, default=str)}\n"
        f"RESULT: {json.dumps(result, default=str)}"
    )

    raw = await _stream_collect_json(client, prompt, SUMMARIZE_SYSTEM, correlation_id)

    # Strip possible markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        data = json.loads(cleaned)
        return _render_structured(data)
    except Exception:
        # Last resort: just print whatever came back
        _print(f"\n[green]Assistant:[/green] {cleaned}\n")
        return cleaned



def normalize_plan(plan: Any) -> dict:
    if isinstance(plan, list):
        return {"type": "plan", "steps": plan}
    if not isinstance(plan, dict):
        return {"type": "ask", "text": "I couldn't understand the plan. Please rephrase."}
    if "type" not in plan:
        if "steps" in plan and isinstance(plan["steps"], list):
            plan["type"] = "plan"
        elif "text" in plan:
            plan["type"] = "ask"
        else:
            plan["type"] = "ask"
            plan["text"] = "I need a bit more detail to proceed."
    return plan


# ---------------------------------------------------------------------------
# Step executor
# ---------------------------------------------------------------------------
async def execute_step(
    session: ClientSession,
    toolbox: dict[str, ToolMeta],
    state: AgentState,
    memory: list[dict],
    audit: AuditLog,
    step: dict,
    correlation_id: str,
) -> tuple[bool, Optional[Any]]:
    """
    Execute one plan step. Returns (success, result_data).
    On failure records to memory and returns (False, None).
    """
    tool_name: str = step.get("tool", "")
    args: dict = dict(step.get("args") or {})

    # Unknown tool
    if not tool_name or tool_name not in toolbox:
        msg = f"Unknown tool: '{tool_name}'"
        add_memory(memory, {"error": msg, "step": step, "cid": correlation_id})
        log.warning("unknown_tool", extra={"tool": tool_name, "cid": correlation_id})
        return False, None

    meta = toolbox[tool_name]

    # Circuit breaker
    cb = _circuit_breakers[tool_name]
    if cb.is_open():
        msg = f"Tool '{tool_name}' is temporarily disabled (circuit open after repeated failures)."
        _print(f"[bold red]Assistant:[/bold red] {msg}\n")
        add_memory(memory, {"error": msg, "tool": tool_name, "cid": correlation_id})
        return False, None

    # Auto-inject organization_id
    if "organization_id" in meta.required and "organization_id" not in args:
        if state.organization_id:
            args["organization_id"] = state.organization_id

    # Schema validation
    unknown, missing = validate_args(meta, args)
    if unknown or missing:
        add_memory(memory, {
            "tool": tool_name,
            "args": args,
            "cid": correlation_id,
            "schema_error": {
                "unknown_keys": unknown,
                "missing_required": missing,
                "allowed": meta.allowed[:40],
                "required": meta.required,
            },
        })
        log.warning("schema_error", extra={"tool": tool_name, "unknown": unknown, "missing": missing})
        return False, None

    # Persist org_id
    if org := args.get("organization_id"):
        state.organization_id = str(org)

    # Call MCP tool with timeout + retry
    log.info("tool_call", extra={"tool": tool_name, "cid": correlation_id})
    t0 = time.monotonic()
    try:
        result = await retry_async(
            asyncio.wait_for,
            session.call_tool(tool_name, args),
            timeout=TOOL_TIMEOUT,
            retries=2,
            base_delay=1.0,
            label=tool_name,
        )
    except asyncio.TimeoutError:
        msg = f"Tool '{tool_name}' timed out after {TOOL_TIMEOUT}s."
        add_memory(memory, {"tool": tool_name, "error": msg, "cid": correlation_id})
        cb.record_failure()
        log.error("tool_timeout", extra={"tool": tool_name, "cid": correlation_id})
        _print(f"[bold yellow]Assistant:[/bold yellow] {msg}\n")
        return False, None
    except Exception as exc:
        msg = str(exc)
        add_memory(memory, {"tool": tool_name, "args": args, "error": msg, "cid": correlation_id})
        cb.record_failure()
        log.error("tool_error", extra={"tool": tool_name, "error": msg, "cid": correlation_id})
        _print(f"[bold red]Assistant:[/bold red] Tool call failed — {msg}\n")
        return False, None

    elapsed = round(time.monotonic() - t0, 2)
    cb.record_success()
    result_data = result.model_dump() if hasattr(result, "model_dump") else result

    add_memory(memory, {
        "tool": tool_name,
        "args": {k: args[k] for k in list(args)[:12]},
        "result": result_data,
        "elapsed_s": elapsed,
        "cid": correlation_id,
    })
    audit.write("tool_called", tool=tool_name, args=args, elapsed_s=elapsed, cid=correlation_id)
    log.info("tool_success", extra={"tool": tool_name, "elapsed_s": elapsed, "cid": correlation_id})
    return True, result_data


async def execute_plan(
    session: ClientSession,
    toolbox: dict[str, ToolMeta],
    state: AgentState,
    memory: list[dict],
    audit: AuditLog,
    gemini: genai.Client,
    steps: list[dict],
    correlation_id: str,
) -> tuple[bool, Optional[Any]]:
    """
    Execute all steps. Handles parallel batches. Returns (all_ok, last_result_data).
    """
    last_result: Optional[Any] = None
    last_tool: str = ""

    i = 0
    while i < len(steps):
        step = steps[i]

        # Collect a parallel batch
        if step.get("parallel"):
            batch = [step]
            j = i + 1
            while j < len(steps) and steps[j].get("parallel"):
                batch.append(steps[j])
                j += 1

            log.info("parallel_batch", extra={"size": len(batch), "cid": correlation_id})
            tasks = [
                execute_step(session, toolbox, state, memory, audit, s, correlation_id)
                for s in batch
            ]
            results = await asyncio.gather(*tasks)

            for (ok, rdata), s in zip(results, batch):
                if not ok:
                    return False, None
                last_result = rdata
                last_tool = s.get("tool", "")

            i = j
            continue

        # Sequential step
        ok, rdata = await execute_step(session, toolbox, state, memory, audit, step, correlation_id)
        if not ok:
            return False, None
        last_result = rdata
        last_tool = step.get("tool", "")
        i += 1

    return True, last_result


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def is_number_string(s: str) -> bool:
    return bool(re.fullmatch(r"\d{6,}", (s or "").strip()))


def validate_env() -> None:
    missing = [k for k in REQUIRED_ENV_VARS if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------
async def main() -> None:
    validate_env()

    gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    zoho_url = ""
    while not zoho_url:
        zoho_url = (await asyncio.get_event_loop().run_in_executor(
            None, input, "Enter MCP URL: "
        )).strip()
        if not zoho_url:
            _print("[yellow]MCP URL cannot be empty.[/yellow]")
    zoho_bearer: Optional[str] = os.getenv("ZOHO_MCP_BEARER")
    default_org_id: Optional[str] = os.getenv("ZOHO_ORG_ID", "60065733225")

    headers: dict[str, str] = {}
    if zoho_bearer:
        headers["Authorization"] = f"Bearer {zoho_bearer}"

    state = AgentState(organization_id=default_org_id)
    memory: list[dict] = []
    audit = AuditLog(AUDIT_LOG_PATH)
    pending_confirm: Optional[dict] = None
    replan_attempts: int = 0
    shutdown_event = asyncio.Event()

    # Graceful shutdown
    def _handle_signal(sig: int, _frame: Any) -> None:
        _print("\n[bold yellow]Shutting down gracefully…[/bold yellow]")
        shutdown_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info("agent_starting", extra={"model": MODEL, "org_id": default_org_id})

    async with httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(TOOL_TIMEOUT + 10)) as http_client:
        async with streamable_http_client(zoho_url, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                list_tools_result = await session.list_tools()
                toolbox = build_tool_catalog(list_tools_result)

                _print("\n[bold green]✓ Zoho Books Assistant ready[/bold green]")
                _print(f"  [dim]Tools loaded: {len(toolbox)} | Model: {MODEL} | Org: {state.organization_id}[/dim]")
                _print("  [dim]Type 'quit' to exit | 'tools' to list available tools | 'memory' to inspect context[/dim]\n")

                while not shutdown_event.is_set():
                    # Async-friendly input
                    try:
                        user_msg = await asyncio.get_event_loop().run_in_executor(None, input, "You: ")
                    except EOFError:
                        break
                    user_msg = user_msg.strip()

                    if not user_msg:
                        continue

                    # Built-in commands
                    if user_msg.lower() in {"quit", "exit", "q"}:
                        break

                    if user_msg.lower() == "tools":
                        _print("\n[bold]Available tools:[/bold]")
                        for name, meta in sorted(toolbox.items()):
                            tag = "[dim](read)[/dim]" if meta.read_only else "[yellow](write)[/yellow]"
                            _print(f"  {tag} {name} — {meta.desc}")
                        _print("")
                        continue

                    if user_msg.lower() == "memory":
                        _print(f"\n[bold]Memory ({len(memory)} entries, ~{sum(_estimate_tokens(m) for m in memory)} tokens):[/bold]")
                        _print(json.dumps(memory, indent=2, default=str))
                        _print("")
                        continue

                    if user_msg.lower() == "state":
                        _print("\n[bold]State:[/bold]")
                        _print(json.dumps(state.to_dict(), indent=2))
                        _print("")
                        continue

                    # Bare org ID shortcut
                    if is_number_string(user_msg) and not state.organization_id:
                        state.organization_id = user_msg
                        _print(f"[green]Assistant:[/green] Organization ID saved: {user_msg}. What would you like to do?\n")
                        audit.write("org_id_set", org_id=user_msg)
                        continue

                    # New correlation ID per user turn
                    cid = uuid.uuid4().hex[:10]
                    audit.write("user_message", msg=user_msg, cid=cid)
                    log.info("user_turn", extra={"cid": cid, "msg_len": len(user_msg)})

                    # ----------------------------------------------------------
                    # Handle pending confirmation
                    # ----------------------------------------------------------
                    plan: dict
                    if pending_confirm:
                        answer = user_msg.lower()
                        if answer in {"yes", "y"}:
                            plan = pending_confirm["on_yes"]
                            pending_confirm = None
                            audit.write("confirm_accepted", cid=cid)
                        elif answer in {"no", "n"}:
                            no_obj: dict = pending_confirm.get("on_no") or {}
                            pending_confirm = None
                            audit.write("confirm_rejected", cid=cid)
                            _print(f"[green]Assistant:[/green] {no_obj.get('text', 'Cancelled. What would you like instead?')}\n")
                            continue
                        else:
                            _print("[green]Assistant:[/green] Please reply [bold]YES[/bold] or [bold]NO[/bold].\n")
                            continue
                    else:
                        raw = await gemini_plan(gemini, toolbox, state, memory, user_msg, cid)
                        plan = normalize_plan(raw)
                        replan_attempts = 0

                    # ----------------------------------------------------------
                    # Dispatch
                    # ----------------------------------------------------------
                    ptype = plan.get("type")

                    if ptype == "ask":
                        state.update_from_dict(plan.get("save") or {})
                        _print(f"[green]Assistant:[/green] {plan.get('text', 'I need more details.')}\n")
                        audit.write("ask", text=plan.get("text"), cid=cid)
                        continue

                    if ptype == "confirm":
                        pending_confirm = {
                            "on_yes": plan.get("on_yes"),
                            "on_no": plan.get("on_no"),
                        }
                        _print(f"[bold yellow]Assistant:[/bold yellow] {plan.get('text', 'Confirm?')} [bold](YES / NO)[/bold]\n")
                        audit.write("confirm_requested", text=plan.get("text"), cid=cid)
                        continue

                    if ptype != "plan":
                        _print("[green]Assistant:[/green] I couldn't determine an action. Please rephrase.\n")
                        continue

                    steps: list[dict] = plan.get("steps") or []
                    if not steps:
                        _print("[green]Assistant:[/green] Plan had no steps. Please rephrase.\n")
                        continue

                    # Risky check before any execution
                    risky_steps = [s for s in steps if is_risky(s.get("tool", ""))]
                    if risky_steps and not pending_confirm:
                        risky_names = ", ".join(s["tool"] for s in risky_steps)
                        pending_confirm = {
                            "on_yes": plan,
                            "on_no": {"type": "ask", "text": "Cancelled. What would you like instead?"},
                        }
                        _print(
                            f"[bold yellow]Assistant:[/bold yellow] This involves sensitive operations: "
                            f"[bold]{risky_names}[/bold]. Reply [bold]YES[/bold] to confirm or [bold]NO[/bold] to cancel.\n"
                        )
                        audit.write("risky_confirm_requested", tools=risky_names, cid=cid)
                        continue

                    # Execute
                    _print("[dim]Assistant: Working…[/dim]")
                    ok, last_result = await execute_plan(
                        session, toolbox, state, memory, audit, gemini, steps, cid
                    )

                    if ok and last_result is not None:
                        last_tool = steps[-1].get("tool", "")
                        await gemini_summarize(gemini, last_tool, steps[-1].get("args", {}), last_result, state, cid)
                        state.clear_workflow()
                        replan_attempts = 0
                        continue

                    # ----------------------------------------------------------
                    # Replan on failure
                    # ----------------------------------------------------------
                    if replan_attempts >= MAX_REPLAN_ATTEMPTS:
                        _print("[bold red]Assistant:[/bold red] I've tried multiple times but can't complete this. Please rephrase or provide more information.\n")
                        replan_attempts = 0
                        continue

                    replan_attempts += 1
                    log.info("replanning", extra={"attempt": replan_attempts, "cid": cid})

                    replan_raw = await gemini_plan(
                        gemini,
                        toolbox,
                        state,
                        memory,
                        "Fix the plan using MEMORY errors. Ask only for genuinely missing required fields.",
                        cid,
                    )
                    replan = normalize_plan(replan_raw)
                    rtype = replan.get("type")

                    if rtype == "ask":
                        state.update_from_dict(replan.get("save") or {})
                        _print(f"[green]Assistant:[/green] {replan.get('text', 'I need more details.')}\n")

                    elif rtype == "confirm":
                        pending_confirm = {
                            "on_yes": replan.get("on_yes"),
                            "on_no": replan.get("on_no"),
                        }
                        _print(f"[bold yellow]Assistant:[/bold yellow] {replan.get('text', 'Confirm?')} [bold](YES / NO)[/bold]\n")

                    elif rtype == "plan":
                        replan_steps = replan.get("steps") or []
                        has_risky = any(is_risky(s.get("tool", "")) for s in replan_steps)
                        tools_preview = " → ".join(s.get("tool", "?") for s in replan_steps)
                        if has_risky:
                            pending_confirm = {
                                "on_yes": replan,
                                "on_no": {"type": "ask", "text": "Cancelled. Tell me what to change."},
                            }
                            _print(f"[bold yellow]Assistant:[/bold yellow] Replanned with sensitive steps: {tools_preview}. Reply [bold]YES[/bold] to proceed.\n")
                        else:
                            pending_confirm = {
                                "on_yes": replan,
                                "on_no": {"type": "ask", "text": "Okay — tell me what to change."},
                            }
                            _print(f"[green]Assistant:[/green] Replanned: {tools_preview}. Reply [bold]YES[/bold] to proceed or [bold]NO[/bold] to change.\n")
                    else:
                        _print("[bold red]Assistant:[/bold red] I'm stuck. Please rephrase what you'd like to do.\n")

    audit.close()
    _print("[dim]Session ended. Goodbye.[/dim]")
    log.info("agent_stopped")


if __name__ == "__main__":

    asyncio.run(main())

