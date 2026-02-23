"""
zoho_agent.py — Production-ready Zoho Books MCP Agent
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
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
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
    from rich import box
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

MODEL: str                   = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
PLANNER_MODEL: str           = os.getenv("GEMINI_PLANNER_MODEL", MODEL)
MAX_MEMORY_TOKENS: int       = int(os.getenv("MAX_MEMORY_TOKENS", "24000"))
MAX_MEMORY_ENTRIES: int      = int(os.getenv("MAX_MEMORY_ENTRIES", "60"))
MEMORY_SUMMARIZE_THRESHOLD   = int(os.getenv("MEMORY_SUMMARIZE_THRESHOLD", "45"))
MAX_TOOL_PROPS: int          = int(os.getenv("MAX_TOOL_PROPS", "60"))
MAX_REPLAN_ATTEMPTS: int     = int(os.getenv("MAX_REPLAN_ATTEMPTS", "3"))
AUDIT_LOG_PATH: str          = os.getenv("AUDIT_LOG_PATH", "audit.jsonl")
LOG_LEVEL: str               = os.getenv("LOG_LEVEL", "WARNING")
TOOL_TIMEOUT: float          = float(os.getenv("TOOL_TIMEOUT_SECONDS", "30"))
GEMINI_TIMEOUT: float        = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "60"))
CIRCUIT_BREAK_THRESHOLD: int = int(os.getenv("CIRCUIT_BREAK_THRESHOLD", "3"))
CIRCUIT_BREAK_RESET_S: int   = int(os.getenv("CIRCUIT_BREAK_RESET_SECONDS", "120"))
SUMMARIZE_INPUT_CHAR_LIMIT: int = int(os.getenv("SUMMARIZE_INPUT_CHAR_LIMIT", "400000"))

REQUIRED_ENV_VARS = ["GEMINI_API_KEY"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_RECORD_BUILTINS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName", "asctime",
})

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extra = {
            k: v for k, v in record.__dict__.items()
            if k not in _LOG_RECORD_BUILTINS and not k.startswith("_")
        }
        out: dict = {
            "ts":     datetime.now(timezone.utc).isoformat(),
            "level":  record.levelname,
            "msg":    record.getMessage(),
            "logger": record.name,
            **extra,
        }
        if record.exc_info and record.exc_info[0] is not None:
            out["exception"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)

def _setup_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    logger = logging.getLogger("zoho_agent")
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.WARNING))
    logger.addHandler(handler)
    logger.propagate = False
    return logger

log = _setup_logging()

def _resolve_value(template: str, step_results: list[Any]) -> Any:
    """
    Resolve a template string like "{{steps[0].contacts[0].contact_id}}"
    against the list of previous step results.
    """
    match = re.fullmatch(r'\{\{(.+?)\}\}', template.strip())
    if not match:
        return template

    path = match.group(1).strip()
    tokens = path.split('.')
    current: Any = {"steps": step_results}

    for token in tokens:
        arr_match = re.fullmatch(r'(\w+)\[(\d+)\]', token)
        try:
            if arr_match:
                key, idx = arr_match.group(1), int(arr_match.group(2))
                if isinstance(current, dict):
                    current = current[key][idx]
                elif isinstance(current, list):
                    current = current[int(idx)]
            else:
                if isinstance(current, dict):
                    current = current[token]
                elif isinstance(current, list):
                    current = current[int(token)]
        except (KeyError, IndexError, TypeError, ValueError):
            return template

    return current


def _resolve_args(args: dict, step_results: list[Any]) -> dict:
    """Walk all args and resolve any {{...}} template values."""
    resolved = {}
    for k, v in args.items():
        if isinstance(v, str) and '{{' in v:
            resolved[k] = _resolve_value(v, step_results)
        elif isinstance(v, dict):
            resolved[k] = _resolve_args(v, step_results)
        elif isinstance(v, list):
            resolved[k] = [
                _resolve_value(item, step_results) if isinstance(item, str) and '{{' in item
                else item
                for item in v
            ]
        else:
            resolved[k] = v
    return resolved

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
    "create_vendor_payment", "refund_excess_vendor_payment", "refund_vendor_credit",
    "create_customer_payment_refund", "delete_contact", "delete_invoice",
    "delete_bill", "delete_item", "delete_expense", "delete_customer_payment",
    "delete_vendor_payment", "delete_account", "void_invoice", "void_bill",
    "void_salesorder",
})
RISKY_SUBSTR: frozenset[str] = frozenset({"delete", "void", "write_off", "refund"})

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
            log.warning("retry", extra={"label": label, "attempt": attempt + 1,
                                        "wait_s": wait, "error": str(exc)})
            await asyncio.sleep(wait)
        except Exception:
            raise
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
            "desc":      self.desc,
            "required":  self.required,
            "allowed":   self.allowed[:40],
            "schema":    self.schema_compact,
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
        if self.organization_id:  d["organization_id"] = self.organization_id
        if self.pending_intent:   d["pending_intent"]  = self.pending_intent
        if self.pending_question: d["pending_question"] = self.pending_question
        d.update(self.extra)
        return d

    def update_from_dict(self, data: dict) -> None:
        for k, v in data.items():
            if v in (None, ""):
                continue
            if k == "organization_id":  self.organization_id = str(v)
            elif k == "pending_intent":   self.pending_intent  = str(v)
            elif k == "pending_question": self.pending_question = str(v)
            else:                         self.extra[k] = v

    def clear_workflow(self) -> None:
        self.pending_intent  = None
        self.pending_question = None


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _slim_memory_for_planner(memory_entries: list) -> list:
    """
    Prepare memory entries for the planner context window.

    Since add_memory now stores compact summaries (not raw result arrays),
    entries are already small enough to send as-is. We just pass them
    through, dropping any oversized fields defensively.
    """
    slimmed = []
    for entry in memory_entries:
        if not isinstance(entry, dict):
            slimmed.append(entry)
            continue
        # Entries already have result_summary (compact) instead of result (raw).
        # Drop any stale raw "result" key if somehow still present.
        e = {k: v for k, v in entry.items() if k != "result"}
        slimmed.append(e)
    return slimmed


def compact_schema(schema: dict, max_props: int = MAX_TOOL_PROPS) -> dict:
    if not schema:
        return {}
    props    = schema.get("properties") or {}
    req      = schema.get("required") or []
    keys     = list(props.keys())[:max_props]
    overflow = len(props) - len(keys)
    slim     = {k: {"type": props[k].get("type")} if isinstance(props[k], dict) else {}
                for k in keys}
    out: dict = {"type": schema.get("type"), "required": [k for k in req if k in slim],
                 "properties": slim}
    if overflow > 0:
        out["_truncated"] = overflow
    return out


def build_tool_catalog(list_tools_result: Any) -> dict[str, ToolMeta]:
    catalog: dict[str, ToolMeta] = {}
    for t in list_tools_result.tools:
        schema   = t.inputSchema or {}
        props    = schema.get("properties") or {}
        required = schema.get("required") or []
        allowed  = sorted(props.keys())

        body_key: Optional[str] = None
        body_required: list[str] = []
        for key in ("body", "JSONString"):
            if key in props and isinstance(props[key], dict):
                body_key      = key
                body_required = props[key].get("required") or []
                break

        catalog[t.name] = ToolMeta(
            name           = t.name,
            desc           = (t.description or "").strip()[:160],
            required       = [k for k in required if k in props],
            allowed        = allowed,
            body_key       = body_key,
            body_required  = body_required,
            schema_compact = compact_schema(schema),
            read_only      = is_read_only(t.name),
        )
    log.info("catalog_built", extra={"tool_count": len(catalog)})
    return catalog


def validate_args(meta: ToolMeta, args: dict) -> tuple[list[str], list[str]]:
    allowed_set = set(meta.allowed)
    unknown     = [k for k in args if k not in allowed_set]
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
# Memory
# ---------------------------------------------------------------------------
def _estimate_tokens(obj: Any) -> int:
    return max(1, len(json.dumps(obj, default=str)) // 4)


MEMORY_SUMMARIZE_SYSTEM = """
You are a memory compressor for a Zoho Books AI agent.
You receive compact tool-call summaries (tool name, args, result_summary with record counts
and sample IDs, errors). Raw result arrays are never stored in memory.

Produce a single compact JSON with the key facts needed to continue the conversation.
Preserve all entity IDs and numbers — they are needed for single-entity follow-up queries.

Return ONLY this JSON — no prose, no markdown:
{
  "type": "summary",
  "covers_entries": <count>,
  "entity_ids": {
    "invoices":  [{"invoice_id":"...","invoice_number":"...","customer_name":"..."},...],
    "bills":     [{"bill_id":"...","bill_number":"...","vendor_name":"..."},...],
    "contacts":  [{"contact_id":"...","contact_name":"...","email":"..."},...],
    "items":     [{"item_id":"...","name":"..."},...],
    "other":     [{"entity":"...","id":"..."}]
  },
  "last_fetch": {
    "tool": "<tool_name>",
    "record_count": <N>,
    "note": "<what was asked>"
  },
  "actions_taken": ["<verb> <entity> <id_or_number>", ...],
  "errors": ["<brief>", ...]
}
""".strip()


async def summarize_memory_async(
    client: genai.Client,
    entries: list[dict],
    cid: str = "",
) -> dict:
    prompt = "Compress these memory entries:\n" + json.dumps(entries, default=str)
    try:
        raw     = await _stream_collect_json(client, prompt, MEMORY_SUMMARIZE_SYSTEM, cid)
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
        parsed  = json.loads(cleaned)
        parsed["covers_entries"] = len(entries)
        log.info("memory_summarized", extra={"entries": len(entries), "cid": cid})
        return parsed
    except Exception as exc:
        log.warning("memory_summarize_failed", extra={"error": str(exc), "cid": cid})
        actions, errors = [], []
        for e in entries:
            if e.get("tool"): actions.append(e["tool"])
            if "error" in e:  errors.append(str(e["error"])[:120])
        return {"type": "summary", "covers_entries": len(entries),
                "facts": {"actions_taken": actions, "errors": errors}}


def add_memory(
    memory: list,
    entry: dict,
    gemini_client=None,
    cid: str = "",
):
    """
    Store a CONVERSATION-LEVEL memory entry.

    Architecture (new):
    - Memory stores WHAT WAS ASKED and WHAT WAS FOUND (counts, IDs, names).
    - Memory does NOT store raw API result arrays — those are too large and
      cause the "only 10 records" problem where trimmed samples were reused.
    - Fresh data is ALWAYS fetched from MCP when a query needs it.
    - The planner uses memory only to know: what entities exist, what IDs
      were seen, what was done, and what errors occurred.

    Entry shapes stored:
      Tool call:   {"tool": "...", "args": {...}, "record_count": N,
                    "entity_ids": [...], "error": "..." (if failed)}
      User turn:   {"role": "user", "content": "..."}
      Bot turn:    {"role": "assistant", "summary": "...", "tool": "..."}
      Error:       {"error": "...", "tool": "..."}
    """
    # If this is a tool-result entry, strip the raw result array and keep
    # only a compact summary (record count + a few IDs for chaining).
    if isinstance(entry, dict) and "result" in entry and isinstance(entry.get("result"), dict):
        result = entry["result"]
        best_key, best_arr = "", []
        for k, v in result.items():
            if isinstance(v, list) and len(v) > len(best_arr):
                best_key, best_arr = k, v

        # Build a compact summary instead of storing the raw array
        compact: dict = {}
        if best_arr:
            compact["record_count"] = len(best_arr)
            compact["record_key"]   = best_key
            # Keep up to 5 IDs/names so the planner can reference them
            id_fields = ("invoice_id","bill_id","contact_id","item_id",
                         "payment_id","expense_id","estimate_id",
                         "salesorder_id","purchaseorder_id","creditnote_id")
            name_fields = ("invoice_number","bill_number","contact_name",
                           "customer_name","vendor_name","name","number")
            samples = []
            for rec in best_arr[:5]:
                if not isinstance(rec, dict):
                    continue
                s = {}
                for f in (*id_fields, *name_fields, "total", "balance", "status"):
                    if f in rec:
                        s[f] = rec[f]
                if s:
                    samples.append(s)
            if samples:
                compact["sample_ids"] = samples
        else:
            # Single-record result (get_invoice etc.) — keep key fields
            compact = {k: v for k, v in result.items()
                       if not isinstance(v, (list, dict)) and k not in
                       ("organization_id","template_id","color_code")}

        entry = {k: v for k, v in entry.items() if k != "result"}
        entry["result_summary"] = compact

    memory.append(entry)

    # Trim old entries to stay within token budget
    total_tokens = sum(_estimate_tokens(m) for m in memory)
    over_entries = len(memory) > MAX_MEMORY_ENTRIES
    over_tokens  = total_tokens > MAX_MEMORY_TOKENS

    if not (over_entries or over_tokens):
        return None

    # Trim oldest entries first (keep most recent context)
    while len(memory) > max(4, MAX_MEMORY_ENTRIES):
        memory.pop(0)
    while len(memory) > 4 and sum(_estimate_tokens(m) for m in memory) > MAX_MEMORY_TOKENS:
        memory.pop(0)
    return None


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

# FIX: Completely rewritten PLANNER_SYSTEM.
#
# The original had three problems that caused wrong tool selection:
#
# 1. No guidance on HOW to choose between similar tools.
#    e.g. list_invoices vs search_invoices vs get_invoice — the model guessed.
#
# 2. "NEVER pass filters" instruction was contradicted by the summarizer
#    prompt which told Gemini to filter. The planner didn't understand
#    that it fetches everything and the summarizer filters later.
#
# 3. No instruction to use memory entity IDs for chained operations,
#    so follow-up queries like "now get details for the first one" would
#    re-fetch everything instead of using the ID from memory.
#
# Fixed version:
# - Explicit tool-selection hierarchy (list > search > get)
# - Clear explanation of the fetch-all-then-filter pattern
# - Explicit instruction to use IDs from memory for single-entity lookups
# - Better handling of multi-step plans with dependency references

PLANNER_SYSTEM = """
You are a Zoho Books agent planner. Given a user request, MEMORY of past
conversation context, and a TOOLBOX, produce a JSON execution plan.

MEMORY contains compact summaries of past tool calls — record counts,
sample IDs, and what actions were taken. It does NOT contain raw data.
All raw data must be fetched fresh from the MCP tools when needed.

==============================================
OUTPUT FORMAT — return exactly ONE JSON shape:
==============================================

  Ask for missing info:
    {"type":"ask","text":"<question>","save":{}}

  Execute tools:
    {"type":"plan","steps":[{"tool":"<name>","args":{...},"note":"<user question verbatim>"}]}

  Confirm risky operations (delete/void/refund/send):
    {"type":"confirm","text":"<what will happen>","on_yes":{...plan...},"on_no":{...}}

========================================
TOOL SELECTION — follow this hierarchy:
========================================

1. SINGLE ENTITY — user names a specific record by ID or number:
   → Check MEMORY for the ID (result_summary.sample_ids)
   → If found: use get_<entity> directly with that ID
   → If not found: use search_<entity> to find it first
   → NEVER use list_<entity> when you only need one record

2. LISTING / FILTERING / AGGREGATION — user wants a set or a total:
   → ALWAYS fetch fresh: use list_<entity> with per_page=200
   → Do NOT pass filter params to Zoho (date, customer_name, status, etc.)
   → The summarizer receives the full result and applies all filters
   → This applies even if the same entity was listed in a previous turn —
     always re-fetch to get accurate, complete data
   → Exception: if MEMORY shows the previous turn listed the same entity
     AND the user is refining (e.g. "now filter that for January") AND
     the session is less than 5 minutes old — you MAY re-use by adding
     a note "REUSE_LAST_RESULT" to the step (server handles it)

3. SEARCH BY NAME — user mentions a customer/vendor without an ID:
   → Search memory first for a matching contact_id
   → If not in memory, use search_contacts to get the contact_id

4. MULTI-STEP — chain results between steps:
   → Use {{steps[N].field_name}} to reference earlier step results

5. PAGINATION — always include per_page=200 in list_* calls.

================================
WHEN TO FETCH VS WHEN TO REUSE:
================================

ALWAYS FETCH FRESH:
  - "total amount from punjab national bank" after listing invoices
    → re-fetch list_invoices (the previous list was trimmed in memory,
      fresh fetch guarantees all records are included)
  - Any aggregate or filter query on a large dataset
  - First time asking about an entity type

USE MEMORY IDs (skip list, use get_<entity>):
  - "get details for INV-001" → get_invoice with invoice_id from memory
  - "update that contact" → get contact_id from memory
  - "void invoice #INV-123" → find invoice_id in memory

============================
ARGUMENT CONSTRUCTION RULES:
============================

* Check schema "required" — include every required arg.
* organization_id is almost always required — take from STATE.
* Some tools need a "body" or "JSONString" wrapper — check schema.
* Never invent arg names not in "allowed".
* Set "note" to the user's exact question verbatim.
""".strip()


def _slim_toolbox(toolbox: dict[str, ToolMeta], user_msg: str, memory: list[dict]) -> dict:
    """
    Return a toolbox dict for the planner with full schemas for relevant tools
    and brief descriptions for everything else.

    FIX over original: The original returned only {"desc", "read_only"} for
    non-relevant tools, meaning if domain-hint detection missed, the planner
    got zero schema info and hallucinated arguments.

    New approach:
    - Relevant tools (domain match OR recently used): full schema
    - All other tools: description + required args only (not just desc)
      so the planner can at least construct valid calls if it chooses them
    """
    user_lower = user_msg.lower()
    recent_tools = {m.get("tool", "") for m in memory[-10:] if "tool" in m}

    # Broader domain hints — more keywords, catch more edge cases
    DOMAIN_HINTS = [
        ({"invoice", "invoic", "receivable", "receive", "customer payment",
          "outstanding", "overdue", "collect"}, "invoice"),
        ({"bill", "payable", "vendor payment", "pay vendor", "supplier"}, "bill"),
        ({"contact", "customer", "vendor", "supplier", "client"}, "contact"),
        ({"item", "product", "service", "inventory", "sku"}, "item"),
        ({"expense"}, "expense"),
        ({"payment", "paid", "receipt"}, "payment"),
        ({"credit", "credit note", "creditnote"}, "credit"),
        ({"bank", "transaction", "reconcil", "statement"}, "bank"),
        ({"report", "summary", "p&l", "profit", "loss", "balance sheet",
          "trial", "aging", "ageing"}, "report"),
        ({"account", "chart of accounts", "ledger"}, "account"),
        ({"tax", "gst", "vat", "tds"}, "tax"),
        ({"estimate", "quote", "quotation"}, "estimate"),
        ({"purchase order", "po ", "purchaseorder"}, "purchaseorder"),
        ({"sales order", "so ", "salesorder", "delivery"}, "salesorder"),
        ({"journal", "journal entry"}, "journal"),
    ]

    relevant_substrings: set[str] = set()
    for keywords, tool_substr in DOMAIN_HINTS:
        if any(kw in user_lower for kw in keywords):
            relevant_substrings.add(tool_substr)

    slim: dict = {}
    for name, meta in toolbox.items():
        name_lower = name.lower()
        is_recent  = name in recent_tools
        is_relevant = not relevant_substrings or any(s in name_lower for s in relevant_substrings)

        if is_recent or is_relevant:
            # Full schema for tools the planner is likely to use
            slim[name] = meta.to_planner_dict()
        else:
            # Fallback: description + required args so the planner can still
            # use this tool correctly if it decides to — just with less detail
            slim[name] = {
                "desc":      meta.desc[:100],
                "required":  meta.required,
                "read_only": meta.read_only,
            }

    return slim


def _build_planner_payload(
    toolbox: dict,
    state,
    memory: list,
    user_msg: str,
) -> str:
    # Send last 20 memory entries to the planner (up from 12).
    # A 3-step plan consumes 3 memory slots; with replanning that's 6+.
    # 12 entries was too few for multi-turn conversations.
    return json.dumps(
        {
            "TOOLBOX": _slim_toolbox(toolbox, user_msg, memory),
            "STATE":   state.to_dict(),
            "MEMORY":  _slim_memory_for_planner(memory[-20:]),
            "USER":    user_msg,
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
                model=PLANNER_MODEL,
                contents=[types.Content(role="user", parts=[types.Part(text=payload)])],
                config=types.GenerateContentConfig(
                    system_instruction=PLANNER_SYSTEM,
                    temperature=0.3,  # lower = more deterministic tool selection
                    response_mime_type="application/json",
                    max_output_tokens=2048,
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
        log.error("planner_failed", extra={"cid": correlation_id, "error": str(exc),
                                           "error_type": type(exc).__name__})
        return {"type": "ask", "text": "I ran into a problem thinking through that. Please rephrase."}


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------

# FIX: SUMMARIZE_SYSTEM rewritten to improve response quality and formatting.
#
# Problems with the original:
# 1. The filtering rules were correct but underspecified — partial name matches
#    would fail silently and return "no records" for valid queries.
# 2. The output format section was too rigid; Gemini would sometimes pick the
#    wrong format (e.g. "answer" for a list query).
# 3. Currency formatting was ₹ only — no flexibility for other orgs.
# 4. No guidance on what to do when the result is a single record (detail view).
# 5. Missing instruction for action results (create/update/delete responses).

SUMMARIZE_SYSTEM = """
You are a Zoho Books data formatter. You receive a RAW RESULT (already fetched from
the API) and the user's question. Your ONLY job is to filter the data and format it.

YOU ARE NOT A PLANNER. YOU MUST NEVER:
  ✗ Say "I will filter..." or "I will get back to you"
  ✗ Say "Let me..." or "I'll now..." or describe future actions
  ✗ Return format="status" for list/filter/aggregate queries
  ✗ Return an empty or deferred response
  ✗ Ask for clarification — work with what you have

YOU MUST ALWAYS return a fully populated JSON object with actual data from RESULT.
The data is already in RESULT. Filter it and show it NOW.

TODAY'S DATE is provided in the prompt.

══════════════════════════════════
STEP 1 — CLASSIFY THE QUERY
══════════════════════════════════

  A) LIST / FILTER  → user wants to see records: "give me", "show", "list", "get",
                      "what are", "invoices of", "for [month/customer]"
  B) AGGREGATE      → user wants a number: "total", "sum", "how much", "how many",
                      "count", "amount owed"
  C) DETAIL         → RESULT has exactly 1 record, or user asked about one specific entity
  D) ACTION RESULT  → TOOL name contains create/update/delete/void/send

  When in doubt between A and B: choose A (show the table).
  NEVER choose D (status) for a list or filter query.

══════════════════════════════════
STEP 2 — FILTER THE RAW DATA
══════════════════════════════════

The RESULT contains ALL records from Zoho — you must filter them yourself.
Apply ALL relevant filters from USER_QUESTION at the same time:

  By customer/vendor name:
    Match customer_name, vendor_name, or contact_name against the name in the question.
    Use case-insensitive partial match.
    "punjab national bank" matches "Punjab National Bank Ltd" ✓
    "fonly punjab national bank" → extract the real name → "punjab national bank" ✓

  By month/year:
    "january 2026"   → keep records where date starts with "2026-01"
    "in 2026"        → keep records where date starts with "2026"
    "this month"     → keep records where date starts with current YYYY-MM
    "last month"     → keep records where date starts with previous YYYY-MM
    "last 30 days"   → keep records where date >= (today minus 30 days)
    "between X and Y"→ keep records where date is between X and Y inclusive

  By status:
    "pending / unpaid / outstanding / to collect / to receive"
      → keep only: balance > 0 AND status NOT IN [paid, void]
    "paid"    → status = paid
    "overdue" → status = overdue OR due_date < today
    "draft"   → status = draft
    no status word → no status filter

  Combine all matching filters. If no records survive filtering:
    Return: {"format":"answer","question":"...","answer":"No matching records found for [filters applied].","breakdown":[],"note":"Filters: [describe exactly what was applied]"}

══════════════════════════════════
STEP 3 — FORMAT THE OUTPUT
══════════════════════════════════

  LIST / FILTER (type A):
    Use "table" format.
    Columns: invoice/bill number, customer/vendor name, date, total, balance, status.
    Sort by date descending. Include ALL filtered records (up to 200 rows).
    Footer: "N records · Total: ₹X,XX,XXX.XX" (sum the total/balance column).

  AGGREGATE (type B):
    Use "answer" format.
    Compute the metric yourself from filtered records.
    Show breakdown by top 10 entities if >1 result.
    Always state record count.

  DETAIL (type C):
    Use "panel" format with all meaningful fields.
    Skip: template_id, color_code, created_time, is_emailed, exchange_rate.

  ACTION (type D):
    Use "status" format. State what was done and the entity number/ID.

══════════════════════════════════
OUTPUT — valid JSON only:
══════════════════════════════════

{"format":"table",
 "title":"Invoices – Punjab National Bank – January 2026",
 "columns":["Invoice #","Customer","Date","Total","Balance","Status"],
 "rows":[["INV-001","Punjab National Bank","01 Jan 2026","₹1,00,000.00","₹50,000.00","Open"],...],
 "footer":"3 records · Total: ₹3,00,000.00"}

{"format":"answer",
 "question":"Total outstanding from Punjab National Bank",
 "answer":"₹1,50,000.00 across 3 invoices",
 "breakdown":[["INV-001","₹50,000.00"],["INV-002","₹1,00,000.00"]],
 "note":"Filtered by: customer = Punjab National Bank, status = unpaid"}

{"format":"panel",
 "title":"Invoice #INV-001",
 "fields":[["Customer","Punjab National Bank"],["Date","01 Jan 2026"],...],
 "note":""}

{"format":"status",
 "ok":true,
 "headline":"Invoice #INV-001 created",
 "detail":"Invoice ID: 12345678901"}

ABSOLUTE RULES:
• ALWAYS filter and populate the output with real data from RESULT — never defer.
• Never output format="status" for list/filter/aggregate queries.
• Never use future tense ("I will...", "Let me...", "I'll get back...").
• Currency: ₹ with Indian comma formatting (₹12,45,678.00).
• Dates in output: DD MMM YYYY.
• No markdown ** or backticks inside JSON string values.
""".strip()


def _render_structured(data: dict) -> str:
    fmt = data.get("format", "status")

    if fmt == "answer":
        question  = data.get("question", "")
        answer    = data.get("answer", "Done.")
        breakdown = data.get("breakdown") or []
        note      = data.get("note", "")
        if RICH_AVAILABLE:
            _print("")
            if question:
                _print(f"  [dim]{rich_escape(question)}[/dim]")
            _print(f"\n  [bold green]{rich_escape(answer)}[/bold green]\n")
            if breakdown:
                for label, value in breakdown:
                    _print(f"  [dim]{rich_escape(str(label))}:[/dim]  {rich_escape(str(value))}")
            if note:
                _print(f"\n  [dim italic]{rich_escape(note)}[/dim italic]")
            _print("")
        else:
            print(f"\n{answer}")
            for label, value in breakdown:
                print(f"  {label}: {value}")
            if note:
                print(f"  ({note})")
            print()
        return answer

    if fmt == "table" and RICH_AVAILABLE:
        cols   = data.get("columns") or []
        rows   = data.get("rows") or []
        title  = data.get("title", "")
        footer = data.get("footer", "")

        tbl = Table(title=title, box=box.ROUNDED, show_header=True,
                    header_style="bold cyan", border_style="bright_black",
                    pad_edge=True, expand=False, show_lines=False)
        for col in cols:
            numeric = any(k in col.lower() for k in
                          ("amount", "total", "balance", "qty", "price", "rate", "tax"))
            tbl.add_column(col, justify="right" if numeric else "left", no_wrap=True)

        for row in rows:
            styled = []
            for cell in row:
                s  = str(cell)
                sl = s.lower()
                if any(w in sl for w in ("overdue", "due today", "unpaid", "⚠")):
                    styled.append(f"[bold red]{rich_escape(s)}[/bold red]")
                elif sl in ("paid", "completed", "accepted", "closed"):
                    styled.append(f"[green]{rich_escape(s)}[/green]")
                elif sl in ("open", "pending", "draft", "sent"):
                    styled.append(f"[yellow]{rich_escape(s)}[/yellow]")
                elif sl in ("void", "voided", "cancelled", "canceled"):
                    styled.append(f"[dim]{rich_escape(s)}[/dim]")
                else:
                    styled.append(rich_escape(s))
            tbl.add_row(*styled)

        _print_raw(tbl)
        if footer:
            _print(f"  [dim]{footer}[/dim]")
        _print("")
        return title

    elif fmt == "panel" and RICH_AVAILABLE:
        title  = data.get("title", "Details")
        fields = data.get("fields") or []
        note   = data.get("note", "")
        max_label = max((len(str(f[0])) for f in fields), default=10)
        lines = []
        for label, value in fields:
            v  = str(value)
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
            lines.append(f"  [bold]{rich_escape(str(label).ljust(max_label))}[/bold]  {v_styled}")

        body = "\n".join(lines)
        if note:
            body += f"\n\n  [dim italic]{rich_escape(note)}[/dim italic]"
        _print_raw(Panel(body, title=f"[bold cyan]{rich_escape(title)}[/bold cyan]",
                         border_style="bright_black", padding=(0, 1)))
        _print("")
        return title

    else:
        ok       = data.get("ok", True)
        headline = str(data.get("headline", "Done."))
        detail   = str(data.get("detail", ""))
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


async def _stream_collect_json(
    client: genai.Client,
    prompt: str,
    system: str,
    cid: str,
    timeout: Optional[float] = None,
) -> str:
    effective_timeout = timeout or GEMINI_TIMEOUT
    full = ""
    try:
        async with asyncio.timeout(effective_timeout):
            async for chunk in await client.aio.models.generate_content_stream(
                model=MODEL,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.2,
                    response_mime_type="application/json",
                    max_output_tokens=65536,
                ),
            ):
                full += chunk.text or ""
        return full
    except Exception as exc:
        log.warning("stream_collect_failed", extra={
            "cid": cid, "error": str(exc),
            "error_type": type(exc).__name__, "collected_chars": len(full)
        })
        resp = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=MODEL,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.2,
                    response_mime_type="application/json",
                    max_output_tokens=65536,
                ),
            ),
            timeout=effective_timeout,
        )
        return resp.text or "{}"


_ZOHO_NOISE_FIELDS: frozenset[str] = frozenset({
    "created_time", "last_modified_time", "created_by", "last_modified_by",
    "submitted_date", "submitted_by", "approved_date", "approved_by",
    "color_code", "current_sub_status_id", "current_sub_status",
    "template_id", "template_name", "template_type",
    "is_viewed_by_client", "client_viewed_time",
    "payment_made", "payment_expected_date",
    "salesperson_id", "salesperson_name",
    "shipping_charge", "adjustment", "write_off_amount",
    "exchange_rate", "currency_id", "currency_code", "currency_symbol",
    "documents", "attachments", "comments",
    "page_context", "zcrm_potential_id", "zcrm_potential_name",
    "salesorder_id", "purchaseorder_id", "estimate_id",
    "source", "reference_number", "reason",
    "ach_payment_initiated", "tax_rounding",
    "is_emailed", "is_viewed_by_client", "is_inclusive_tax",
    "payment_options", "line_item_total", "bcy_total", "bcy_balance",
    "bcy_discount_total", "bcy_adjustment", "bcy_sub_total",
})


def _strip_noise(obj: Any, depth: int = 0) -> Any:
    if depth > 4:
        return obj
    if isinstance(obj, dict):
        return {
            k: _strip_noise(v, depth + 1)
            for k, v in obj.items()
            if k not in _ZOHO_NOISE_FIELDS
        }
    if isinstance(obj, list):
        return [_strip_noise(i, depth + 1) for i in obj]
    return obj


def _safe_result_str(result: Any, char_limit: int = SUMMARIZE_INPUT_CHAR_LIMIT) -> str:
    cleaned = _strip_noise(result)
    full = json.dumps(cleaned, default=str, ensure_ascii=False)
    if len(full) <= char_limit:
        return full

    if isinstance(cleaned, dict):
        best_key, best_arr = "", []
        for k, v in cleaned.items():
            if isinstance(v, list) and len(v) > len(best_arr):
                best_key, best_arr = k, v
        if best_arr:
            trimmed = json.dumps({best_key: best_arr}, default=str, ensure_ascii=False)
            if len(trimmed) <= char_limit:
                log.info("result_trimmed_to_array",
                         extra={"key": best_key, "rows": len(best_arr), "chars": len(trimmed)})
                return trimmed

    log.warning("result_hard_truncated",
                extra={"original_chars": len(full), "limit": char_limit})
    return full[:char_limit] + "\n...[TRUNCATED — partial result]"


async def gemini_summarize(
    client: genai.Client,
    tool: str,
    args: dict,
    result: Any,
    state: AgentState,
    correlation_id: str = "",
    user_question: str = "",
) -> str:
    result_str = _safe_result_str(result)
    prompt = (
        f"TODAY: {date.today().isoformat()}\n"
        f"USER_QUESTION: {user_question or '(not specified — use best judgement on format)'}\n"
        f"TOOL: {tool}\n"
        f"ARGS: {json.dumps(args, default=str)}\n"
        f"RESULT: {result_str}"
    )
    timeout = max(GEMINI_TIMEOUT, min(180.0, len(result_str) / 2000))
    raw     = await _stream_collect_json(client, prompt, SUMMARIZE_SYSTEM,
                                         correlation_id, timeout=timeout)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
        return _render_structured(data)
    except Exception:
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
# Auto-pagination helper
# ---------------------------------------------------------------------------
MAX_AUTOPAGINATE_PAGES: int = int(os.getenv("MAX_AUTOPAGINATE_PAGES", "20"))
_PAGEABLE_TOOL_SUBSTRINGS = (
    "list_invoices", "list_bills", "list_contacts", "list_items",
    "list_expenses", "list_payments", "list_creditnotes", "list_estimates",
    "list_salesorders", "list_purchaseorders", "list_transactions",
    "list_bank", "list_journals", "search_invoices", "search_contacts",
)


def _is_pageable(tool_name: str) -> bool:
    tl = tool_name.lower()
    return any(sub in tl for sub in _PAGEABLE_TOOL_SUBSTRINGS)


def _extract_records(result: Any) -> tuple[str, list]:
    """
    Return (key, records_list) for the largest meaningful list in a Zoho result dict.
    Skips MCP envelope keys so we never treat the transport wrapper as records.
    """
    if not isinstance(result, dict):
        return "", []

    _MCP_ENVELOPE_KEYS = frozenset({"content", "meta", "isError", "_meta"})

    best_key, best_arr = "", []
    for k, v in result.items():
        if k in _MCP_ENVELOPE_KEYS:
            continue
        if k == "page_context":
            continue
        if isinstance(v, list) and len(v) > len(best_arr):
            best_key, best_arr = k, v
    return best_key, best_arr


def _has_more_pages(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    pc = result.get("page_context") or {}
    if isinstance(pc, dict) and pc.get("has_more_page"):
        return True
    for v in result.values():
        if isinstance(v, dict) and v.get("has_more_page"):
            return True
    return False


def _unwrap_mcp_result(raw: Any) -> Any:
    """
    Unwrap the MCP tool call envelope to get the actual Zoho API response.
    MCP returns: {"content": [{"type": "text", "text": "<JSON string>"}], "isError": false}
    """
    if not isinstance(raw, dict):
        return raw

    content = raw.get("content")
    if not isinstance(content, list) or not content:
        return raw

    first = content[0] if content else {}
    if not isinstance(first, dict) or first.get("type") != "text":
        return raw

    text_value = first.get("text", "")
    if not isinstance(text_value, str) or not text_value.strip():
        return raw

    try:
        parsed = json.loads(text_value)
        log.debug("mcp_envelope_unwrapped", extra={
            "keys": list(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__
        })
        return parsed
    except json.JSONDecodeError:
        log.warning("mcp_text_not_json", extra={"preview": text_value[:120]})
        return raw


def _parse_mcp_result(raw: Any) -> Any:
    """Convert a raw MCP tool call result into a plain Python dict."""
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    raw = _unwrap_mcp_result(raw)
    return raw


async def _autopaginate(
    session: ClientSession,
    tool_name: str,
    base_args: dict,
    first_result: Any,
    audit: "AuditLog",
    correlation_id: str,
) -> Any:
    record_key, all_records = _extract_records(first_result)
    if not record_key or not all_records:
        return first_result

    page = 2
    while page <= MAX_AUTOPAGINATE_PAGES:
        if not _has_more_pages(first_result if page == 2 else page_result):  # type: ignore[possibly-undefined]
            break

        page_args = {**base_args, "page": page}
        log.info("autopaginate_fetch", extra={
            "tool": tool_name, "page": page, "cid": correlation_id
        })
        try:
            raw = await asyncio.wait_for(
                session.call_tool(tool_name, page_args),
                timeout=TOOL_TIMEOUT,
            )
            page_result = _parse_mcp_result(raw)
        except Exception as exc:
            log.warning("autopaginate_error", extra={
                "tool": tool_name, "page": page,
                "error": str(exc), "cid": correlation_id
            })
            break

        _, page_records = _extract_records(page_result)
        if not page_records:
            break
        all_records.extend(page_records)
        audit.write("tool_paginated", tool=tool_name, page=page,
                    records=len(page_records), cid=correlation_id)
        page += 1

    merged = dict(first_result)
    merged[record_key] = all_records
    if "page_context" in merged and isinstance(merged["page_context"], dict):
        merged["page_context"] = {
            **merged["page_context"],
            "has_more_page": False,
            "total": len(all_records),
        }
    log.info("autopaginate_complete", extra={
        "tool": tool_name, "total_records": len(all_records),
        "pages": page - 1, "cid": correlation_id
    })
    return merged


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
    gemini_client: Optional[Any] = None,
) -> tuple[bool, Optional[Any]]:
    tool_name: str = step.get("tool", "")
    args: dict     = dict(step.get("args") or {})

    if not tool_name or tool_name not in toolbox:
        msg = f"Unknown tool: '{tool_name}'"
        add_memory(memory, {"error": msg, "step": step, "cid": correlation_id},
                   gemini_client, correlation_id)
        log.warning("unknown_tool", extra={"tool": tool_name, "cid": correlation_id})
        return False, None

    meta = toolbox[tool_name]

    cb = _circuit_breakers[tool_name]
    if cb.is_open():
        msg = f"Tool '{tool_name}' is temporarily disabled (circuit open)."
        _print(f"[bold red]Assistant:[/bold red] {msg}\n")
        add_memory(memory, {"error": msg, "tool": tool_name, "cid": correlation_id},
                   gemini_client, correlation_id)
        return False, None

    if "organization_id" in meta.required and "organization_id" not in args:
        if state.organization_id:
            args["organization_id"] = state.organization_id

    unknown, missing = validate_args(meta, args)
    if unknown or missing:
        add_memory(memory, {
            "tool": tool_name, "args": args, "cid": correlation_id,
            "schema_error": {
                "unknown_keys": unknown, "missing_required": missing,
                "allowed": meta.allowed[:40], "required": meta.required,
            },
        }, gemini_client, correlation_id)
        log.warning("schema_error",
                    extra={"tool": tool_name, "unknown": unknown, "missing": missing})
        return False, None

    if org := args.get("organization_id"):
        state.organization_id = str(org)

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
        add_memory(memory, {"tool": tool_name, "error": msg, "cid": correlation_id},
                   gemini_client, correlation_id)
        cb.record_failure()
        log.error("tool_timeout", extra={"tool": tool_name, "cid": correlation_id})
        _print(f"[bold yellow]Assistant:[/bold yellow] {msg}\n")
        return False, None
    except Exception as exc:
        msg = str(exc)
        add_memory(memory, {"tool": tool_name, "args": args, "error": msg, "cid": correlation_id},
                   gemini_client, correlation_id)
        cb.record_failure()
        log.error("tool_error", extra={"tool": tool_name, "error": msg,
                                       "error_type": type(exc).__name__, "cid": correlation_id})
        _print(f"[bold red]Assistant:[/bold red] Tool call failed — {msg}\n")
        return False, None

    elapsed     = round(time.monotonic() - t0, 2)
    cb.record_success()

    result_data = _parse_mcp_result(result)

    if _is_pageable(tool_name) and _has_more_pages(result_data):
        result_data["__paginate__"] = {
            "tool":   tool_name,
            "args":   args,
            "page1":  True,
        }

    # Store a compact conversation entry — NOT the raw result array.
    # add_memory will extract record_count + sample_ids from result_data.
    add_memory(memory, {
        "tool":      tool_name,
        "args":      {k: args[k] for k in list(args)[:8]
                      if k not in ("organization_id",)},
        "result":    result_data,   # add_memory strips this to a compact summary
        "elapsed_s": elapsed,
        "cid":       correlation_id,
    }, gemini_client, correlation_id)
    audit.write("tool_called", tool=tool_name, args=args, elapsed_s=elapsed, cid=correlation_id)
    log.info("tool_success",
             extra={"tool": tool_name, "elapsed_s": elapsed, "cid": correlation_id})
    return True, result_data


async def execute_plan(
    session,
    toolbox: dict,
    state,
    memory: list,
    audit,
    gemini,
    steps: list[dict],
    correlation_id: str,
) -> tuple[bool, Optional[Any], list]:
    """
    Execute a plan's steps sequentially (or in parallel batches).

    Returns (True, last_result, all_step_results) on success.
    Returns (False, error_dict, []) on failure.

    all_step_results contains every step's raw API result — needed for
    multi-tool queries like profit (invoices result + bills result).
    """
    last_result: Optional[Any] = None
    step_results: list[Any] = []

    def _last_error() -> Optional[Any]:
        """Extract the most recent error from memory for surfacing to the user."""
        for entry in reversed(memory):
            if not isinstance(entry, dict):
                continue
            if "error" in entry:
                return {"__execution_error__": True,
                        "tool":    entry.get("tool", "unknown"),
                        "message": str(entry["error"])}
            if "schema_error" in entry:
                se = entry["schema_error"]
                tool = entry.get("tool", "unknown")
                parts = []
                if se.get("missing_required"):
                    parts.append(f"missing required args: {se['missing_required']}")
                if se.get("unknown_keys"):
                    parts.append(f"unknown args: {se['unknown_keys']}")
                return {"__execution_error__": True,
                        "tool":    tool,
                        "message": f"Schema error on {tool} — {'; '.join(parts)}"}
        return {"__execution_error__": True, "tool": "unknown",
                "message": "Execution failed — check server logs for details."}

    i = 0
    while i < len(steps):
        step = steps[i]

        if step_results:
            original_args = step.get("args") or {}
            resolved_args = _resolve_args(original_args, step_results)
            if resolved_args != original_args:
                step = {**step, "args": resolved_args}

        if step.get("parallel"):
            batch = [step]
            j = i + 1
            while j < len(steps) and steps[j].get("parallel"):
                ps = steps[j]
                if step_results:
                    resolved = _resolve_args(ps.get("args") or {}, step_results)
                    if resolved != (ps.get("args") or {}):
                        ps = {**ps, "args": resolved}
                batch.append(ps)
                j += 1

            tasks = [
                execute_step(session, toolbox, state, memory, audit, s, correlation_id, gemini)
                for s in batch
            ]
            results = await asyncio.gather(*tasks)
            for (ok, rdata), s in zip(results, batch):
                if not ok:
                    return False, _last_error(), []
                last_result = rdata
                step_results.append(rdata)
            i = j
            continue

        ok, rdata = await execute_step(
            session, toolbox, state, memory, audit, step, correlation_id, gemini
        )
        if not ok:
            return False, _last_error(), []

        last_result = rdata
        step_results.append(rdata)
        i += 1

    return True, last_result, step_results


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def is_number_string(s: str) -> bool:
    return bool(re.fullmatch(r"\d{6,}", (s or "").strip()))


def validate_env() -> None:
    missing = [k for k in REQUIRED_ENV_VARS if not os.getenv(k)]
    if missing:
        raise RuntimeError("Missing required environment variables: " + ", ".join(missing))


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------
async def main() -> None:
    validate_env()
    gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    zoho_url = ""
    while not zoho_url:
        zoho_url = (await asyncio.get_event_loop().run_in_executor(
            None, input, "Enter MCP URL: "
        )).strip()
        if not zoho_url:
            _print("[yellow]MCP URL cannot be empty.[/yellow]")

    zoho_bearer: Optional[str] = os.getenv("ZOHO_MCP_BEARER")
    headers: dict[str, str] = {}
    if zoho_bearer:
        headers["Authorization"] = f"Bearer {zoho_bearer}"

    default_org_id: Optional[str] = os.getenv("ZOHO_ORG_ID") or None
    state = AgentState(organization_id=default_org_id)

    if not state.organization_id:
        while True:
            oid = (await asyncio.get_event_loop().run_in_executor(
                None, input, "Enter Zoho Organisation ID: "
            )).strip()
            if re.fullmatch(r"\d{4,}", oid):
                state.organization_id = oid
                break
            _print("[yellow]Organisation ID must be numeric (e.g. 60065733225).[/yellow]")

    memory: list[dict] = []
    audit  = AuditLog(AUDIT_LOG_PATH)
    pending_confirm: Optional[dict] = None
    replan_attempts: int = 0
    shutdown_event = asyncio.Event()

    def _handle_signal(sig: int, _frame: Any) -> None:
        _print("\n[bold yellow]Shutting down gracefully…[/bold yellow]")
        shutdown_event.set()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info("agent_starting", extra={"model": MODEL, "org_id": state.organization_id})

    async with httpx.AsyncClient(headers=headers,
                                  timeout=httpx.Timeout(TOOL_TIMEOUT + 10)) as http_client:
        async with streamable_http_client(zoho_url, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                toolbox = build_tool_catalog(await session.list_tools())

                _print("\n[bold green]✓ Zoho Books Assistant ready[/bold green]")
                _print(f"  [dim]Tools: {len(toolbox)} | Model: {MODEL} | Org: {state.organization_id}[/dim]")
                _print("  [dim]Commands: quit, tools, memory, state[/dim]\n")

                while not shutdown_event.is_set():
                    try:
                        user_msg = await asyncio.get_event_loop().run_in_executor(
                            None, input, "You: "
                        )
                    except EOFError:
                        break
                    user_msg = user_msg.strip()
                    if not user_msg:
                        continue

                    if user_msg.lower() in {"quit", "exit", "q"}:
                        break

                    if user_msg.lower() == "tools":
                        for name, meta in sorted(toolbox.items()):
                            tag = "[dim](read)[/dim]" if meta.read_only else "[yellow](write)[/yellow]"
                            _print(f"  {tag} {name} — {meta.desc}")
                        _print("")
                        continue

                    if user_msg.lower() == "memory":
                        _print(f"\n[bold]Memory ({len(memory)} entries):[/bold]")
                        _print(json.dumps(memory, indent=2, default=str))
                        _print("")
                        continue

                    if user_msg.lower() == "state":
                        _print(json.dumps(state.to_dict(), indent=2))
                        continue

                    if is_number_string(user_msg) and not state.organization_id:
                        state.organization_id = user_msg
                        _print(f"[green]Assistant:[/green] Org ID saved: {user_msg}\n")
                        audit.write("org_id_set", org_id=user_msg)
                        continue

                    cid = uuid.uuid4().hex[:10]
                    audit.write("user_message", msg=user_msg, cid=cid)

                    plan: dict
                    if pending_confirm:
                        answer = user_msg.lower()
                        if answer in {"yes", "y"}:
                            plan = pending_confirm["on_yes"]
                            pending_confirm = None
                            audit.write("confirm_accepted", cid=cid)
                        elif answer in {"no", "n"}:
                            no_obj = pending_confirm.get("on_no") or {}
                            pending_confirm = None
                            audit.write("confirm_rejected", cid=cid)
                            _print(f"[green]Assistant:[/green] {no_obj.get('text', 'Cancelled.')}\n")
                            continue
                        else:
                            _print("[green]Assistant:[/green] Please reply YES or NO.\n")
                            continue
                    else:
                        raw  = await gemini_plan(gemini_client, toolbox, state, memory, user_msg, cid)
                        plan = normalize_plan(raw)
                        replan_attempts = 0

                    ptype = plan.get("type")

                    if ptype == "ask":
                        state.update_from_dict(plan.get("save") or {})
                        _print(f"[green]Assistant:[/green] {plan.get('text', 'I need more details.')}\n")
                        audit.write("ask", text=plan.get("text"), cid=cid)
                        continue

                    if ptype == "confirm":
                        pending_confirm = {"on_yes": plan.get("on_yes"), "on_no": plan.get("on_no")}
                        _print(f"[bold yellow]Assistant:[/bold yellow] {plan.get('text', 'Confirm?')} (YES / NO)\n")
                        audit.write("confirm_requested", text=plan.get("text"), cid=cid)
                        continue

                    if ptype != "plan":
                        _print("[green]Assistant:[/green] Couldn't determine an action. Please rephrase.\n")
                        continue

                    steps: list[dict] = plan.get("steps") or []
                    if not steps:
                        _print("[green]Assistant:[/green] Plan had no steps. Please rephrase.\n")
                        continue

                    risky_steps = [s for s in steps if is_risky(s.get("tool", ""))]
                    if risky_steps and not pending_confirm:
                        risky_names = ", ".join(s["tool"] for s in risky_steps)
                        pending_confirm = {
                            "on_yes": plan,
                            "on_no":  {"type": "ask", "text": "Cancelled. What would you like instead?"},
                        }
                        _print(f"[bold yellow]Assistant:[/bold yellow] Sensitive: {risky_names}. YES to confirm / NO to cancel.\n")
                        audit.write("risky_confirm_requested", tools=risky_names, cid=cid)
                        continue

                    _print("[dim]Assistant: Working…[/dim]")
                    ok, last_result = await execute_plan(
                        session, toolbox, state, memory, audit, gemini_client, steps, cid
                    )

                    if ok and last_result is not None:
                        last_step = steps[-1]
                        await gemini_summarize(gemini_client, last_step.get("tool", ""),
                                               last_step.get("args", {}), last_result, state, cid,
                                               user_question=last_step.get("note", user_msg))
                        state.clear_workflow()
                        replan_attempts = 0
                        continue

                    if replan_attempts >= MAX_REPLAN_ATTEMPTS:
                        _print("[bold red]Assistant:[/bold red] Tried multiple times but couldn't complete this. Please rephrase.\n")
                        replan_attempts = 0
                        continue

                    replan_attempts += 1
                    replan_raw = await gemini_plan(
                        gemini_client, toolbox, state, memory,
                        "Fix the plan using MEMORY errors. Ask only for genuinely missing required fields.",
                        cid,
                    )
                    replan = normalize_plan(replan_raw)
                    rtype  = replan.get("type")

                    if rtype == "ask":
                        state.update_from_dict(replan.get("save") or {})
                        _print(f"[green]Assistant:[/green] {replan.get('text', 'I need more details.')}\n")
                    elif rtype == "confirm":
                        pending_confirm = {"on_yes": replan.get("on_yes"), "on_no": replan.get("on_no")}
                        _print(f"[bold yellow]Assistant:[/bold yellow] {replan.get('text', 'Confirm?')} (YES / NO)\n")
                    elif rtype == "plan":
                        tools_preview = " → ".join(s.get("tool", "?") for s in (replan.get("steps") or []))
                        has_risky = any(is_risky(s.get("tool", "")) for s in (replan.get("steps") or []))
                        pending_confirm = {
                            "on_yes": replan,
                            "on_no":  {"type": "ask", "text": "Okay — tell me what to change."},
                        }
                        label = "[bold yellow]" if has_risky else "[green]"
                        _print(f"{label}Assistant:[/bold yellow if has_risky else /green] Replanned: {tools_preview}. YES to proceed / NO to change.\n")
                    else:
                        _print("[bold red]Assistant:[/bold red] I'm stuck. Please rephrase.\n")

    audit.close()
    _print("[dim]Session ended.[/dim]")
    log.info("agent_stopped")


if __name__ == "__main__":
    asyncio.run(main())
