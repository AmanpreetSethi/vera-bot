from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from composer import ClaudeComposer, classify_reply, trigger_template_and_send_as
from models import (
    ContextPushRequest,
    ContextPushResponse,
    HealthResponse,
    MetadataResponse,
    ReplyEndResponse,
    ReplyRequest,
    ReplyResponse,
    ReplySendResponse,
    ReplyWaitResponse,
    TickAction,
    TickRequest,
    TickResponse,
)
from state import AppState

load_dotenv()

app = FastAPI(title="Vera Bot", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
state = AppState()
composer: Optional[ClaudeComposer] = None
boot_time = time.monotonic()


@app.on_event("startup")
def startup() -> None:
    global composer
    base_dir = Path(__file__).resolve().parent
    state.preload_category_contexts(base_dir=base_dir)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        composer = ClaudeComposer(api_key=api_key)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_trigger_payload(trigger_id: str) -> Optional[Dict[str, Any]]:
    return state.get_context_payload("trigger", trigger_id)


def _latest_trigger_for_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    parts = conversation_id.split("_", 2)
    if len(parts) < 3:
        return None
    trigger_id = parts[2]
    return _extract_trigger_payload(trigger_id)


def _is_auto_reply(message: str, history: List[Dict[str, str]]) -> bool:
    lowered = message.lower()
    auto_markers = ["thank you for contacting", "dhanyawad", "we will get back"]
    if any(marker in lowered for marker in auto_markers):
        return True

    last_two_merchant_messages: List[str] = [
        entry["body"].strip().lower()
        for entry in history[-2:]
        if entry.get("from") in {"merchant", "customer"} and "body" in entry
    ]
    return lowered.strip() in last_two_merchant_messages


@app.get("/v1/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    uptime_seconds = int(time.monotonic() - boot_time)
    return HealthResponse(
        status="ok",
        uptime_seconds=uptime_seconds,
        contexts_loaded=state.context_counts(),  # type: ignore[arg-type]
    )


@app.get("/v1/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    return MetadataResponse(
        team_name="Team VeraForge",
        team_members=["HP", "Codex Assistant"],
        model="claude-sonnet-4-20250514",
        approach="4-context composer with trigger routing",
        contact_email="team.veraforge@example.com",
        version="1.0.0",
        submitted_at=datetime.fromisoformat("2026-05-02T00:00:00+00:00"),
    )


@app.post("/v1/context", response_model=ContextPushResponse)
def push_context(payload: ContextPushRequest) -> ContextPushResponse:
    accepted = state.put_context_if_newer(
        scope=payload.scope,
        context_id=payload.context_id,
        version=payload.version,
        payload=payload.payload,
        delivered_at_iso=payload.delivered_at.isoformat(),
    )
    if not accepted:
        return ContextPushResponse(accepted=False, reason="stale_version")
    return ContextPushResponse(
        accepted=True,
        ack_id=f"ack_{payload.context_id}_v{payload.version}",
        stored_at=datetime.now(timezone.utc),
    )


@app.post("/v1/tick", response_model=TickResponse)
def tick(req: TickRequest) -> TickResponse:
    if composer is None:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured")

    actions: List[TickAction] = []
    for trigger_id in req.available_triggers:
        trigger_ctx = _extract_trigger_payload(trigger_id)
        if not trigger_ctx:
            continue

        suppression_key = trigger_ctx.get("suppression_key")
        suppression_key_str = str(suppression_key) if suppression_key else None
        if state.is_suppressed(suppression_key_str):
            continue

        merchant_id = trigger_ctx.get("merchant_id")
        if not merchant_id:
            continue

        merchant_ctx = state.get_context_payload("merchant", str(merchant_id))
        if not merchant_ctx:
            continue

        category_slug = merchant_ctx.get("category_slug")
        category_ctx = state.get_context_payload("category", str(category_slug))
        customer_id = trigger_ctx.get("customer_id")
        customer_ctx = state.get_context_payload("customer", str(customer_id)) if customer_id else None

        try:
            body = composer.compose_tick_message(
                category_ctx=category_ctx,
                merchant_ctx=merchant_ctx,
                trigger_ctx=trigger_ctx,
                customer_ctx=customer_ctx,
            )
        except Exception:
            continue

        trigger_kind = str(trigger_ctx.get("kind", "general"))
        template_name, send_as = trigger_template_and_send_as(trigger_ctx)
        merchant_name = str(
            merchant_ctx.get("name")
            or merchant_ctx.get("identity", {}).get("name")
            or merchant_id
        )
        conversation_id = f"conv_{merchant_id}_{trigger_id}"

        action = TickAction(
            conversation_id=conversation_id,
            merchant_id=str(merchant_id),
            customer_id=str(customer_id) if customer_id else None,
            send_as=send_as,  # type: ignore[arg-type]
            trigger_id=str(trigger_id),
            template_name=template_name,
            template_params=[merchant_name, trigger_kind],
            body=body,
            cta="open_ended",
            suppression_key=str(suppression_key_str or ""),
            rationale=f"Triggered by {trigger_kind} with merchant/category context fusion",
        )
        actions.append(action)
        state.mark_suppression_sent(suppression_key_str)

        state.append_conversation_message(
            conversation_id=conversation_id,
            from_role="vera",
            body=body,
            ts_iso=_now_iso(),
        )

    return TickResponse(actions=actions)


@app.post("/v1/reply", response_model=ReplyResponse)
def reply(req: ReplyRequest) -> ReplyResponse:
    if composer is None:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured")

    prior_history = state.get_conversation(req.conversation_id)
    state.append_conversation_message(
        conversation_id=req.conversation_id,
        from_role=req.from_role,
        body=req.message,
        ts_iso=req.received_at.isoformat(),
    )

    if _is_auto_reply(req.message, prior_history):
        return ReplyWaitResponse(
            action="wait",
            wait_seconds=3600,
            rationale="Auto-reply detected, backing off",
        )
    history = state.get_conversation(req.conversation_id)

    merchant_ctx = state.get_context_payload("merchant", req.merchant_id)
    trigger_ctx = _latest_trigger_for_conversation(req.conversation_id)
    category_ctx = None
    if merchant_ctx:
        category_slug = merchant_ctx.get("category_slug")
        category_ctx = state.get_context_payload("category", str(category_slug))
    customer_ctx = state.get_context_payload("customer", req.customer_id) if req.customer_id else None

    reply_type = classify_reply(req.message)
    if reply_type == "rejecting":
        return ReplyEndResponse(
            action="end",
            rationale="Merchant rejected continuation, conversation ended gracefully",
        )

    try:
        result = composer.compose_reply_action(
            conversation_history=history,
            category_ctx=category_ctx,
            merchant_ctx=merchant_ctx,
            trigger_ctx=trigger_ctx,
            customer_ctx=customer_ctx,
            reply_type=reply_type,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reply generation failed: {exc}") from exc

    action = result.get("action")
    if action == "wait":
        wait_seconds = int(result.get("wait_seconds", 1800))
        return ReplyWaitResponse(
            action="wait",
            wait_seconds=max(wait_seconds, 1),
            rationale=str(result.get("rationale", "Need time before next follow-up")),
        )
    if action == "end":
        return ReplyEndResponse(
            action="end",
            rationale=str(result.get("rationale", "Conversation completed")),
        )

    body = str(result.get("body", "")).strip()
    if not body:
        raise HTTPException(status_code=500, detail="Claude returned empty send body")

    state.append_conversation_message(
        conversation_id=req.conversation_id,
        from_role="vera",
        body=body,
        ts_iso=_now_iso(),
    )
    return ReplySendResponse(
        action="send",
        body=body,
        cta="open_ended",
        rationale=str(result.get("rationale", "Responded to merchant intent with context")),
    )


@app.post("/v1/teardown")
def teardown() -> Dict[str, bool]:
    state.wipe_all()
    return {"wiped": True}
