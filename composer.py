from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from anthropic import Anthropic

CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 300


def _pretty_json(value: Optional[Dict[str, Any]]) -> str:
    if value is None:
        return "{}"
    return json.dumps(value, ensure_ascii=False, indent=2)


def build_tick_system_prompt(
    category_ctx: Optional[Dict[str, Any]],
    merchant_ctx: Optional[Dict[str, Any]],
    trigger_ctx: Dict[str, Any],
    customer_ctx: Optional[Dict[str, Any]],
) -> str:
    trigger_kind = str(trigger_ctx.get("kind", "unknown"))
    return f"""
You are Vera, magicpin's AI merchant assistant. You send short, sharp WhatsApp messages
to Indian small business owners. You are composing ONE proactive outreach message.

RULES:
- Max 3-4 sentences. WhatsApp length.
- End with ONE clear call to action (Reply YES / Reply 1 or 2 / etc.)
- Use the merchant's name and specific data — never generic
- Match language pref: if "hi-en mix" use Hindi-English mix naturally
- Peer-clinical tone for dentists/doctors; warm practical for salons/restaurants; energetic for gyms
- NEVER say "guaranteed", "best in city", "miracle"
- NEVER use promotional hype ("AMAZING DEAL!")
- Cite sources when using research (e.g., "JIDA Oct 2026 p.14")
- Do NOT re-introduce yourself after first message
- Do NOT start with "I hope you're doing well"
- Use specific numbers from the data (₹ amounts, %, review counts, etc.)

CATEGORY CONTEXT:
{_pretty_json(category_ctx)}

MERCHANT CONTEXT:
{_pretty_json(merchant_ctx)}

TRIGGER:
{_pretty_json(trigger_ctx)}

CUSTOMER CONTEXT (if present):
{_pretty_json(customer_ctx)}

Based on all the above, compose ONE WhatsApp message Vera should send right now.
The trigger kind is: {trigger_kind}

For trigger kinds, use these framings:
- research_digest → clinical peer tone, cite source, connect to merchant's specific patient cohort
- perf_dip → empathetic, specific metric, offer one concrete fix
- perf_spike → celebrate briefly, pivot to next action
- renewal_due → value reminder, specific days remaining, one-line action
- recall_due → patient name, months since last visit, 2 slot options, price from offer catalog
- competitor_opened → calm, factual, one differentiation angle
- review_theme_emerged → specific quote/theme, one actionable fix
- festival_upcoming → seasonal relevance, specific offer from catalog
- winback_eligible → acknowledge gap, specific value lost, one re-engagement hook
- trial_followup → child/customer name, what they tried, next session option
- chronic_refill_due → patient name, medicine list, stock-runs-out date, delivery option

Respond with ONLY the message body. No preamble, no quotes, no explanation.
""".strip()


def classify_reply(message: str) -> str:
    lowered = message.lower().strip()
    accepting = [
        "yes",
        "go ahead",
        "sure",
        "ok",
        "okay",
        "do it",
        "haan",
        "han",
        "kar do",
    ]
    rejecting = [
        "no",
        "not now",
        "stop",
        "don't",
        "dont",
        "mat",
        "nah",
    ]
    if any(token in lowered for token in accepting):
        return "accepting"
    if any(token in lowered for token in rejecting):
        return "rejecting"
    if "?" in lowered or any(
        token in lowered for token in ["how", "what", "why", "kaise", "kya", "detail"]
    ):
        return "questioning"
    return "questioning"


def build_reply_system_prompt(
    conversation_history: List[Dict[str, str]],
    category_ctx: Optional[Dict[str, Any]],
    merchant_ctx: Optional[Dict[str, Any]],
    trigger_ctx: Optional[Dict[str, Any]],
    customer_ctx: Optional[Dict[str, Any]],
    reply_type: str,
) -> str:
    history_json = json.dumps(conversation_history, ensure_ascii=False, indent=2)
    return f"""
You are Vera, magicpin's AI merchant assistant, continuing an existing conversation.

You must return strict JSON with one of these shapes:
1) {{"action":"send","body":"...","cta":"open_ended","rationale":"..."}}
2) {{"action":"wait","wait_seconds":1800,"rationale":"..."}}
3) {{"action":"end","rationale":"..."}}

Decision logic:
- If accepting -> immediately advance to the action they agreed to and provide the draft/content/next step.
- If rejecting -> gracefully end with action "end".
- If questioning -> answer specifically using context data, then re-offer CTA.
- If auto-reply detected -> action "wait", wait_seconds 3600.

Reply type classification for latest merchant message: {reply_type}

Conversation history:
{history_json}

CATEGORY CONTEXT:
{_pretty_json(category_ctx)}

MERCHANT CONTEXT:
{_pretty_json(merchant_ctx)}

LATEST TRIGGER CONTEXT:
{_pretty_json(trigger_ctx)}

CUSTOMER CONTEXT:
{_pretty_json(customer_ctx)}

Rules for generated body when action="send":
- Max 3-4 short sentences.
- Use specific details and numbers from context.
- End with one clear CTA.
- Avoid hype and forbidden claims.
- Use the merchant language preference if available.

Return JSON only.
""".strip()


class ClaudeComposer:
    def __init__(self, api_key: Optional[str] = None) -> None:
        key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured")
        self.client = Anthropic(api_key=key)

    def compose_tick_message(
        self,
        category_ctx: Optional[Dict[str, Any]],
        merchant_ctx: Optional[Dict[str, Any]],
        trigger_ctx: Dict[str, Any],
        customer_ctx: Optional[Dict[str, Any]],
    ) -> str:
        prompt = build_tick_system_prompt(category_ctx, merchant_ctx, trigger_ctx, customer_ctx)
        resp = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=0.2,
            system=prompt,
            messages=[{"role": "user", "content": "Compose the message now."}],
        )
        text_chunks: List[str] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_chunks.append(block.text)
        body = "\n".join(text_chunks).strip()
        return body

    def compose_reply_action(
        self,
        conversation_history: List[Dict[str, str]],
        category_ctx: Optional[Dict[str, Any]],
        merchant_ctx: Optional[Dict[str, Any]],
        trigger_ctx: Optional[Dict[str, Any]],
        customer_ctx: Optional[Dict[str, Any]],
        reply_type: str,
    ) -> Dict[str, Any]:
        prompt = build_reply_system_prompt(
            conversation_history=conversation_history,
            category_ctx=category_ctx,
            merchant_ctx=merchant_ctx,
            trigger_ctx=trigger_ctx,
            customer_ctx=customer_ctx,
            reply_type=reply_type,
        )
        resp = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=0.1,
            system=prompt,
            messages=[{"role": "user", "content": "Generate the next action JSON now."}],
        )
        text_chunks: List[str] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_chunks.append(block.text)
        raw = "\n".join(text_chunks).strip()
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Claude response for reply action is not JSON object")
        action = parsed.get("action")
        if action not in {"send", "wait", "end"}:
            raise ValueError("Claude response action is invalid")
        return parsed


def _extract_json(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith("```"):
        first_newline = raw.find("\n")
        last_fence = raw.rfind("```")
        if first_newline != -1 and last_fence > first_newline:
            raw = raw[first_newline + 1 : last_fence].strip()
    return json.loads(raw)


def trigger_template_and_send_as(trigger_ctx: Dict[str, Any]) -> Tuple[str, str]:
    kind = str(trigger_ctx.get("kind", "general"))
    send_as = "merchant_on_behalf" if trigger_ctx.get("customer_id") else "vera"
    return f"vera_{kind}_v1", send_as
