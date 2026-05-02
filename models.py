from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ScopeType = Literal["category", "merchant", "customer", "trigger"]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    uptime_seconds: int
    contexts_loaded: Dict[ScopeType, int]


class MetadataResponse(BaseModel):
    team_name: str
    team_members: List[str]
    model: str
    approach: str
    contact_email: str
    version: str
    submitted_at: datetime


class ContextPushRequest(BaseModel):
    scope: ScopeType
    context_id: str = Field(min_length=1)
    version: int = Field(ge=0)
    payload: Dict[str, Any]
    delivered_at: datetime


class ContextPushResponse(BaseModel):
    accepted: bool
    ack_id: Optional[str] = None
    stored_at: Optional[datetime] = None
    reason: Optional[str] = None


class TickRequest(BaseModel):
    now: datetime
    available_triggers: List[str]


class TickAction(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    send_as: Literal["vera", "merchant_on_behalf"]
    trigger_id: str
    template_name: str
    template_params: List[str]
    body: str
    cta: Literal["open_ended"]
    suppression_key: str
    rationale: str


class TickResponse(BaseModel):
    actions: List[TickAction]


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    from_role: Literal["merchant", "customer", "vera"]
    message: str = Field(min_length=1)
    received_at: datetime
    turn_number: int = Field(ge=1)


class ReplySendResponse(BaseModel):
    action: Literal["send"]
    body: str
    cta: Literal["open_ended"]
    rationale: str


class ReplyWaitResponse(BaseModel):
    action: Literal["wait"]
    wait_seconds: int = Field(ge=1)
    rationale: str


class ReplyEndResponse(BaseModel):
    action: Literal["end"]
    rationale: str


ReplyResponse = ReplySendResponse | ReplyWaitResponse | ReplyEndResponse
