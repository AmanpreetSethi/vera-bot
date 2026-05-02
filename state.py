from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ContextEntry = Dict[str, Any]
ContextStore = Dict[Tuple[str, str], ContextEntry]
ConversationMessage = Dict[str, str]


class AppState:
    def __init__(self) -> None:
        self.started_at: datetime = datetime.now(timezone.utc)
        self.contexts: ContextStore = {}
        self.context_lock = threading.Lock()
        self.conversations: Dict[str, List[ConversationMessage]] = {}
        self.conversations_lock = threading.Lock()
        self.sent_suppressions: set[str] = set()
        self.suppressions_lock = threading.Lock()

    def preload_category_contexts(self, base_dir: Path) -> None:
        category_dir = base_dir / "dataset" / "categories"
        slugs = ["dentists", "gyms", "pharmacies", "restaurants", "salons"]

        with self.context_lock:
            for slug in slugs:
                file_path = category_dir / f"{slug}.json"
                if not file_path.exists():
                    continue
                with file_path.open("r", encoding="utf-8") as fp:
                    payload = json.load(fp)

                self.contexts[("category", slug)] = {
                    "scope": "category",
                    "context_id": slug,
                    "version": 0,
                    "payload": payload,
                    "delivered_at": datetime.now(timezone.utc).isoformat(),
                    "stored_at": datetime.now(timezone.utc).isoformat(),
                }

    def context_counts(self) -> Dict[str, int]:
        counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
        with self.context_lock:
            for (scope, _), _entry in self.contexts.items():
                counts[scope] += 1
        return counts

    def put_context_if_newer(
        self,
        scope: str,
        context_id: str,
        version: int,
        payload: Dict[str, Any],
        delivered_at_iso: str,
    ) -> bool:
        key = (scope, context_id)
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.context_lock:
            existing = self.contexts.get(key)
            if existing and version <= int(existing["version"]):
                return False
            self.contexts[key] = {
                "scope": scope,
                "context_id": context_id,
                "version": version,
                "payload": payload,
                "delivered_at": delivered_at_iso,
                "stored_at": now_iso,
            }
        return True

    def get_context_payload(self, scope: str, context_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not context_id:
            return None
        with self.context_lock:
            entry = self.contexts.get((scope, context_id))
            if not entry:
                return None
            return entry.get("payload")

    def check_and_mark_suppression(self, suppression_key: Optional[str]) -> bool:
        if not suppression_key:
            return True
        with self.suppressions_lock:
            if suppression_key in self.sent_suppressions:
                return False
            self.sent_suppressions.add(suppression_key)
            return True

    def is_suppressed(self, suppression_key: Optional[str]) -> bool:
        if not suppression_key:
            return False
        with self.suppressions_lock:
            return suppression_key in self.sent_suppressions

    def mark_suppression_sent(self, suppression_key: Optional[str]) -> None:
        if not suppression_key:
            return
        with self.suppressions_lock:
            self.sent_suppressions.add(suppression_key)

    def append_conversation_message(
        self,
        conversation_id: str,
        from_role: str,
        body: str,
        ts_iso: str,
    ) -> None:
        with self.conversations_lock:
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []
            self.conversations[conversation_id].append(
                {"from": from_role, "body": body, "ts": ts_iso}
            )

    def get_conversation(self, conversation_id: str) -> List[ConversationMessage]:
        with self.conversations_lock:
            return list(self.conversations.get(conversation_id, []))

    def wipe_all(self) -> None:
        with self.context_lock:
            self.contexts.clear()
        with self.conversations_lock:
            self.conversations.clear()
        with self.suppressions_lock:
            self.sent_suppressions.clear()
