from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple


BASE_URL = "http://localhost:8080"


def _request(
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Tuple[int, Dict[str, Any]]:
    url = f"{BASE_URL}{path}"
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url=url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.getcode()
            body = resp.read().decode("utf-8")
            return status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        parsed = json.loads(body) if body else {}
        return exc.code, parsed


def _print_result(name: str, status: int, body: Dict[str, Any]) -> None:
    print(f"=== {name} ===")
    print(f"status: {status}")
    print(json.dumps(body, indent=2, ensure_ascii=False))
    print()


def main() -> int:
    now_iso = datetime.now(timezone.utc).isoformat()

    checks = [
        ("GET /v1/healthz", "GET", "/v1/healthz", None),
        ("GET /v1/metadata", "GET", "/v1/metadata", None),
        (
            "POST /v1/context (merchant)",
            "POST",
            "/v1/context",
            {
                "scope": "merchant",
                "context_id": "m_001",
                "version": 1,
                "payload": {
                    "merchant_id": "m_001",
                    "name": "Dr. Mehta Dental Care",
                    "category_slug": "dentists",
                    "language_pref": "hi-en mix",
                },
                "delivered_at": now_iso,
            },
        ),
        (
            "POST /v1/context (trigger)",
            "POST",
            "/v1/context",
            {
                "scope": "trigger",
                "context_id": "t_001",
                "version": 1,
                "payload": {
                    "trigger_id": "t_001",
                    "kind": "perf_dip",
                    "merchant_id": "m_001",
                    "suppression_key": "sup_m_001_perf_dip",
                    "urgency": "medium",
                },
                "delivered_at": now_iso,
            },
        ),
        ("POST /v1/tick", "POST", "/v1/tick", {"now": now_iso, "available_triggers": ["t_001"]}),
        (
            "POST /v1/reply",
            "POST",
            "/v1/reply",
            {
                "conversation_id": "conv_m_001_t_001",
                "merchant_id": "m_001",
                "customer_id": None,
                "from_role": "merchant",
                "message": "Yes, please do this",
                "received_at": now_iso,
                "turn_number": 1,
            },
        ),
        ("POST /v1/teardown", "POST", "/v1/teardown", {}),
        ("GET /v1/healthz (after teardown)", "GET", "/v1/healthz", None),
    ]

    ok = True
    for name, method, path, payload in checks:
        status, body = _request(method, path, payload)
        _print_result(name, status, body)
        if path in {"/v1/healthz", "/v1/metadata", "/v1/context", "/v1/teardown"} and status >= 400:
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
