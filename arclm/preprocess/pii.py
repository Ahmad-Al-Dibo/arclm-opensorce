from __future__ import annotations

import re

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def redact_pii(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = IP_RE.sub("[IP]", text)
    return text


def pii_reasons(text: str) -> list[str]:
    reasons = []
    if EMAIL_RE.search(text):
        reasons.append("contains_email")
    if PHONE_RE.search(text):
        reasons.append("contains_phone")
    if IP_RE.search(text):
        reasons.append("contains_ip")
    return reasons
