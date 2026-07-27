from __future__ import annotations

import html
import re
import unicodedata

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
WHITESPACE_RE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    if BeautifulSoup is not None:
        return BeautifulSoup(text, "html.parser").get_text(" ")
    return re.sub(r"<[^>]+>", " ", text)


def normalize_text(text: str, *, remove_html: bool = True, normalize_unicode: bool = True, lowercase: bool = False) -> str:
    if remove_html:
        text = strip_html(text)
    text = html.unescape(text)
    if normalize_unicode:
        text = unicodedata.normalize("NFKC", text)
    if lowercase:
        text = text.lower()
    return WHITESPACE_RE.sub(" ", text).strip()


def redact_patterns(text: str, *, urls: bool = False, emails: bool = True, phones: bool = False) -> str:
    if urls:
        text = URL_RE.sub("[URL]", text)
    if emails:
        text = EMAIL_RE.sub("[EMAIL]", text)
    if phones:
        text = PHONE_RE.sub("[PHONE]", text)
    return WHITESPACE_RE.sub(" ", text).strip()
