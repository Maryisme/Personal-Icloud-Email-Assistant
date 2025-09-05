# clean_from_sources.py
import json
import os
import re
import unicodedata
from html import unescape
from typing import Dict, Any, Optional, List, Iterable
from email.header import decode_header, make_header

# ---- CONFIG: keep raw HTML/text in output for debugging/reference?
KEEP_RAW_BODIES = True  # set False if you only care about text_plain

# ---------- RFC 2047 / Unicode helpers ----------
def _decode_rfc2047(s: Optional[str]) -> str:
    if not s:
        return ""
    try:
        return str(make_header(decode_header(s)))
    except Exception:
        return s

def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", unescape(s or ""))

# ---------- Address helpers ----------
def _decode_addr(d: Dict[str, Any]) -> Dict[str, Any]:
    if not d:
        return {"name": "", "email": ""}
    return {
        "name": _nfc(_decode_rfc2047(d.get("name") or "")),
        "email": (d.get("email") or "").strip(),
    }

def _decode_list(lst, *, keep_empty=True):
    out: List[Any] = []
    for it in (lst or []):
        if isinstance(it, dict):
            out.append(_decode_addr(it))
        else:
            out.append(_nfc(_decode_rfc2047(str(it))))
    if keep_empty:
        return out
    # (unused now) legacy filtered behavior:
    cleaned: List[Any] = []
    for x in out:
        if isinstance(x, dict):
            if (x.get("email") or x.get("name") or "").strip():
                cleaned.append(x)
        else:
            if str(x).strip():
                cleaned.append(x)
    return cleaned


# ---------- “Weird characters” / control-char scrub ----------
CTRL_SAFE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
ZERO_WIDTH = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]")

def strip_controls(s: str) -> str:
    """Remove zero-width and non-printable control chars, but keep \n and \t."""
    if not s:
        return ""
    # First remove explicit CR to avoid \r\n escapes in JSON; normalize newlines to \n
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Remove zero-width formatting marks
    s = ZERO_WIDTH.sub("", s)
    # Remove other control chars (except \n and \t which we kept by excluding them in the regex)
    s = CTRL_SAFE.sub("", s)
    return s

# ---------- Binary-ish detector ----------
def looks_binaryish(s: str) -> bool:
    """Heuristic: treat as binary if too many control chars present."""
    if not s:
        return False
    if "\x00" in s:
        return True
    ctrl = sum(1 for ch in s if ord(ch) < 32 and ch not in ("\n", "\t"))
    return (ctrl > 64) or (ctrl / max(len(s), 1) > 0.01)

# ---------- HTML → text (robust, no content stripping) ----------
def html_to_text(html: str) -> str:
    """Convert HTML to readable text like a mail client (no spam/junk stripping)."""
    if not html:
        return ""
    text = ""
    try:
        from bs4 import BeautifulSoup
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")
        # remove script/style noise but otherwise keep all visible text
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    except Exception:
        # Fallback without bs4
        tmp = re.sub(r"(?i)<\s*(br|/p|/div|p|div)\s*>", "\n", html)
        text = re.sub(r"<[^>]+>", "", tmp)
    # light whitespace normalization
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return _nfc(text).strip()

# ---------- Core normalizer ----------
def minimal_normalize_email(rec: Dict[str, Any], *, make_plain_from_html: bool = True) -> Dict[str, Any]:
    out = dict(rec)  # shallow copy

    # Headers / names
    out["subject"] = strip_controls(_nfc(_decode_rfc2047(rec.get("subject"))))
    out["from"] = _decode_addr(rec.get("from") or {})
    out["to"]  = _decode_list(rec.get("to"),  keep_empty=True)   # keep as-is (decoded), no dropping
    out["cc"]  = _decode_list(rec.get("cc"),  keep_empty=False)  # OK to keep filtering for cc/bcc
    out["bcc"] = _decode_list(rec.get("bcc"), keep_empty=False)


    # Attachments: decode filename only
    atts = []
    for a in (rec.get("attachments") or []):
        a2 = dict(a)
        a2["filename"] = strip_controls(_nfc(_decode_rfc2047(a.get("filename") or "")))
        atts.append(a2)
    out["attachments"] = atts

    # Bodies (raw)
    raw_text = rec.get("body_text") or ""
    raw_html = rec.get("body_html") or ""

    # Clean raw text minimally: drop if binary-like; normalize newlines; strip control chars
    if looks_binaryish(raw_text):
        cleaned_text = ""
    else:
        cleaned_text = strip_controls(_nfc(raw_text))

    # Always produce a human-readable plain text
    if cleaned_text:
        text_plain = cleaned_text
    elif make_plain_from_html and raw_html:
        text_plain = strip_controls(html_to_text(raw_html))
    else:
        text_plain = ""

    # Optionally keep raw bodies; else drop them
    if KEEP_RAW_BODIES:
        out["body_text"] = cleaned_text  # cleaned version of original text part
        out["body_html"] = raw_html      # untouched HTML
    else:
        out.pop("body_text", None)
        out.pop("body_html", None)

    out["text_plain"] = text_plain
    return out

# ---------- JSONL I/O ----------
def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def clean_file(src_path: str, make_plain_from_html: bool = True) -> str:
    base, _ = os.path.splitext(src_path)
    dst_path = base + ".clean.jsonl"
    cleaned = (minimal_normalize_email(rec, make_plain_from_html=make_plain_from_html)
               for rec in iter_jsonl(src_path))
    write_jsonl(dst_path, cleaned)
    return dst_path

# ---------- Main ----------
if __name__ == "__main__":
    sources = [
        "mailstore/normalized/INBOX.jsonl",
        "mailstore/normalized/Sent Messages.jsonl",
    ]
    for src in sources:
        dst = clean_file(src, make_plain_from_html=True)
        print(f"[OK] {src} -> {dst} (KEEP_RAW_BODIES={KEEP_RAW_BODIES})")
