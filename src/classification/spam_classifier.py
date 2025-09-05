"""
Spam classifier via local Ollama (Qwen models), labeling JSONL emails.
Requires OLLAMA_URL and a local model pulled (e.g., `ollama pull qwen3:8b`).
"""
from __future__ import annotations
import os
import requests, json
from typing import Callable

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

_SYSTEM = (
    "You are a strict email spam classifier. "
    "Return ONLY JSON with key 'label' as 'spam' or 'not_spam'."
)

def process_jsonl(src_path: str, dst_path: str, classify_fn: Callable[[str], str]) -> None:
    bad_out = src_path + ".badlines.txt"
    wrote = 0
    with open(src_path, "r", encoding="utf-8") as fin,              open(dst_path, "w", encoding="utf-8") as fout,              open(bad_out, "w", encoding="utf-8") as bad:
        for i, line in enumerate(fin, start=1):
            s = line.rstrip("\n")
            if not s.strip(): continue
            try:
                rec = json.loads(s)
            except Exception as e:
                bad.write(f"{i}\t{e}\t{s[:500]}\n"); continue
            text = (rec.get("text") or rec.get("text_plain") or rec.get("body_text") or "")
            rec["spam_classification"] = classify_fn(text)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wrote += 1
    print(f"[OK] wrote {wrote} labeled rows to {dst_path}")
    print(f"[INFO] bad lines logged to {bad_out}")

def classify_spam_ollama(text: str, model: str = MODEL) -> str:
    payload = {
        "model": model,
        "prompt": f"System: {_SYSTEM}\nUser: Classify the email. Return JSON only.\nEMAIL:\n{text}\nAssistant:",
        "stream": False,
        "format": {
            "type": "object",
            "properties": { "label": { "type": "string", "enum": ["spam","not_spam"] } },
            "required": ["label"],
            "additionalProperties": False
        }
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    resp = r.json().get("response", "")
    try:
        obj = json.loads(resp) if isinstance(resp, str) else resp
        lbl = str(obj.get("label", "")).lower()
        return "spam" if lbl == "spam" else "not_spam"
    except Exception:
        low = str(resp).lower()
        return "spam" if ("spam" in low and "not_spam" not in low) else "not_spam"

if __name__ == "__main__":
    process_jsonl(
        "mailstore/normalized/merged.clean.jsonl",
        "mailstore/normalized/merged.clean.labeled.jsonl",
        classify_spam_ollama
    )
