
import json
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
HISTORY_FILE = ROOT_DIR / "data" / "chat_history.json"


def _normalize_entry(entry: dict, index: int) -> dict:
    meta = entry.get("meta") or {}
    sources = entry.get("sources") or []
    return {
        "id": entry.get("id", index + 1),
        "question": entry.get("question", ""),
        "answer": entry.get("answer", entry.get("ans", "")),
        "sources": sources if isinstance(sources, list) else [],
        "meta": meta if isinstance(meta, dict) else {},
        "timestamp": entry.get("timestamp", ""),
    }


def _load_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        raw = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            return []
        return [_normalize_entry(entry, idx) for idx, entry in enumerate(raw)]
    except Exception as exc:
        print(f"Error loading history: {exc}")
        return []

def _save(history: list):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
def add_entry(question: str, ans: str, sources: list | None = None, meta: dict | None = None) -> dict:
    history = _load_history()
    entry = {
        "id": len(history) +1 ,
        "question": question,
        "answer": ans or "",
        "sources": sources or [],
        "meta": meta or {},
        "timestamp": datetime.now().isoformat(),
    }
    history.append(entry)
    _save(history)
    return entry

def get_all_history() -> list:
    return _load_history()
def get_by_id(entry_id: int)->dict | None:
    return next((e for e in _load_history() if e.get("id") == entry_id), None)
def clear() -> None:
    _save([])
