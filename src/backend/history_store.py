
import json
from datetime import datetime
from pathlib import Path

HISTORY_FILE = Path("data/chat_history.json")

def _load_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Error loading history: {exc}")
        return []

def _save(history: list):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
def  add_entry(question : str , ans : str , sources: list , meta : list) -> dict:
    history = _load_history()
    entry = {
        "id": len(history) +1 ,
        "question": question,
        "answer": ans,
        "sources": sources,
        "meta": meta,
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