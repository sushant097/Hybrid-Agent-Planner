import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional logging from agent.py
try:
    from agent import log
except ImportError:  # Fallback logger
    import datetime

    def log(stage: str, msg: str) -> None:
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [{stage}] {msg}")


# ---------- Data model ----------

@dataclass
class HistoricalExample:
    session_id: str
    turn_index: int
    user_query: str
    final_answer: str
    tools_used: List[str]
    successful_tools: List[str]
    tags: List[str]
    keywords: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "user_query": self.user_query,
            "final_answer": self.final_answer,
            "tools_used": self.tools_used,
            "successful_tools": self.successful_tools,
            "tags": self.tags,
            "keywords": self.keywords,
        }


# ---------- Paths & helpers ----------

def _find_memory_root(memory_path: Path) -> Path:
    """
    Given the per-session memory JSON path, walk up until we find the 'memory'
    directory. If not found, fall back to its parent directory.
    """
    for parent in [memory_path] + list(memory_path.parents):
        if parent.name == "memory":
            return parent
    return memory_path.parent


def _get_index_path(memory_path: Path) -> Path:
    """
    Global historical index stored as:
        memory/historical_conversation_store.json
    """
    memory_root = _find_memory_root(memory_path)
    return memory_root / "historical_conversation_store.json"


_STOPWORDS = {
    "the", "is", "a", "an", "of", "and", "or", "to", "in", "on", "for",
    "with", "at", "by", "from", "as", "about", "what", "which", "who",
    "how", "much", "many", "when", "where", "why", "do", "does", "did",
    "his", "her", "their", "its", "this", "that", "these", "those",
}


def _normalize_text(text: str) -> List[str]:
    """
    Very lightweight tokenizer + stopword removal for keyword extraction.
    """
    text = text.lower()
    # Keep words and numbers
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t not in _STOPWORDS]


def _load_index(index_path: Path) -> List[Dict[str, Any]]:
    if not index_path.exists():
        return []
    try:
        with index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        log("history", "Index file not a list; resetting.")
        return []
    except Exception as e:
        log("history", f"Failed to load historical index: {e}")
        return []


def _save_index(index_path: Path, items: List[Dict[str, Any]]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


# ---------- Parsing session memory ----------

def _extract_user_query(run_text: str) -> Optional[str]:
    """
    Example run_metadata text:
      "Started new session with input: What is X? at 2025-11-28T20:07:31.568388"
    """
    marker = "Started new session with input:"
    if marker not in run_text:
        return None
    after = run_text.split(marker, 1)[1].strip()
    # Strip trailing " at YYYY..." part if present
    if " at " in after:
        after = after.split(" at ", 1)[0].strip()
    return after or None


def _extract_final_answer(entry: Dict[str, Any]) -> Optional[str]:
    """
    Look for a clean FINAL_ANSWER inside a tool_output entry.
    We avoid indexing partial / failed runs.
    """
    # 1) Prefer explicit final_answer field if present
    fa = entry.get("final_answer")
    if isinstance(fa, str) and fa.startswith("FINAL_ANSWER:"):
        return fa

    # 2) Otherwise look into tool_result.result
    tool_result = entry.get("tool_result") or {}
    result_text = tool_result.get("result")
    if isinstance(result_text, str) and result_text.startswith("FINAL_ANSWER:"):
        return result_text

    return None


def _should_index_final_answer(fa: str) -> bool:
    """
    Filter out "junk" FINAL_ANSWER strings like:
      - FINAL_ANSWER: [Could not generate valid solve()]
      - FINAL_ANSWER: [unknown]
      - Diagnostics-only messages
    """
    if not fa.startswith("FINAL_ANSWER:"):
        return False
    lowered = fa.lower()
    junk_patterns = [
        "could not generate valid solve",
        "max steps reached",
        "unknown",
        "unexpected",
    ]
    return not any(pat in lowered for pat in junk_patterns)


def _parse_session_memory(
    memory_path: Path,
    session_id: str,
) -> List[HistoricalExample]:
    """
    Read the per-session memory file and extract (query, FINAL_ANSWER) pairs.
    """
    try:
        with memory_path.open("r", encoding="utf-8") as f:
            events = json.load(f)
    except Exception as e:
        log("history", f"Failed to read memory file {memory_path}: {e}")
        return []

    if not isinstance(events, list):
        log("history", f"Memory file {memory_path} not a list; skipping.")
        return []

    examples: List[HistoricalExample] = []
    i = 0
    turn_index = 0

    while i < len(events):
        evt = events[i]
        i += 1

        # We only care about run_start metadata
        if evt.get("type") != "run_metadata":
            continue

        txt = evt.get("text") or ""
        if "Started new session with input:" not in txt:
            continue

        user_query = _extract_user_query(txt)
        if not user_query:
            continue

        # Find the next tool_output entry for this run
        final_answer: Optional[str] = None
        tools_used: List[str] = []
        successful_tools: List[str] = []
        tags: List[str] = []

        j = i
        while j < len(events):
            evt2 = events[j]
            if evt2.get("type") == "run_metadata":
                # Start of the next run
                break

            if evt2.get("type") == "tool_output":
                tool_name = evt2.get("tool_name")
                if tool_name:
                    tools_used.append(tool_name)
                    if evt2.get("success") is True:
                        successful_tools.append(tool_name)
                fa = _extract_final_answer(evt2)
                if fa is not None:
                    final_answer = fa

                # Merge tags
                evt_tags = evt2.get("tags") or []
                if isinstance(evt_tags, list):
                    tags.extend(str(t) for t in evt_tags)

            j += 1

        # Move pointer
        i = j

        if not final_answer or not _should_index_final_answer(final_answer):
            continue

        keywords = _normalize_text(user_query)
        example = HistoricalExample(
            session_id=session_id,
            turn_index=turn_index,
            user_query=user_query,
            final_answer=final_answer,
            tools_used=list(dict.fromkeys(tools_used)),         # dedupe, keep order
            successful_tools=list(dict.fromkeys(successful_tools)),
            tags=list(dict.fromkeys(tags)),
            keywords=keywords,
        )
        examples.append(example)
        turn_index += 1

    return examples


# ---------- Public API ----------

def update_index_for_session(memory_path_str: str, session_id: str) -> None:
    """
    Called from loop.py after each run to keep the global index fresh.

    - memory_path_str: path to the per-session JSON log (from MemoryManager).
    - session_id: unique ID for this logical session (already built in loop.py).
    """
    memory_path = Path(memory_path_str)
    index_path = _get_index_path(memory_path)

    try:
        new_examples = _parse_session_memory(memory_path, session_id)
        if not new_examples:
            log("history", f"No indexable examples found for session {session_id}.")
            return

        index = _load_index(index_path)

        # Build a set of (session_id, turn_index) already present
        existing_keys = {
            (item.get("session_id"), item.get("turn_index"))
            for item in index
            if isinstance(item, dict)
        }

        added = 0
        for ex in new_examples:
            key = (ex.session_id, ex.turn_index)
            if key in existing_keys:
                continue
            index.append(ex.to_dict())
            existing_keys.add(key)
            added += 1

        if added:
            _save_index(index_path, index)
            log("history", f"✅ Updated historical index for session {session_id} (added {added} entries).")
        else:
            log("history", f"Historical index already up-to-date for session {session_id}.")
    except Exception as e:
        log("history", f"Failed to update historical index for session {session_id}: {e}")


def _jaccard_similarity(a: List[str], b: List[str]) -> float:
    """
    Simple Jaccard similarity over keyword sets.
    """
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def load_similar_examples(
    user_query: str,
    top_k: int = 3,
    memory_root: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return up to `top_k` historical examples most similar to `user_query`.

    Used in:
      - decision.py (to build {historical_examples} / {memory_texts} sections)
      - loop.py's `find_cached_answer` (which then does exact-string match
        on `user_query` to safely reuse a cached FINAL_ANSWER).
    """
    # Locate index file (default: ./memory/historical_conversation_store.json)
    if memory_root is None:
        memory_root_path = Path("memory")
    else:
        memory_root_path = Path(memory_root)

    index_path = memory_root_path / "historical_conversation_store.json"

    index = _load_index(index_path)
    if not index:
        return []

    q_keywords = _normalize_text(user_query)

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for item in index:
        if not isinstance(item, dict):
            continue

        fa = item.get("final_answer")
        uq = item.get("user_query")
        if not isinstance(fa, str) or not isinstance(uq, str):
            continue
        if not fa.startswith("FINAL_ANSWER:"):
            continue

        if fa.strip() == "FINAL_ANSWER: [Could not generate valid solve()]":
            continue

        kw = item.get("keywords")
        if not isinstance(kw, list) or not kw:
            kw = _normalize_text(uq)

        sim = _jaccard_similarity(q_keywords, kw)
        if sim <= 0:
            continue

        scored.append((sim, item))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    top_items = [item for _, item in scored[:top_k]]

    # Log for debugging
    try:
        dbg = ", ".join(
            f'"{i.get("user_query")}" (sim={s:.2f})'
            for s, i in scored[:min(len(scored), top_k)]
        )
        log("history", f"Similar examples for '{user_query}': {dbg}")
    except Exception:
        pass

    return top_items
