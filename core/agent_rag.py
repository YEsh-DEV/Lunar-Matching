"""
core/agent_rag.py
=================
Lightweight, dependency-free keyword-based RAG engine for LUNA-MATCH.
Works fully offline with zero external vector DB or embedding requirements.
"""

from pathlib import Path
import re
from typing import List, Dict, Any

KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"


def _load_all_chunks() -> List[Dict[str, Any]]:
    """
    Load all .md files from knowledge/ directory.
    Split each into chunks of ~300 words with 50-word overlap.
    Each chunk: {text, source_file, section_title}.
    section_title = nearest preceding '## ' heading.
    Cache result at module level (computed once at import time).
    """
    chunks: List[Dict[str, Any]] = []
    if not KNOWLEDGE_DIR.exists():
        return chunks

    for md_path in sorted(KNOWLEDGE_DIR.rglob("*.md")):
        source_file = f"{md_path.parent.name}/{md_path.name}"
        try:
            content = md_path.read_text(encoding="utf-8")
        except Exception:
            continue

        lines = content.splitlines()
        current_section = "Overview"
        current_lines: List[str] = []

        def add_chunk_from_lines(sec_title: str, sec_lines: List[str]):
            text = "\n".join(sec_lines).strip()
            if not text:
                return
            words = text.split()
            if len(words) > 300:
                step = 250
                for i in range(0, len(words), step):
                    chunk_words = words[i : i + 300]
                    chunk_text = " ".join(chunk_words)
                    chunks.append({
                        "text": chunk_text,
                        "source_file": source_file,
                        "section_title": sec_title,
                    })
                    if i + 300 >= len(words):
                        break
            else:
                chunks.append({
                    "text": text,
                    "source_file": source_file,
                    "section_title": sec_title,
                })

        for line in lines:
            if line.startswith("## "):
                if current_lines:
                    add_chunk_from_lines(current_section, current_lines)
                    current_lines = []
                current_section = line[3:].strip()
                current_lines.append(line)
            else:
                current_lines.append(line)

        if current_lines:
            add_chunk_from_lines(current_section, current_lines)

    return chunks


_CHUNKS: List[Dict[str, Any]] = []
_CHUNKS_LOADED = False


def _ensure_loaded() -> None:
    global _CHUNKS, _CHUNKS_LOADED
    if _CHUNKS_LOADED:
        return
    loaded = _load_all_chunks()
    _CHUNKS.clear()
    _CHUNKS.extend(loaded)
    _CHUNKS_LOADED = True


def retrieve(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Keyword-based retrieval. Score each chunk by counting how many
    query words (lowercased, split by whitespace, len > 3) appear in
    the chunk text (lowercased). Return top_k by score.
    If score == 0 for all chunks: return top_k chunks from the most
    relevant source file (match query words against source_file name).
    Never return empty list if chunks exist — always return top_k.
    Each returned dict: {text, source_file, section_title, score}.
    """
    _ensure_loaded()
    if not _CHUNKS:
        return []

    # Query words (lowercased, split by whitespace, len > 3)
    query_words = [re.sub(r"^\W+|\W+$", "", w).lower() for w in query.split()]
    query_words = [w for w in query_words if len(w) > 3]

    scored_chunks: List[Dict[str, Any]] = []
    for chunk in _CHUNKS:
        text_lower = f"{chunk['section_title'].lower()} {chunk['text'].lower()}"
        sf_lower = chunk["source_file"].lower()
        score = float(sum(1 for qw in query_words if qw in text_lower))
        score += float(sum(2 for qw in query_words if qw in sf_lower))
        scored_chunks.append({
            "text": chunk["text"],
            "source_file": chunk["source_file"],
            "section_title": chunk["section_title"],
            "score": score,
        })

    # Sort descending by score
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    if all(c["score"] == 0 for c in scored_chunks):
        def file_score(c):
            sf_lower = c["source_file"].lower()
            return sum(1 for qw in query_words if qw in sf_lower)

        scored_by_file = sorted(scored_chunks, key=file_score, reverse=True)
        return scored_by_file[:top_k]

    return scored_chunks[:top_k]


def format_for_prompt(chunks: List[Dict[str, Any]], max_chars: int = 1200) -> str:
    """
    Format retrieved chunks as a context block for the LLM prompt.
    Format: "Source: {source_file} — {section_title}\n{text}\n\n"
    Truncate total output to max_chars.
    """
    blocks = []
    for c in chunks:
        blocks.append(f"Source: {c['source_file']} — {c['section_title']}\n{c['text']}\n\n")
    full_text = "".join(blocks)
    if len(full_text) > max_chars:
        return full_text[:max_chars]
    return full_text
