"""
tests/test_agent_rag.py
=======================
Unit tests for the lightweight keyword-based RAG system in core/agent_rag.py.
"""

import pytest
from core.agent_rag import retrieve, format_for_prompt, _ensure_loaded, _CHUNKS


def test_chunks_loaded_from_knowledge_dir():
    _ensure_loaded()
    assert len(_CHUNKS) > 0


def test_retrieve_returns_top_k():
    results = retrieve("RMSE inlier", top_k=4)
    assert len(results) == 4
    for r in results:
        assert r["score"] > 0
        assert "text" in r
        assert "source_file" in r
        assert "section_title" in r


def test_retrieve_ohrc_query():
    results = retrieve("OHRC resolution", top_k=4)
    assert len(results) > 0
    match = any("ohrc" in r["source_file"].lower() or "sensors" in r["source_file"].lower() for r in results)
    assert match, f"Expected 'ohrc' or 'sensors' in one of: {[r['source_file'] for r in results]}"


def test_format_for_prompt_truncates():
    results = retrieve("RMSE inlier ratio lunar surface crater", top_k=4)
    ctx = format_for_prompt(results, max_chars=1200)
    assert len(ctx) <= 1200
    assert len(ctx) > 0
