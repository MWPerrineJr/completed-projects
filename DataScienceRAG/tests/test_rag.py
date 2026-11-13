"""Unit tests for rag.py: no network, no API keys, no cost.

OpenAI and Supabase are replaced by fakes. An autouse fixture makes every real
client factory raise, so a test that forgets to patch something fails loudly
instead of spending money.

Run from the datascience_rag folder:
    python -m pytest -q -p no:dash tests/
"""
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableGenerator, RunnableLambda

import rag

REAL_GET_SUPABASE = rag.get_supabase  # kept for the env-var test


# --- Helpers -----------------------------------------------------------------
def hit(i=1, library="pandas", section="Reshaping and pivot tables", page=100, sim=0.6, **meta):
    """A fake row as returned by the match_documents RPC."""
    return {"id": f"{library}-p{i:04d}", "content": f"chunk text {i}",
            "metadata": {"library": library, "version": "1.0", "section": section, "page": page, **meta},
            "similarity": sim}


FIVE_HITS = [hit(i, section=f"Section {i}", page=100 + i, sim=0.7 - i / 100) for i in range(1, 6)]
USAGE = {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100}


def fake_llm(answer: rag.Answer, usage=USAGE):
    """Stands in for get_llm(): returns what with_structured_output(..., include_raw=True) returns."""
    return RunnableLambda(lambda _: {"parsed": answer, "raw": AIMessage(content="", usage_metadata=usage),
                                     "parsing_error": None})


def fake_stream_llm(parts):
    """Stands in for get_stream_llm(): yields partial dicts like the json_schema parser does."""
    def gen(inputs):
        for _ in inputs:        # consume the prompt value
            pass
        for p in parts:
            yield dict(p)
    return RunnableGenerator(gen)


def boom(*args, **kwargs):
    raise AssertionError("real client used in a unit test (network call)")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    for name in ("get_embeddings", "get_supabase", "get_llm", "get_stream_llm"):
        monkeypatch.setattr(rag, name, boom)


@pytest.fixture
def returns(monkeypatch):
    """returns(hits): make rag.search return these hits."""
    def _set(hits):
        monkeypatch.setattr(rag, "search", lambda *a, **k: hits)
    return _set


# --- 1. cite() -------------------------------------------------------------------
def test_cite_single_page_and_missing_url():
    c = rag.cite(hit(page=100))
    assert c["pages"] == "100"          # not "100-None" (Session 5 bug)
    assert c["url"] is None
    assert c["sim"] == 0.6


def test_cite_page_range_and_url():
    c = rag.cite(hit(page=5, page_end=35, source_url="https://example.org/doc"))
    assert c["pages"] == "5-35"
    assert c["url"] == "https://example.org/doc"


# --- 2. dedupe() -----------------------------------------------------------------
def test_dedupe_keeps_first_per_section():
    best, worse = rag.cite(hit(1, sim=0.58)), rag.cite(hit(2, sim=0.49))   # same library + section
    other = rag.cite(hit(3, section="Group by"))
    assert rag.dedupe([best, worse, other]) == [best, other]


# --- 3-4. Layer 1: similarity gate -------------------------------------------------
@pytest.mark.parametrize("hits", [[hit(sim=0.30)], []], ids=["below_min_sim", "no_hits"])
def test_gate_refuses_without_llm(returns, hits):
    returns(hits)                        # get_llm / get_stream_llm still raise if touched
    r = rag.ask("anything")
    assert r["refused_by"] == "gate" and r["confident"] is False
    assert r["cost"] == 0.0 and r["context"] is None
    assert r["answer"] == rag.REFUSAL

    events = list(rag.ask_stream("anything"))
    assert [kind for kind, _ in events] == ["refused"]
    assert events[0][1]["refused_by"] == "gate"


def test_gate_boundary_passes_at_min_sim(returns, monkeypatch):
    returns([hit(sim=rag.MIN_SIM)])      # exactly MIN_SIM is NOT refused (< not <=)
    monkeypatch.setattr(rag, "get_llm", lambda: fake_llm(rag.Answer(in_context=True, answer="ok", cited=[1])))
    assert rag.ask("q")["refused_by"] is None


# --- 5-6. Layer 2: LLM in_context and citations --------------------------------------
def test_llm_refusal_keeps_note_and_nearest_sources(returns, monkeypatch):
    returns(FIVE_HITS)
    note = "The context covers pandas reshaping, not Polars."
    monkeypatch.setattr(rag, "get_llm", lambda: fake_llm(rag.Answer(in_context=False, answer=note, cited=[])))
    r = rag.ask("How do I read a CSV lazily with Polars?")
    assert r["refused_by"] == "llm" and r["confident"] is False
    assert r["answer"] == rag.REFUSAL and r["llm_note"] == note
    assert [s["section"] for s in r["sources"]] == ["Section 1", "Section 2", "Section 3"]  # top-3 nearest
    assert r["cost"] == pytest.approx((1000 * 0.10 + 100 * 0.50) / 1e6)


def test_answer_ignores_out_of_range_citations(returns, monkeypatch):
    returns(FIVE_HITS)
    monkeypatch.setattr(rag, "get_llm",
                        lambda: fake_llm(rag.Answer(in_context=True, answer="Use pivot.", cited=[1, 9, 0])))
    r = rag.ask("pivot?")
    assert r["refused_by"] is None and r["confident"] is True
    assert [s["section"] for s in r["sources"]] == ["Section 1"]   # 9 and 0 dropped, no IndexError
    assert "[1] pandas · Section 1" in r["context"]


# --- 7-8. Streaming ------------------------------------------------------------------
def test_stream_tokens_join_to_final_answer(returns, monkeypatch):
    returns(FIVE_HITS)
    parts = [{"in_context": True},
             {"in_context": True, "answer": "Use "},
             {"in_context": True, "answer": "Use `df.pivot`"},
             {"in_context": True, "answer": "Use `df.pivot`."},
             {"in_context": True, "answer": "Use `df.pivot`.", "cited": [2]}]
    monkeypatch.setattr(rag, "get_stream_llm", lambda: fake_stream_llm(parts))
    events = list(rag.ask_stream("pivot?"))

    tokens = [x for kind, x in events if kind == "token"]
    kind, result = events[-1]
    assert kind == "done" and [k for k, _ in events[:-1]] == ["token"] * len(tokens)
    assert "".join(tokens) == result["answer"] == "Use `df.pivot`."
    assert [s["section"] for s in result["sources"]] == ["Section 2"]


def test_stream_refusal_shows_no_partial_text(returns, monkeypatch):
    returns(FIVE_HITS)
    note = "The context covers scikit-learn, not XGBoost."
    parts = [{"in_context": False},
             {"in_context": False, "answer": "The context covers"},
             {"in_context": False, "answer": note, "cited": []}]
    monkeypatch.setattr(rag, "get_stream_llm", lambda: fake_stream_llm(parts))
    events = list(rag.ask_stream("XGBoost early stopping?"))

    assert [kind for kind, _ in events] == ["refused"]   # zero token events
    assert events[0][1]["refused_by"] == "llm" and events[0][1]["llm_note"] == note


# --- 9. Lazy clients / env vars --------------------------------------------------------
def test_missing_env_vars_raise_clear_error(monkeypatch):
    for var in rag.REQUIRED_ENV:
        monkeypatch.delenv(var, raising=False)
    REAL_GET_SUPABASE.cache_clear()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY"):
        REAL_GET_SUPABASE()
