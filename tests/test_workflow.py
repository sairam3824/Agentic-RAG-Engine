from __future__ import annotations

from src.graph.nodes import AgenticRAGNodes
from src.graph.workflow import AgenticRAGEngine
from src.llm import LLMAdapter


class FakeLLM:
    def __init__(self, json_payload=None, text_payload=None):
        self._json_payload = json_payload
        self._text_payload = text_payload

    def json_response(self, system_prompt: str, user_prompt: str):
        return self._json_payload

    def text_response(self, system_prompt: str, user_prompt: str):
        return self._text_payload


class DummyRetriever:
    def __init__(self, docs=None):
        self.docs = docs or []

    def search(self, query: str, k: int = 5):
        return self.docs[:k]


class DummyVectorRetriever(DummyRetriever):
    def add_documents(self, docs):
        self.docs.extend(docs)
        return len(docs)


def make_engine(vector_docs=None, web_docs=None, sql_docs=None):
    nodes = AgenticRAGNodes(
        llm=LLMAdapter(api_key=""),
        vector_retriever=DummyVectorRetriever(vector_docs or []),
        web_retriever=DummyRetriever(web_docs or []),
        sql_retriever=DummyRetriever(sql_docs or []),
        retrieval_threshold=0.8,
    )
    return AgenticRAGEngine(nodes=nodes)


def test_query_without_retrieval_for_simple_math():
    engine = make_engine()
    state = engine.run("2 + 2", sources=["vector", "web", "sql"], max_iterations=3)

    assert state["needs_retrieval"] is False
    assert state["iteration_count"] == 0
    assert "answer" in state
    assert "4" in state["answer"]


def test_retrieval_loops_until_max_iterations():
    poor_docs = [
        {
            "id": "d1",
            "content": "Unrelated text about cooking recipes.",
            "source": "vector_store",
            "score": 0.1,
            "metadata": {},
        }
    ]
    engine = make_engine(vector_docs=poor_docs)

    state = engine.run("What is the latest cloud database benchmark in 2026?", sources=["vector"], max_iterations=3)

    assert state["iteration_count"] == 3
    assert state["evaluation"]["good_enough"] is True
    assert len(state["retrieval_trace"]) >= 7


def test_requested_source_is_respected():
    sql_docs = [
        {
            "id": "sql-1",
            "content": '{"quarter":"2025-Q4","revenue_usd_mn":149.1}',
            "source": "sqlite:kpi_metrics",
            "score": 0.9,
            "metadata": {"table": "kpi_metrics"},
        }
    ]
    engine = make_engine(sql_docs=sql_docs)

    state = engine.run("Compare quarterly revenue performance", sources=["sql"], max_iterations=2)

    retriever_events = [e for e in state["retrieval_trace"] if e["node"] == "RETRIEVER"]
    assert retriever_events
    assert retriever_events[0]["data"]["selected_sources"] == ["sql"]


def test_requested_sources_are_normalized_and_deduplicated():
    sql_docs = [
        {
            "id": "sql-1",
            "content": '{"quarter":"2025-Q4","revenue_usd_mn":149.1}',
            "source": "sqlite:kpi_metrics",
            "score": 0.9,
            "metadata": {"table": "kpi_metrics"},
        }
    ]
    engine = make_engine(sql_docs=sql_docs)

    state = engine.run(
        "Compare quarterly revenue performance",
        sources=[" SQL ", "sql", "INVALID"],
        max_iterations=2,
    )
    retriever_events = [e for e in state["retrieval_trace"] if e["node"] == "RETRIEVER"]
    assert retriever_events
    assert retriever_events[0]["data"]["selected_sources"] == ["sql"]


def test_query_analyzer_coerces_llm_boolean_string():
    nodes = AgenticRAGNodes(
        llm=FakeLLM(
            json_payload={
                "needs_retrieval": "false",
                "optimized_query": "short query",
                "query_type": "FACTUAL",
            }
        ),
        vector_retriever=DummyVectorRetriever([]),
        web_retriever=DummyRetriever([]),
        sql_retriever=DummyRetriever([]),
    )
    result = nodes.query_analyzer({"question": "Give me a quick greeting"})
    assert result["needs_retrieval"] is False
    assert result["query_type"] == "factual"


def test_direct_answer_fallback_evaluates_math_expression():
    engine = make_engine()
    state = engine.run("5 * (3 + 1)", sources=["vector", "web", "sql"], max_iterations=3)
    assert state["needs_retrieval"] is False
    assert "20" in state["answer"]
def test_requested_multiple_sources_are_respected():
    engine = make_engine()
    # "factual" query type usually defaults to ["vector", "web"]
    # We want to ensure that requesting ["web", "sql"] respects BOTH.
    state = engine.run("What is the latest revenue?", sources=["web", "sql"], max_iterations=1)
    
    retriever_events = [e for e in state["retrieval_trace"] if e["node"] == "RETRIEVER"]
    assert retriever_events
    selected = retriever_events[0]["data"]["selected_sources"]
    assert "web" in selected
    assert "sql" in selected
    assert len(selected) == 2
