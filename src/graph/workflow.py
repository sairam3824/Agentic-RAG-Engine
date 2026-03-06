from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.graph.nodes import AgenticRAGNodes
from src.graph.state import RAGState
from src.llm import LLMAdapter
from src.retrievers.sql import SQLiteRetriever
from src.retrievers.vector import VectorRetriever
from src.retrievers.web import TavilyWebRetriever


def _route_after_query_analyzer(state: RAGState) -> str:
    return "RETRIEVER" if state.get("needs_retrieval", True) else "GENERATOR"


def _route_after_evaluator(state: RAGState) -> str:
    evaluation = state.get("evaluation", {})
    if evaluation.get("good_enough", False):
        return "GENERATOR"

    iteration_count = int(state.get("iteration_count", 0))
    max_iterations = int(state.get("max_iterations", 3))
    if iteration_count >= max_iterations:
        return "GENERATOR"

    return "RETRIEVER"


def build_workflow(nodes: AgenticRAGNodes):
    builder = StateGraph(RAGState)

    builder.add_node("QUERY_ANALYZER", nodes.query_analyzer)
    builder.add_node("RETRIEVER", nodes.retriever)
    builder.add_node("EVALUATOR", nodes.evaluator)
    builder.add_node("GENERATOR", nodes.generator)
    builder.add_node("FACT_CHECKER", nodes.fact_checker)

    builder.add_edge(START, "QUERY_ANALYZER")
    builder.add_conditional_edges(
        "QUERY_ANALYZER",
        _route_after_query_analyzer,
        {
            "RETRIEVER": "RETRIEVER",
            "GENERATOR": "GENERATOR",
        },
    )
    builder.add_edge("RETRIEVER", "EVALUATOR")
    builder.add_conditional_edges(
        "EVALUATOR",
        _route_after_evaluator,
        {
            "RETRIEVER": "RETRIEVER",
            "GENERATOR": "GENERATOR",
        },
    )
    builder.add_edge("GENERATOR", "FACT_CHECKER")
    builder.add_edge("FACT_CHECKER", END)

    return builder.compile()


class AgenticRAGEngine:
    def __init__(self, nodes: AgenticRAGNodes | None = None) -> None:
        if nodes is None:
            nodes = AgenticRAGNodes(
                llm=LLMAdapter(),
                vector_retriever=VectorRetriever(),
                web_retriever=TavilyWebRetriever(),
                sql_retriever=SQLiteRetriever(),
            )
        self.nodes = nodes
        self.graph = build_workflow(nodes)

    def run(self, question: str, sources: list[str] | None = None, max_iterations: int = 3) -> dict[str, Any]:
        initial_state: RAGState = {
            "question": question,
            "requested_sources": sources or [],
            "max_iterations": max_iterations,
            "retrieval_trace": [],
            "iteration_count": 0,
            "retrieved_docs": [],
            "sources": [],
        }
        return self.graph.invoke(initial_state)
