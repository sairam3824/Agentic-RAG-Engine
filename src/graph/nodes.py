from __future__ import annotations

import json
import re
from typing import Any

from src.evaluators.completeness import score_completeness, score_freshness
from src.evaluators.relevance import score_relevance
from src.graph.state import RAGState, RetrievedDoc
from src.llm import LLMAdapter
from src.retrievers.sql import SQLiteRetriever
from src.retrievers.vector import VectorRetriever
from src.retrievers.web import TavilyWebRetriever
from src.utils.text import dedupe_by_key


class AgenticRAGNodes:
    def __init__(
        self,
        llm: LLMAdapter,
        vector_retriever: VectorRetriever,
        web_retriever: TavilyWebRetriever,
        sql_retriever: SQLiteRetriever,
        retrieval_threshold: float = 0.62,
        enable_fact_checker: bool = True,
    ) -> None:
        self.llm = llm
        self.vector_retriever = vector_retriever
        self.web_retriever = web_retriever
        self.sql_retriever = sql_retriever
        self.retrieval_threshold = retrieval_threshold
        self.enable_fact_checker = enable_fact_checker

    def _coerce_bool(self, value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    def _normalize_query_type(self, value: Any, default: str) -> str:
        if not isinstance(value, str):
            return default
        lowered = value.strip().lower()
        if lowered in {"factual", "analytical", "comparison", "how-to"}:
            return lowered
        return default

    def _safe_arithmetic_eval(self, expression: str) -> str | None:
        # Evaluate only simple arithmetic expressions (+, -, *, /, parentheses, numbers).
        if not expression or len(expression) > 120:
            return None
        if not re.match(r"^[\d\s+\-*/().]+$", expression):
            return None
        try:
            import ast
            import operator as op

            operators = {
                ast.Add: op.add,
                ast.Sub: op.sub,
                ast.Mult: op.mul,
                ast.Div: op.truediv,
                ast.USub: op.neg,
                ast.UAdd: op.pos,
            }

            def _eval(node: ast.AST) -> float:
                if isinstance(node, ast.Expression):
                    return _eval(node.body)
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    return float(node.value)
                if isinstance(node, ast.BinOp) and type(node.op) in operators:
                    return operators[type(node.op)](_eval(node.left), _eval(node.right))
                if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
                    return operators[type(node.op)](_eval(node.operand))
                raise ValueError("Unsupported expression")

            parsed = ast.parse(expression, mode="eval")
            value = _eval(parsed)
            if value.is_integer():
                return str(int(value))
            return f"{value:.6g}"
        except Exception:
            return None

    def _trace(self, state: RAGState, node: str, message: str, data: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        trace = list(state.get("retrieval_trace", []))
        trace.append({"node": node, "message": message, "data": data or {}})
        return trace

    def _classify_query_type(self, question: str) -> str:
        lower = question.lower()
        if any(token in lower for token in ["difference", "compare", "vs", "versus"]):
            return "comparison"
        if any(token in lower for token in ["how", "steps", "guide", "implement"]):
            return "how-to"
        if any(token in lower for token in ["why", "impact", "analyze", "analysis", "tradeoff"]):
            return "analytical"
        return "factual"

    def _needs_retrieval(self, question: str) -> bool:
        lower = question.lower().strip()
        if self._safe_arithmetic_eval(lower) is not None:
            return False
        if re.match(r"^\s*\d+(\s*[+\-*/]\s*\d+)+\s*$", lower):
            return False
        trivial = {"hi", "hello", "hey", "thanks", "thank you"}
        if lower in trivial:
            return False
        if any(
            token in lower
            for token in [
                "latest",
                "current",
                "according",
                "source",
                "report",
                "document",
                "compare",
                "difference",
                "versus",
                "vs",
                "how",
                "why",
            ]
        ):
            return True
        return len(lower.split()) > 4

    def _source_plan(self, query_type: str, requested_sources: list[str]) -> list[str]:
        default = {
            "factual": ["vector", "web"],
            "analytical": ["vector", "sql", "web"],
            "comparison": ["vector", "sql", "web"],
            "how-to": ["vector", "web"],
        }.get(query_type, ["vector", "web"])

        allowed = {"vector", "web", "sql"}
        requested = []
        for source in requested_sources:
            normalized = str(source).strip().lower()
            if normalized in allowed and normalized not in requested:
                requested.append(normalized)
        if requested:
            return [source for source in default if source in requested] or requested
        return default

    def query_analyzer(self, state: RAGState) -> dict[str, Any]:
        question = state.get("question", "")
        query_type = self._classify_query_type(question)
        needs_retrieval = self._needs_retrieval(question)
        optimized_query = question

        response = self.llm.json_response(
            system_prompt=(
                "You are a query analysis module for agentic RAG. "
                "Return JSON keys: needs_retrieval (bool), optimized_query (string), query_type "
                "(factual|analytical|comparison|how-to)."
            ),
            user_prompt=question,
        )
        if response:
            needs_retrieval = self._coerce_bool(response.get("needs_retrieval"), needs_retrieval)
            raw_query = response.get("optimized_query", question)
            optimized_query = raw_query.strip() if isinstance(raw_query, str) else question
            optimized_query = optimized_query or question
            query_type = self._normalize_query_type(response.get("query_type"), query_type)

        return {
            "query": question,
            "optimized_query": optimized_query,
            "query_type": query_type,
            "needs_retrieval": needs_retrieval,
            "iteration_count": 0,
            "retrieval_trace": self._trace(
                state,
                "QUERY_ANALYZER",
                "Analyzed question and retrieval need.",
                {
                    "needs_retrieval": needs_retrieval,
                    "optimized_query": optimized_query,
                    "query_type": query_type,
                },
            ),
        }

    def retriever(self, state: RAGState) -> dict[str, Any]:
        query = state.get("refined_query") or state.get("optimized_query") or state.get("query") or ""
        query_type = state.get("query_type", "factual")
        requested_sources = state.get("requested_sources", [])
        selected_sources = self._source_plan(query_type, requested_sources)
        docs: list[RetrievedDoc] = []

        for source in selected_sources:
            if source == "vector":
                docs.extend(self.vector_retriever.search(query, k=4))
            elif source == "web":
                docs.extend(self.web_retriever.search(query, k=4))
            elif source == "sql":
                docs.extend(self.sql_retriever.search(query, k=4))

        docs = dedupe_by_key(sorted(docs, key=lambda d: d.get("score", 0.0), reverse=True), "content")[:10]
        next_iteration = int(state.get("iteration_count", 0)) + 1

        return {
            "retrieved_docs": docs,
            "sources": selected_sources,
            "iteration_count": next_iteration,
            "retrieval_trace": self._trace(
                state,
                "RETRIEVER",
                "Retrieved documents from selected sources.",
                {
                    "query": query,
                    "selected_sources": selected_sources,
                    "retrieved_count": len(docs),
                    "iteration": next_iteration,
                },
            ),
        }

    def evaluator(self, state: RAGState) -> dict[str, Any]:
        query = state.get("refined_query") or state.get("optimized_query") or state.get("query") or ""
        docs = state.get("retrieved_docs", [])
        query_type = state.get("query_type", "factual")
        used_sources = [doc.get("source", "") for doc in docs]

        relevance = score_relevance(query, docs)
        completeness = score_completeness(query, docs, query_type)
        freshness = score_freshness(query, docs, used_sources)

        overall = round((relevance * 0.45) + (completeness * 0.35) + (freshness * 0.20), 3)
        max_iterations = int(state.get("max_iterations", 3))
        iteration_count = int(state.get("iteration_count", 0))
        good_enough = overall >= self.retrieval_threshold or iteration_count >= max_iterations

        reason = "sufficient_evidence"
        if relevance < 0.45:
            reason = "low_relevance"
        elif completeness < 0.50:
            reason = "insufficient_coverage"
        elif freshness < 0.50:
            reason = "stale_or_non_current_sources"

        refined_query = state.get("refined_query") or state.get("optimized_query") or state.get("query") or ""
        if not good_enough and iteration_count < max_iterations:
            llm_refinement = self.llm.json_response(
                system_prompt=(
                    "Improve this search query for retrieval. Return JSON keys: refined_query, reason. "
                    "Keep it concise and specific."
                ),
                user_prompt=json.dumps(
                    {
                        "query": query,
                        "reason": reason,
                        "query_type": query_type,
                        "retrieved_sample": [doc.get("content", "")[:160] for doc in docs[:3]],
                    },
                    ensure_ascii=True,
                ),
            )
            if llm_refinement and llm_refinement.get("refined_query"):
                refined_query = str(llm_refinement["refined_query"]).strip() or refined_query
            else:
                refined_query = f"{query} detailed evidence official source"

        evaluation = {
            "relevance": relevance,
            "completeness": completeness,
            "freshness": freshness,
            "overall": overall,
            "good_enough": good_enough,
            "reason": reason,
        }

        return {
            "evaluation": evaluation,
            "refined_query": refined_query,
            "retrieval_trace": self._trace(
                state,
                "EVALUATOR",
                "Evaluated retrieval quality.",
                {
                    "evaluation": evaluation,
                    "next_action": "GENERATOR" if good_enough else "RETRIEVER",
                    "refined_query": refined_query,
                },
            ),
        }

    def generator(self, state: RAGState) -> dict[str, Any]:
        question = state.get("question", "")
        needs_retrieval = bool(state.get("needs_retrieval", True))
        docs = state.get("retrieved_docs", [])

        if not needs_retrieval:
            direct = self.llm.text_response(
                system_prompt="Answer directly and concisely.",
                user_prompt=question,
            )
            if direct:
                answer = direct
            else:
                quick_math = self._safe_arithmetic_eval(question.strip())
                if quick_math is not None:
                    answer = f"No retrieval required. The answer is {quick_math}."
                elif question.strip().lower() in {"hi", "hello", "hey"}:
                    answer = "Hello! Ask me a question and I will retrieve only if needed."
                else:
                    answer = (
                        "No retrieval required, but no LLM provider is configured for a richer direct response. "
                        "Set OPENAI_API_KEY for direct generation."
                    )
            return {
                "answer": answer,
                "confidence": 0.9,
                "unanswerable_points": [],
                "retrieval_trace": self._trace(
                    state,
                    "GENERATOR",
                    "Generated direct answer without retrieval.",
                ),
            }

        if not docs:
            return {
                "answer": "I could not find enough evidence to answer reliably with the selected sources.",
                "confidence": 0.2,
                "unanswerable_points": ["Missing relevant retrieved documents."],
                "retrieval_trace": self._trace(
                    state,
                    "GENERATOR",
                    "Unable to generate grounded answer because retrieval returned no documents.",
                ),
            }

        context_lines = []
        for idx, doc in enumerate(docs[:6], start=1):
            snippet = doc.get("content", "").replace("\n", " ")[:360]
            context_lines.append(f"[{idx}] source={doc.get('source', 'unknown')} | {snippet}")

        llm_output = self.llm.json_response(
            system_prompt=(
                "Synthesize an answer grounded in retrieved context. Return JSON with keys: "
                "answer (string with citation markers like [1]), confidence (0-1 float), "
                "unanswerable_points (array of strings)."
            ),
            user_prompt=f"Question: {question}\n\nContext:\n" + "\n".join(context_lines),
        )

        if llm_output:
            answer = str(llm_output.get("answer", "")).strip()
            confidence = float(llm_output.get("confidence", 0.65) or 0.65)
            unanswerable = llm_output.get("unanswerable_points", [])
            if not isinstance(unanswerable, list):
                unanswerable = []
        else:
            citations = ", ".join(f"[{i}]" for i in range(1, min(4, len(docs)) + 1))
            evidence = " ".join((docs[i].get("content", "") or "")[:130] for i in range(min(3, len(docs))))
            answer = f"Based on retrieved evidence {citations}, {evidence}"
            confidence = min(max(state.get("evaluation", {}).get("overall", 0.6), 0.3), 0.95)
            unanswerable = []

        return {
            "answer": answer,
            "confidence": round(confidence, 3),
            "unanswerable_points": unanswerable,
            "retrieval_trace": self._trace(
                state,
                "GENERATOR",
                "Generated grounded answer with citations.",
                {
                    "confidence": round(confidence, 3),
                    "used_docs": len(docs),
                },
            ),
        }

    def fact_checker(self, state: RAGState) -> dict[str, Any]:
        if not self.enable_fact_checker:
            return {
                "fact_check": [],
                "retrieval_trace": self._trace(
                    state,
                    "FACT_CHECKER",
                    "Fact checker disabled.",
                ),
            }

        answer = state.get("answer", "")
        docs = state.get("retrieved_docs", [])
        if not answer or not docs:
            return {
                "fact_check": [],
                "retrieval_trace": self._trace(
                    state,
                    "FACT_CHECKER",
                    "Skipped fact check due to missing answer or evidence.",
                ),
            }

        context = "\n".join(
            f"source={doc.get('source', '')}: {doc.get('content', '')[:280]}" for doc in docs[:6]
        )
        response = self.llm.json_response(
            system_prompt=(
                "Cross-check answer claims against evidence. Return JSON keys: "
                "unsupported_claims (array of strings), qualifications (array of strings)."
            ),
            user_prompt=f"Answer:\n{answer}\n\nEvidence:\n{context}",
        )

        unsupported_claims: list[str] = []
        qualifications: list[str] = []
        if response:
            raw_unsupported = response.get("unsupported_claims", []) or []
            raw_qualifications = response.get("qualifications", []) or []
            unsupported_claims = (
                [str(item) for item in raw_unsupported if isinstance(item, (str, int, float))]
                if isinstance(raw_unsupported, list)
                else []
            )
            qualifications = (
                [str(item) for item in raw_qualifications if isinstance(item, (str, int, float))]
                if isinstance(raw_qualifications, list)
                else []
            )

        if not response:
            unsupported_claims = [] if "[" in answer and "]" in answer else ["Answer lacks explicit citation markers."]
            qualifications = ["Treat uncited claims as tentative."] if unsupported_claims else []

        return {
            "fact_check": [
                {
                    "unsupported_claims": unsupported_claims,
                    "qualifications": qualifications,
                }
            ],
            "retrieval_trace": self._trace(
                state,
                "FACT_CHECKER",
                "Fact-checked answer against retrieved evidence.",
                {
                    "unsupported_claim_count": len(unsupported_claims),
                },
            ),
        }
