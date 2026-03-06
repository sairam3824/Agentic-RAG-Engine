# Claude Code Guide for agentic-rag-engine

## Quickstart

```bash
pip install -r requirements.txt
uvicorn src.api:app --reload --port 8000
streamlit run src/demo.py
```

## Suggested Claude prompts

1. "Run a review of retrieval routing logic in `src/graph/nodes.py` and identify failure modes."
2. "Add an additional retriever source (e.g., Confluence API) and wire it through the LangGraph flow."
3. "Improve evaluator scoring weights for analytical queries and add regression tests."
4. "Harden citation generation so each claim maps to one or more retrieved evidence snippets."

## Repository map

- Graph state: `src/graph/state.py`
- Node logic: `src/graph/nodes.py`
- Workflow routing: `src/graph/workflow.py`
- Retrievers: `src/retrievers/`
- Evaluators: `src/evaluators/`
- API: `src/api.py`
- Demo: `src/demo.py`
- Tests: `tests/test_workflow.py`

## Agent behavior contract

- Query analyzer decides whether retrieval is needed.
- Retriever chooses among vector/web/sql.
- Evaluator can force re-query, up to 3 iterations.
- Generator must include grounded citations.
- Fact checker flags unsupported claims.
