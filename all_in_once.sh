#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
API_PORT="${API_PORT:-8000}"
DEMO_PORT="${DEMO_PORT:-8501}"
INSTALL_DEPS=1
RUN_TESTS=1
RUN_API=1
RUN_DEMO=1

is_port_in_use() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"$port" -sTCP:LISTEN -n -P >/dev/null 2>&1
    return $?
  fi
  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    "$PYTHON_BIN" - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.2)
    in_use = sock.connect_ex(("127.0.0.1", port)) == 0
sys.exit(0 if in_use else 1)
PY
    return $?
  else
    return 1
  fi
}

find_free_port() {
  local start_port="$1"
  local candidate="$start_port"
  local tries=20
  while (( tries > 0 )); do
    if ! is_port_in_use "$candidate"; then
      echo "$candidate"
      return 0
    fi
    candidate=$((candidate + 1))
    tries=$((tries - 1))
  done
  return 1
}

print_help() {
  cat <<'HELP'
Usage: ./all_in_once.sh [options]

Options:
  --no-install       Skip dependency installation
  --no-tests         Skip pytest
  --no-api           Skip FastAPI server
  --no-demo          Skip Streamlit demo
  --api-port <port>  FastAPI port (default: 8000)
  --demo-port <port> Streamlit port (default: 8501)
  --python <bin>     Python executable to use (default: python3)
  -h, --help         Show help

Examples:
  ./all_in_once.sh
  ./all_in_once.sh --no-tests
  ./all_in_once.sh --api-port 9000 --demo-port 8601
  ./all_in_once.sh --python python3.13
HELP
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-install)
      INSTALL_DEPS=0
      shift
      ;;
    --no-tests)
      RUN_TESTS=0
      shift
      ;;
    --no-api)
      RUN_API=0
      shift
      ;;
    --no-demo)
      RUN_DEMO=0
      shift
      ;;
    --api-port)
      API_PORT="$2"
      shift 2
      ;;
    --demo-port)
      DEMO_PORT="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      print_help
      exit 1
      ;;
  esac
done

if [[ ! -f "requirements.txt" ]]; then
  echo "requirements.txt not found. Run this from the project root."
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] Python executable not found: $PYTHON_BIN"
  exit 1
fi

PY_VER="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PY_VER" == "3.14" ]]; then
  echo "[warn] Python 3.14 can trigger compatibility warnings in current LangGraph stack."
  if command -v python3.13 >/dev/null 2>&1 && [[ "$PYTHON_BIN" != "python3.13" ]]; then
    echo "[warn] Auto-switching to python3.13 for better runtime compatibility."
    PYTHON_BIN="python3.13"
  fi
fi

echo "[1/5] Creating/updating virtual environment: $VENV_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

if [[ "$INSTALL_DEPS" -eq 1 ]]; then
  echo "[2/5] Installing dependencies"
  pip install --upgrade pip
  pip install -r requirements.txt
else
  echo "[2/5] Skipped dependency installation"
fi

if [[ "$RUN_TESTS" -eq 1 ]]; then
  echo "[3/5] Running tests"
  pytest -q
else
  echo "[3/5] Skipped tests"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[warn] OPENAI_API_KEY is not set. LLM calls will use fallback behavior."
fi
if [[ -z "${TAVILY_API_KEY:-}" ]]; then
  echo "[warn] TAVILY_API_KEY is not set. Web retrieval will be disabled."
fi

mkdir -p logs
API_PID=""
DEMO_PID=""

cleanup() {
  if [[ -n "$API_PID" ]] && kill -0 "$API_PID" >/dev/null 2>&1; then
    kill "$API_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$DEMO_PID" ]] && kill -0 "$DEMO_PID" >/dev/null 2>&1; then
    kill "$DEMO_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "$RUN_API" -eq 1 ]]; then
  if is_port_in_use "$API_PORT"; then
    NEW_API_PORT="$(find_free_port "$API_PORT")" || {
      echo "[error] Could not find free API port near $API_PORT"
      exit 1
    }
    echo "[warn] Port $API_PORT is busy. Using $NEW_API_PORT for FastAPI."
    API_PORT="$NEW_API_PORT"
  fi
  echo "[4/5] Starting FastAPI on http://localhost:$API_PORT"
  uvicorn src.api:app --host 127.0.0.1 --port "$API_PORT" > logs/api.log 2>&1 &
  API_PID=$!
  sleep 1
  if ! kill -0 "$API_PID" >/dev/null 2>&1; then
    echo "[error] FastAPI failed to start. Last logs:"
    tail -n 80 logs/api.log || true
    exit 1
  fi
else
  echo "[4/5] Skipped FastAPI"
fi

if [[ "$RUN_DEMO" -eq 1 ]]; then
  if is_port_in_use "$DEMO_PORT"; then
    NEW_DEMO_PORT="$(find_free_port "$DEMO_PORT")" || {
      echo "[error] Could not find free demo port near $DEMO_PORT"
      exit 1
    }
    echo "[warn] Port $DEMO_PORT is busy. Using $NEW_DEMO_PORT for Streamlit."
    DEMO_PORT="$NEW_DEMO_PORT"
  fi
  echo "[5/5] Starting Streamlit on http://localhost:$DEMO_PORT"
  streamlit run src/demo.py --server.port "$DEMO_PORT" --server.headless true > logs/demo.log 2>&1 &
  DEMO_PID=$!
  sleep 2
  if ! kill -0 "$DEMO_PID" >/dev/null 2>&1; then
    echo "[error] Streamlit failed to start. Last logs:"
    tail -n 80 logs/demo.log || true
    exit 1
  fi
else
  echo "[5/5] Skipped Streamlit"
fi

echo
echo "Agentic RAG Engine is ready."
if [[ "$RUN_API" -eq 1 ]]; then
  echo "- API:  http://localhost:$API_PORT"
  echo "- Logs: $ROOT_DIR/logs/api.log"
fi
if [[ "$RUN_DEMO" -eq 1 ]]; then
  echo "- Demo: http://localhost:$DEMO_PORT"
  echo "- Logs: $ROOT_DIR/logs/demo.log"
fi

echo
echo "Press Ctrl+C to stop running services."

if [[ "$RUN_API" -eq 1 ]] || [[ "$RUN_DEMO" -eq 1 ]]; then
  wait
else
  echo "Nothing to run (both API and demo were skipped)."
fi
