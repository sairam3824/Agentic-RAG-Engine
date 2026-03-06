from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture()
def client():
    return TestClient(app)


def test_root_route_exists(client):
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "agentic-rag-engine"
    assert payload["docs"] == "/docs"


def test_health_route(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

