from fastapi.testclient import TestClient

from triage.api import create_app


def _client(agent):
    return TestClient(create_app(agent))


def test_health_and_model(agent):
    c = _client(agent)
    assert c.get("/health").json()["status"] == "ok"
    assert "min_confidence" in c.get("/model").json()


def test_triage_endpoint_returns_decision_and_trace(agent):
    r = _client(agent).post(
        "/triage",
        json={"subject": "Error in the app", "body": "I get a 500 error when calling the API.",
              "ticket_id": "T-42"},
    )
    body = r.json()
    assert r.status_code == 200 and body["ticket_id"] == "T-42"
    assert body["category"] == "technical" and len(body["trace"]) == 5


def test_validation(agent):
    assert _client(agent).post("/triage", json={"subject": "", "body": "x"}).status_code == 422
