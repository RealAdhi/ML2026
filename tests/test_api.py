import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL = os.path.join(ROOT, "models", "final", "model.joblib")
DATA = os.path.join(ROOT, "student_test")

pytestmark = pytest.mark.skipif(not (os.path.exists(MODEL) and os.path.isdir(DATA)),
                                reason="needs models/final/model.joblib and the student_test folder")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_trips_lists_ten_trips(client):
    trips = client.get("/trips").json()["trips"]
    assert len(trips) == 10
    assert all({"trip_id", "n_gps"} <= set(t) for t in trips)


def test_unknown_trip_returns_404(client):
    assert client.post("/predict", json={"trip_id": "does-not-exist"}).status_code == 404


def test_predict_returns_route_and_metrics_but_no_ground_truth(client):
    trip_id = client.get("/trips").json()["trips"][0]["trip_id"]
    r = client.post("/predict", json={"trip_id": trip_id})
    assert r.status_code == 200
    body = r.json()
    assert body["trip_id"] == trip_id and len(body["gps"]) == body["n_obs"]
    assert len(body["predicted_route"]) > 0
    for key in ("acc", "continuity", "route_f1"):
        assert 0.0 <= body["metrics"]["hmm"][key] <= 1.0
    assert not any("reference" in k or k == "ref_arc" for k in body)
