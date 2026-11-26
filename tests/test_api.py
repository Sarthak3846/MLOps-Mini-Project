import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.main import FEATURE_ORDER, get_model

VALID = {
    "flu_cases_reported": 50,
    "temp_c": 20.5,
    "nearby_clinics": 4,
    "discount_pct": 12.5,
    "stock_lines": 1500,
    "distance_to_hospital_km": 3.2,
    "shop_type": "chain",
    "locality": "Old Town",
}


def tiny_model():
    """Small model built inside the test: no CSVs, no real model file needed."""
    rows = [
        [10, 15.0, 1, 5.0, 500, 1.0, "chain", "Old Town"],
        [60, 25.0, 5, 15.0, 1500, 4.0, "independent", "New Town"],
        [90, 30.0, 9, 20.0, 2500, 8.0, "hospital_store", "Cantonment"],
        [40, 20.0, 3, 10.0, 1000, 2.0, "chain", "New Town"],
        [75, 28.0, 7, 18.0, 2000, 6.0, "independent", "Cantonment"],
    ]
    X = pd.DataFrame(rows, columns=FEATURE_ORDER)
    y = [100, 400, 700, 250, 550]
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), ["shop_type", "locality"])],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("model", LinearRegression())]).fit(X, y)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    joblib.dump(tiny_model(), path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    get_model.cache_clear()
    from app.main import app

    with TestClient(app) as c:
        yield c
    get_model.cache_clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_valid_returns_number(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    assert isinstance(r.json()["prediction"], float)


def test_stock_lines_above_5000_rejected(client):
    r = client.post("/predict", json={**VALID, "stock_lines": 5001})
    assert r.status_code == 422


def test_stock_lines_exactly_5000_accepted(client):
    assert client.post("/predict", json={**VALID, "stock_lines": 5000}).status_code == 200


def test_negative_distance_rejected(client):
    r = client.post("/predict", json={**VALID, "distance_to_hospital_km": -0.1})
    assert r.status_code == 422


def test_missing_field_rejected(client):
    bad = {k: v for k, v in VALID.items() if k != "temp_c"}
    assert client.post("/predict", json=bad).status_code == 422


def test_unknown_locality_rejected(client):
    assert client.post("/predict", json={**VALID, "locality": "Nowhere"}).status_code == 422


def test_prediction_is_logged_as_json(client):
    import io
    import json
    import logging

    from app.logging_config import JsonFormatter

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("mp046-api")
    logger.addHandler(handler)
    try:
        client.post("/predict", json=VALID)
    finally:
        logger.removeHandler(handler)

    records = [json.loads(line) for line in buf.getvalue().splitlines()]
    assert any(r.get("event") == "prediction" and "prediction" in r for r in records)
