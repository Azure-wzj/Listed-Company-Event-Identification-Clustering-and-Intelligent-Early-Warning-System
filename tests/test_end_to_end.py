"""Exercise extraction, clustering, scoring and API using synthetic news."""
import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from src.event_ner.extractor import EventExtractor
from src.services.extracted_event_adapter import ExtractedEventAdapter
from src.services.group_builder import GroupBuilder
from src.services.event_scorer import EventScorer
from src.services.alert_engine import AlertEngine
from src.clustering.cluster import EventClusterer
from scripts.pipeline_end_to_end import load_or_encode_tfidf_embeddings


def test_synthetic_news_to_api(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    news = json.loads((root / "examples/news.json").read_text(encoding="utf-8"))
    extracted = EventExtractor(use_llm=False).extract_batch(news)
    assert all(item["method"] == "rule" for item in extracted)
    events = ExtractedEventAdapter.from_records(extracted)
    vectors = load_or_encode_tfidf_embeddings(events)
    labels = EventClusterer(method="graph").fit_predict(vectors)
    groups = GroupBuilder.build_groups(events, labels, vectors)
    assert sum(group.article_count for group in groups) == len(news)
    assert any(group.article_count == 1 for group in groups)
    scored = EventScorer.score_batch(groups, use_event_classifier=False)
    alerts = AlertEngine.apply_batch(scored)
    assert len(alerts) == len(groups)

    import src.api.data_store as store
    monkeypatch.setattr(store, "WEB_DIR", tmp_path / "web")
    monkeypatch.setattr(store, "RESULTS_DIR", tmp_path / "results")
    store.clear_cache()
    store.RESULTS_DIR.mkdir()
    (store.RESULTS_DIR / "alerts.json").write_text(json.dumps([a.model_dump(mode="json") for a in alerts], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("EVENTRADAR_WARMUP", "0")
    from src.api.main import app
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/v1/events/").json()["total"] == len(groups)
        assert client.get("/api/v1/alert/stats").json()["total"] == len(groups)
        assert client.get("/api/v1/timeline/").json()["total"] == len(groups)
        ranked = client.get("/api/v1/alert/", params={"sort_by": "impact"}).json()["alerts"]
        impacts = [item["impact_score"] for item in ranked]
        assert impacts == sorted(impacts, reverse=True)
    store.clear_cache()


def test_web_alerts_take_precedence_over_stale_event_results(tmp_path, monkeypatch):
    import src.api.data_store as store
    web = tmp_path / "web"
    results = tmp_path / "results"
    web.mkdir()
    results.mkdir()
    (web / "alerts.json").write_text('[{"group_id": "current"}]', encoding="utf-8")
    (results / "event_groups.json").write_text('[{"group_id": "stale"}]', encoding="utf-8")
    monkeypatch.setattr(store, "WEB_DIR", web)
    monkeypatch.setattr(store, "RESULTS_DIR", results)
    store.clear_cache()
    assert store.load_results("event_groups")[0]["group_id"] == "current"
    assert store.load_results("alerts")[0]["group_id"] == "current"
    store.clear_cache()
