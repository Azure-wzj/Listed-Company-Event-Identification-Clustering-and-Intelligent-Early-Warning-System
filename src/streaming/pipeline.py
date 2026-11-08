"""增量事件归组与预警流程。"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.api.schemas.cluster import EventGroupSchema
from src.api.schemas.event import EventSchema
from src.api.schemas.scoring import ScoredEventGroupSchema
from src.features.embedding import EmbeddingEngine
from src.services.alert_engine import AlertEngine
from src.services.event_scorer import EventScorer
from src.services.event_classifier import EventClassifier, get_default_event_classifier
from src.services.group_builder import GroupBuilder
from src.services.incremental_clusterer import IncrementalClusterer


class StreamPipeline:
    """Maintain event groups and emit a fully scored alert for every event."""

    def __init__(
        self,
        embedder: Any | None = None,
        classifier: EventClassifier | None = None,
        threshold: float | None = None,
    ) -> None:
        self.embedder = embedder or EmbeddingEngine()
        self.classifier = classifier
        self.clusterer = IncrementalClusterer(threshold)
        self.events: list[EventSchema] = []
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.groups: list[ScoredEventGroupSchema] = []

    @staticmethod
    def _text(event: EventSchema) -> str:
        return " ".join(
            value
            for value in (
                event.article_title,
                event.content,
                event.event_name,
                event.event_impact_direction,
                event.secu_abbr,
                event.industry_name,
            )
            if value
        )

    def _classifier(self) -> EventClassifier:
        if self.classifier is None:
            self.classifier = get_default_event_classifier()
        return self.classifier

    def _rebuild_group(
        self,
        group_id: str,
        previous: ScoredEventGroupSchema | None,
    ) -> ScoredEventGroupSchema:
        member_ids = {event.article_file_id for event in previous.events} if previous else set()
        member_ids.add(self.events[-1].article_file_id)
        indices = [idx for idx, event in enumerate(self.events) if event.article_file_id in member_ids]
        subset_events = [self.events[idx] for idx in indices]
        subset_embeddings = self.embeddings[indices]
        group: EventGroupSchema = GroupBuilder.build_groups(
            subset_events,
            np.zeros(len(subset_events), dtype=np.int32),
            subset_embeddings,
        )[0].model_copy(update={"group_id": group_id})
        scored = EventScorer.score(group, classifier=self._classifier())
        if previous:
            scored = scored.model_copy(update={"alert_level": previous.alert_level})
        return AlertEngine.apply(scored)

    def process_event(self, event: dict[str, Any] | EventSchema) -> dict[str, Any]:
        """Validate, assign, rebuild, score and alert one incoming event."""
        parsed = event if isinstance(event, EventSchema) else EventSchema.model_validate(event)
        vector = np.asarray(self.embedder.encode([self._text(parsed)]), dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        assignment = self.clusterer.assign(
            parsed,
            vector[0],
            self.groups,
            [item.article_file_id for item in self.events],
            self.embeddings,
        )
        self.events.append(parsed)
        self.embeddings = vector if not self.embeddings.size else np.vstack([self.embeddings, vector])

        previous = next(
            (group for group in self.groups if group.group_id == assignment.matched_group_id),
            None,
        )
        group_id = previous.group_id if previous else f"S{len(self.events) - 1:06d}"
        updated = self._rebuild_group(group_id, previous)
        if previous:
            self.groups = [updated if group.group_id == previous.group_id else group for group in self.groups]
        else:
            self.groups.append(updated)

        return {
            "assignment": assignment.model_dump(),
            "group": updated.model_dump(mode="json"),
            "alert": {
                "group_id": updated.group_id,
                "level": updated.alert_level,
                "reason": updated.alert_reason,
                "changed_reason": updated.alert_changed_reason,
            },
        }
