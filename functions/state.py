from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class PipelineState:
    data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Path] = field(default_factory=dict)

    def put_data(self, **kwargs: Any) -> None:
        self.data.update(kwargs)

    def put_metrics(self, **kwargs: Any) -> None:
        self.metrics.update(kwargs)

    def put_models(self, **kwargs: Any) -> None:
        self.models.update(kwargs)

    def put_artifacts(self, **kwargs: Path) -> None:
        self.artifacts.update(kwargs)
