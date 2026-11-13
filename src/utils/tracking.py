"""Experiment tracking wrapper.

Uses **MLflow** when it is installed (params, metrics, tags and artifacts go to
the tracking store configured in ``config.yaml``). If MLflow is not available
the same calls are recorded as JSON files under ``reports/tracking/`` so the
pipeline never fails because of tracking.
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

log = get_logger(__name__)

try:
    import mlflow  # type: ignore
    MLFLOW_OK = True
except Exception:  # pragma: no cover - depends on environment
    mlflow = None
    MLFLOW_OK = False


class ExperimentTracker:
    """Thin facade over MLflow with a JSON fallback."""

    def __init__(self, cfg: Dict[str, Any]):
        tcfg = cfg.get("tracking", {})
        self.enabled = bool(tcfg.get("enabled", True))
        self.use_mlflow = self.enabled and MLFLOW_OK
        self.root: Path = cfg["root"]
        self.fallback_dir = cfg["paths"]["reports"] / "tracking"
        self._run_id: Optional[str] = None
        self._fallback_record: Dict[str, Any] = {}

        if self.use_mlflow:
            uri = tcfg.get("tracking_uri", "sqlite:///mlflow.db")
            if uri.startswith("sqlite:///") and not Path(uri[len("sqlite:///"):]).is_absolute():
                # Anchor relative sqlite paths at the repo root (works on Windows too).
                uri = "sqlite:///" + (self.root / uri[len("sqlite:///"):]).as_posix()
            mlflow.set_tracking_uri(uri)
            name = tcfg.get("experiment_name", "default")
            if mlflow.get_experiment_by_name(name) is None:
                # Store artifacts in <repo>/mlruns regardless of the working directory.
                mlflow.create_experiment(name, artifact_location=(self.root / "mlruns").as_uri())
            mlflow.set_experiment(name)
            log.info("Experiment tracking: MLflow (%s), experiment '%s'", uri, name)
        else:
            log.info("Experiment tracking: JSON fallback in %s (MLflow not installed or disabled)", self.fallback_dir)

    # ------------------------------------------------------------------ runs
    @contextmanager
    def start_run(self, run_name: str, run_id: Optional[str] = None):
        """Context manager that opens a new run, or resumes ``run_id`` if given."""
        if self.use_mlflow:
            with mlflow.start_run(run_name=None if run_id else run_name, run_id=run_id) as run:
                self._run_id = run.info.run_id
                yield self._run_id
        else:
            self._run_id = run_id or uuid.uuid4().hex[:12]
            path = self.fallback_dir / f"{self._run_id}.json"
            self._fallback_record = json.loads(path.read_text()) if path.exists() else {
                "run_id": self._run_id, "run_name": run_name, "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "params": {}, "metrics": {}, "tags": {}, "artifacts": []}
            try:
                yield self._run_id
            finally:
                self.fallback_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(self._fallback_record, indent=2, default=str))
        self._run_id = None

    # --------------------------------------------------------------- logging
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyper-parameters / configuration values."""
        clean = {k: (str(v) if v is None else v) for k, v in params.items()}
        if self.use_mlflow:
            mlflow.log_params(clean)
        else:
            self._fallback_record["params"].update(clean)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log numeric metrics (NaNs are skipped)."""
        clean = {k: float(v) for k, v in metrics.items() if v is not None and v == v}
        if self.use_mlflow:
            mlflow.log_metrics(clean)
        else:
            self._fallback_record["metrics"].update(clean)

    def set_tags(self, tags: Dict[str, Any]) -> None:
        """Attach descriptive tags to the active run."""
        if self.use_mlflow:
            mlflow.set_tags({k: str(v) for k, v in tags.items()})
        else:
            self._fallback_record["tags"].update({k: str(v) for k, v in tags.items()})

    def log_artifact(self, path: Path) -> None:
        """Attach a file (model, table, plot) to the active run."""
        path = Path(path)
        if not path.exists():
            return
        if self.use_mlflow:
            mlflow.log_artifact(str(path))
        else:
            dest = self.fallback_dir / "artifacts" / str(self._run_id)
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest / path.name)
            self._fallback_record["artifacts"].append(str(dest / path.name))
