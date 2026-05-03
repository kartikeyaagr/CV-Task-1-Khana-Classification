"""MLflow experiment tracker."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import mlflow
import mlflow.pytorch


class Tracker:
    def __init__(self, experiment: str, uri="sqlite:///mlruns.db", system_metrics=True):
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        if system_metrics:
            mlflow.enable_system_metrics_logging()

    @contextmanager
    def run(self, name=None, params=None):
        with mlflow.start_run(run_name=name) as r:
            if params:
                items = list(params.items())
                for i in range(0, len(items), 100):
                    mlflow.log_params(dict(items[i:i+100]))
            yield self

    def log(self, metrics: dict, step: int):
        mlflow.log_metrics(metrics, step=step)

    def artifact(self, path):
        mlflow.log_artifact(str(path))

    def model(self, m, name="model"):
        mlflow.pytorch.log_model(m, name=name)
