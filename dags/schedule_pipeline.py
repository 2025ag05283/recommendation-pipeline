"""Schedule the pipeline to run automatically (daily at 02:00) with Prefect.

Usage (needs a running Prefect server - see README):

    python dags/schedule_pipeline.py

This registers a deployment called ``daily-recommendation-pipeline`` and keeps
a worker process alive that runs the flow on the cron schedule. Press Ctrl+C
to stop. You can also trigger a run immediately from the Prefect UI
(Deployments -> daily-recommendation-pipeline -> Run).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dags.pipeline_flow import PREFECT_OK, recommendation_pipeline  # noqa: E402

if __name__ == "__main__":
    if not PREFECT_OK:
        sys.exit("Prefect is not installed: pip install prefect  (or use cron / Windows Task Scheduler with "
                 "'python run_pipeline.py')")
    recommendation_pipeline.serve(name="daily-recommendation-pipeline", cron="0 2 * * *")
