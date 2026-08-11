from pathlib import Path

import mlflow
import pytest

from ml_platform.tracking.tracker import ExperimentTracker


@pytest.fixture
def tracker(tmp_path: Path) -> ExperimentTracker:
    database_path = tmp_path / "mlflow.db"
    tracking_uri = f"sqlite:///{database_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    return ExperimentTracker(
        experiment_name="test-experiment",
    )


def test_tracker_creates_experiment(tracker: ExperimentTracker) -> None:
    experiment_id = tracker._get_or_create_experiment()

    experiment = mlflow.get_experiment(experiment_id)

    assert experiment is not None
    assert experiment.name == "test-experiment"


def test_tracker_starts_run(tracker: ExperimentTracker) -> None:
    with tracker.start_run() as run:
        assert run.info.run_id is not None


def test_tracker_logs_params(
    tracker: ExperimentTracker,
) -> None:
    with tracker.start_run() as run:
        tracker.log_params(
            {
                "learning_rate": 0.01,
                "epochs": 10,
            }
        )

        logged_run = mlflow.get_run(run.info.run_id)

        assert logged_run.data.params["learning_rate"] == "0.01"
        assert logged_run.data.params["epochs"] == "10"


def test_tracker_logs_metrics(
    tracker: ExperimentTracker,
) -> None:
    with tracker.start_run() as run:
        tracker.log_metrics(
            {
                "accuracy": 0.95,
                "loss": 0.12,
            }
        )

        logged_run = mlflow.get_run(run.info.run_id)

        assert logged_run.data.metrics["accuracy"] == 0.95
        assert logged_run.data.metrics["loss"] == 0.12


def test_tracker_logs_artifact(
    tracker: ExperimentTracker,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "metrics.json"
    artifact.write_text(
        '{"accuracy": 0.95}',
        encoding="utf-8",
    )

    with tracker.start_run() as run:
        tracker.log_artifact(artifact)

        artifacts = mlflow.artifacts.list_artifacts(
            run_id=run.info.run_id,
        )

        assert any(item.path == "metrics.json" for item in artifacts)


def test_run_finishes_successfully(
    tracker: ExperimentTracker,
) -> None:
    with tracker.run(run_name="successful-run") as run:
        run_id = run.info.run_id

    logged_run = mlflow.get_run(run_id)

    assert logged_run.info.status == "FINISHED"


def test_run_is_marked_failed(
    tracker: ExperimentTracker,
) -> None:
    with pytest.raises(ValueError, match="Training failed"):
        with tracker.run(run_name="failed-run") as run:
            run_id = run.info.run_id

            raise ValueError("Training failed")

    logged_run = mlflow.get_run(run_id)

    assert logged_run.info.status == "FAILED"


def test_run_name_is_logged(
    tracker: ExperimentTracker,
) -> None:
    with tracker.run(run_name="experiment-run") as run:
        run_id = run.info.run_id

    logged_run = mlflow.get_run(run_id)

    assert logged_run.info.run_name == "experiment-run"


def test_run_tags_are_logged(
    tracker: ExperimentTracker,
) -> None:
    with tracker.run(
        run_name="tagged-run",
        tags={
            "model": "random_forest",
            "environment": "test",
        },
    ) as run:
        run_id = run.info.run_id

    logged_run = mlflow.get_run(run_id)

    assert logged_run.data.tags["model"] == "random_forest"
    assert logged_run.data.tags["environment"] == "test"


def test_get_active_run_id(
    tracker: ExperimentTracker,
) -> None:
    assert tracker.get_active_run_id() is None

    with tracker.run() as run:
        assert tracker.get_active_run_id() == run.info.run_id

    assert tracker.get_active_run_id() is None
