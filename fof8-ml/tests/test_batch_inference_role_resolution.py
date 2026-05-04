import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

MODULE_PATH = Path(__file__).resolve().parents[2] / "pipelines" / "batch_inference.py"
spec = importlib.util.spec_from_file_location("batch_inference", MODULE_PATH)
assert spec and spec.loader
batch_inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch_inference)


def _run_with_name(run_id: str, run_name: str, tags: dict[str, str] | None = None):
    run_tags = {"mlflow.runName": run_name}
    if tags:
        run_tags.update(tags)
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id),
        data=SimpleNamespace(tags=run_tags),
    )


def test_resolve_classifier_run_uses_flattened_role_tag():
    client = Mock()
    client.get_run.return_value = _run_with_name(
        "pipeline-1",
        "Classifier_catboost_classifier",
        tags={"model_role": "classifier"},
    )

    run = batch_inference.resolve_classifier_run(
        client=client, run_id="pipeline-1", experiment_ids=["0"]
    )

    assert run is not None
    assert run.info.run_id == "pipeline-1"


def test_resolve_classifier_run_rejects_other_role():
    client = Mock()
    client.get_run.return_value = _run_with_name(
        "pipeline-1",
        "Regressor_catboost_tweedie_regressor",
        tags={"model_role": "regressor"},
    )

    run = batch_inference.resolve_classifier_run(
        client=client, run_id="pipeline-1", experiment_ids=["0"]
    )

    assert run is None
