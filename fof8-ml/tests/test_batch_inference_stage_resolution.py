import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

MODULE_PATH = Path(__file__).resolve().parents[2] / "pipelines" / "batch_inference.py"
spec = importlib.util.spec_from_file_location("batch_inference", MODULE_PATH)
assert spec and spec.loader
batch_inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch_inference)


def _run_with_name(run_id: str, run_name: str):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id),
        data=SimpleNamespace(tags={"mlflow.runName": run_name}),
    )


def test_resolve_stage1_run_uses_canonical_stage_name():
    client = Mock()
    client.search_runs.return_value = [
        _run_with_name("child-1", "Other_Name"),
        _run_with_name("child-2", "Stage1_Sieve_Classifier"),
    ]

    run = batch_inference.resolve_stage1_run(
        client=client, parent_run_id="parent-1", experiment_ids=["0"]
    )
    assert run is not None
    assert run.info.run_id == "child-2"
