from types import SimpleNamespace
from unittest.mock import Mock

import mlflow
from fof8_ml.orchestration.experiment_logger import resolve_model_role_name
from fof8_ml.orchestration.sweep_manager import SweepContext, SweepManager
from omegaconf import OmegaConf


def _run_with_name(run_id: str, run_name: str):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id),
        data=SimpleNamespace(tags={"mlflow.runName": run_name}, metrics={}),
    )


def _artifact(path: str):
    return SimpleNamespace(path=path)


def _logged_model(
    model_id: str,
    name: str = "regressor_model",
    source_run_id: str = "role-run",
    last_updated_timestamp: int = 1,
):
    return SimpleNamespace(
        model_id=model_id,
        name=name,
        source_run_id=source_run_id,
        last_updated_timestamp=last_updated_timestamp,
    )


def test_resolve_model_role_name_uses_canonical_contract():
    assert resolve_model_role_name("classifier") == "Classifier"
    assert resolve_model_role_name("regressor") == "Regressor"


def test_resolve_flat_role_run_matches_parent_role_tag():
    client = Mock()
    mgr = SweepManager(client=client, experiment_id="exp1", exp_root=".")
    client.get_run.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="pipeline-1"),
        data=SimpleNamespace(tags={"model_role": "regressor"}, metrics={}),
    )

    found = mgr._resolve_flat_role_run(pipeline_run_id="pipeline-1", role_name="regressor")

    assert found == "pipeline-1"


def test_resolve_flat_role_run_rejects_other_role():
    client = Mock()
    mgr = SweepManager(client=client, experiment_id="exp1", exp_root=".")
    client.get_run.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="pipeline-1"),
        data=SimpleNamespace(tags={"model_role": "classifier"}, metrics={}),
    )

    found = mgr._resolve_flat_role_run(pipeline_run_id="pipeline-1", role_name="regressor")

    assert found is None


def test_update_champion_registers_classifier_champion(monkeypatch):
    client = Mock()
    mgr = SweepManager(client=client, experiment_id="exp1", exp_root=".")
    ctx = SweepContext(
        is_sweep=True,
        sweep_name="Sweep_1",
        sweep_run_id="sweep-parent",
        quiet=True,
        tags={},
    )
    cfg = OmegaConf.create(
        {
            "optimization": {"metric": "classifier_oof_pr_auc", "direction": "maximize"},
            "model": {"name": "catboost_classifier", "family": "catboost", "params": {"depth": 6}},
        }
    )

    parent_run = SimpleNamespace(
        data=SimpleNamespace(metrics={}, tags={}),
        info=SimpleNamespace(run_id="sweep-parent"),
    )
    classifier_run = SimpleNamespace(
        data=SimpleNamespace(metrics={}, tags={"model_role": "classifier"}),
        info=SimpleNamespace(run_id="pipeline-123", experiment_id="exp1"),
    )
    client.get_run.side_effect = [parent_run, classifier_run]
    client.search_runs.return_value = []
    client.list_artifacts.return_value = [_artifact("classifier_model/MLmodel")]

    register_model_mock = Mock(return_value=SimpleNamespace(version="3"))
    monkeypatch.setattr(mlflow, "register_model", register_model_mock)

    is_new_best = mgr.update_champion(
        ctx=ctx,
        pipeline_run_id="pipeline-123",
        current_score=0.52,
        cfg=cfg,
    )

    assert is_new_best is True
    register_model_mock.assert_called_once_with(
        model_uri="runs:/pipeline-123/classifier_model",
        name="fof8-scout-classifier",
    )
    best_params_calls = [c for c in client.set_tag.call_args_list if c.args[1] == "best_params"]
    assert best_params_calls, "Expected best_params tag to be set"
    assert "catboost_classifier" in best_params_calls[0].args[2]


def test_registration_target_resolves_classifier_and_regressor():
    client = Mock()
    mgr = SweepManager(client=client, experiment_id="exp1", exp_root=".")

    classifier_cfg = OmegaConf.create({"model": {"name": "catboost_classifier"}})
    regressor_cfg = OmegaConf.create({"model": {"name": "catboost_tweedie_regressor"}})

    assert mgr._registration_target(classifier_cfg) == (
        "classifier",
        "classifier_model",
        "fof8-scout-classifier",
    )
    assert mgr._registration_target(regressor_cfg) == (
        "regressor",
        "regressor_model",
        "fof8-scout-regressor",
    )


def test_champion_model_summary_resolves_family_from_registry():
    client = Mock()
    mgr = SweepManager(client=client, experiment_id="exp1", exp_root=".")
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "catboost_tweedie_regressor",
                "params": {"loss_function": "Tweedie", "variance_power": 1.5},
            }
        }
    )

    summary = mgr._champion_model_summary(cfg)

    assert summary["model_name"] == "catboost_tweedie_regressor"
    assert summary["model_family"] == "catboost"


def test_resolve_model_uri_prefers_run_artifact_when_mlmodel_exists():
    client = Mock()
    mgr = SweepManager(client=client, experiment_id="exp1", exp_root=".")
    client.list_artifacts.return_value = [_artifact("regressor_model/MLmodel")]

    uri = mgr._resolve_model_uri("role-run", "regressor_model")

    assert uri == "runs:/role-run/regressor_model"
    client.search_logged_models.assert_not_called()


def test_resolve_model_uri_uses_mlflow3_logged_model_when_run_artifact_missing():
    client = Mock()
    mgr = SweepManager(client=client, experiment_id="exp1", exp_root=".")
    client.list_artifacts.return_value = []
    client.get_run.return_value = SimpleNamespace(info=SimpleNamespace(experiment_id="exp1"))
    client.search_logged_models.return_value = [
        _logged_model("older", last_updated_timestamp=1),
        _logged_model("newer", last_updated_timestamp=2),
        _logged_model("wrong-name", name="other_model", last_updated_timestamp=3),
    ]

    uri = mgr._resolve_model_uri("role-run", "regressor_model")

    assert uri == "models:/newer"


def test_register_role_model_records_model_version_tags(monkeypatch):
    client = Mock()
    mgr = SweepManager(client=client, experiment_id="exp1", exp_root=".")
    monkeypatch.setattr(mgr, "_resolve_model_uri", Mock(return_value="models:/model-id"))
    register_model_mock = Mock(return_value=SimpleNamespace(version="7"))
    monkeypatch.setattr(mlflow, "register_model", register_model_mock)

    mgr._register_role_model(
        pipeline_run_id="pipeline-run",
        role_run_id="role-run",
        artifact_path="regressor_model",
        registered_name="fof8-scout-regressor",
    )

    register_model_mock.assert_called_once_with(
        model_uri="models:/model-id",
        name="fof8-scout-regressor",
    )
    client.set_tag.assert_any_call(
        "pipeline-run", "registered_model_name", "fof8-scout-regressor"
    )
    client.set_tag.assert_any_call("pipeline-run", "registered_model_uri", "models:/model-id")
    client.set_tag.assert_any_call("pipeline-run", "registered_model_version", "7")
