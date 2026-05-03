from types import SimpleNamespace
from unittest.mock import Mock

import mlflow
from fof8_ml.orchestration.experiment_logger import resolve_stage_run_name
from fof8_ml.orchestration.sweep_manager import SweepContext, SweepManager
from omegaconf import OmegaConf


def _run_with_name(run_id: str, run_name: str):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id),
        data=SimpleNamespace(tags={"mlflow.runName": run_name}, metrics={}),
    )


def test_resolve_stage_run_name_uses_canonical_contract():
    assert resolve_stage_run_name("stage1") == "Stage1_Sieve_Classifier"
    assert resolve_stage_run_name("stage2") == "Stage2_Intensity_Regressor"


def test_find_stage_run_matches_canonical_nested_name():
    client = Mock()
    mgr = SweepManager(client=client, experiment_id="exp1", exp_root=".")
    client.search_runs.return_value = [
        _run_with_name("nested-a", "Some_Other_Stage"),
        _run_with_name("nested-b", "Stage2_Intensity_Regressor"),
    ]

    found = mgr._find_stage_run(parent_run_id="parent-1", stage_name="stage2")
    assert found == "nested-b"


def test_update_champion_uses_unified_model_config_and_no_stage2_registration(monkeypatch):
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
            "optimization": {"metric": "s1_oof_pr_auc", "direction": "maximize"},
            "model": {"name": "s1_catboost", "family": "catboost", "params": {"depth": 6}},
        }
    )

    parent_run = SimpleNamespace(
        data=SimpleNamespace(metrics={}, tags={}),
        info=SimpleNamespace(run_id="sweep-parent"),
    )
    client.get_run.return_value = parent_run
    client.search_runs.return_value = []

    register_model_mock = Mock()
    monkeypatch.setattr(mlflow, "register_model", register_model_mock)

    is_new_best = mgr.update_champion(
        ctx=ctx,
        pipeline_run_id="pipeline-123",
        current_score=0.52,
        cfg=cfg,
    )

    assert is_new_best is True
    register_model_mock.assert_not_called()
    best_params_calls = [c for c in client.set_tag.call_args_list if c.args[1] == "best_params"]
    assert best_params_calls, "Expected best_params tag to be set"
    assert "s1_catboost" in best_params_calls[0].args[2]
