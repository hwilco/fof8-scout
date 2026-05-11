from pathlib import Path
from types import SimpleNamespace

from fof8_ml.orchestration.experiment_logger import ExperimentLogger
from fof8_ml.orchestration.experiment_matrix import (
    MatrixRunResult,
    resolve_matrix_candidates,
    run_experiment_matrix,
)
from fof8_ml.reporting.matrix_report import export_matrix_report
from omegaconf import OmegaConf


class _FakeMlflowClient:
    def __init__(self, metrics_by_run_id):
        self._metrics_by_run_id = metrics_by_run_id

    def get_run(self, run_id: str):
        return SimpleNamespace(data=SimpleNamespace(metrics=self._metrics_by_run_id[run_id]))


def test_resolve_matrix_candidates_filters_requested_ids():
    cfg = OmegaConf.create(
        {
            "candidates": [
                {
                    "candidate_id": "A1",
                    "label": "one",
                    "regressor": {
                        "model": "catboost_regressor_tweedie",
                        "target_col": "Positive_Career_Merit_Cap_Share",
                        "target_space": "raw",
                    },
                },
                {
                    "candidate_id": "A2",
                    "label": "two",
                    "regressor": {
                        "model": "catboost_regressor_rmse",
                        "target_col": "Positive_Career_Merit_Cap_Share",
                        "target_space": "log",
                    },
                    "adjustment": {"method": "position_group_multiplier_proxy"},
                    "ablation": {"toggles": {"no_college": False}},
                },
            ]
        }
    )

    candidates = resolve_matrix_candidates(cfg, ["A2"])

    assert [candidate.candidate_id for candidate in candidates] == ["A2"]
    assert candidates[0].regressor_target_space == "log"
    assert candidates[0].ablation_toggles == {"no_college": False}
    assert candidates[0].adjustment_method == "position_group_multiplier_proxy"


def test_experiment_matrix_and_report_smoke(monkeypatch, tmp_path):
    cfg = OmegaConf.create(
        {
            "candidate_ids": [],
            "matrix": {
                "matrix_name": "economic_target_loss",
                "output_subdir": "matrices",
                "shared": {
                    "classifier_source": "train_per_candidate",
                    "fixed_classifier_run_id": None,
                    "classifier": {
                        "experiment_name": "Matrix_Classifier",
                        "model": "catboost_classifier",
                        "target_config": "economic",
                    },
                    "regressor": {
                        "experiment_name": "Matrix_Regressor",
                        "target_config": "economic",
                    },
                    "complete_model": {
                        "experiment_name": "Matrix_Complete_Model",
                        "target_config": "economic",
                    },
                    "tags": {"phase": "4", "experiment_set": "A"},
                    "runtime": {"refit_final_model": False},
                },
                "candidates": [
                    {
                        "candidate_id": "A1",
                        "label": "positive_merit_tweedie_raw",
                        "regressor": {
                            "model": "catboost_regressor_tweedie",
                            "target_col": "Positive_Career_Merit_Cap_Share",
                            "target_space": "raw",
                        },
                        "ablation": {"toggles": {"no_scout": False}},
                    },
                    {
                        "candidate_id": "A2",
                        "label": "positive_merit_rmse_log",
                        "adjustment": {"method": "second_stage_utility_proxy"},
                        "regressor": {
                            "model": "catboost_regressor_rmse",
                            "target_col": "Positive_Career_Merit_Cap_Share",
                            "target_space": "log",
                        },
                    },
                ],
            },
            "manifest_path": None,
            "output_path": None,
        }
    )

    target_cfg = OmegaConf.create(
        {
            "classifier_sieve": {"target_col": "Economic_Success"},
            "regressor_intensity": {
                "target_col": "Positive_Career_Merit_Cap_Share",
                "target_space": "raw",
            },
            "outcome_scorecard": {
                "elite": {
                    "enabled": True,
                    "source_column": "Career_Merit_Cap_Share",
                    "quantile": 0.95,
                    "scope": "position_group",
                    "fallback_scope": "global",
                }
            },
        }
    )
    complete_pipeline_cfg = OmegaConf.create(
        {
            "data": {"raw_path": "fof8-ml/data/processed/features.parquet"},
            "target": target_cfg,
            "ablation": {
                "toggles": {
                    "no_position": False,
                    "no_position_group": False,
                    "no_combine": False,
                    "no_interviewed": True,
                    "no_zscores": False,
                    "no_scout": True,
                    "no_delta": True,
                    "no_college": True,
                }
            },
        }
    )

    def fake_load_config(_exp_root, *parts):
        if parts[-1] == "complete_model_pipeline.yaml":
            return complete_pipeline_cfg.copy()
        if parts[-2] == "target":
            return target_cfg.copy()
        raise AssertionError(f"Unexpected config request: {parts}")

    def fake_prepare_pipeline_cfg(
        _exp_root,
        *,
        pipeline_config_name,
        experiment_name,
        target_config_name,
        model_config_name,
        runtime_refit_final_model,
        run_tags,
        regressor_target_col=None,
        regressor_target_space=None,
        ablation_toggles=None,
    ):
        _ = target_config_name, runtime_refit_final_model, run_tags
        base_ablation = {
            "toggles": {
                "no_position": False,
                "no_position_group": False,
                "no_combine": False,
                "no_interviewed": True,
                "no_zscores": False,
                "no_scout": True,
                "no_delta": True,
                "no_college": True,
            },
            "toggle_to_group": {
                "no_position": "no_position",
                "no_position_group": "no_position_group",
                "no_combine": "no_combine",
                "no_interviewed": "no_interviewed",
                "no_zscores": "no_zscores",
                "no_scout": "no_scout",
                "no_delta": "no_delta",
                "no_college": "no_college",
            },
            "groups": {
                "no_position": ["Position"],
                "no_position_group": ["Position_Group"],
                "no_combine": ["Dash"],
                "no_interviewed": ["Interviewed"],
                "no_zscores": ["*_Z"],
                "no_scout": ["Scout_*"],
                "no_delta": ["Delta_*"],
                "no_college": ["College_*", "College"],
            },
            "invalid_combinations": [],
        }
        if ablation_toggles:
            base_ablation["toggles"].update(ablation_toggles)
        if pipeline_config_name == "classifier_pipeline.yaml":
            return OmegaConf.create(
                {
                    "experiment_name": experiment_name,
                    "optimization": {"metric": "classifier_val_pr_auc"},
                    "model": {"name": model_config_name, "params": {"loss_function": "Logloss"}},
                    "target": target_cfg.copy(),
                    "ablation_signature": "default",
                    "ablation": base_ablation,
                }
            )
        loss_by_model = {
            "catboost_regressor_tweedie": "Tweedie",
            "catboost_regressor_rmse": "RMSE",
        }
        return OmegaConf.create(
            {
                "experiment_name": experiment_name,
                "optimization": {"metric": "regressor_val_draft_value_score"},
                "model": {
                    "name": model_config_name,
                    "params": {"loss_function": loss_by_model[model_config_name]},
                },
                "target": {
                    "classifier_sieve": {"target_col": "Economic_Success"},
                    "regressor_intensity": {
                        "target_col": regressor_target_col,
                        "target_space": regressor_target_space,
                    },
                    "outcome_scorecard": target_cfg.outcome_scorecard,
                },
                "ablation_signature": "default",
                "ablation": base_ablation,
            }
        )

    classifier_calls = []
    regressor_calls = []
    complete_calls = []

    def fake_train_classifier(_exp_root, _cfg):
        classifier_calls.append(dict(_cfg.ablation.toggles))
        return MatrixRunResult(
            run_id=f"cls-{len(classifier_calls)}",
            experiment_name="Matrix_Classifier",
            optimization_metric="classifier_val_pr_auc",
            optimization_score=0.6,
            metrics={"classifier_val_pr_auc": 0.6},
        )

    def fake_train_regressor(_exp_root, cfg_obj):
        regressor_calls.append((cfg_obj.model.name, dict(cfg_obj.ablation.toggles)))
        run_id = f"reg-{len(regressor_calls)}"
        return MatrixRunResult(
            run_id=run_id,
            experiment_name="Matrix_Regressor",
            optimization_metric="regressor_val_draft_value_score",
            optimization_score=0.4,
            metrics={"regressor_val_draft_value_score": 0.4},
        )

    complete_metric_map = {
        "reg-1": {
            "complete_draft_value_score": 0.51,
            "complete_mean_ndcg_at_64": 0.71,
            "complete_top64_weighted_mae_normalized": 0.12,
            "complete_top64_bias": 0.01,
            "complete_top64_calibration_slope": 0.95,
            "complete_top64_actual_value": 18.0,
            "complete_bust_rate_at_32": 0.22,
            "complete_elite_precision_at_32": 0.18,
            "complete_elite_recall_at_64": 0.41,
            "complete_econ_mean_ndcg_at_64": 0.74,
            "complete_talent_mean_ndcg_at_64": 0.66,
            "complete_longevity_mean_ndcg_at_64": 0.61,
        },
        "reg-2": {
            "complete_draft_value_score": 0.49,
            "complete_mean_ndcg_at_64": 0.69,
            "complete_top64_weighted_mae_normalized": 0.11,
            "complete_top64_bias": -0.02,
            "complete_top64_calibration_slope": 1.01,
            "complete_top64_actual_value": 17.5,
            "complete_bust_rate_at_32": 0.24,
            "complete_elite_precision_at_32": 0.17,
            "complete_elite_recall_at_64": 0.39,
            "complete_econ_mean_ndcg_at_64": 0.72,
            "complete_talent_mean_ndcg_at_64": 0.68,
            "complete_longevity_mean_ndcg_at_64": 0.6,
        },
    }

    def fake_complete_eval(_exp_root, cfg_obj):
        complete_calls.append((cfg_obj.regressor_run_id, dict(cfg_obj.ablation.toggles)))
        regressor_run_id = str(cfg_obj.regressor_run_id)
        return MatrixRunResult(
            run_id=f"complete-{regressor_run_id}",
            experiment_name="Matrix_Complete_Model",
            optimization_metric="complete_draft_value_score",
            optimization_score=complete_metric_map[regressor_run_id]["complete_draft_value_score"],
            metrics=complete_metric_map[regressor_run_id],
        )

    monkeypatch.setattr(
        "fof8_ml.orchestration.experiment_matrix._load_config",
        fake_load_config,
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.experiment_matrix._prepare_pipeline_cfg",
        fake_prepare_pipeline_cfg,
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.experiment_matrix._train_classifier_pipeline",
        fake_train_classifier,
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.experiment_matrix._train_regressor_pipeline",
        fake_train_regressor,
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.experiment_matrix._evaluate_complete_model_pipeline",
        fake_complete_eval,
    )

    result = run_experiment_matrix(cfg, exp_root=str(tmp_path))

    assert result["candidate_count"] == 2
    assert len(classifier_calls) == 2
    assert classifier_calls[0]["no_scout"] is False
    assert classifier_calls[1]["no_scout"] is True
    assert regressor_calls == [
        ("catboost_regressor_tweedie", classifier_calls[0]),
        ("catboost_regressor_rmse", classifier_calls[1]),
    ]
    assert complete_calls == [
        ("reg-1", classifier_calls[0]),
        ("reg-2", classifier_calls[1]),
    ]

    manifest_dir = Path(result["output_dir"])
    assert (manifest_dir / "A1.json").exists()
    assert (manifest_dir / "A2.json").exists()
    assert (manifest_dir / "matrix_manifest.json").exists()

    metrics_by_run_id = {
        "complete-reg-1": complete_metric_map["reg-1"],
        "complete-reg-2": complete_metric_map["reg-2"],
    }

    def fake_init_tracking(self):
        self.client = _FakeMlflowClient(metrics_by_run_id)
        self.experiment_id = "exp-1"

    monkeypatch.setattr(ExperimentLogger, "init_tracking", fake_init_tracking)

    report_result = export_matrix_report(cfg, exp_root=str(tmp_path))

    assert report_result["row_count"] == 2
    csv_path = Path(report_result["output_path"])
    assert csv_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "candidate_id" in csv_text
    assert "complete_draft_value_score" in csv_text
    assert "adjustment_method" in csv_text
    assert "A1" in csv_text
    assert "A2" in csv_text
