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
    assert candidates[0].regressor_target_config is None
    assert candidates[0].ablation_toggles == {"no_college": False}
    assert candidates[0].classifier_ablation_toggles == {"no_college": False}
    assert candidates[0].regressor_ablation_toggles == {"no_college": False}
    assert candidates[0].adjustment_method == "position_group_multiplier_proxy"


def test_set_f_mlp_talent_matrix_resolves_top3_raw_candidates():
    matrix_path = (
        Path(__file__).resolve().parents[2]
        / "pipelines"
        / "conf"
        / "matrix"
        / "set_f_mlp_talent.yaml"
    )
    cfg = OmegaConf.load(matrix_path)

    candidates = resolve_matrix_candidates(cfg)

    assert [candidate.candidate_id for candidate in candidates] == ["F1", "F2", "F3"]
    assert [candidate.regressor_model for candidate in candidates] == [
        "catboost_regressor_rmse",
        "sklearn_mlp_regressor",
        "sklearn_mlp_regressor",
    ]
    assert {candidate.regressor_target_col for candidate in candidates} == {
        "Top3_Mean_Current_Overall"
    }
    assert {candidate.regressor_target_space for candidate in candidates} == {"raw"}
    assert candidates[2].ablation_toggles == {}
    assert candidates[2].classifier_ablation_toggles == {}
    assert candidates[2].regressor_ablation_toggles == {"no_position": True}


def test_set_g_talent_matrix_resolves_per_candidate_target_configs():
    matrix_path = (
        Path(__file__).resolve().parents[2]
        / "pipelines"
        / "conf"
        / "matrix"
        / "set_g_talent_targets.yaml"
    )
    cfg = OmegaConf.load(matrix_path)

    candidates = resolve_matrix_candidates(cfg)

    assert [candidate.candidate_id for candidate in candidates] == ["G1", "G2"]
    assert [candidate.regressor_target_config for candidate in candidates] == [
        "talent_top3",
        "talent_control_window",
    ]
    assert [candidate.regressor_target_col for candidate in candidates] == [
        "Top3_Mean_Current_Overall",
        "Control_Window_Mean_Current_Overall",
    ]


def test_position_scope_matrix_resolves_candidate_positions():
    matrix_path = (
        Path(__file__).resolve().parents[2]
        / "pipelines"
        / "conf"
        / "matrix"
        / "set_i_talent_position_scope.yaml"
    )
    cfg = OmegaConf.load(matrix_path)

    candidates = resolve_matrix_candidates(cfg)

    assert [candidate.candidate_id for candidate in candidates] == ["I1", "I2", "I3", "I4"]
    assert candidates[0].positions is None
    assert candidates[1].positions == ["QB"]
    assert candidates[2].positions == ["RB"]
    assert candidates[3].positions == [
        "QB",
        "RB",
        "WR",
        "TE",
        "C",
        "G",
        "T",
        "DE",
        "DT",
        "OLB",
        "ILB",
        "CB",
        "S",
    ]


def test_set_j_mlp_position_scope_resolves_candidate_positions_and_model():
    matrix_path = (
        Path(__file__).resolve().parents[2]
        / "pipelines"
        / "conf"
        / "matrix"
        / "set_j_mlp_position_scope.yaml"
    )
    cfg = OmegaConf.load(matrix_path)

    candidates = resolve_matrix_candidates(cfg)

    assert [candidate.candidate_id for candidate in candidates] == ["J1", "J2", "J3", "J4"]
    assert {candidate.regressor_model for candidate in candidates} == {
        "sklearn_mlp_regressor_control_window"
    }
    assert {candidate.regressor_target_config for candidate in candidates} == {
        "talent_control_window"
    }
    assert {candidate.regressor_target_col for candidate in candidates} == {
        "Control_Window_Mean_Current_Overall"
    }
    assert candidates[0].positions is None
    assert candidates[1].positions == ["QB"]
    assert candidates[2].positions == ["RB"]
    assert candidates[3].positions == [
        "QB",
        "RB",
        "WR",
        "TE",
        "C",
        "G",
        "T",
        "DE",
        "DT",
        "OLB",
        "ILB",
        "CB",
        "S",
    ]


def test_talent_matrices_use_regressor_only_mode():
    for filename in [
        "set_h_talent_target_standardization.yaml",
        "set_i_talent_position_scope.yaml",
        "set_j_mlp_position_scope.yaml",
        "set_k_talent_sample_weighting.yaml",
    ]:
        matrix_path = (
            Path(__file__).resolve().parents[2] / "pipelines" / "conf" / "matrix" / filename
        )
        cfg = OmegaConf.load(matrix_path)
        assert cfg.shared.classifier_source == "none"


def test_set_k_sample_weighting_matrix_resolves_target_variants():
    matrix_path = (
        Path(__file__).resolve().parents[2]
        / "pipelines"
        / "conf"
        / "matrix"
        / "set_k_talent_sample_weighting.yaml"
    )
    cfg = OmegaConf.load(matrix_path)

    candidates = resolve_matrix_candidates(cfg)

    assert [candidate.candidate_id for candidate in candidates] == ["K1", "K2", "K3", "K4", "K5"]
    assert {candidate.regressor_model for candidate in candidates} == {"catboost_regressor_rmse"}
    assert candidates[0].regressor_target_config == "talent_control_window"
    assert [candidate.regressor_target_config for candidate in candidates[1:]] == [
        "talent_control_window_weighted_top25",
        "talent_control_window_weighted_top10",
        "talent_control_window_weighted_top10_top5",
        "talent_control_window_weighted_position_top10",
    ]


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
        positions_override=None,
        ablation_toggles=None,
    ):
        _ = target_config_name, runtime_refit_final_model, run_tags, positions_override
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


def test_experiment_matrix_supports_regressor_only_mode(monkeypatch, tmp_path):
    cfg = OmegaConf.create(
        {
            "candidate_ids": [],
            "matrix": {
                "matrix_name": "talent_only",
                "output_subdir": "matrices",
                "shared": {
                    "classifier_source": "none",
                    "fixed_classifier_run_id": None,
                    "classifier": {
                        "experiment_name": "Unused_Classifier",
                        "model": "catboost_classifier",
                        "target_config": "economic",
                    },
                    "regressor": {
                        "experiment_name": "Matrix_Regressor",
                        "target_config": "talent_top3",
                    },
                    "complete_model": {
                        "experiment_name": "Unused_Complete_Model",
                        "target_config": "economic",
                    },
                    "tags": {"phase": "4", "experiment_set": "H"},
                    "runtime": {"refit_final_model": False},
                },
                "candidates": [
                    {
                        "candidate_id": "H1",
                        "label": "top3_raw_all_positions",
                        "regressor": {
                            "model": "catboost_regressor_rmse",
                            "target_config": "talent_top3",
                            "target_col": "Top3_Mean_Current_Overall",
                            "target_space": "raw",
                        },
                    }
                ],
            },
            "manifest_path": None,
            "output_path": None,
        }
    )

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
        positions_override=None,
        ablation_toggles=None,
    ):
        _ = (
            pipeline_config_name,
            target_config_name,
            runtime_refit_final_model,
            run_tags,
            positions_override,
            ablation_toggles,
        )
        return OmegaConf.create(
            {
                "experiment_name": experiment_name,
                "optimization": {"metric": "regressor_val_top32_target_capture_ratio"},
                "model": {"name": model_config_name, "params": {"loss_function": "RMSE"}},
                "target": {
                    "classifier_sieve": {"target_col": "unused"},
                    "regressor_intensity": {
                        "target_col": regressor_target_col,
                        "target_space": regressor_target_space,
                    },
                },
                "ablation_signature": "default",
            }
        )

    regressor_calls = []

    def fake_train_regressor(_exp_root, cfg_obj):
        regressor_calls.append(cfg_obj.target.regressor_intensity.target_col)
        return MatrixRunResult(
            run_id="reg-1",
            experiment_name="Matrix_Regressor",
            optimization_metric="regressor_val_top32_target_capture_ratio",
            optimization_score=0.75,
            metrics={"regressor_val_top32_target_capture_ratio": 0.75},
        )

    monkeypatch.setattr(
        "fof8_ml.orchestration.experiment_matrix._prepare_pipeline_cfg",
        fake_prepare_pipeline_cfg,
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.experiment_matrix._train_regressor_pipeline",
        fake_train_regressor,
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.experiment_matrix._train_classifier_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("should not train classifier")
        ),
    )
    monkeypatch.setattr(
        "fof8_ml.orchestration.experiment_matrix._evaluate_complete_model_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("should not evaluate complete model")
        ),
    )

    result = run_experiment_matrix(cfg, exp_root=str(tmp_path))

    assert regressor_calls == ["Top3_Mean_Current_Overall"]
    assert result["candidate_count"] == 1


def test_export_matrix_report_supports_regressor_only_manifests(monkeypatch, tmp_path):
    matrix_dir = tmp_path / "outputs" / "matrices" / "talent_only"
    matrix_dir.mkdir(parents=True)

    candidate_manifest = {
        "candidate_id": "H1",
        "label": "top3_raw_all_positions",
        "classifier_source": "none",
        "classifier_run_id": "",
        "regressor_run_id": "reg-1",
        "complete_run_id": "",
        "classifier_target_col": "",
        "regressor_target_col": "Top3_Mean_Current_Overall",
        "regressor_target_space": "raw",
        "regressor_model": "catboost_regressor_rmse",
        "regressor_loss_function": "RMSE",
        "adjustment_method": None,
        "ablation_signature": "default",
        "elite_config": {},
    }
    candidate_path = matrix_dir / "H1.json"
    candidate_path.write_text(__import__("json").dumps(candidate_manifest), encoding="utf-8")
    (matrix_dir / "matrix_manifest.json").write_text(
        __import__("json").dumps(
            {
                "matrix_name": "talent_only",
                "candidates": [
                    {
                        "candidate_id": "H1",
                        "label": "top3_raw_all_positions",
                        "manifest_path": str(candidate_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    metrics_by_run_id = {
        "reg-1": {
            "regressor_val_top32_target_capture_ratio": 0.81,
            "regressor_val_top64_target_capture_ratio": 0.87,
            "regressor_val_mean_ndcg_at_32": 0.72,
            "regressor_val_mean_ndcg_at_64": 0.75,
            "regressor_val_rmse": 8.4,
            "regressor_val_mae": 5.9,
            "regressor_test_draft_value_score": 0.0,
        }
    }

    def fake_init_tracking(self):
        self.client = _FakeMlflowClient(metrics_by_run_id)
        self.experiment_id = "exp-1"

    monkeypatch.setattr(ExperimentLogger, "init_tracking", fake_init_tracking)

    cfg = OmegaConf.create(
        {
            "manifest_path": None,
            "output_path": None,
            "matrix": {
                "matrix_name": "talent_only",
                "output_subdir": "matrices",
            },
        }
    )

    result = export_matrix_report(cfg, exp_root=str(tmp_path))

    assert result["row_count"] == 1
    csv_text = Path(result["output_path"]).read_text(encoding="utf-8")
    assert "regressor_val_top32_target_capture_ratio" in csv_text
    assert "0.81" in csv_text


def test_fixed_classifier_complete_eval_keeps_classifier_feature_schema(monkeypatch, tmp_path):
    cfg = OmegaConf.create(
        {
            "candidate_ids": [],
            "matrix": {
                "matrix_name": "fixed_classifier_regressor_ablation",
                "output_subdir": "matrices",
                "shared": {
                    "classifier_source": "fixed_run",
                    "fixed_classifier_run_id": "fixed-cls",
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
                    "tags": {},
                    "runtime": {"refit_final_model": False},
                },
                "candidates": [
                    {
                        "candidate_id": "F3",
                        "label": "mlp_no_position",
                        "ablation": {"toggles": {"no_position": True}},
                        "regressor": {
                            "model": "sklearn_mlp_regressor",
                            "target_col": "Top3_Mean_Current_Overall",
                            "target_space": "raw",
                        },
                    }
                ],
            },
        }
    )
    target_cfg = OmegaConf.create(
        {
            "classifier_sieve": {"target_col": "Economic_Success"},
            "regressor_intensity": {"target_col": "Top3_Mean_Current_Overall"},
            "outcome_scorecard": {"elite": {"enabled": False}},
        }
    )
    complete_pipeline_cfg = OmegaConf.create(
        {
            "target": target_cfg.copy(),
            "ablation": {
                "toggles": {"no_position": False},
                "toggle_to_group": {"no_position": "no_position"},
                "groups": {"no_position": ["Position"]},
                "invalid_combinations": [],
            },
        }
    )

    def fake_load_config(_exp_root, *parts):
        if parts[-1] == "complete_model_pipeline.yaml":
            return complete_pipeline_cfg.copy()
        if parts[-2] == "target":
            return target_cfg.copy()
        raise AssertionError(f"Unexpected config request: {parts}")

    def fake_prepare_pipeline_cfg(_exp_root, **kwargs):
        ablation = complete_pipeline_cfg.ablation.copy()
        if kwargs.get("ablation_toggles"):
            ablation.toggles.update(kwargs["ablation_toggles"])
        return OmegaConf.create(
            {
                "experiment_name": kwargs["experiment_name"],
                "optimization": {"metric": "regressor_val_draft_value_score"},
                "model": {"name": kwargs["model_config_name"], "params": {}},
                "target": target_cfg.copy(),
                "ablation": ablation,
                "ablation_signature": "regressor-no-position",
            }
        )

    regressor_toggles = []
    complete_toggles = []

    monkeypatch.setattr("fof8_ml.orchestration.experiment_matrix._load_config", fake_load_config)
    monkeypatch.setattr(
        "fof8_ml.orchestration.experiment_matrix._prepare_pipeline_cfg",
        fake_prepare_pipeline_cfg,
    )

    def fake_train_regressor(_exp_root, cfg_obj):
        regressor_toggles.append(dict(cfg_obj.ablation.toggles))
        return MatrixRunResult(
            run_id="reg-f3",
            experiment_name="Matrix_Regressor",
            optimization_metric="regressor_val_draft_value_score",
            optimization_score=0.1,
            metrics={"regressor_val_draft_value_score": 0.1},
        )

    monkeypatch.setattr(
        "fof8_ml.orchestration.experiment_matrix._train_regressor_pipeline",
        fake_train_regressor,
    )

    def fake_complete_eval(_exp_root, cfg_obj):
        complete_toggles.append(dict(cfg_obj.ablation.toggles))
        return MatrixRunResult(
            run_id="complete-f3",
            experiment_name="Matrix_Complete_Model",
            optimization_metric="complete_draft_value_score",
            optimization_score=0.2,
            metrics={
                "complete_draft_value_score": 0.2,
                "complete_mean_ndcg_at_64": 0.2,
                "complete_top64_weighted_mae_normalized": 0.2,
                "complete_top64_bias": 0.0,
                "complete_top64_calibration_slope": 1.0,
                "complete_top64_actual_value": 1.0,
                "complete_bust_rate_at_32": 0.0,
                "complete_elite_precision_at_32": 0.0,
                "complete_elite_recall_at_64": 0.0,
                "complete_econ_mean_ndcg_at_64": 0.2,
                "complete_talent_mean_ndcg_at_64": 0.2,
                "complete_longevity_mean_ndcg_at_64": 0.2,
            },
        )

    monkeypatch.setattr(
        "fof8_ml.orchestration.experiment_matrix._evaluate_complete_model_pipeline",
        fake_complete_eval,
    )

    run_experiment_matrix(cfg, exp_root=str(tmp_path))

    assert regressor_toggles == [{"no_position": True}]
    assert complete_toggles == [{"no_position": False}]
