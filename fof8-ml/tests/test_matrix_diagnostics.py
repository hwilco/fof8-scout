from types import SimpleNamespace

import numpy as np
import polars as pl
from fof8_ml.orchestration.experiment_logger import ExperimentLogger
from fof8_ml.reporting.matrix_diagnostics import export_matrix_diagnostics
from omegaconf import OmegaConf


class _FakeMlflowClient:
    def __init__(self, artifact_map):
        self.artifact_map = artifact_map

    def download_artifacts(self, run_id: str, artifact_path: str):
        return self.artifact_map[(run_id, artifact_path)]

    def list_artifacts(self, run_id: str, artifact_path: str):
        _ = artifact_path
        return [
            SimpleNamespace(path=path)
            for (artifact_run_id, path), _local in self.artifact_map.items()
            if artifact_run_id == run_id
        ]


def test_export_matrix_diagnostics_writes_second_pass_artifacts(monkeypatch, tmp_path):
    matrix_dir = tmp_path / "outputs" / "matrices" / "economic_target_loss"
    matrix_dir.mkdir(parents=True)

    candidate_a_manifest = {
        "candidate_id": "A1",
        "label": "candidate-a",
        "complete_run_id": "complete-a",
        "board_artifact_path": "complete_model_holdout_board.csv",
        "position_group_metrics_artifact_path": "complete_model_position_group_metrics.csv",
    }
    candidate_b_manifest = {
        "candidate_id": "A2",
        "label": "candidate-b",
        "complete_run_id": "complete-b",
        "board_artifact_path": "complete_model_holdout_board.csv",
        "position_group_metrics_artifact_path": "complete_model_position_group_metrics.csv",
    }

    candidate_a_path = matrix_dir / "A1.json"
    candidate_b_path = matrix_dir / "A2.json"
    candidate_a_path.write_text(__import__("json").dumps(candidate_a_manifest), encoding="utf-8")
    candidate_b_path.write_text(__import__("json").dumps(candidate_b_manifest), encoding="utf-8")
    (matrix_dir / "matrix_manifest.json").write_text(
        __import__("json").dumps(
            {
                "matrix_name": "economic_target_loss",
                "candidates": [
                    {
                        "candidate_id": "A1",
                        "label": "candidate-a",
                        "manifest_path": str(candidate_a_path),
                    },
                    {
                        "candidate_id": "A2",
                        "label": "candidate-b",
                        "manifest_path": str(candidate_b_path),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    board_a = pl.DataFrame(
        {
            "Universe": ["U1", "U1", "U1"],
            "Player_ID": [1, 2, 3],
            "Year": [2020, 2020, 2020],
            "First_Name": ["A", "B", "C"],
            "Last_Name": ["A", "B", "C"],
            "Position_Group": ["QB", "WR", "QB"],
            "complete_prediction": [0.9, 0.8, 0.1],
            "rank_within_year": [1, 2, 3],
        }
    )
    board_b = pl.DataFrame(
        {
            "Universe": ["U1", "U1", "U1"],
            "Player_ID": [1, 2, 3],
            "Year": [2020, 2020, 2020],
            "First_Name": ["A", "B", "C"],
            "Last_Name": ["A", "B", "C"],
            "Position_Group": ["QB", "WR", "QB"],
            "complete_prediction": [0.7, 0.95, 0.2],
            "rank_within_year": [2, 1, 3],
        }
    )
    pos_a = pl.DataFrame(
        {
            "slice_column": ["Position_Group", "Position_Group"],
            "slice_value": ["QB", "WR"],
            "n_players": [2, 1],
            "n_draft_classes": [1, 1],
            "complete_mean_ndcg_at_64": [0.8, 0.7],
        }
    )
    pos_b = pl.DataFrame(
        {
            "slice_column": ["Position_Group", "Position_Group"],
            "slice_value": ["QB", "WR"],
            "n_players": [2, 1],
            "n_draft_classes": [1, 1],
            "complete_mean_ndcg_at_64": [0.75, 0.9],
        }
    )

    board_a_path = tmp_path / "board_a.csv"
    board_b_path = tmp_path / "board_b.csv"
    pos_a_path = tmp_path / "pos_a.csv"
    pos_b_path = tmp_path / "pos_b.csv"
    board_a.write_csv(board_a_path)
    board_b.write_csv(board_b_path)
    pos_a.write_csv(pos_a_path)
    pos_b.write_csv(pos_b_path)

    artifact_map = {
        ("complete-a", "complete_model_holdout_board.csv"): str(board_a_path),
        ("complete-b", "complete_model_holdout_board.csv"): str(board_b_path),
        ("complete-a", "complete_model_position_group_metrics.csv"): str(pos_a_path),
        ("complete-b", "complete_model_position_group_metrics.csv"): str(pos_b_path),
    }

    def fake_init_tracking(self):
        self.client = _FakeMlflowClient(artifact_map)
        self.experiment_id = "exp-1"

    monkeypatch.setattr(ExperimentLogger, "init_tracking", fake_init_tracking)

    cfg = OmegaConf.create(
        {
            "manifest_path": None,
            "output_dir": None,
            "matrix": {
                "matrix_name": "economic_target_loss",
                "output_subdir": "matrices",
            },
        }
    )

    result = export_matrix_diagnostics(cfg, exp_root=str(tmp_path))

    assert result["candidate_count"] == 2
    for filename in [
        "position_group_summary.csv",
        "board_overlap_summary.csv",
        "board_rank_deltas.csv",
        "board_position_mix.csv",
    ]:
        assert (matrix_dir / filename).exists(), filename

    overlap_text = (matrix_dir / "board_overlap_summary.csv").read_text(encoding="utf-8")
    assert "overlap_count" in overlap_text
    rank_text = (matrix_dir / "board_rank_deltas.csv").read_text(encoding="utf-8")
    assert "abs_rank_delta" in rank_text


def test_export_matrix_diagnostics_writes_regressor_only_position_slice_summary(
    monkeypatch, tmp_path
):
    matrix_dir = tmp_path / "outputs" / "matrices" / "talent_only"
    matrix_dir.mkdir(parents=True)

    candidate_manifest = {
        "candidate_id": "H1",
        "label": "top3_raw_all_positions",
        "classifier_source": "none",
        "regressor_run_id": "reg-1",
        "regressor_target_col": "Control_Window_Mean_Current_Overall",
        "complete_run_id": "",
    }
    candidate_manifest_b = {
        "candidate_id": "H2",
        "label": "top3_qb_only",
        "classifier_source": "none",
        "regressor_run_id": "reg-2",
        "regressor_target_col": "Control_Window_Mean_Current_Overall",
        "complete_run_id": "",
    }
    candidate_path = matrix_dir / "H1.json"
    candidate_b_path = matrix_dir / "H2.json"
    candidate_path.write_text(__import__("json").dumps(candidate_manifest), encoding="utf-8")
    candidate_b_path.write_text(__import__("json").dumps(candidate_manifest_b), encoding="utf-8")
    (matrix_dir / "matrix_manifest.json").write_text(
        __import__("json").dumps(
            {
                "matrix_name": "talent_only",
                "candidates": [
                    {
                        "candidate_id": "H1",
                        "label": "top3_raw_all_positions",
                        "manifest_path": str(candidate_path),
                    },
                    {
                        "candidate_id": "H2",
                        "label": "top3_qb_only",
                        "manifest_path": str(candidate_b_path),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_init_tracking(self):
        self.client = object()
        self.experiment_id = "exp-1"

    monkeypatch.setattr(ExperimentLogger, "init_tracking", fake_init_tracking)

    prepared = SimpleNamespace(
        X_val=pl.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0]}),
        y_cls_val=np.array([1, 1, 1, 1]),
        y_reg_val=np.array([10.0, 8.0, 7.0, 3.0]),
        meta_val=pl.DataFrame(
            {
                "Universe": ["U1", "U1", "U2", "U2"],
                "Year": [2020, 2020, 2021, 2021],
                "Position_Group": ["QB", "RB", "QB", "RB"],
            }
        ),
    )

    monkeypatch.setattr(
        "fof8_ml.reporting.matrix_diagnostics._prepare_regressor_only_validation_frame",
        lambda cfg, exp_root, reference_candidate: (
            prepared.X_val,
            prepared.y_reg_val,
            (
                prepared.meta_val.get_column("Universe")
                + ":"
                + prepared.meta_val.get_column("Year").cast(pl.Utf8)
            ).to_numpy(),
            prepared.meta_val,
        ),
    )
    monkeypatch.setattr(
        "fof8_ml.reporting.matrix_diagnostics._load_role_bundle",
        lambda client, run_id, role, artifact_path, exp_root=None: {"run_id": run_id},
    )

    def fake_predict(bundle, X_full):
        _ = X_full
        if bundle["run_id"] == "reg-1":
            return np.array([9.0, 5.0, 6.0, 2.0])
        return np.array([8.0, 4.0, 9.0, 1.0])

    monkeypatch.setattr("fof8_ml.reporting.matrix_diagnostics._predict_regressor", fake_predict)

    cfg = OmegaConf.create(
        {
            "manifest_path": None,
            "output_dir": None,
            "matrix": {
                "matrix_name": "talent_only",
                "output_subdir": "matrices",
                "candidates": [
                    {
                        "candidate_id": "H1",
                        "label": "top3_raw_all_positions",
                        "regressor": {
                            "model": "catboost_regressor_rmse",
                            "target_config": "talent_control_window",
                            "target_col": "Control_Window_Mean_Current_Overall",
                            "target_space": "raw",
                        },
                    },
                    {
                        "candidate_id": "H2",
                        "label": "top3_qb_only",
                        "positions": ["QB"],
                        "regressor": {
                            "model": "catboost_regressor_rmse",
                            "target_config": "talent_control_window",
                            "target_col": "Control_Window_Mean_Current_Overall",
                            "target_space": "raw",
                        },
                    },
                ],
            },
        }
    )

    result = export_matrix_diagnostics(cfg, exp_root=str(tmp_path))
    assert result["candidate_count"] == 2
    slice_path = matrix_dir / "regressor_position_slice_summary.csv"
    assert slice_path.exists()
    summary = pl.read_csv(slice_path)
    assert set(summary.get_column("evaluation_position_group").to_list()) == {"all", "QB", "RB"}
    assert set(summary.get_column("training_scope").to_list()) == {"all", "QB"}
