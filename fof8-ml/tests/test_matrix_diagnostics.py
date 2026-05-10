from types import SimpleNamespace

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
