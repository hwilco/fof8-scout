import pytest
from fof8_ml.orchestration.data_loader import resolve_feature_ablation_config
from omegaconf import OmegaConf


def _make_cfg():
    return OmegaConf.create(
        {
            "include_features": None,
            "exclude_features": [],
            "ablation": {
                "toggles": {
                    "no_position": False,
                    "no_combine": False,
                    "no_interviewed": False,
                    "no_zscores": False,
                },
                "toggle_to_group": {
                    "no_position": "no_position",
                    "no_combine": "no_combine",
                    "no_interviewed": "no_interviewed",
                    "no_zscores": "no_zscores",
                },
                "groups": {
                    "no_position": ["Position", "Scout_*", "Delta_*"],
                    "no_combine": ["Dash", "Strength"],
                    "no_interviewed": ["Interviewed"],
                    "no_zscores": ["*_Z"],
                },
                "invalid_combinations": [],
            },
        }
    )


def test_resolve_feature_ablation_combines_enabled_toggle_groups():
    cfg = _make_cfg()
    cfg.ablation.toggles.no_position = True
    cfg.ablation.toggles.no_interviewed = True

    resolved = resolve_feature_ablation_config(cfg)
    assert resolved["enabled_toggles"] == ["no_position", "no_interviewed"]
    assert resolved["exclude_features"] == ["Position", "Scout_*", "Delta_*", "Interviewed"]


def test_resolve_feature_ablation_signature_is_deterministic():
    cfg_a = _make_cfg()
    cfg_a.ablation.toggles.no_zscores = True
    cfg_b = _make_cfg()
    cfg_b.ablation.toggles.no_zscores = True

    resolved_a = resolve_feature_ablation_config(cfg_a)
    resolved_b = resolve_feature_ablation_config(cfg_b)

    assert resolved_a["signature"] == resolved_b["signature"]


def test_resolve_feature_ablation_rejects_invalid_combinations():
    cfg = _make_cfg()
    cfg.ablation.invalid_combinations = [["no_position", "no_combine"]]
    cfg.ablation.toggles.no_position = True
    cfg.ablation.toggles.no_combine = True

    with pytest.raises(ValueError) as exc_info:
        resolve_feature_ablation_config(cfg)
    assert "Invalid ablation toggle combination enabled" in str(exc_info.value)
