from contextlib import contextmanager
from pathlib import Path

from fof8_gen.automation_runner import AutomationRunner


@contextmanager
def _noop_sleep_guard():
    yield


def test_snapshot_only_dispatch(monkeypatch, tmp_path: Path):
    runner = AutomationRunner(wait_for_image_fn=lambda *args, **kwargs: True)

    calls = []

    def _fake_export_data(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("fof8_gen.automation_runner.prevent_system_sleep", _noop_sleep_guard)
    monkeypatch.setattr("fof8_gen.automation_runner.time.sleep", lambda _n: None)
    monkeypatch.setattr(runner, "export_data", _fake_export_data)

    runner.snapshot_only(
        fof8_dir="/fake/fof8",
        league_name="League",
        output_dir=tmp_path,
    )

    assert calls == [
        {
            "fof8_dir": "/fake/fof8",
            "league_name": "League",
            "output_dir": tmp_path,
        }
    ]


def test_generate_universes_runs_workflows_in_order(monkeypatch, tmp_path: Path):
    base_metadata = tmp_path / "base_metadata.yaml"
    base_metadata.write_text(
        "new_game_options:\n  league_name: TEMPLATE\n  coach_names_file: false\n",
        encoding="utf-8",
    )
    runner = AutomationRunner(wait_for_image_fn=lambda *args, **kwargs: True)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr("fof8_gen.automation_runner.prevent_system_sleep", _noop_sleep_guard)
    monkeypatch.setattr("fof8_gen.automation_runner.time.sleep", lambda _n: None)
    monkeypatch.setattr(
        runner.workflows,
        "save_current_universe",
        lambda required=True: calls.append(("save_current_universe", required)) or True,
    )
    monkeypatch.setattr(
        runner.workflows,
        "start_new_game",
        lambda league_name, begin_with_coach_names_file: calls.append(
            ("start_new_game", league_name, begin_with_coach_names_file)
        ),
    )
    monkeypatch.setattr(
        runner.workflows,
        "complete_initial_staff_draft",
        lambda: calls.append(("complete_initial_staff_draft",)),
    )
    monkeypatch.setattr(
        runner,
        "_run_iterations",
        lambda fof8_dir, league_name, output_dir, num_iterations: calls.append(
            ("run_iterations", fof8_dir, league_name, output_dir, num_iterations)
        ),
    )

    runner.generate_universes(
        fof8_dir="/fake/fof8",
        base_metadata_path=base_metadata,
        universe_names=["DRAFT009", "DRAFT010"],
        output_root=tmp_path / "generated",
        num_iterations=3,
    )

    assert calls == [
        ("save_current_universe", False),
        ("start_new_game", "DRAFT009", False),
        ("complete_initial_staff_draft",),
        ("run_iterations", "/fake/fof8", "DRAFT009", tmp_path / "generated" / "DRAFT009", 3),
        ("save_current_universe", True),
        ("save_current_universe", False),
        ("start_new_game", "DRAFT010", False),
        ("complete_initial_staff_draft",),
        ("run_iterations", "/fake/fof8", "DRAFT010", tmp_path / "generated" / "DRAFT010", 3),
        ("save_current_universe", True),
    ]


def test_generate_universes_skips_initial_staff_draft_for_coach_names_file(
    monkeypatch, tmp_path: Path
):
    base_metadata = tmp_path / "base_metadata.yaml"
    base_metadata.write_text(
        "new_game_options:\n  league_name: TEMPLATE\n  coach_names_file: true\n",
        encoding="utf-8",
    )
    runner = AutomationRunner(wait_for_image_fn=lambda *args, **kwargs: True)
    calls: list[str] = []

    monkeypatch.setattr("fof8_gen.automation_runner.prevent_system_sleep", _noop_sleep_guard)
    monkeypatch.setattr("fof8_gen.automation_runner.time.sleep", lambda _n: None)
    monkeypatch.setattr(runner.workflows, "save_current_universe", lambda required=True: True)
    monkeypatch.setattr(
        runner.workflows,
        "start_new_game",
        lambda league_name, begin_with_coach_names_file: calls.append(
            f"start_new_game:{league_name}:{begin_with_coach_names_file}"
        ),
    )
    monkeypatch.setattr(
        runner.workflows,
        "complete_initial_staff_draft",
        lambda: calls.append("complete_initial_staff_draft"),
    )
    monkeypatch.setattr(
        runner,
        "_run_iterations",
        lambda fof8_dir, league_name, output_dir, num_iterations: calls.append(
            f"run_iterations:{league_name}"
        ),
    )

    runner.generate_universes(
        fof8_dir="/fake/fof8",
        base_metadata_path=base_metadata,
        universe_names=["DRAFT009"],
        output_root=tmp_path / "generated",
        num_iterations=1,
    )

    assert calls == [
        "start_new_game:DRAFT009:True",
        "run_iterations:DRAFT009",
    ]
