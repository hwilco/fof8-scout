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
