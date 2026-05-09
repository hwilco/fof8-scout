from pathlib import Path

from fof8_gen import automation


class _FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def snapshot_only(self, **kwargs) -> None:
        self.calls.append(("snapshot_only", kwargs))

    def run(self, **kwargs) -> None:
        self.calls.append(("run", kwargs))

    def generate_universes(self, **kwargs) -> None:
        self.calls.append(("generate_universes", kwargs))


def test_cli_dispatches_generate_universes(monkeypatch, tmp_path: Path):
    metadata = tmp_path / "metadata.yaml"
    metadata.write_text(
        "new_game_options:\n  league_name: TEMPLATE\n  coach_names_file: false\n",
        encoding="utf-8",
    )
    fake_runner = _FakeRunner()
    local_app_data = tmp_path / "LocalAppData"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))

    monkeypatch.setattr(
        automation,
        "AutomationRunner",
        lambda: fake_runner,
        raising=False,
    )
    monkeypatch.setattr(
        "fof8_gen.automation_runner.AutomationRunner",
        lambda: fake_runner,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "gather-data",
            "--metadata",
            str(metadata),
            "--generate-universes",
            "DRAFT009:DRAFT010",
            "--output-root",
            str(tmp_path / "raw"),
            "--iterations",
            "7",
            "--overwrite-metadata",
        ],
    )

    automation.main()

    assert fake_runner.calls == [
        (
            "generate_universes",
            {
                "fof8_dir": local_app_data / "Solecismic Software" / "Front Office Football Eight",
                "base_metadata_path": metadata.resolve(),
                "universe_names": ["DRAFT009", "DRAFT010"],
                "output_root": tmp_path / "raw",
                "num_iterations": 7,
                "overwrite_metadata": True,
            },
        )
    ]
