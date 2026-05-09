"""High-level automation runner for FOF8 collection workflows."""

import time
from collections.abc import Callable
from pathlib import Path

from .metadata import clone_metadata_for_league, load_metadata
from .screen import _PyAutoGuiLike, prevent_system_sleep, wait_for_image
from .snapshot import create_league_snapshot
from .workflows import AutomationWorkflows


class _LazyPyAutoGUI:
    """Import pyautogui only when it is first used."""

    def __getattr__(self, name: str) -> object:
        import pyautogui

        return getattr(pyautogui, name)


class AutomationRunner:
    def __init__(
        self,
        wait_for_image_fn: Callable[..., bool] = wait_for_image,
        pyautogui_module: _PyAutoGuiLike | None = None,
    ) -> None:
        self.wait_for_image = wait_for_image_fn
        self.pyautogui: _PyAutoGuiLike = (
            pyautogui_module if pyautogui_module is not None else _LazyPyAutoGUI()  # type: ignore[assignment]
        )
        self.workflows = AutomationWorkflows(
            wait_for_image=self.wait_for_image,
            export_data=self.export_data,
            pyautogui_module=self.pyautogui,
        )

    def export_data(
        self,
        fof8_dir: str,
        league_name: str,
        output_dir: Path,
        file_filter: list[str] | dict[str, str] | None = None,
        rename_map: dict[str, str] | None = None,
    ) -> None:
        self.wait_for_image("export_data_btn.png")
        self.wait_for_image("ok_btn.png", timeout=120)

        self.wait_for_image("export_scouting_data_btn.png")
        self.wait_for_image("yes_btn.png")
        self.wait_for_image("ok_btn.png", timeout=120)

        print(f"Creating snapshot for {league_name}...")
        create_league_snapshot(
            fof8_dir,
            league_name,
            str(output_dir),
            file_filter=file_filter,
            rename_map=rename_map,
        )

    def snapshot_only(self, fof8_dir: str, league_name: str, output_dir: Path) -> None:
        with prevent_system_sleep():
            print("Starting manual export in 5 seconds... switch to the game window!")
            time.sleep(5)
            self.export_data(fof8_dir=fof8_dir, league_name=league_name, output_dir=output_dir)
            print("Manual export and snapshot complete.")

    def _validate_run_target(self, league_name: str, output_dir: Path) -> bool:
        metadata_path = output_dir / "metadata.yaml"
        if not metadata_path.exists():
            print(f"\nERROR: Metadata file not found at {metadata_path}")
            print("Please create this file manually to describe your generation settings.")
            print("You can use 'templates/metadata.example.yaml' as a starting point.")
            return False

        if league_name != output_dir.name:
            print(f"\n[!] WARNING: League name in metadata ('{league_name}') does not match ")
            print(f"    the output directory name ('{output_dir.name}').")
            print("    This often means the automation will snapshot the wrong league files.")

            try:
                response = input("\nDo you want to continue anyway? (y/N): ").strip().lower()
                if response != "y":
                    print("Aborting.")
                    return False
            except EOFError:
                print("No terminal input available. Aborting due to league name mismatch.")
                return False
        return True

    def _run_iterations(
        self,
        fof8_dir: str,
        league_name: str,
        output_dir: Path,
        num_iterations: int,
    ) -> None:
        total_start_time = time.time()
        for iteration in range(1, num_iterations + 1):
            iter_start_time = time.time()
            print(f"--- Starting Iteration {iteration} of {num_iterations} ---")

            self.workflows.export_draft_snapshot(fof8_dir, league_name, output_dir)
            self.workflows.create_one_year_history()
            self.workflows.export_post_sim_snapshot(fof8_dir, league_name, output_dir)
            self.workflows.advance_to_staff_draft()
            self.workflows.complete_staff_draft()
            self.workflows.begin_free_agency()

            iter_duration = time.time() - iter_start_time
            total_elapsed = time.time() - total_start_time
            msg = (
                f"--- Finished Iteration {iteration} in {iter_duration:.2f}s "
                f"(Total elapsed: {total_elapsed:.2f}s) ---"
            )
            print(msg)

        print(f"Automation complete! Total time: {time.time() - total_start_time:.2f}s")

    def run(self, fof8_dir: str, league_name: str, output_dir: Path, num_iterations: int) -> None:
        if not self._validate_run_target(league_name=league_name, output_dir=output_dir):
            return

        with prevent_system_sleep():
            print("Starting FOF8 Automation in 5 seconds... switch to the game window!")
            time.sleep(5)
            self._run_iterations(
                fof8_dir=fof8_dir,
                league_name=league_name,
                output_dir=output_dir,
                num_iterations=num_iterations,
            )

    def generate_universes(
        self,
        fof8_dir: str,
        base_metadata_path: Path,
        universe_names: list[str],
        output_root: Path,
        num_iterations: int,
        overwrite_metadata: bool = False,
    ) -> None:
        base_metadata, _ = load_metadata(base_metadata_path)
        begin_with_coach_names_file = bool(
            base_metadata.get("new_game_options", {}).get("coach_names_file", False)
        )

        with prevent_system_sleep():
            print(
                "Starting FOF8 multi-universe automation in 5 seconds... switch to the game window!"
            )
            time.sleep(5)

            for index, universe_name in enumerate(universe_names, start=1):
                output_dir = output_root / universe_name
                metadata_path = clone_metadata_for_league(
                    base_metadata_path=base_metadata_path,
                    league_name=universe_name,
                    output_dir=output_dir,
                    overwrite=overwrite_metadata,
                )
                print(
                    f"=== Starting generated universe {index} of {len(universe_names)}: "
                    f"{universe_name} ==="
                )
                print(f"Generated metadata at {metadata_path}")

                self.workflows.save_current_universe(required=False)
                self.workflows.start_new_game(
                    league_name=universe_name,
                    begin_with_coach_names_file=begin_with_coach_names_file,
                )
                if not begin_with_coach_names_file:
                    self.workflows.complete_initial_staff_draft()

                if not self._validate_run_target(league_name=universe_name, output_dir=output_dir):
                    return

                self._run_iterations(
                    fof8_dir=fof8_dir,
                    league_name=universe_name,
                    output_dir=output_dir,
                    num_iterations=num_iterations,
                )
                self.workflows.save_current_universe()
