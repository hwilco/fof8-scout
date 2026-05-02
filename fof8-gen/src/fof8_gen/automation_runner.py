"""High-level automation runner for FOF8 collection workflows."""

import time
from pathlib import Path

from .screen import prevent_system_sleep, wait_for_image
from .snapshot import create_league_snapshot
from .workflows import AutomationWorkflows


class _LazyPyAutoGUI:
    """Import pyautogui only when it is first used."""

    def __getattr__(self, name):
        import pyautogui

        return getattr(pyautogui, name)


class AutomationRunner:
    def __init__(self, wait_for_image_fn=wait_for_image, pyautogui_module=None):
        self.wait_for_image = wait_for_image_fn
        self.pyautogui = pyautogui_module if pyautogui_module is not None else _LazyPyAutoGUI()
        self.workflows = AutomationWorkflows(
            wait_for_image=self.wait_for_image,
            export_data=self.export_data,
            pyautogui_module=self.pyautogui,
        )

    def export_data(self, fof8_dir, league_name, output_dir, file_filter=None, rename_map=None):
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

    def snapshot_only(self, fof8_dir, league_name, output_dir):
        with prevent_system_sleep():
            print("Starting manual export in 5 seconds... switch to the game window!")
            time.sleep(5)
            self.export_data(fof8_dir=fof8_dir, league_name=league_name, output_dir=output_dir)
            print("Manual export and snapshot complete.")

    def run(self, fof8_dir, league_name, output_dir: Path, num_iterations):
        metadata_path = output_dir / "metadata.yaml"
        if not metadata_path.exists():
            print(f"\nERROR: Metadata file not found at {metadata_path}")
            print("Please create this file manually to describe your generation settings.")
            print("You can use 'templates/metadata.example.yaml' as a starting point.")
            return

        if league_name != output_dir.name:
            print(f"\n[!] WARNING: League name in metadata ('{league_name}') does not match ")
            print(f"    the output directory name ('{output_dir.name}').")
            print("    This often means the automation will snapshot the wrong league files.")

            try:
                response = input("\nDo you want to continue anyway? (y/N): ").strip().lower()
                if response != "y":
                    print("Aborting.")
                    return
            except EOFError:
                print("No terminal input available. Aborting due to league name mismatch.")
                return

        with prevent_system_sleep():
            print("Starting FOF8 Automation in 5 seconds... switch to the game window!")
            time.sleep(5)

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
