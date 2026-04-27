import argparse
import ctypes
import os
import time
from importlib import resources
from pathlib import Path

import pyautogui

from .snapshot import create_league_snapshot

# Windows constants for preventing sleep/screen timeout
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

pyautogui.FAILSAFE = True  # (shoving mouse to corner of screen aborts the program)


def prevent_sleep():
    """Prevents the system from sleeping or turning off the display. Returns the previous state."""
    print("Preventing screen timeout...")
    # SetThreadExecutionState returns the previous state
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    return ctypes.windll.kernel32.SetThreadExecutionState(flags)


def allow_sleep(previous_state):
    """Restores the system to its previous execution state."""
    print("Restoring screen timeout settings...")
    if previous_state is not None:
        ctypes.windll.kernel32.SetThreadExecutionState(previous_state)


def wait_for_image(image_names, timeout=60, confidence=0.95, required=True, click=True):
    """
    Waits for provided images to appear on screen and optionally clicks the first one found.

    :param image_names: A single filename string or a list of filename strings.
    :param timeout: How long to wait before giving up (seconds)
    :param confidence: How closely the image needs to match the screen (0.0 to 1.0)
    :param required: If True, raises a RuntimeError on timeout. If False, returns False.
    :param click: If True, clicks the center of the found image.
    :return: True if found (and clicked if requested), False if it timed out.
    :raises RuntimeError: If `required` is True and the image is not found within the timeout.
    """
    if isinstance(image_names, str):
        image_names = [image_names]

    print(f"Waiting for {', '.join(image_names)}...")

    # Use importlib.resources to find the images relative to the package
    images_resource = resources.files("fof8_gen.resources.images")

    start_time = time.time()
    while time.time() - start_time < timeout:
        for name in image_names:
            image_path = str(images_resource.joinpath(name))

            try:
                # locateCenterOnScreen uses OpenCV under the hood for the confidence parameter
                location = pyautogui.locateCenterOnScreen(image_path, confidence=confidence)
                if location:
                    if click:
                        print(f"Found {name}! Clicking...")
                        pyautogui.click(location)
                    else:
                        print(f"Found {name}!")
                    return True
            except (pyautogui.ImageNotFoundException, Exception):
                pass  # Image not found yet or error, try the next one or loop

        time.sleep(1)  # Wait 1 second before checking again

    error_msg = f"Error: Timed out waiting for {', '.join(image_names)}"
    print(error_msg)
    if required:
        raise RuntimeError(error_msg)
    return False


def find_project_root():
    """Finds the project root by searching upwards for pyproject.toml."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path(__file__).resolve().parents[3]  # Fallback


# Resolve project root
PROJECT_ROOT = find_project_root()


def load_metadata(path: Path):
    """Loads and validates the metadata YAML file."""
    try:
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise RuntimeError(f"Metadata YAML is malformed: {e}")
    except OSError as e:
        raise RuntimeError(f"Could not read metadata file: {e}")

    if not data:
        raise ValueError("Metadata file is empty")

    league_name = data.get("new_game_options", {}).get("league_name")
    if not league_name:
        raise ValueError("Metadata file is missing 'new_game_options: league_name'")

    return data, league_name


def export_data(fof8_dir, league_name, output_dir, file_filter=None, rename_map=None):
    """Exports data and takes a snapshot for the current year.

    :param fof8_dir: Path to the FOF8 data directory.
    :param league_name: Name of the league folder.
    :param output_dir: Directory where snapshots will be saved.
    :param file_filter: Optional list or dict of specific filenames to snapshot
                        (e.g. ["rookies.csv", "draft_personal.csv"]).
                        If None, snapshots all relevant files.
    :param rename_map: Optional dict for renaming files during a full snapshot.
    """
    # Click export data
    wait_for_image("export_data_btn.png")

    # Wait for it to complete and hit ok
    wait_for_image("ok_btn.png", timeout=120)

    # Click export scouting data
    wait_for_image("export_scouting_data_btn.png")

    # Click yes in the confirmation window
    wait_for_image("yes_btn.png")

    # Wait for it to complete and hit ok
    wait_for_image("ok_btn.png", timeout=120)

    # Run the snapshot logic
    print(f"Creating snapshot for {league_name}...")
    create_league_snapshot(
        fof8_dir,
        league_name,
        str(output_dir),
        file_filter=file_filter,
        rename_map=rename_map,
    )


def generate_data(fof8_dir, league_name, output_dir, num_iterations):
    """Generates a new league file and runs all the automation steps to generate data in FOF8.

    :param fof8_dir: Path to the FOF8 data directory.
    :param league_name: Name of the league folder.
    :param output_dir: Directory where snapshots will be saved.
    :param num_iterations: Number of seasons to simulate.
    """
    # Validate that metadata.yaml exists in the output directory
    metadata_path = output_dir / "metadata.yaml"
    if not metadata_path.exists():
        print(f"\nERROR: Metadata file not found at {metadata_path}")
        print("Please create this file manually to describe your generation settings.")
        print("You can use 'templates/metadata.example.yaml' as a starting point.")
        return

    # Verify league name matches output directory name to prevent copying the wrong data
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

    previous_state = prevent_sleep()
    try:
        print(f"Project Root: {PROJECT_ROOT}")

        print("Starting FOF8 Automation in 5 seconds... switch to the game window!")
        time.sleep(5)

        total_start_time = time.time()

        for iteration in range(1, num_iterations + 1):
            iter_start_time = time.time()
            print(f"--- Starting Iteration {iteration} of {num_iterations} ---")

            # 0. Grab draft files from the most recent off-season draft.
            #    These are available now (after the previous iteration's draft
            #    or the initial begin-with-draft setup) and will be stored in
            #    the snapshot directory for the current year.
            print("Exporting draft files (rookies + draft_personal + player_information)...")
            export_data(
                fof8_dir,
                league_name,
                output_dir,
                file_filter={
                    "rookies.csv": "rookies.csv",
                    "draft_personal.csv": "draft_personal.csv",
                    "player_information.csv": "player_information_pre_draft.csv",
                },
            )

            # 1. Click "create history"
            wait_for_image("create_history_btn.png")

            # 2. Enter 1 in the textbox that appears
            time.sleep(1)  # Give the box a moment to appear and gain focus
            pyautogui.write("1")

            # 3. Click "create history" in that new box
            wait_for_image("create_history_confirm_btn.png")

            # 4. Click yes in the confirmation window
            wait_for_image("yes_btn.png")

            # 5. Wait for the history to be created (End Season indicator)
            print("Waiting for history to generate...")
            wait_for_image("end_season_indicator_and_btn.png", timeout=600, click=False)

            # 6. Export data
            export_data(
                fof8_dir,
                league_name,
                output_dir,
                rename_map={"player_information.csv": "player_information_post_sim.csv"},
            )

            # 7. Click the "End Season" button
            wait_for_image("end_season_indicator_and_btn.png")

            # 8. Click yes in the confirmation window
            wait_for_image("yes_btn.png")

            # 9. Click continue in the overview window
            wait_for_image("continue_btn.png")

            # 10. Click the "Retain Staff" button
            wait_for_image("retain_staff_btn.png")

            # 11. Click Finished with Renegotiation
            wait_for_image("finished_renegotiation_btn.png")

            # 12. If a window pop up asking for confirmation, click yes (only sometimes)
            print("Checking for optional confirmation window...")
            if wait_for_image("yes_btn.png", timeout=3, required=False):
                print("Optional confirmation found and clicked.")
            else:
                print("No optional confirmation window appeared.")

            # 13. Click the "Staff Draft" button
            wait_for_image("staff_draft_btn.png")

            # 14. Click the Draft Speed dropdown and select "fast"
            wait_for_image("draft_speed_dropdown.png")
            time.sleep(0.5)  # wait half a second for the dropdown to open
            wait_for_image(["fast_option.png", "fast_option_selected.png"])

            # 15. Click the "Have Staff Finish Draft" button
            wait_for_image("have_staff_finish_draft_btn.png")

            # 16. Wait for the staff draft to finish, then hit "close window"
            print("Waiting for staff draft to finish...")
            wait_for_image("draft_completed_indicator.png", timeout=600)
            wait_for_image("close_window_btn.png")

            # 17. Click the Begin Free Agency button
            wait_for_image("begin_free_agency_btn.png")

            # 18. Click yes in the confirmation window
            wait_for_image("yes_btn.png")

            iter_duration = time.time() - iter_start_time
            total_elapsed = time.time() - total_start_time
            msg = (
                f"--- Finished Iteration {iteration} in {iter_duration:.2f}s "
                f"(Total elapsed: {total_elapsed:.2f}s) ---"
            )
            print(msg)

        print(f"Automation complete! Total time: {time.time() - total_start_time:.2f}s")
    finally:
        allow_sleep(previous_state)


def main():
    default_fof8_dir = (
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Solecismic Software"
        / "Front Office Football Eight"
    )

    parser = argparse.ArgumentParser(description="Automate data generation for FOF8.")
    parser.add_argument(
        "-m",
        "--metadata",
        type=str,
        metavar="PATH",
        required=True,
        help="path to the metadata.yaml file (auto-detects league and output directory)",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        metavar="N",
        default=100,
        help="number of seasons to simulate (default: 100)",
    )
    parser.add_argument(
        "-f",
        "--fof8-dir",
        type=str,
        metavar="DIR",
        default=default_fof8_dir,
        help=f"path to the FOF8 data directory (default: '{default_fof8_dir}')",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        metavar="DIR",
        help="override the default snapshot output directory (default: <metadata_directory>)",
    )
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        help="only run the export and snapshot logic, then exit (useful for manual recovery)",
    )

    args = parser.parse_args()

    if not args.metadata:
        print("ERROR: Metadata file path is required. Use --metadata <path>")
        return

    metadata_path = Path(args.metadata).resolve()
    if not metadata_path.exists():
        print(f"ERROR: Specified metadata file does not exist: {metadata_path}")
        return

    try:
        _, league_name = load_metadata(metadata_path)
    except (RuntimeError, ValueError) as e:
        print(f"ERROR: {e}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else metadata_path.parent

    if args.snapshot_only:
        previous_state = prevent_sleep()
        try:
            print("Starting manual export in 5 seconds... switch to the game window!")
            time.sleep(5)
            export_data(
                fof8_dir=args.fof8_dir,
                league_name=league_name,
                output_dir=output_dir,
            )
            print("Manual export and snapshot complete.")
        finally:
            allow_sleep(previous_state)
        return

    generate_data(
        fof8_dir=args.fof8_dir,
        league_name=league_name,
        output_dir=output_dir,
        num_iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
