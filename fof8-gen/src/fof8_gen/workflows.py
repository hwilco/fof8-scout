"""Named season workflow steps used by the automation runner."""

import time
from collections.abc import Callable
from pathlib import Path

from .screen import _PyAutoGuiLike

LEAGUE_NAME_FIELD_LENGTH = 8


class AutomationWorkflows:
    """Named workflow steps for the season automation loop."""

    def __init__(
        self,
        wait_for_image: Callable[..., bool],
        export_data: Callable[..., None],
        pyautogui_module: _PyAutoGuiLike,
    ) -> None:
        self.wait_for_image = wait_for_image
        self.export_data = export_data
        self.pyautogui = pyautogui_module

    def export_draft_snapshot(self, fof8_dir: str, league_name: str, output_dir: Path) -> None:
        print("Exporting draft files (rookies + draft_personal + player_information)...")
        self.export_data(
            fof8_dir,
            league_name,
            output_dir,
            file_filter={
                "rookies.csv": "rookies.csv",
                "draft_personal.csv": "draft_personal.csv",
                "player_information.csv": "player_information_pre_draft.csv",
            },
        )

    def create_one_year_history(self) -> None:
        self.wait_for_image("create_history_btn.png")
        time.sleep(1)
        self.pyautogui.write("1")
        self.wait_for_image("create_history_confirm_btn.png")
        self.wait_for_image("yes_btn.png")
        print("Waiting for history to generate...")
        self.wait_for_image("end_season_indicator_and_btn.png", timeout=600, click=False)

    def export_post_sim_snapshot(self, fof8_dir: str, league_name: str, output_dir: Path) -> None:
        self.export_data(
            fof8_dir,
            league_name,
            output_dir,
            rename_map={"player_information.csv": "player_information_post_sim.csv"},
        )

    def advance_to_staff_draft(self) -> None:
        self.wait_for_image("end_season_indicator_and_btn.png")
        self.wait_for_image("yes_btn.png")
        self.wait_for_image("continue_btn.png")
        self.wait_for_image("retain_staff_btn.png")
        self.wait_for_image("finished_renegotiation_btn.png")

        print("Checking for optional confirmation window...")
        if self.wait_for_image("yes_btn.png", timeout=3, required=False):
            print("Optional confirmation found and clicked.")
        else:
            print("No optional confirmation window appeared.")

        self.wait_for_image("staff_draft_btn.png")

    def complete_staff_draft(self) -> None:
        self.wait_for_image("draft_speed_dropdown.png")
        time.sleep(0.5)
        self.wait_for_image(["fast_option.png", "fast_option_selected.png"])
        self.wait_for_image("have_staff_finish_draft_btn.png")

        print("Waiting for staff draft to finish...")
        self.wait_for_image("draft_completed_indicator.png", timeout=600)
        self.wait_for_image("close_window_btn.png")

    def begin_free_agency(self) -> None:
        self.wait_for_image("begin_free_agency_btn.png")
        self.wait_for_image("yes_btn.png")

    def save_current_universe(
        self, required: bool = True, timeout: float = 1.0, wait_for_save_secs: float = 4.0
    ) -> bool:
        if not self.wait_for_image("save_btn.png", required=required, timeout=timeout):
            return False

        time.sleep(wait_for_save_secs)  # Allow time for the game to finish saving before proceeding
        return True

    def start_new_game(self, league_name: str, begin_with_coach_names_file: bool) -> None:
        if len(league_name) != LEAGUE_NAME_FIELD_LENGTH:
            raise ValueError(
                f"League name must be exactly {LEAGUE_NAME_FIELD_LENGTH} characters long."
            )
        if begin_with_coach_names_file:
            raise NotImplementedError(
                "Automation for 'coach_names_file: true' is not implemented. "
                "Use a template with coach_names_file: false for generated universes."
            )
        self.wait_for_image("new_game_btn.png")
        self.wait_for_image("submit_btn.png")
        self.wait_for_image("league_name_box.png")
        time.sleep(0.5)
        # Clear the league name box
        for _ in range(LEAGUE_NAME_FIELD_LENGTH):
            self.pyautogui.press("backspace")
            self.pyautogui.press("delete")
        self.pyautogui.write(league_name)
        self.wait_for_image("submit_btn.png")

        print("Checking for optional end-current-game confirmation...")
        if self.wait_for_image("yes_btn.png", timeout=3, required=False):
            print("End-current-game confirmation found and clicked.")
        else:
            print("No end-current-game confirmation window appeared.")

    def complete_initial_staff_draft(self) -> None:
        self.wait_for_image("draft_speed_dropdown.png")
        time.sleep(0.5)
        self.wait_for_image("fastest_option.png")
        self.wait_for_image("have_scout_finish_draft_btn.png")

        print("Waiting for initial staff draft to finish...")
        self.wait_for_image("begin_training_camp_btn.png", timeout=600, click=False)
