"""Named season workflow steps used by the automation runner."""

import time


class AutomationWorkflows:
    """Named workflow steps for the season automation loop."""

    def __init__(self, wait_for_image, export_data, pyautogui_module):
        self.wait_for_image = wait_for_image
        self.export_data = export_data
        self.pyautogui = pyautogui_module

    def export_draft_snapshot(self, fof8_dir, league_name, output_dir):
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

    def create_one_year_history(self):
        self.wait_for_image("create_history_btn.png")
        time.sleep(1)
        self.pyautogui.write("1")
        self.wait_for_image("create_history_confirm_btn.png")
        self.wait_for_image("yes_btn.png")
        print("Waiting for history to generate...")
        self.wait_for_image("end_season_indicator_and_btn.png", timeout=600, click=False)

    def export_post_sim_snapshot(self, fof8_dir, league_name, output_dir):
        self.export_data(
            fof8_dir,
            league_name,
            output_dir,
            rename_map={"player_information.csv": "player_information_post_sim.csv"},
        )

    def advance_to_staff_draft(self):
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

    def complete_staff_draft(self):
        self.wait_for_image("draft_speed_dropdown.png")
        time.sleep(0.5)
        self.wait_for_image(["fast_option.png", "fast_option_selected.png"])
        self.wait_for_image("have_staff_finish_draft_btn.png")

        print("Waiting for staff draft to finish...")
        self.wait_for_image("draft_completed_indicator.png", timeout=600)
        self.wait_for_image("close_window_btn.png")

    def begin_free_agency(self):
        self.wait_for_image("begin_free_agency_btn.png")
        self.wait_for_image("yes_btn.png")
