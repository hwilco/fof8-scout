# Multi-Universe Generation Plan

## Goal

Add a generation mode that can create multiple new FOF8 universes from one metadata template, run each universe through the initial staff draft setup, then hand off to the existing season snapshot loop.

Example target workflow:

```bash
gather-data --metadata templates/metadata.yaml --generate-universes DRAFT009:DRAFT014 --iterations 30
```

This should create six new universe output directories, `DRAFT009` through `DRAFT014`, each with a generated `metadata.yaml` whose `new_game_options.league_name` matches the universe name.

## Current State

The current `fof8-gen` flow assumes the league already exists.

- `fof8_gen.automation.main()` accepts one metadata path.
- `metadata.load_metadata()` reads `new_game_options.league_name`.
- `AutomationRunner.run()` validates that `league_name == output_dir.name`.
- `AutomationRunner.run()` loops existing-season work:
  - export draft snapshot
  - create one-year history
  - export post-sim snapshot
  - advance to staff draft
  - complete staff draft
  - begin free agency

The new feature should extend this flow without changing the existing single-league behavior.

## New Screenshots

Add these image assets under `fof8-gen/src/fof8_gen/resources/images/`:

- `save_btn.PNG`
- `new_game_btn.PNG`
- `submit_btn.PNG`
- `league_name_box.PNG`
- `begin_draft_btn.PNG`
- `begin_training_camp_btn.PNG`

Existing reusable assets:

- `yes_btn.PNG`
- `draft_speed_dropdown.PNG`
- `fast_option.PNG`
- `fast_option_selected.PNG`
- `have_staff_finish_draft_btn.PNG`

Confirm whether the current `fast_option` image truly represents the fastest draft speed. If it only means "Fast", capture a dedicated `fastest_option.PNG` and use that in the new-universe setup.

## CLI Design

Add a separate mode to the existing `gather-data` command:

```bash
gather-data \
  --metadata /path/to/base_metadata.yaml \
  --generate-universes DRAFT009:DRAFT014 \
  --output-root data/raw \
  --iterations 30
```

Proposed args:

- `--generate-universes START:END`: inclusive universe range using a shared prefix and zero-padded numeric suffix.
- `--output-root DIR`: parent directory where one directory per universe will be created. Default to the metadata file parent if omitted.
- `--overwrite-metadata`: optional, default false. If false, abort when a generated universe metadata file already exists.
- Keep `--snapshot-only` incompatible with `--generate-universes`.

Implementation detail:

- Parse `DRAFT009:DRAFT014` into `["DRAFT009", ..., "DRAFT014"]`.
- Require matching prefix and numeric width.
- Reject descending ranges.

## Metadata Handling

Add helper functions in `fof8_gen.metadata`:

- `clone_metadata_for_league(base_metadata_path, league_name, output_dir, overwrite=False) -> Path`
- `write_metadata(path, data) -> None`
- `parse_universe_range(value: str) -> list[str]`

For each generated universe:

1. Load the base metadata YAML.
2. Set `new_game_options.league_name` to the generated name.
3. Create `<output-root>/<league-name>/`.
4. Write `<output-root>/<league-name>/metadata.yaml`.

Do not mutate the source template file.

## Workflow Changes

Add new methods to `AutomationWorkflows`.

### Save Current Universe

`save_current_universe()`

Steps:

1. Click `save_btn.PNG`.
2. Wait for confirmation if the game presents one.
3. Click `yes_btn.PNG` only when applicable.

The confirmation should be optional because save behavior may differ depending on current screen/state.

### Start New Game

`start_new_game(league_name: str, begin_with_coach_names_file: bool)`

Steps:

1. Click `new_game_btn.PNG`.
2. Click `submit_btn.PNG` to submit the reused/default new-game settings.
3. Click `league_name_box.PNG`.
4. Select or clear existing text.
5. Type `league_name`.
6. Click `submit_btn.PNG`.
7. Click `yes_btn.PNG` to end the current game if that confirmation appears.
8. Wait for universe generation to complete.

Use `begin_with_coach_names_file = metadata["new_game_options"]["coach_names_file"]`.

If `coach_names_file` is false, continue into the initial staff draft setup described below. If true, document the expected manual or alternate path before implementing that branch.

### Complete Initial Staff Draft

`complete_initial_staff_draft()`

Steps:

1. Click `draft_speed_dropdown.PNG`.
2. Click fastest speed option.
3. Click `begin_draft_btn.PNG`.
4. Click `have_staff_finish_draft_btn.PNG`.
5. Wait for `begin_training_camp_btn.PNG`.

This is distinct from the existing `complete_staff_draft()`, which currently expects the staff draft screen reached from the season loop and waits for `draft_completed_indicator.PNG` plus `close_window_btn.PNG`.

### Run Generated Universe

Add `AutomationRunner.generate_universes(...)`.

For each universe:

1. Create/write universe metadata.
2. Save the current universe if there is an active game.
3. Start a new game using the generated league name.
4. If `coach_names_file` is false, complete the initial staff draft.
5. Once `begin_training_camp_btn.PNG` appears, call the existing `run(...)` for that universe.
6. After the final iteration, save the universe before moving to the next generated universe.

The first generated universe may or may not have an active game open. Treat the "end current game" confirmation as optional.

## Control Flow

Target structure:

```text
automation.main()
  if --snapshot-only:
    snapshot_only(...)
  elif --generate-universes:
    runner.generate_universes(...)
  else:
    runner.run(...)
```

Keep `AutomationRunner.run()` as the existing single-universe season loop. The multi-universe method should call it instead of duplicating the season loop.

## Testing Plan

Unit tests:

- `parse_universe_range("DRAFT009:DRAFT014")` returns six names.
- Range parser rejects mismatched prefixes, mismatched padding, and descending ranges.
- Metadata cloning writes one metadata file per league and updates only `new_game_options.league_name`.
- CLI dispatch calls `generate_universes()` when `--generate-universes` is present.
- `generate_universes()` invokes workflow methods in order using fake `wait_for_image` and fake `pyautogui`.

Workflow tests should use fakes and should not require pyautogui or real screenshots.

Manual smoke test:

1. Capture the new screenshots at the same resolution/scaling used by the current automation.
2. Run a one-universe range, for example `DRAFT009:DRAFT009`, with `--iterations 1`.
3. Verify the generated directory, metadata, draft snapshot, post-sim snapshot, and final save.
4. Run a two-universe range only after the one-universe smoke test passes.

## Open Questions

- Is the "new game" flow guaranteed to preserve all settings from the previous game/template, or do some metadata options need to be clicked explicitly?
- Does `coach_names_file: true` require a different first-screen path before the staff draft?
- Does the save action always use the same button from every screen where the runner may stop?
- Should generated universes start from an already-open template game, or from the main menu?
- Should `--iterations 30` mean 30 draft classes exported per universe after initial universe creation, or 30 total years including the initial generated draft?
