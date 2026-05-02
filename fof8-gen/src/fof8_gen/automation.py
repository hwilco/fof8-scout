"""CLI entrypoint for FOF8 data automation.

This module intentionally stays thin so the console script target remains stable:
`gather-data = fof8_gen.automation:main`.
"""

import argparse
import os
from pathlib import Path

from .metadata import load_metadata


def main() -> None:
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

    from .automation_runner import AutomationRunner

    runner = AutomationRunner()
    if args.snapshot_only:
        runner.snapshot_only(fof8_dir=args.fof8_dir, league_name=league_name, output_dir=output_dir)
        return

    runner.run(
        fof8_dir=args.fof8_dir,
        league_name=league_name,
        output_dir=output_dir,
        num_iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
