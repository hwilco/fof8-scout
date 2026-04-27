import shutil
from pathlib import Path


def convert_in_place(path: Path):
    """
    Reads file as cp1252, writes back as utf-8, and preserves metadata.
    Skips if already valid UTF-8.
    """
    try:
        # Check if already valid UTF-8
        try:
            path.read_text(encoding="utf-8")
            return  # Skip already converted files
        except UnicodeDecodeError:
            pass

        # Read the content
        content = path.read_text(encoding="cp1252")

        # Create a temporary file to preserve metadata easily
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(content, encoding="utf-8")

        # Copy metadata from original to temp
        shutil.copystat(path, temp_path)

        # Replace original with temp
        temp_path.replace(path)
        print(f"Successfully converted: {path}")
    except Exception as e:
        print(f"Failed to convert {path}: {e}")


def main():
    # Target directory relative to this script or workspace root
    # Based on the environment, we'll look for data/raw relative to the workspace root
    base_dir = Path("data/raw")

    if not base_dir.exists():
        # Try relative to the script location if run from data-generation
        base_dir = Path(__file__).parent / "data" / "raw"

    if not base_dir.exists():
        print(f"Error: Could not find directory {base_dir}")
        return

    print(f"Scanning for CSV files in {base_dir.resolve()}...")

    csv_files = list(base_dir.rglob("*.csv"))
    total = len(csv_files)
    print(f"Found {total} CSV files.")

    for i, csv_file in enumerate(csv_files, 1):
        convert_in_place(csv_file)
        if i % 100 == 0 or i == total:
            print(f"--- Progress: {i}/{total} files processed ({(i / total) * 100:.1f}%) ---")

    print("All conversions complete.")


if __name__ == "__main__":
    main()
