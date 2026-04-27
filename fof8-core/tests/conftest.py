import pytest
from fof8_core.loader import FOF8Loader


@pytest.fixture
def temp_league_dir(tmp_path):
    """Creates a temporary league structure."""
    league_name = "DRAFT003"
    league_dir = tmp_path / league_name
    league_dir.mkdir(parents=True)
    (league_dir / "2020").mkdir()
    return tmp_path


@pytest.fixture
def mock_loader(temp_league_dir):
    """Provides a loader instance pointing to the temporary directory."""
    return FOF8Loader(temp_league_dir, league_name="DRAFT003")
