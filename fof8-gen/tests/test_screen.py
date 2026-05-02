import pytest
from fof8_gen import screen


class _FakePyAutoGUI:
    ImageNotFoundException = RuntimeError

    def __init__(self, locate_results):
        self._locate_results = list(locate_results)
        self.click_calls = []

    def locateCenterOnScreen(self, _image_path, confidence=0.95):
        if self._locate_results:
            return self._locate_results.pop(0)
        return None

    def click(self, location):
        self.click_calls.append(location)


def test_wait_for_image_clicks_on_success(monkeypatch):
    fake_pg = _FakePyAutoGUI([None, (10, 20)])
    monkeypatch.setattr(screen, "_pyautogui_cache", fake_pg)

    class _FakeFiles:
        def joinpath(self, name):
            return f"/fake/{name}"

    monkeypatch.setattr(screen.resources, "files", lambda _pkg: _FakeFiles())
    monkeypatch.setattr(screen.time, "sleep", lambda _n: None)

    times = iter([0.0, 0.1, 0.2])
    monkeypatch.setattr(screen.time, "time", lambda: next(times, 10.0))

    found = screen.wait_for_image("ok_btn.png", timeout=5)

    assert found is True
    assert fake_pg.click_calls == [(10, 20)]


def test_wait_for_image_timeout_required_false(monkeypatch):
    fake_pg = _FakePyAutoGUI([None, None])
    monkeypatch.setattr(screen, "_pyautogui_cache", fake_pg)

    class _FakeFiles:
        def joinpath(self, name):
            return f"/fake/{name}"

    monkeypatch.setattr(screen.resources, "files", lambda _pkg: _FakeFiles())
    monkeypatch.setattr(screen.time, "sleep", lambda _n: None)

    times = iter([0.0, 2.0])
    monkeypatch.setattr(screen.time, "time", lambda: next(times, 2.0))

    found = screen.wait_for_image("never.png", timeout=1, required=False)

    assert found is False


def test_wait_for_image_timeout_required_true(monkeypatch):
    fake_pg = _FakePyAutoGUI([None])
    monkeypatch.setattr(screen, "_pyautogui_cache", fake_pg)

    class _FakeFiles:
        def joinpath(self, name):
            return f"/fake/{name}"

    monkeypatch.setattr(screen.resources, "files", lambda _pkg: _FakeFiles())
    monkeypatch.setattr(screen.time, "sleep", lambda _n: None)

    times = iter([0.0, 2.0])
    monkeypatch.setattr(screen.time, "time", lambda: next(times, 2.0))

    with pytest.raises(RuntimeError, match="Timed out"):
        screen.wait_for_image("never.png", timeout=1, required=True)
