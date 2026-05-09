"""Screen interaction helpers for image-based FOF8 GUI automation."""

import ctypes
import time
from contextlib import contextmanager
from importlib import resources
from typing import Any, Iterator, Protocol, Sequence, cast

# Windows constants for preventing sleep/screen timeout
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

_pyautogui_cache = None


class _PyAutoGuiLike(Protocol):
    ImageNotFoundException: type[Exception]

    def locateCenterOnScreen(self, image_path: str, confidence: float = 0.95) -> object: ...
    def click(self, location: object) -> None: ...
    def hotkey(self, *keys: str) -> None: ...
    def press(self, key: str) -> None: ...
    def write(self, text: str) -> None: ...


def _get_pyautogui() -> _PyAutoGuiLike:
    global _pyautogui_cache
    if _pyautogui_cache is None:
        import pyautogui

        pyautogui.FAILSAFE = True  # shoving mouse to corner of screen aborts the program
        _pyautogui_cache = pyautogui
    return cast(_PyAutoGuiLike, _pyautogui_cache)


def prevent_sleep() -> int:
    """Prevent the system from sleeping or turning off the display."""
    print("Preventing screen timeout...")
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    windll = cast(Any, getattr(ctypes, "windll"))
    return cast(int, windll.kernel32.SetThreadExecutionState(flags))


def allow_sleep(previous_state: int | None) -> None:
    """Restore the system to its previous execution state."""
    print("Restoring screen timeout settings...")
    if previous_state is not None:
        windll = cast(Any, getattr(ctypes, "windll"))
        windll.kernel32.SetThreadExecutionState(previous_state)


@contextmanager
def prevent_system_sleep() -> Iterator[None]:
    """Context manager that prevents sleep and reliably restores prior state."""
    previous_state = prevent_sleep()
    try:
        yield
    finally:
        allow_sleep(previous_state)


def wait_for_image(
    image_names: str | Sequence[str],
    timeout: int = 60,
    confidence: float = 0.95,
    required: bool = True,
    click: bool = True,
) -> bool:
    """Wait for one or more images to appear and optionally click the first match."""
    if isinstance(image_names, str):
        image_names = [image_names]

    print(f"Waiting for {', '.join(image_names)}...")

    images_resource = resources.files("fof8_gen.resources.images")
    pyautogui = _get_pyautogui()

    start_time = time.time()
    while time.time() - start_time < timeout:
        for name in image_names:
            image_path = str(images_resource.joinpath(name))
            try:
                location = pyautogui.locateCenterOnScreen(image_path, confidence=confidence)
                if location:
                    if click:
                        print(f"Found {name}! Clicking...")
                        pyautogui.click(location)
                    else:
                        print(f"Found {name}!")
                    return True
            except (pyautogui.ImageNotFoundException, Exception):
                pass

        time.sleep(1)

    error_msg = f"Error: Timed out waiting for {', '.join(image_names)}"
    print(error_msg)
    if required:
        raise RuntimeError(error_msg)
    return False
