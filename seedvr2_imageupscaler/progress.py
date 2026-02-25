"""Progress tracking module for ComfyUI upscaling operations."""

from __future__ import annotations

from typing import Any

try:
    from comfy.utils import ProgressBar as ComfyProgressBar
except ImportError:
    ComfyProgressBar = None

try:
    import comfy.utils as comfy_utils
except ImportError:
    comfy_utils = None


class Progress:
    """Comfy-safe progress reporter with optional verbose debug output."""

    def __init__(self, total_steps: int, sub_steps_per_tile: int = 3, enable_debug: bool = False) -> None:
        self.total_steps = max(int(total_steps), 0)
        self.sub_steps_per_tile = max(int(sub_steps_per_tile), 1)
        self.total_sub_steps = max(self.total_steps * self.sub_steps_per_tile, 1)
        self.current_step = 0
        self.current_sub_step = 0
        self.enable_debug = enable_debug
        self._last_percent = -1

        self._progress_bar: Any | None = None
        if ComfyProgressBar is not None:
            try:
                self._progress_bar = ComfyProgressBar(100)
            except Exception:
                self._progress_bar = None

        print(f"[SeedVR2 Tiling] Starting upscale process ({self.total_steps} tiles)", flush=True)
        self._emit_progress(force=True)

    def _emit_progress(self, force: bool = False) -> None:
        """Emit progress using Comfy's native progress channel."""
        percent = int((self.current_sub_step / self.total_sub_steps) * 100)
        percent = max(0, min(100, percent))

        if not force and percent == self._last_percent:
            return

        self._last_percent = percent

        if self._progress_bar is not None:
            try:
                self._progress_bar.update_absolute(percent, 100)
                return
            except Exception:
                self._progress_bar = None

        if comfy_utils is not None and hasattr(comfy_utils, "report_progress"):
            try:
                comfy_utils.report_progress(self.current_sub_step, self.total_sub_steps)
            except Exception:
                pass

    def update(self, sub_progress_step: int | None = None) -> None:
        if sub_progress_step is not None:
            step = max(0, min(int(sub_progress_step), self.sub_steps_per_tile))
            self.current_sub_step = min(
                self.total_sub_steps,
                self.current_step * self.sub_steps_per_tile + step,
            )
        else:
            self.current_step = min(self.total_steps, self.current_step + 1)
            self.current_sub_step = min(
                self.total_sub_steps,
                self.current_step * self.sub_steps_per_tile,
            )

        if self.enable_debug:
            print(
                (
                    f"[SeedVR2 Tiling][debug] Progress "
                    f"tile={self.current_step}/{self.total_steps}, "
                    f"step={self.current_sub_step}/{self.total_sub_steps}"
                ),
                flush=True,
            )

        self._emit_progress()

    def update_sub_progress(self, step_name: str, step_number: int) -> None:
        step = max(0, min(int(step_number), self.sub_steps_per_tile))
        self.current_sub_step = min(
            self.total_sub_steps,
            self.current_step * self.sub_steps_per_tile + step,
        )

        if self.enable_debug:
            print(
                (
                    f"[SeedVR2 Tiling][debug] {step_name} "
                    f"(step {step}/{self.sub_steps_per_tile}, tile {self.current_step}/{self.total_steps})"
                ),
                flush=True,
            )

        self._emit_progress()

    def initialize_websocket_progress(self) -> None:
        # Kept for compatibility with existing call sites.
        self.current_step = 0
        self.current_sub_step = 0
        self._emit_progress(force=True)

    def finalize_websocket_progress(self) -> None:
        self.current_step = self.total_steps
        self.current_sub_step = self.total_sub_steps
        self._emit_progress(force=True)
        print(
            f"[SeedVR2 Tiling] Upscale completed successfully ({self.total_steps} tiles)",
            flush=True,
        )
