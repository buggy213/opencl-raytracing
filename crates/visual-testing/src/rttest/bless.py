"""Interactive blessing workflow with matplotlib visualization."""

import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib

# try interactive backends in order of preference
_BACKENDS = ["TkAgg", "Qt5Agg", "GTK3Agg", "WXAgg", "macosx"]
_plt = None
for _backend in _BACKENDS:
    try:
        matplotlib.use(_backend)
        import matplotlib.pyplot as _plt
        import matplotlib.colors as _mcolors
        # test that the backend actually works by accessing the figure manager
        _plt.figure()
        _plt.close()
        break
    except Exception:
        _plt = None
        continue

if _plt is None:
    print("error: no interactive matplotlib backend available", file=sys.stderr)
    print("install one of: python3-tk (apt install python3-tk)", file=sys.stderr)
    sys.exit(2)

plt = _plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button

from .runner import TestResult
from .diff import load_exr


def estimate_exposure(img: np.ndarray) -> float:
    """Estimate a good exposure divisor based on image statistics.
    
    Returns a value to divide by so that bright regions map to ~0.7.
    """
    if img.shape[-1] >= 3:
        luminance = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    else:
        luminance = np.mean(img, axis=-1)
    
    # use 90th percentile so bright regions map to ~0.7
    p90 = np.percentile(luminance[luminance > 0], 90) if np.any(luminance > 0) else 1.0
    return p90 / 0.7 if p90 > 0 else 1.0


def tonemap_with_exposure(img: np.ndarray, exposure: float) -> np.ndarray:
    """Simple exposure adjustment by division."""
    return img / exposure


def apply_gamma(img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Apply gamma correction (linear -> sRGB-ish)."""
    return np.power(np.clip(img, 0, 1), 1.0 / gamma)


def prepare_for_display(img: np.ndarray, exposure: float, is_beauty: bool) -> np.ndarray:
    """Prepare image for display with exposure control."""
    if img.shape[-1] == 2:
        # UV: display as RG with B=0
        zeros = np.zeros((*img.shape[:-1], 1), dtype=img.dtype)
        img = np.concatenate([img, zeros], axis=-1)
    
    result = np.clip(tonemap_with_exposure(img, exposure), 0, 1)
    
    if is_beauty:
        result = apply_gamma(result)
    
    return result


def get_max_value(img: np.ndarray) -> float:
    """Get the maximum value in the image for slider range."""
    return float(np.max(np.abs(img)))


class BlessingViewer:
    """Interactive matplotlib viewer for blessing workflow."""
    
    def __init__(self, output: np.ndarray, reference: np.ndarray | None, scene_name: str, channels: list[str]):
        self.output = output
        self.reference = reference
        self.scene_name = scene_name
        self.channels = channels
        self.is_beauty = channels == ["R", "G", "B"]
        self.result = None  # 'bless', 'skip', or 'quit'
        
        # auto-detect exposure from output
        self.exposure = estimate_exposure(output)
        
        # compute max value for slider range
        max_val = get_max_value(output)
        if reference is not None:
            max_val = max(max_val, get_max_value(reference))
        self.max_exposure = max(max_val * 2, 10.0)  # allow seeing beyond max
        
        self._setup_figure()
    
    def _setup_figure(self):
        has_reference = self.reference is not None
        n_cols = 3 if has_reference else 1
        
        self.fig, self.axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        if n_cols == 1:
            self.axes = [self.axes]
        
        self.fig.canvas.manager.set_window_title(f"Bless: {self.scene_name}")
        
        # leave room for slider and buttons at bottom
        self.fig.subplots_adjust(bottom=0.25)
        
        # initial display
        self._update_display()
        
        # exposure slider with dynamic range
        ax_slider = self.fig.add_axes([0.2, 0.12, 0.6, 0.03])
        self.slider = Slider(
            ax_slider, "Exposure", 
            0.01, self.max_exposure, 
            valinit=self.exposure,
        )
        self.slider.on_changed(self._on_exposure_change)
        
        # buttons
        ax_bless = self.fig.add_axes([0.2, 0.03, 0.2, 0.05])
        ax_skip = self.fig.add_axes([0.45, 0.03, 0.15, 0.05])
        ax_quit = self.fig.add_axes([0.65, 0.03, 0.15, 0.05])
        
        self.btn_bless = Button(ax_bless, "Bless (y)")
        self.btn_skip = Button(ax_skip, "Skip (n)")
        self.btn_quit = Button(ax_quit, "Quit (q)")
        
        self.btn_bless.on_clicked(lambda _: self._on_action("bless"))
        self.btn_skip.on_clicked(lambda _: self._on_action("skip"))
        self.btn_quit.on_clicked(lambda _: self._on_action("quit"))
        
        # keyboard shortcuts
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
    
    def _update_display(self):
        out_display = prepare_for_display(self.output, self.exposure, self.is_beauty)
        
        if self.reference is not None:
            ref_display = prepare_for_display(self.reference, self.exposure, self.is_beauty)
            diff = np.abs(self.output - self.reference)
            diff_magnitude = np.sum(diff, axis=-1)
            
            self.axes[0].clear()
            self.axes[0].imshow(ref_display)
            self.axes[0].set_title("Reference")
            self.axes[0].axis("off")
            
            self.axes[1].clear()
            self.axes[1].imshow(out_display)
            self.axes[1].set_title("Output (new)")
            self.axes[1].axis("off")
            
            # diff plot doesn't need to be redrawn on exposure change
            if not hasattr(self, '_diff_drawn'):
                if diff_magnitude.max() > 0:
                    # use log scale for better visibility of small differences
                    vmin = diff_magnitude[diff_magnitude > 0].min()
                    vmax = diff_magnitude.max()
                    norm = mcolors.LogNorm(vmin=max(vmin, 1e-8), vmax=vmax)
                    im = self.axes[2].imshow(diff_magnitude, cmap="inferno", norm=norm)
                    self.fig.colorbar(im, ax=self.axes[2], fraction=0.046, pad=0.04)
                else:
                    self.axes[2].imshow(np.zeros_like(diff_magnitude), cmap="inferno")
                self.axes[2].set_title("Difference (log)")
                self.axes[2].axis("off")
                self._diff_drawn = True
        else:
            self.axes[0].clear()
            self.axes[0].imshow(out_display)
            self.axes[0].set_title(f"Output: {self.scene_name}")
            self.axes[0].axis("off")
        
        self.fig.canvas.draw_idle()
    
    def _on_exposure_change(self, val):
        self.exposure = val
        self._update_display()
    
    def _on_action(self, action: str):
        self.result = action
        plt.close(self.fig)
    
    def _on_key(self, event):
        if event.key in ("y", "Y", "enter"):
            self._on_action("bless")
        elif event.key in ("n", "N"):
            self._on_action("skip")
        elif event.key in ("q", "Q", "escape"):
            self._on_action("quit")
    
    def show(self) -> str:
        """Show the viewer and return the user's choice."""
        plt.show()
        return self.result or "skip"


def interactive_bless(
    results: list[TestResult],
    output_dir: Path,
    reference_dir: Path,
):
    """Interactively bless test results with matplotlib viewer."""
    print()
    print("=" * 60)
    print("INTERACTIVE BLESSING")
    print("=" * 60)
    print()
    print(f"Found {len(results)} result(s) to review.")
    print("Use the matplotlib window to inspect and bless each result.")
    print()

    blessed_count = 0
    for result in results:
        if not result.output_path or not result.output_path.exists():
            print(f"  ! {result.scene}: no output to bless")
            continue
        
        # load output
        try:
            output, _, _, channels = load_exr(result.output_path)
        except Exception as e:
            print(f"  ! {result.scene}: failed to load output: {e}")
            continue
        
        # load reference if it exists
        reference = None
        if result.reference_path and result.reference_path.exists():
            try:
                reference, _, _, _ = load_exr(result.reference_path)
            except Exception:
                pass  # no reference is fine
        
        # show interactive viewer
        viewer = BlessingViewer(output, reference, result.scene, channels)
        action = viewer.show()
        
        if action == "bless":
            bless_result(result, reference_dir)
            blessed_count += 1
            print(f"  ✓ Blessed {result.scene}")
        elif action == "skip":
            print(f"  ✗ Skipped {result.scene}")
        elif action == "quit":
            print()
            print(f"Blessed {blessed_count} result(s), quit early.")
            return
    
    print()
    print(f"Blessed {blessed_count} of {len(results)} result(s).")


def bless_result(result: TestResult, reference_dir: Path):
    """Copy the output to the reference directory."""
    if result.output_path and result.output_path.exists():
        reference_dir.mkdir(parents=True, exist_ok=True)
        dest = reference_dir / result.output_path.name
        shutil.copy(result.output_path, dest)
