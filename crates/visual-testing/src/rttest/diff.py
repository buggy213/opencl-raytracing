"""Image comparison."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import OpenEXR


@dataclass
class ComparisonResult:
    mse: float
    max_diff: float


# channel groups we know how to compare, in order of preference
CHANNEL_GROUPS = [
    ["R", "G", "B"],                           # beauty
    ["Normal.X", "Normal.Y", "Normal.Z"],       # normals
    ["U", "V"],                                 # UVs
]


def load_exr(path: Path) -> tuple[np.ndarray, int, int, list[str]]:
    """Load an EXR file and return (pixels, width, height, channel_names).
    
    Returns pixels as a (H, W, C) float32 array where C is the number of channels.
    Automatically detects and loads known channel groups.
    """
    exr_file = OpenEXR.InputFile(str(path))
    header = exr_file.header()
    
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    available_channels = set(header["channels"].keys())
    
    # find a channel group we can load
    channels_to_load = None
    for group in CHANNEL_GROUPS:
        if all(c in available_channels for c in group):
            channels_to_load = group
            break
    
    if channels_to_load is None:
        raise ValueError(
            f"No recognized channel group found. "
            f"Available: {sorted(available_channels)}"
        )
    
    # load the channels
    channel_data = []
    for channel in channels_to_load:
        raw = exr_file.channel(channel)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
        channel_data.append(arr)
    
    pixels = np.stack(channel_data, axis=-1)
    return pixels, width, height, channels_to_load


def compare_images(output_path: Path, reference_path: Path) -> ComparisonResult:
    """Compare output image against reference."""
    output, out_w, out_h, out_channels = load_exr(output_path)
    reference, ref_w, ref_h, ref_channels = load_exr(reference_path)
    
    if (out_w, out_h) != (ref_w, ref_h):
        raise ValueError(
            f"resolution mismatch: output is {out_w}x{out_h}, "
            f"reference is {ref_w}x{ref_h}"
        )
    
    if out_channels != ref_channels:
        raise ValueError(
            f"channel mismatch: output has {out_channels}, "
            f"reference has {ref_channels}"
        )
    
    # compute per-pixel difference
    diff = output - reference
    abs_diff = np.abs(diff)
    
    # compute metrics
    mse = float(np.mean(diff ** 2))
    max_diff = float(np.max(abs_diff))
    
    return ComparisonResult(mse=mse, max_diff=max_diff)
