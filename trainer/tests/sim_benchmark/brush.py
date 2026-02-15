"""
Annotation painting utilities for simulated corrective annotation.

Provides low-level tools to paint on RGBA annotation arrays matching
RootPainter's format: channel 0 = foreground, channel 1 = background,
channels 2-3 unused.
"""
import numpy as np


def disk(radius):
    """Return a boolean 2D array (disk structuring element) of diameter 2*radius+1."""
    size = 2 * radius + 1
    y, x = np.ogrid[:size, :size]
    return (x - radius) ** 2 + (y - radius) ** 2 <= radius ** 2


def paint(annot, center, radius, channel):
    """Paint a filled circle on the given channel of an annotation array.

    Args:
        annot: (H, W, 4) uint8 array (modified in-place)
        center: (row, col) tuple
        radius: brush radius in pixels
        channel: 0 for foreground, 1 for background
    """
    row, col = center
    h, w = annot.shape[:2]
    d = disk(radius)
    r0 = row - radius
    c0 = col - radius
    d_size = 2 * radius + 1

    # clip to image bounds
    dr_start = max(0, -r0)
    dc_start = max(0, -c0)
    dr_end = min(d_size, h - r0)
    dc_end = min(d_size, w - c0)

    if dr_start >= dr_end or dc_start >= dc_end:
        return

    annot[r0 + dr_start:r0 + dr_end,
          c0 + dc_start:c0 + dc_end,
          channel][d[dr_start:dr_end, dc_start:dc_end]] = 255


def paint_stroke(annot, start, end, brush_radius, channel, forbidden=None):
    """Paint brush dabs along a line from start to end.

    Args:
        annot: (H, W, 4) uint8 array (modified in-place)
        start: (row, col) tuple
        end: (row, col) tuple
        brush_radius: radius of each dab
        channel: 0 for foreground, 1 for background
        forbidden: optional (H, W) bool mask — skip dabs whose center is here
    """
    r0, c0 = start
    r1, c1 = end
    h, w = annot.shape[:2]
    dist = max(abs(r1 - r0), abs(c1 - c0))
    if dist == 0:
        if forbidden is None or not _on_forbidden(r0, c0, h, w, forbidden):
            paint(annot, start, brush_radius, channel)
        return
    # Space dabs by ~brush_radius for continuous coverage
    num_dabs = max(1, int(dist / max(1, brush_radius))) + 1
    for i in range(num_dabs + 1):
        t = i / max(num_dabs, 1)
        r = int(r0 + (r1 - r0) * t)
        c = int(c0 + (c1 - c0) * t)
        if forbidden is not None and _on_forbidden(r, c, h, w, forbidden):
            continue
        paint(annot, (r, c), brush_radius, channel)


def _on_forbidden(r, c, h, w, forbidden):
    """Check if (r, c) lands on forbidden pixels."""
    if r < 0 or r >= h or c < 0 or c >= w:
        return False
    return forbidden[r, c]


def new_annot(height, width):
    """Return a blank (H, W, 4) uint8 annotation array (all zeros)."""
    return np.zeros((height, width, 4), dtype=np.uint8)
