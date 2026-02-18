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

    brush = d[dr_start:dr_end, dc_start:dc_end]
    annot[r0 + dr_start:r0 + dr_end,
          c0 + dc_start:c0 + dc_end,
          channel][brush] = 255


def paint_stroke(annot, from_pt, to_pt, radius, channel):
    """Paint a capsule (line with round caps) between two points.

    This matches QPainter.drawLine with a round-cap pen — a filled
    rectangle connecting the two points with semicircles on each end.
    """
    r0, c0 = int(from_pt[0]), int(from_pt[1])
    r1, c1 = int(to_pt[0]), int(to_pt[1])
    h, w = annot.shape[:2]

    # Bounding box with radius margin, clipped to image
    min_r = max(0, min(r0, r1) - radius)
    max_r = min(h, max(r0, r1) + radius + 1)
    min_c = max(0, min(c0, c1) - radius)
    max_c = min(w, max(c0, c1) + radius + 1)

    if min_r >= max_r or min_c >= max_c:
        return

    rr = np.arange(min_r, max_r)[:, None]
    cc = np.arange(min_c, max_c)[None, :]

    dr = r1 - r0
    dc = c1 - c0
    seg_len_sq = dr * dr + dc * dc

    if seg_len_sq == 0:
        dist_sq = (rr - r0) ** 2 + (cc - c0) ** 2
    else:
        # Project each pixel onto line segment, clamp t to [0, 1]
        t = ((rr - r0) * dr + (cc - c0) * dc) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)
        pr = r0 + t * dr
        pc = c0 + t * dc
        dist_sq = (rr - pr) ** 2 + (cc - pc) ** 2

    mask = dist_sq <= radius * radius
    annot[min_r:max_r, min_c:max_c, channel][mask] = 255


def new_annot(height, width):
    """Return a blank (H, W, 4) uint8 annotation array (all zeros)."""
    return np.zeros((height, width, 4), dtype=np.uint8)
