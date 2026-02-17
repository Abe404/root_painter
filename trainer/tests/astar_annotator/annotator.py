"""
A*-based corrective annotator.

Uses pathfinding to navigate error regions. The GT class mask
defines walkable terrain — the path cannot spill.

Approach:
  1. Pick brush radius from GT class region depth (50% of max)
  2. Erode GT mask so full brush disk stays inside the class
  3. A* to nearest uncovered error pixel, paint along path, repeat
  4. Halve brush radius and repeat for errors the big brush couldn't reach
"""
import heapq
from collections import deque

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.morphology import disk

from sim_benchmark.brush import paint, new_annot
from sim_benchmark.sim_user import (
    mouse_travel, FG_STROKE_DURATION, BG_STROKE_DURATION,
    STROKE_DURATION_SPREAD,
)


def _neighbors(r, c, h, w):
    """8-connected neighbors."""
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _nearest_walkable(r, c, walkable, h, w, max_dist=None):
    """BFS for nearest walkable pixel.

    If max_dist is set, only search within that Euclidean distance.
    """
    visited = set()
    q = deque([(r, c)])
    visited.add((r, c))
    while q:
        cr, cc = q.popleft()
        if max_dist is not None:
            if ((cr - r)**2 + (cc - c)**2) > max_dist**2:
                continue
        if walkable[cr, cc]:
            return cr, cc
        for nr, nc in _neighbors(cr, cc, h, w):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))
    return None, None


def astar(start, goal, walkable, h, w, brush_radius=None):
    """A* shortest path on the walkable grid.

    Returns list of (r, c) from start to goal, or [] if no path.
    If the goal isn't walkable, finds the nearest walkable pixel
    within brush_radius (so the brush edge can still reach it).
    """
    sr, sc = int(start[0]), int(start[1])
    gr, gc = int(goal[0]), int(goal[1])

    if not walkable[sr, sc]:
        sr, sc = _nearest_walkable(sr, sc, walkable, h, w)
        if sr is None:
            return []
    if not walkable[gr, gc]:
        wr, wc = _nearest_walkable(
            gr, gc, walkable, h, w, max_dist=brush_radius)
        if wr is None:
            return []
        gr, gc = wr, wc

    open_set = [(0.0, sr, sc)]
    came_from = {}
    g_score = {(sr, sc): 0.0}
    closed = set()

    while open_set:
        _, r, c = heapq.heappop(open_set)
        if (r, c) in closed:
            continue
        closed.add((r, c))

        if r == gr and c == gc:
            path = [(r, c)]
            while (r, c) in came_from:
                r, c = came_from[(r, c)]
                path.append((r, c))
            return path[::-1]

        for nr, nc in _neighbors(r, c, h, w):
            if (nr, nc) in closed or not walkable[nr, nc]:
                continue
            move_cost = 1.414 if (nr != r and nc != c) else 1.0
            new_g = g_score[(r, c)] + move_cost
            if new_g < g_score.get((nr, nc), float('inf')):
                g_score[(nr, nc)] = new_g
                h_cost = ((nr - gr)**2 + (nc - gc)**2) ** 0.5
                heapq.heappush(open_set, (new_g + h_cost, nr, nc))
                came_from[(nr, nc)] = (r, c)

    return []


def erode_safe(mask, radius):
    """Erode mask without shrinking at image borders.

    The brush can extend past image edges (paint() clips to bounds),
    so we pad with edge values before eroding — only class boundaries
    cause erosion, not image edges.
    """
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode='edge')
    eroded = binary_erosion(padded, structure=disk(radius))
    return eroded[radius:-radius, radius:-radius]


def pick_brush_radius(gt_class_mask):
    """50% of the max inscribed radius of the class region.

    The distance transform gives the depth (distance to nearest
    boundary) at every pixel. Half the max depth gives a brush
    that fits comfortably inside the region.
    """
    dt = distance_transform_edt(gt_class_mask)
    max_depth = float(np.max(dt))
    return max(1, int(max_depth * 0.5))


def corrective_annotation(ground_truth, prediction):
    """Create corrective annotations using A* pathfinding.

    Returns (annot, trajectory) or (None, []) if no errors.
    """
    fn_mask = (ground_truth == 1) & (prediction == 0)
    fp_mask = (ground_truth == 0) & (prediction == 1)

    if not np.any(fn_mask) and not np.any(fp_mask):
        return None, []

    h, w = ground_truth.shape
    annot = new_annot(h, w)
    trajectory = []
    rng = np.random.RandomState(0)
    mouse_pos = (h // 2, w // 2)

    for error_mask, gt_class_mask, channel in [
        (fn_mask, ground_truth == 1, 0),  # FN → paint FG
        (fp_mask, ground_truth == 0, 1),  # FP → paint BG
    ]:
        if not np.any(error_mask):
            continue

        duration_base = (FG_STROKE_DURATION if channel == 0
                         else BG_STROKE_DURATION)
        br = pick_brush_radius(gt_class_mask)

        # Multi-pass: start with the ideal brush, halve for remaining
        # errors that the big brush couldn't reach.
        while br >= 1:
            uncovered = error_mask & (annot[:, :, channel] == 0)
            if not np.any(uncovered):
                break

            walkable = erode_safe(gt_class_mask, br)
            if not np.any(walkable):
                br //= 2
                continue

            # Move mouse to walkable area near uncovered errors
            if not walkable[mouse_pos[0], mouse_pos[1]]:
                unc_r, unc_c = np.where(uncovered)
                mid_r, mid_c = int(np.mean(unc_r)), int(np.mean(unc_c))
                wr, wc = _nearest_walkable(mid_r, mid_c, walkable, h, w)
                if wr is not None:
                    mouse_travel(trajectory, mouse_pos, (wr, wc), rng)
                    mouse_pos = (wr, wc)

            # Paint along A* paths to each uncovered error pixel.
            stroke_points = []
            first_path = True

            for _ in range(2000):
                if not np.any(uncovered):
                    break

                # Nearest uncovered error pixel
                unc_r, unc_c = np.where(uncovered)
                dists = ((unc_r - mouse_pos[0])**2
                         + (unc_c - mouse_pos[1])**2)
                nearest = int(np.argmin(dists))
                target = (int(unc_r[nearest]), int(unc_c[nearest]))

                path = astar(mouse_pos, target, walkable, h, w,
                             brush_radius=br)
                if not path:
                    # Unreachable with this brush — skip for now,
                    # a smaller brush pass will try again.
                    uncovered[target[0], target[1]] = False
                    continue

                if first_path:
                    start_pt = path[0]
                    if start_pt != mouse_pos:
                        mouse_travel(trajectory, mouse_pos, start_pt, rng)
                        mouse_pos = start_pt
                    first_path = False

                for r, c in path:
                    paint(annot, (r, c), br, channel)
                    stroke_points.append((r, c))

                    # Mark brush footprint as covered
                    r0, r1 = max(0, r - br), min(h, r + br + 1)
                    c0, c1 = max(0, c - br), min(w, c + br + 1)
                    rr = np.arange(r0, r1)[:, None]
                    cc = np.arange(c0, c1)[None, :]
                    bmask = ((rr - r)**2 + (cc - c)**2) <= br**2
                    uncovered[r0:r1, c0:c1] &= ~bmask

                mouse_pos = path[-1]

            # Trajectory timing: emit all points for the video renderer,
            # but only give nonzero dt every step_size points.
            # Each ~100px segment gets one stroke duration.
            if stroke_points:
                step_size = max(1, br // 2)
                n_timed = max(1, len(stroke_points) // step_size)
                steps_per_stroke = max(1, 100 // step_size)
                n_equiv_strokes = max(1, n_timed // steps_per_stroke)
                total_dur = sum(
                    duration_base * rng.lognormal(0, STROKE_DURATION_SPREAD)
                    for _ in range(n_equiv_strokes))
                dt_per_step = total_dur / n_timed
                for idx, (r, c) in enumerate(stroke_points):
                    trajectory.append({
                        'r': r, 'c': c, 'painting': True,
                        'channel': channel, 'brush_radius': br,
                        'dt': dt_per_step if idx % step_size == 0 else 0.0,
                    })

            br //= 2

    if not np.any(annot[:, :, 0] > 0) and not np.any(annot[:, :, 1] > 0):
        return None, []

    return annot, trajectory
