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

from sim_benchmark.brush import paint, paint_stroke, new_annot

# Mouse input model — approximates OS mouse event delivery during painting.
# macOS coalesces mouse events to the display refresh rate regardless of
# mouse hardware polling rate. Measured ~119Hz on a 120Hz ProMotion display
# using measure_mouse_rate.py. Qt receives one mouseMoveEvent per display frame.
MOUSE_POLL_HZ = 120       # Hz, measured on macOS with 120Hz ProMotion display
MOUSE_PAINT_SPEED = 400   # px/s, moderate speed for corrective annotation
MOUSE_TRAVEL_SPEED = 800  # px/s, fast straight-line repositioning (mouse-up)

# Decision and assessment timings.
# Scene gist: >80% accuracy after 36ms (Larson & Loschky, 2014).
# Two-choice decision: ~190ms for trained gamers (Hyde & von Bastian, 2024).
VISUAL_SEARCH_MS = 200    # ms, find next error target and decide to go there
ASSESSMENT_PAUSE_MS = 200  # ms, post-annotation scan to confirm no errors remain


def mouse_travel(trajectory, from_pos, to_pos):
    """Record mouse-up repositioning from one position to another.

    Decomposes the gap into visual search time (finding the next target)
    and physical travel time (moving the cursor there).
    """
    r0, c0 = from_pos
    r1, c1 = to_pos
    dist = np.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2)
    travel_time = dist / MOUSE_TRAVEL_SPEED
    total_time = VISUAL_SEARCH_MS / 1000 + travel_time
    num_steps = max(1, int(dist / 8.0))
    dt = total_time / num_steps
    for i in range(1, num_steps + 1):
        t = i / num_steps
        trajectory.append({
            'r': int(round(r0 + (r1 - r0) * t)),
            'c': int(round(c0 + (c1 - c0) * t)),
            'painting': False, 'channel': -1, 'brush_radius': 0, 'dt': dt,
        })


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


def pick_brush_radius(gt_class_mask, error_mask):
    """Largest brush that's useful for covering errors, constrained by class.

    Finds the error pixel deepest inside the class region (most room
    around it), then measures how far other errors extend from there
    within the class circle. Adds 50% margin for imperfect aim.

    The class circle constraint means errors on opposite sides of a
    foreground region are measured separately — the circle can't cross
    into the wrong class.
    """
    class_dt = distance_transform_edt(gt_class_mask)
    max_r = max(1, int(np.max(class_dt) * 0.5))

    err_coords = np.argwhere(error_mask)
    if len(err_coords) == 0:
        return max_r

    # Best center: error pixel with most room in the class
    err_depths = class_dt[err_coords[:, 0], err_coords[:, 1]]
    best_idx = int(np.argmax(err_depths))
    cr, cc = err_coords[best_idx]
    circle_r = class_dt[cr, cc]

    # Error pixels within the class circle at that center
    dists = np.sqrt(np.sum((err_coords - [cr, cc]) ** 2, axis=1))
    in_circle = dists <= circle_r
    if not np.any(in_circle):
        return max_r

    # How far do errors extend within this circle?
    needed_r = max(1, int(np.percentile(dists[in_circle], 95)))

    # Add margin, cap by class constraint
    return max(1, min(int(needed_r * 1.5), max_r))


def coarse_waypoint_indices(path, max_spacing):
    """Indices into path at approximately max_spacing apart.

    This is the fastest the user can move their mouse — limited by
    physical hand speed. The OS polls at a fixed rate, so faster
    movement means wider spacing between samples.
    """
    if len(path) <= 1:
        return list(range(len(path)))
    indices = [0]
    dist_acc = 0.0
    for i in range(1, len(path)):
        dr = path[i][0] - path[i - 1][0]
        dc = path[i][1] - path[i - 1][1]
        dist_acc += (dr * dr + dc * dc) ** 0.5
        if dist_acc >= max_spacing:
            indices.append(i)
            dist_acc = 0.0
    if indices[-1] != len(path) - 1:
        indices.append(len(path) - 1)
    return indices


def line_within_walkable(p1, p2, walkable):
    """True if every pixel on the straight line from p1 to p2 is walkable.

    The walkable mask is the GT class eroded by brush radius, so if the
    line stays walkable, the full brush disk stays within the class at
    every point — no spill.
    """
    r0, c0 = p1
    r1, c1 = p2
    steps = max(abs(r1 - r0), abs(c1 - c0))
    if steps <= 1:
        return True
    for i in range(1, steps):
        r = r0 + round((r1 - r0) * i / steps)
        c = c0 + round((c1 - c0) * i / steps)
        if not walkable[r, c]:
            return False
    return True


def _refine_segment(path, idx_a, idx_b, walkable, result):
    """Bisect a segment until the straight line stays walkable."""
    if idx_b - idx_a <= 1:
        result.append(idx_b)
        return
    if line_within_walkable(path[idx_a], path[idx_b], walkable):
        result.append(idx_b)
        return
    mid = (idx_a + idx_b) // 2
    _refine_segment(path, idx_a, mid, walkable, result)
    _refine_segment(path, mid, idx_b, walkable, result)


def refine_for_safety(path, indices, walkable):
    """Add waypoints where straight lines between existing ones leave walkable.

    Uses bisection: if the straight line between two waypoints crosses
    non-walkable pixels, insert the midpoint from the original A* path
    and check both halves. This models a user slowing down on curves
    near class boundaries — more mouse samples where precision matters.
    """
    refined = [indices[0]]
    for i in range(1, len(indices)):
        _refine_segment(path, refined[-1], indices[i], walkable, refined)
    return refined


def safe_waypoints(path, max_spacing, walkable):
    """Subsample path to waypoints safe to connect with straight brush strokes.

    1. Coarse pass: space waypoints at max_spacing (mouse speed limit)
    2. Refine: add waypoints where straight lines would leave walkable mask

    The result models a user who moves fast on straight sections and
    slows down on curves near class boundaries.
    """
    indices = coarse_waypoint_indices(path, max_spacing)
    indices = refine_for_safety(path, indices, walkable)
    return [path[i] for i in indices]


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
    mouse_pos = (h // 2, w // 2)

    for error_mask, gt_class_mask, channel in [
        (fn_mask, ground_truth == 1, 0),  # FN → paint FG
        (fp_mask, ground_truth == 0, 1),  # FP → paint BG
    ]:
        if not np.any(error_mask):
            continue

        br = pick_brush_radius(gt_class_mask, error_mask)
        wp_spacing = MOUSE_PAINT_SPEED / MOUSE_POLL_HZ
        dt_per_wp = 1.0 / MOUSE_POLL_HZ

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
                    mouse_travel(trajectory, mouse_pos, (wr, wc))
                    mouse_pos = (wr, wc)

            # Paint along A* paths using waypoints sampled at poll rate.
            all_waypoints = []
            first_path = True
            prev_wp = None

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
                        mouse_travel(trajectory, mouse_pos, start_pt)
                        mouse_pos = start_pt
                    first_path = False

                waypoints = safe_waypoints(path, wp_spacing, walkable)

                for wp in waypoints:
                    if prev_wp is not None:
                        paint_stroke(annot, prev_wp, wp, br, channel)
                    else:
                        paint(annot, wp, br, channel)

                    # Mark brush circle coverage at waypoint
                    r, c = wp
                    r0, r1 = max(0, r - br), min(h, r + br + 1)
                    c0, c1 = max(0, c - br), min(w, c + br + 1)
                    rr = np.arange(r0, r1)[:, None]
                    cc = np.arange(c0, c1)[None, :]
                    bmask = ((rr - r)**2 + (cc - c)**2) <= br**2
                    uncovered[r0:r1, c0:c1] &= ~bmask

                    prev_wp = wp

                all_waypoints.extend(waypoints)
                mouse_pos = waypoints[-1]

            # Emit trajectory — one waypoint per mouse poll event
            if all_waypoints:
                for r, c in all_waypoints:
                    trajectory.append({
                        'r': r, 'c': c, 'painting': True,
                        'channel': channel, 'brush_radius': br,
                        'dt': dt_per_wp,
                    })

            br //= 2

    if not np.any(annot[:, :, 0] > 0) and not np.any(annot[:, :, 1] > 0):
        return None, []

    # Assessment pause — user scans the image for remaining errors,
    # sees it looks good, and moves on.
    trajectory.append({
        'r': mouse_pos[0], 'c': mouse_pos[1],
        'painting': False, 'channel': -1, 'brush_radius': 0,
        'dt': ASSESSMENT_PAUSE_MS / 1000,
    })

    return annot, trajectory
