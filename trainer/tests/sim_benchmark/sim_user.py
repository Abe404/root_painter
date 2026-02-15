"""
Simulated annotator for corrective annotation benchmarking.

Creates initial and corrective annotations following the RootPainter
protocol using a simulated mouse.

Brush selection principle: use the largest brush that fits the
constraints (region geometry and pixel budget). Bigger brush = more
pixels per dab = less time. No magic scale parameters — the brush
radius is derived from the region and the task.

Speed-jitter tradeoff: the user's aim jitter is proportional to their
painting speed. In open regions they go fast (more jitter, but plenty
of margin). In tight regions they slow down (less jitter, precise
placement). One parameter (JITTER_RATE) governs the whole tradeoff.

Every event is recorded with a simulated time cost (dt in seconds).
"""
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from sim_benchmark.brush import disk, paint, new_annot

# Simulated mouse speeds (pixels per second).
TRAVEL_SPEED = 800.0       # mouse-up repositioning (fast, straight line)
MIN_PAINT_SPEED = 30.0     # careful painting near boundaries
MAX_PAINT_SPEED = 200.0    # confident painting in open areas

# Speed-jitter coupling: aim σ = paint_speed * JITTER_RATE.
# At 100 px/s → σ ≈ 4 px.  At 200 px/s → σ ≈ 8 px.  At 30 px/s → σ ≈ 1.2 px.
JITTER_RATE = 0.04         # seconds (σ in pixels per px/s of speed)


def choose_paint_speed(eroded):
    """Choose painting speed based on the safe zone's size.

    Larger safe zone → faster → more jitter (but within margin).
    Tight zone → slower → precise placement.

    The user picks the fastest speed where 3σ of jitter fits within
    the equivalent radius of the eroded region.

    Returns (paint_speed, aim_sigma).
    """
    eroded_area = int(np.sum(eroded))
    if eroded_area == 0:
        speed = MIN_PAINT_SPEED
        return speed, speed * JITTER_RATE
    margin = np.sqrt(eroded_area / np.pi)
    # Speed where 3σ fits within margin: 3 * speed * JITTER_RATE <= margin
    safe_speed = margin / (3 * JITTER_RATE)
    speed = float(np.clip(safe_speed, MIN_PAINT_SPEED, MAX_PAINT_SPEED))
    sigma = speed * JITTER_RATE
    return speed, sigma


def fit_brush(mask, max_radius, min_interior=50):
    """Find the largest brush radius <= max_radius that fits inside mask.

    "Fits" means binary erosion by (radius+1) still leaves at least
    min_interior pixels of interior. Uses binary search for precision.
    Returns (brush_radius, eroded_mask) or (0, None) if nothing fits.
    """
    lo, hi = 1, max_radius
    best_radius, best_eroded = 0, None
    while lo <= hi:
        mid = (lo + hi) // 2
        eroded = binary_erosion(mask, structure=disk(mid + 1))
        if np.sum(eroded) >= min_interior:
            best_radius, best_eroded = mid, eroded
            lo = mid + 1
        else:
            hi = mid - 1
    return best_radius, best_eroded


def mouse_travel(trajectory, from_pos, to_pos, speed=8.0):
    """Record mouse-up movement from one position to another."""
    r0, c0 = from_pos
    r1, c1 = to_pos
    dist = np.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2)
    num_steps = max(1, int(dist / speed))
    step_dist = dist / num_steps if num_steps > 0 else 0
    dt = step_dist / TRAVEL_SPEED
    for i in range(1, num_steps + 1):
        t = i / num_steps
        r = r0 + (r1 - r0) * t
        c = c0 + (c1 - c0) * t
        trajectory.append({
            'r': int(round(r)), 'c': int(round(c)),
            'painting': False, 'channel': -1, 'brush_radius': 0,
            'dt': dt,
        })


def mouse_stroke(annot, start, eroded, brush_radius, channel, rng,
                  trajectory, num_steps=20, step_size=None, max_dabs=None,
                  paint_speed=None):
    """Simulate a mouse-down drag: smooth curve staying within the region.

    Moves with momentum and slight angular noise. Only paints when
    inside the eroded region (guarantees no cross-class painting).
    Stops early if max_dabs is reached (user lifts pen when done).
    """
    if step_size is None:
        step_size = max(1, brush_radius)
    if paint_speed is None:
        paint_speed = MAX_PAINT_SPEED

    dt = step_size / paint_speed

    h, w = eroded.shape
    r, c = float(start[0]), float(start[1])
    angle = rng.uniform(0, 2 * np.pi)
    dabs = 0

    for _ in range(num_steps):
        ir, ic = int(round(r)), int(round(c))
        if 0 <= ir < h and 0 <= ic < w and eroded[ir, ic]:
            paint(annot, (ir, ic), brush_radius, channel)
            trajectory.append({
                'r': ir, 'c': ic,
                'painting': True, 'channel': channel,
                'brush_radius': brush_radius,
                'dt': dt,
            })
            dabs += 1
            if max_dabs is not None and dabs >= max_dabs:
                break

        angle += rng.normal(0, 0.1)
        next_r = r + np.sin(angle) * step_size
        next_c = c + np.cos(angle) * step_size

        nr, nc = int(round(next_r)), int(round(next_c))
        if 0 <= nr < h and 0 <= nc < w and eroded[nr, nc]:
            r, c = next_r, next_c
        else:
            rows, cols = np.where(eroded)
            dists = (rows - r) ** 2 + (cols - c) ** 2
            nearby = dists < (step_size * 5) ** 2
            if np.any(nearby):
                nearby_idx = np.where(nearby)[0]
                pick = rng.choice(nearby_idx)
                dr = rows[pick] - r
                dc = cols[pick] - c
            else:
                dr = np.mean(rows) - r
                dc = np.mean(cols) - c
            angle = np.arctan2(dr, dc)
            r += np.sin(angle) * step_size
            c += np.cos(angle) * step_size


def initial_annotation(ground_truth, coverage=0.05, seed=None):
    """Create initial annotations using simulated mouse strokes on FG/BG.

    Brush selection:
    - FG: largest brush that fits in the FG region. No pixel cap —
      more FG annotation is always fine, so overshoot is harmless.
    - BG: largest brush where one dab doesn't exceed the pixel budget
      (balance constraint: BG <= 10x FG). Avoids wasting balance budget.

    Returns:
        (annot, trajectory)
    """
    rng = np.random.RandomState(seed)
    h, w = ground_truth.shape
    annot = new_annot(h, w)
    trajectory = []

    fg_mask = ground_truth == 1
    bg_mask = ground_truth == 0
    fg_area = int(np.sum(fg_mask))
    bg_area = int(np.sum(bg_mask))

    if fg_area == 0 or bg_area == 0:
        return annot, trajectory

    mouse_pos = (h // 2, w // 2)

    def paint_class(mask, area, target_pixels, channel, max_dab_pixels=None):
        """Paint strokes on mask to reach target_pixels.

        max_dab_pixels: if set, cap brush so pi*r^2 <= max_dab_pixels.
        """
        nonlocal mouse_pos
        if target_pixels <= 0:
            return 0

        equiv_radius = np.sqrt(area / np.pi)

        # User picks brush to maximize painting efficiency (pixels/second).
        # Big brush = fewer dabs but less margin, forcing slower painting.
        # Sweet spot: biggest brush that still allows fast, comfortable painting.
        margin_for_max_speed = 3 * MAX_PAINT_SPEED * JITTER_RATE
        if equiv_radius > 2 * margin_for_max_speed:
            target_brush = max(1, int(equiv_radius - margin_for_max_speed))
        else:
            target_brush = max(1, int(equiv_radius / 2))

        if max_dab_pixels is not None:
            # Majority class: user paints a few long strokes, not giant dabs.
            # Size brush so it takes ~2 strokes to reach the target.
            dabs_per_stroke = max(5, int(equiv_radius / 20))
            budget_per_dab = max_dab_pixels / max(1, 2 * dabs_per_stroke)
            budget_max = max(3, int(np.sqrt(budget_per_dab / np.pi)))
            brush_max = min(target_brush, budget_max)
        else:
            brush_max = target_brush

        brush_radius, eroded = fit_brush(mask, brush_max)
        if brush_radius == 0 or eroded is None:
            return 0

        # Speed adapts to how much room the user has
        paint_speed, sigma = choose_paint_speed(eroded)

        rows, cols = np.where(eroded)
        dab_area = max(1, int(np.pi * brush_radius ** 2))
        total_dabs_needed = max(1, target_pixels // dab_area)
        # Continuous dab spacing (~50% brush diameter, like real painting)
        step_size = max(1, brush_radius // 2)
        # Stroke covers roughly one equiv_radius of distance
        stroke_steps = max(5, int(equiv_radius / max(1, step_size)))

        attempts = 0
        max_attempts = max(10, total_dabs_needed * 3)
        while attempts < max_attempts:
            painted = int(np.sum(annot[:, :, channel] > 0))
            remaining = target_pixels - painted
            if painted > 0 and remaining <= dab_area:
                break
            dabs_left = max(1, remaining // dab_area)

            # Start on unpainted ground — don't waste time re-annotating
            fresh = eroded & (annot[:, :, channel] == 0)
            fresh_rows, fresh_cols = np.where(fresh)
            if len(fresh_rows) == 0:
                break  # everything reachable is already painted

            if max_dab_pixels is not None:
                idx = rng.choice(len(fresh_rows))
            else:
                dists = (fresh_rows - mouse_pos[0]) ** 2 + (fresh_cols - mouse_pos[1]) ** 2
                nearest = np.argsort(dists)[:max(1, len(fresh_rows) // 5)]
                idx = rng.choice(nearest)

            jr = rng.normal(0, sigma)
            jc = rng.normal(0, sigma)
            start = (int(fresh_rows[idx] + jr), int(fresh_cols[idx] + jc))

            size_noise = abs(rng.normal(0, max(1, sigma * 0.5)))
            jittered_radius = max(1, brush_radius - int(size_noise))

            mouse_travel(trajectory, mouse_pos, start)
            mouse_stroke(annot, start, fresh, jittered_radius, channel, rng,
                          trajectory, num_steps=stroke_steps,
                          step_size=step_size, max_dabs=dabs_left,
                          paint_speed=paint_speed)

            if trajectory and trajectory[-1]['painting']:
                mouse_pos = (trajectory[-1]['r'], trajectory[-1]['c'])
            else:
                mouse_pos = start

            attempts += 1

        return int(np.sum(annot[:, :, channel] > 0))

    # Annotate minority class first, then majority up to 10x minority
    if fg_area <= bg_area:
        min_mask, min_area, min_ch = fg_mask, fg_area, 0
        maj_mask, maj_area, maj_ch = bg_mask, bg_area, 1
    else:
        min_mask, min_area, min_ch = bg_mask, bg_area, 1
        maj_mask, maj_area, maj_ch = fg_mask, fg_area, 0

    # Minority class: paint as much as convenient (strokes will naturally
    # cover a subset — no need for a conservative coverage cap)
    min_target = min_area
    min_annotated = paint_class(min_mask, min_area, min_target, min_ch)

    # Move to a clearly-majority area before painting (user wouldn't
    # paint BG right next to the FG object they just annotated)
    maj_rows, maj_cols = np.where(maj_mask)
    if len(maj_rows) > 0:
        pick = rng.randint(len(maj_rows))
        mouse_pos = (int(maj_rows[pick]), int(maj_cols[pick]))

    # Majority: aim for up to 10x minority, capped by available area
    maj_target = min(10 * max(1, min_annotated), maj_area)
    paint_class(maj_mask, maj_area, maj_target, maj_ch,
                max_dab_pixels=maj_target)

    return annot, trajectory


def corrective_annotation(ground_truth, prediction):
    """Create corrective annotations in the neighborhood of errors.

    Brush selection: largest brush that fits in the GT class region
    and still has paintable area near the errors. The user paints
    the correct class in the error neighborhood — FG strokes may
    land on correctly-predicted FG, which is realistic.

    Returns:
        (annot, trajectory) or (None, []) if no clear errors
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

    def cover_errors(error_mask, gt_class_mask, channel):
        nonlocal mouse_pos
        error_area = int(np.sum(error_mask))
        if error_area == 0:
            return

        # Brush sized for the error region, constrained to fit GT class
        error_equiv = np.sqrt(error_area / np.pi)
        class_area = int(np.sum(gt_class_mask))
        class_equiv = np.sqrt(class_area / np.pi)
        # Start from error-appropriate size (capped by class geometry)
        brush_radius = max(1, int(min(error_equiv, class_equiv) / 2))
        paint_region = None
        while True:
            safe = binary_erosion(
                gt_class_mask, structure=disk(brush_radius + 1))
            # Neighborhood: one brush-width beyond errors
            nbhd = binary_dilation(
                error_mask, structure=disk(brush_radius))
            candidate = safe & nbhd
            if np.any(candidate):
                paint_region = candidate
                break
            if brush_radius <= 1:
                break
            brush_radius = max(1, brush_radius // 2)

        if paint_region is None:
            return

        # Speed adapts to how tight the paint region is
        paint_speed, sigma = choose_paint_speed(paint_region)

        rows, cols = np.where(paint_region)
        # A few strokes across the error region, each a few dabs long
        brush_widths = max(1, int(error_equiv / max(1, brush_radius)))
        stroke_steps = max(3, brush_widths)
        num_strokes = max(1, min(5, brush_widths))

        for _ in range(num_strokes):
            # Start near current position to minimize travel
            dists = (rows - mouse_pos[0]) ** 2 + (cols - mouse_pos[1]) ** 2
            nearest = np.argsort(dists)[:max(1, len(rows) // 5)]
            idx = rng.choice(nearest)
            jr = rng.normal(0, sigma)
            jc = rng.normal(0, sigma)
            start = (int(rows[idx] + jr), int(cols[idx] + jc))

            # Brush size jitter scaled to current sigma
            size_noise = abs(rng.normal(0, max(1, sigma * 0.5)))
            jittered_radius = max(1, brush_radius - int(size_noise))

            mouse_travel(trajectory, mouse_pos, start)
            mouse_stroke(annot, start, paint_region, jittered_radius, channel,
                          rng, trajectory, num_steps=stroke_steps,
                          paint_speed=paint_speed)

            if trajectory and trajectory[-1]['painting']:
                mouse_pos = (trajectory[-1]['r'], trajectory[-1]['c'])
            else:
                mouse_pos = start

    cover_errors(fn_mask, ground_truth == 1, channel=0)
    cover_errors(fp_mask, ground_truth == 0, channel=1)

    if not np.any(annot[:, :, 0] > 0) and not np.any(annot[:, :, 1] > 0):
        return None, []

    return annot, trajectory
