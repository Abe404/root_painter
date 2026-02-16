# Simulated Corrective Annotation Benchmark — Method

## Overview

This benchmark simulates a human annotator following RootPainter's corrective
annotation protocol. It measures how quickly a U-Net model learns from
iterative human feedback, without requiring an actual human in the loop.

A simulated user processes images one at a time in two phases:

1. **Initial phase** — annotate clear foreground and background examples
   without seeing model output
2. **Corrective phase** — view the model's segmentation, correct all clear
   errors, skip images where the prediction looks good

Training begins after the first two images are annotated (matching the
protocol's recommendation). The transition from initial to corrective phase
is adaptive: the simulated user keeps providing initial annotations until the
model achieves a minimum validation F1, indicating it has learned enough to
produce useful segmentations worth correcting.

## Mouse Simulation

All annotation is performed through a simulated mouse. The mouse has two
states: **mouse-up** (travelling, not painting) and **mouse-down** (painting
brush dabs as it moves). Every mouse event is recorded in a trajectory with
a simulated time cost (`dt` in seconds), enabling:

- Video playback at real FPS (e.g. 6 FPS based on simulated time)
- Future comparison against logged human annotation sessions
- Future calibration from measured human annotation speeds

### Speed-jitter tradeoff

The user's aim jitter is proportional to their painting speed. One parameter
(`JITTER_RATE = 0.04`) governs the coupling:

    aim_sigma = paint_speed * JITTER_RATE

The user picks the fastest speed where 3σ of jitter fits within their
available margin (the equivalent radius of the safe interior after erosion):

    safe_speed = margin / (3 * JITTER_RATE)
    paint_speed = clip(safe_speed, MIN_PAINT_SPEED, MAX_PAINT_SPEED)

| Constant | Default | Description |
|----------|---------|-------------|
| `TRAVEL_SPEED` | 800 px/s | Mouse-up repositioning (fast, straight line) |
| `MIN_PAINT_SPEED` | 30 px/s | Careful painting in tight regions |
| `MAX_PAINT_SPEED` | 200 px/s | Confident painting in open areas |
| `JITTER_RATE` | 0.04 s | σ per unit speed (aim σ = speed × rate) |

This produces natural behaviour:

- **Open regions** (large BG area): fast painting (~200 px/s), more jitter
  (σ ≈ 8 px), but plenty of margin so jitter is harmless
- **Tight regions** (small FG, error corrections): slow painting (~30–80 px/s),
  precise placement (σ ≈ 1–3 px), takes more time per dab but accuracy is needed

Each trajectory entry stores `dt = step_size / paint_speed`.

### Mouse-up travel

When moving between stroke locations, the mouse travels in a straight line
at `TRAVEL_SPEED`. The path is discretised into steps of ~8 pixels so the
cursor movement is visible in video playback.

### Mouse-down strokes

When painting, the mouse drags in a smooth curve with slight angular noise
(Gaussian perturbation with σ=0.1 radians per step). This produces
gently curving strokes rather than perfectly straight lines.

If the next step would leave the safe interior of the target region, the
mouse steers back toward a nearby interior pixel. This keeps strokes
contained without sharp clipping artifacts — the simulated user simply
changes direction before hitting the boundary, as a real user would.

Dabs are spaced at ~50% of brush diameter (step_size = brush_radius / 2),
producing continuous coverage along the stroke path — matching how real
painting tools space brush dabs.

### Aim jitter

Stroke start positions are offset by Gaussian noise with σ equal to the
current aim sigma (derived from painting speed via the speed-jitter
tradeoff). Fast painting → more offset, careful painting → precise placement.

### Brush size jitter (jitter awareness)

The user knows they have aim jitter, so they compensate by picking a brush
slightly smaller than the theoretical optimum. The compensation scales with
sigma: `size_noise = abs(normal(0, sigma * 0.5))`, and the brush is always
reduced (never increased). When going fast with high jitter, the user picks
a noticeably smaller brush. When going slow with low jitter, they use
near-optimal size.

## Simulated User Behaviour

### Brush selection

The simulated user picks a brush that maximizes painting efficiency —
pixels covered per second. The efficiency-optimal brush balances coverage
per dab against painting speed:

- **Large regions** (equiv_radius > 2 × margin_for_max_speed): use the
  biggest brush that still allows painting at max speed.
  `target = equiv_radius - 3 * MAX_PAINT_SPEED * JITTER_RATE`
- **Small regions**: balance brush size vs speed. `target = equiv_radius / 2`

For the majority class, the brush is additionally capped to encourage
stroke-based painting (a few long strokes) rather than giant single dabs.

`fit_brush(mask, max_radius)` finds the largest radius ≤ `max_radius` such
that the mask still has usable interior after erosion by `radius + 1`. It
uses binary search for precision.

### Staying inside the region

The target region is eroded (shrunk inward) by `brush_radius + 1` pixels
using a circular structuring element before selecting stroke start positions.
This ensures the circular brush disk never paints outside the region boundary.
The +1 accounts for rounding in the stroke simulation.

Dabs are only placed when the current cursor position is inside the eroded
region. If the mouse wanders outside during a stroke, no paint is applied
until it re-enters — the simulated user lifts the brush near boundaries.

### Initial Phase

The simulated user draws mouse strokes on foreground (FG) and background (BG)
regions of the ground truth mask. This mimics a real user painting sparse
examples on clear areas of a new image, without any model prediction to
react to.

**Minority class first.** The smaller class (by area) is annotated first,
painting as much as convenient (strokes naturally cover a portion of the
region). The larger class is then annotated with up to 10× the minority
annotation (measured in annotated pixels), following the protocol's balance
guidance. This is dataset-independent: if FG is the minority (typical), FG
is annotated first and BG is capped at 10× FG.

**Minority class painting.** The user paints generously — there's no
conservative coverage cap. The brush and stroke dynamics naturally limit
how much gets covered (typically 50–80% of a simple region). More minority
annotation gives the model more signal.

**Majority class painting.** The brush is sized so the user paints a few
long strokes rather than giant single dabs. Stroke start positions are
spread randomly across the region for spatial diversity, and the mouse is
repositioned to a clearly-majority area before painting (avoiding annotating
BG right next to FG).

**Dab-level stopping.** Each stroke's actual pixel coverage is measured
after painting. Strokes continue until the annotated area reaches the target.
The user lifts the pen mid-stroke if the remaining budget is exhausted
(`max_dabs`).

### Corrective Phase

The best model checkpoint segments the current image. The simulated user
compares the segmentation against ground truth to identify errors:

- **False negatives** (GT=1, pred=0): painted as FG annotation
- **False positives** (GT=0, pred=1): painted as BG annotation
- **No clear errors**: image is skipped

**Painting the correct class near errors.** The user paints the correct
class in the *neighbourhood* of error regions, not only on error pixels.
This reflects real user behaviour: when correcting a missed FG region, a
user naturally paints FG over the missed area and some surrounding correctly
predicted FG. Implementation:

1. The error mask is dilated by `brush_radius` to define the error
   neighbourhood
2. The full GT class mask is eroded by `brush_radius + 1` to define the
   safe interior
3. The paint region is the intersection: safe interior ∩ error neighbourhood

This ensures annotations stay on the correct class while covering the error
and its surroundings.

**Brush sizing.** The brush adapts to the remaining error. For chunky
errors (raster fill), the desired radius is based on the error's equivalent
radius (sqrt(area/pi)) capped by the maximum distance transform value
within the error. For thin errors (curved strokes), the brush is sized to
the GT class depth *near* the error — the maximum distance transform in a
20px neighbourhood — so a large brush centered deep inside the GT class
can cover a thin boundary error with just its edge. `_find_brush`
binary-searches for the largest radius where the eroded safe zone
(gt_class_mask eroded by disk(radius)) overlaps with the dilated error.
Erosion ignores image borders (pad with edge values) so the brush can
reach image edges.

**Paint region (safe zone).** The paint region is the eroded GT class mask
intersected with the dilated error neighbourhood. Erosion by the brush
radius guarantees the full circular brush stays inside the correct class —
mathematically safe because `paint()` uses the same disk shape as the
erosion structuring element.

**Raster zigzag fill.** Error regions are filled with parallel scan lines
using the standard raster fill algorithm from coverage path planning — the
same approach used by 3D printing slicers (rectilinear infill in Cura,
PrusaSlicer, etc.) and CNC pocket milling. See Choset (2001)
"Coverage of Known Spaces: The Boustrophedon Cellular Decomposition",
Autonomous Robots 9(3), for the formal robotics treatment. The sweep direction
is computed from the error centroid toward the farthest error pixel, and
scan lines are spaced perpendicular to this at ~85% of brush diameter
(slight overlap to avoid gaps, standard in raster fill). Each scan line
is clipped to the UNCOVERED strokeable area — after each stroke, the
remaining error is updated, so subsequent scan lines target only unpainted
territory. Sweep direction alternates (zigzag / boustrophedon) to minimize
travel between lines, like a human coloring back and forth. The loop
continues until only boundary-ambiguous pixels remain (within 2px of the
GT class edge), or the annotator stalls (3 consecutive iterations
covering less than 5% of remaining error).

**Boundary-aware stopping.** The simulated user does not try to annotate
the last few pixels of error right on the boundary between true foreground
and background. These pixels are genuinely ambiguous — a real user wouldn't
agonize over them either. Error pixels within 2px of the opposite GT class
(computed via `binary_dilation(~gt_class_mask, disk(2))`) are excluded
from the actionable error mask. If all remaining error pixels fall within
this boundary zone, the error region is considered corrected.

**Curved contour-following strokes for thin errors.** When the error is
thin relative to the brush (narrower dimension < 2× brush diameter), the
annotator switches from raster fill to curved strokes that follow the
error contour. Connected components of the actionable error are labeled.
For each component, pixels are ordered via BFS walk from an endpoint,
waypoints are sampled at regular spacing (scaled with brush size), and
snapped to the paintable region. The stroke chains through waypoints,
producing a smooth curve that follows the boundary. These strokes use
"careful" mode: 2× slower duration (less jitter) and full brush radius
(no size reduction), since precision matters near edges.

When no brush fits with safe-zone erosion (error right at the GT class
boundary), a small brush (radius 2) with 1px erosion buffer is used.
If even that fails, radius 1 on the uneroded GT mask is tried.

**Spill detection and eraser.** After each stroke, a check detects any
annotation pixels that landed on the wrong GT class. If spill is found,
those pixels are erased (set to 0) and a warning is printed — simulating
the user noticing the mistake and using the eraser brush to fix it. This
is rare with safe-zone erosion but can occur in boundary mode.

## Training Loop

The benchmark processes images sequentially. Training begins after
`min_initial_images` (default 2) are annotated, matching the protocol's
recommendation to provide at least two clear examples before starting.

For each image:

1. **Phase decision.** If fewer than `min_initial_images` have been
   annotated, or the best validation F1 is below `corrective_f1_threshold`,
   the image receives an initial annotation. Once the threshold is reached,
   the benchmark switches to corrective mode for all subsequent images.

2. **Annotation.** In initial mode, a coverage-based annotation is created.
   In corrective mode, the best model segments the image and errors are
   annotated. If the corrective annotation is None (no clear errors), the
   image is skipped.

3. **Train/val split.** The annotation is saved to train/ or val/ following
   the painter's allocation logic (approximately 5:1 train:val ratio).

4. **Training.** The model trains for `epochs_between_images` epochs on all
   annotations accumulated so far. After each epoch, validation F1 is
   computed. If improved, the model state is saved as the new best
   checkpoint.

The best checkpoint is used for segmentation in subsequent corrective phases.

## Annotation Format

Annotations are RGBA arrays (H, W, 4) matching RootPainter's format:
- Channel 0: foreground (value 255 where annotated)
- Channel 1: background (value 255 where annotated)
- Channels 2-3: unused

`paint(annot, center, radius, channel)` stamps a filled circular disk at
the given position.

## Trajectory Format

The trajectory is a list of event dicts, one per mouse step:

```python
{
    'r': int,           # row position
    'c': int,           # column position
    'painting': bool,   # True = mouse-down (painting), False = mouse-up
    'channel': int,     # 0 = FG, 1 = BG, -1 = not painting
    'brush_radius': int,# brush radius (0 when not painting)
    'dt': float,        # simulated time for this step (seconds)
}
```

Total annotation time for an image is `sum(e['dt'] for e in trajectory)`.

## Verification

Run with `save_video=True` to produce verification frames in
`output_dir/frames/`. Frames are rendered at 6 FPS of simulated time
(capped at 60 seconds), so the video shows the first minute of the
simulated annotation session.

Each frame shows:

- **Left panel**: input image with yellow GT contour, overlaid with
  image index, simulated time, and phase
- **Middle panel**: annotation built up so far (red = FG, green = BG)
  with a cursor showing the current mouse position
- **Right panel** (corrective only): model prediction (blue), with errors
  highlighted (red = false negative, blue = false positive)

Open `frame_000.png` in an image viewer and step through with arrow keys.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_initial_images` | 2 | Minimum images annotated before training starts |
| `corrective_f1_threshold` | 0.2 | Validation F1 needed to switch from initial to corrective phase |
| `epochs_between_images` | 2 | Training epochs after each annotation |
| `batch_size` | 6 | Training batch size |
| `in_w` / `out_w` | 572 / 500 | UNet input/output tile size |
| `lr` | 0.01 | SGD learning rate (with momentum 0.99, Nesterov) |
| `seed` | 1 | Random seed for reproducibility |
| `min_epoch_tiles` | 20 | Minimum tiles per training epoch |
