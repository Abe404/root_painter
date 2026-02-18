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
model produces confident predictions (confidence > 0.8), indicating it has
learned enough to produce useful segmentations worth correcting.

## Mouse Input Model

The simulation models a real mouse as delivered by the OS to Qt on macOS.

### OS event delivery

macOS coalesces mouse events to the display refresh rate regardless of mouse
hardware polling rate — even gaming mice at 500–1000Hz are capped by the
display. Qt receives one `mouseMoveEvent` per display frame.

Measured ~119Hz on a 120Hz ProMotion display using `measure_mouse_rate.py`.
The simulation uses **MOUSE_POLL_HZ = 120**.

### Painting speed

The user moves the cursor at up to **MOUSE_PAINT_SPEED = 400 px/s** during
corrective annotation. At 120Hz polling, this gives a maximum waypoint
spacing of ~3.3px between consecutive mouse events.

### Brush stroke shape

Qt's `QPainter.drawLine` with a round-cap pen draws a **capsule** (Minkowski
sum of a line segment and a disk) between consecutive mouse positions. The
simulation's `paint_stroke()` matches this exactly: for each pixel, project
onto the line segment, check if the distance is within the brush radius.

## Corrective Annotation: A* Pathfinding

The corrective annotator uses A* pathfinding to navigate error regions. The
ground truth class mask defines walkable terrain — the brush path cannot
spill onto the wrong class.

### Brush radius selection

The brush radius is chosen based on two constraints:

1. **Class depth:** The GT class distance transform gives the maximum brush
   radius that fits inside the class region (50% of max depth).
2. **Error extent:** Within the deepest error pixel's class-constrained
   circle, measure how far errors extend. Use 150% of the 95th percentile
   distance, capped by class depth.

This picks the largest useful brush — big enough to cover errors efficiently,
small enough to stay inside the class.

### Safe-zone erosion

The GT class mask is eroded by the brush radius using a circular structuring
element. The eroded mask is the "walkable" area — if the cursor center stays
on walkable pixels, the full brush disk stays inside the class.

Image borders are padded before erosion (edge mode) so the brush can paint
right up to image edges — only class boundaries cause erosion.

### Pathfinding loop

For each error type (false negatives → paint FG, false positives → paint BG):

1. Find nearest uncovered error pixel to current mouse position
2. A* shortest path from mouse to target on the walkable grid
3. Subsample the path to safe waypoints (see below)
4. Paint capsule strokes between consecutive waypoints
5. Mark brush coverage at each waypoint, update uncovered mask
6. Repeat until all errors are covered

If A* finds no path (error unreachable with current brush), that pixel is
skipped and a smaller brush pass will try again.

### Multi-pass brush sizing

Start with the ideal brush radius, halve it after each pass. Smaller brushes
can reach errors closer to class boundaries (less erosion required). The loop
continues until brush radius < 1 or all errors are covered.

### Safe waypoints

The A* path (one point per pixel) is subsampled to waypoints that are safe
to connect with straight capsule strokes. Two passes:

1. **Coarse pass** — space waypoints at the maximum spacing the user's hand
   allows: `max_spacing = MOUSE_PAINT_SPEED / MOUSE_POLL_HZ` (~3.3px).
   This models the physical limit of cursor movement.

2. **Safety refinement** — for each pair of consecutive coarse waypoints,
   check if the straight line between them stays on the walkable mask. If
   any pixel along the line is non-walkable, bisect using the midpoint from
   the original A* path and check both halves recursively. This adds
   waypoints only where geometry requires it — curves near class boundaries.

The result models a user who moves fast on straight sections (few mouse
samples, wide spacing) and slows down on curves near boundaries (many
mouse samples, tight spacing). The adaptive refinement guarantees zero
spill without artificial clipping.

### Why this guarantees zero spill

The walkable mask = GT class eroded by brush radius R. If the straight line
between two waypoints stays on walkable pixels, then at every point along
that line the full brush disk (radius R) is contained within the GT class.
The capsule stroke fills exactly this swept region. Safety refinement ensures
every straight segment satisfies this property.

### Assessment pause

After all painting is complete, the user briefly scans the image to confirm
no errors remain (200ms). This is grounded in scene gist research: humans
recognise the gist of a visual scene at >80% accuracy after just 36ms of
processing (Larson & Loschky, 2014). Esports research shows competitive
players make two-choice decisions ~89ms faster than novices, with advantages
driven by faster stimulus encoding (Hyde & von Bastian, Sheffield, 2024).
For a trained annotator doing a confirmatory scan of a familiar scene they
just painted, 200ms provides comfortable margin.

## Initial Phase

The initial phase uses a separate simulated user (sim_user.py) that draws
mouse strokes on foreground and background regions of the ground truth mask,
without any model prediction. This mimics a real user painting sparse examples
on clear areas of a new image.

The minority class is annotated first, then the majority class up to 10x the
minority annotation. The simulation uses speed-jitter coupling: faster painting
produces more aim jitter, and the user picks the fastest speed where jitter
fits within available margin.

## Training Loop

The benchmark processes images sequentially. Training begins after
`min_initial_images` (default 2) are annotated.

For each image:

1. **Phase decision.** If fewer than `min_initial_images` have been
   annotated, or the model's prediction confidence is below threshold,
   the image receives an initial annotation. Once confidence is sufficient,
   the benchmark switches to corrective mode.

2. **Annotation.** In initial mode, a coverage-based annotation is created.
   In corrective mode, the A* annotator corrects all errors. If no errors
   exist, the image is skipped.

3. **Train/val split.** The annotation is saved to train/ or val/ following
   the painter's allocation logic (approximately 5:1 train:val ratio).

4. **Training.** The model trains for `epochs_between_images` epochs on all
   annotations accumulated so far. After each epoch, validation F1 is
   computed. If improved, the model state is saved as the new best checkpoint.

## Annotation Format

Annotations are RGBA arrays (H, W, 4) matching RootPainter's format:
- Channel 0: foreground (value 255 where annotated)
- Channel 1: background (value 255 where annotated)
- Channels 2-3: unused

`paint(annot, center, radius, channel)` stamps a filled circular disk.
`paint_stroke(annot, from_pt, to_pt, radius, channel)` draws a capsule
(line with round caps) between two points — matching QPainter.drawLine.

## Trajectory Format

The trajectory is a list of event dicts, one per mouse event:

```python
{
    'r': int,           # row position
    'c': int,           # column position
    'painting': bool,   # True = mouse-down (painting), False = mouse-up
    'channel': int,     # 0 = FG, 1 = BG, -1 = not painting
    'brush_radius': int,# brush radius (0 when not painting)
    'dt': float,        # simulated time for this event (seconds)
}
```

For painting events, dt = 1 / MOUSE_POLL_HZ (≈8.3ms). For mouse-up travel,
dt is derived from visual search time + physical travel time, distributed
across interpolated cursor positions. Total annotation time =
`sum(e['dt'] for e in trajectory)`.

## Verification

`test_annotator.py --astar` runs the A* annotator on all saved test cases
and checks:
- **Coverage:** all error pixels are covered (100% FN and FP coverage)
- **Zero spill:** no annotation pixels land on the wrong GT class
- **Corrected F1 = 1.000:** prediction + annotation perfectly matches GT

Video output via `make_video.py` renders frames with four panels (image,
annotation, segmentation, corrected) and chart strips (confidence, val F1,
seg F1, corrected F1) with phase transition and best-model markers.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MOUSE_POLL_HZ` | 120 Hz | OS mouse event rate (measured on macOS 120Hz display) |
| `MOUSE_PAINT_SPEED` | 400 px/s | Maximum cursor speed during painting |
| `ASSESSMENT_PAUSE_MS` | 200 ms | Post-annotation visual confirmation scan |
| `VISUAL_SEARCH_MS` | 200 ms | Find next error and decide to go there (~190ms for trained gamers) |
| `MOUSE_TRAVEL_SPEED` | 800 px/s | Mouse-up repositioning speed (straight line) |
| `min_initial_images` | 2 | Minimum images before training starts |
| `epochs_between_images` | 2 | Training epochs after each annotation |
| `batch_size` | 6 | Training batch size |
| `in_w` / `out_w` | 572 / 500 | UNet input/output tile size |
| `lr` | 0.01 | SGD learning rate (momentum 0.99, Nesterov) |

## References

- Larson, A. M. & Loschky, L. C. (2014). "The Spatiotemporal Dynamics of
  Scene Gist Recognition." Journal of Experimental Psychology: Human
  Perception and Performance. Scene gist recognised at >80% accuracy after
  36ms of uninterrupted processing.
  https://www.k-state.edu/psych/vcl/basic-research/scene-gist.html

- Hyde, E. & von Bastian, C. (2024). University of Sheffield. CS:GO experts
  outperform novices by ~89ms in two-choice decision tasks; advantage driven
  by faster stimulus encoding and evidence accumulation.
  https://osf.io/4kuqc
