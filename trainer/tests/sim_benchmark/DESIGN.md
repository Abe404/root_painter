# Simulated Annotator Design

## Purpose

Benchmark corrective annotation by simulating a human user following the
RootPainter protocol. Measures model quality (F1 on a held-out densely
annotated validation set) vs simulated annotation time. Using an external
validation set allows direct comparison to the dense annotation approach
(and potentially other annotation strategies).

## How a real user annotates in RootPainter

The user sees an image. They select a brush size, pick foreground or
background channel, then paint by clicking and dragging the mouse. A
stroke is one continuous mouse-down → drag → mouse-up motion, producing a
line of paint along the cursor path. The brush is round (QPainter
RoundCap). Between strokes the mouse is up (no paint).

### Initial annotation (no prediction available)

The user looks at the image and identifies foreground and background
regions visually. They paint foreground first, then background (order
doesn't matter — the annotation is only saved to disk at the end).

**Minority class (typically FG):** The user annotates the majority of the
minority class region. They avoid the boundaries — not agonizing over
getting right up to the edge, but also not missing large chunks. The
brush naturally keeps them away from edges since it would spill over. The
user picks the largest brush that fits comfortably and sweeps parallel
strokes to cover the region.

**Majority class (typically BG):** The user annotates as much as they can
following the same boundary-avoidance principle, but does not exceed ~10x
the pixel amount annotated for the minority class. They paint wherever
there is clear background — not artificially restricted to one area.

**Stroke efficiency:** The user does not waste time. If a single big
stroke covers the region, that's what they do — one fast, straight sweep.
They do not click many times when one stroke would suffice, and they do
not squiggle around in random directions. Strokes follow the natural
shape of the region being annotated.

### Switching from initial to corrective

The user switches from initial annotation to corrective annotation when
the model is producing confident, non-fuzzy predictions. Before that
point, initial annotation continues.

**Confidence criterion:** Rather than using an F1 threshold (which
requires ground truth and doesn't reflect what the user sees), the switch
is based on prediction confidence — the fraction of pixels where the
model's foreground probability is clearly high (>0.8) or clearly low
(<0.2). When confidence exceeds a threshold (default 0.8), the model is
producing crisp shapes rather than fuzzy noise, and corrective annotation
becomes meaningful. A model must also have completed at least one training
epoch (best_f1 > 0) before switching — no corrective on random weights.

### Corrective annotation (prediction visible)

The user sees the model's prediction overlaid on the image. They scan for
errors — places where the prediction disagrees with reality. For each
error region, they paint a corrective stroke in the appropriate channel
(FG where the model missed foreground, BG where the model hallucinated
foreground).

**What gets corrected:** All errors that a user could reasonably annotate.
A 10-pixel false positive in the middle of an empty background region is
trivial to fix — just a quick click. A 200-pixel missed root section is a
few confident strokes. The simulated user does not give up on small errors
just because they are small.

**What gets skipped:** Only boundary-ambiguous pixels — error pixels
within ~2px of the GT class edge, where the true boundary between
foreground and background is genuinely hard to pinpoint. A real user
wouldn't try to paint 1–2px slivers right on a complex boundary either.

**Brush size trade-off:** The user picks the largest brush they can use
without spilling onto the wrong class. The brush should be large enough
to cover the area efficiently, but small enough that the full brush disk
stays inside the correct class region when the cursor follows a safe path.

**Skipping:** If the prediction looks good (no clear errors), the image
is skipped entirely. No annotation is saved, and the image is not used in
training (model weights are not updated based on this image).

## Protocol

Two parallel processes run simultaneously:

**User process:** The user annotates images one after another without
stopping. They do not wait for training to finish.

**Trainer process:** Runs in the background. After 2 annotations are on
disk, training starts automatically. The trainer continuously trains on
all annotations currently on disk and updates the best model when
validation improves.

**What the user sees on each image:**

- Images 1–2: No model exists yet. User does initial annotation.
- Images 3, 4, ...: A model may or may not be ready yet (training takes
  time). If no useful model is available, the user continues with initial
  annotation. Once the model is approximately predicting the structure of
  interest, the user switches to corrective annotation.
- In corrective mode: the user sees the prediction, corrects all clear
  errors, or skips if no clear errors exist.

**Note for simulation:** Training runs asynchronously in a background
thread, matching the real tool's architecture. The trainer runs
continuously once started, and the annotation loop grabs the best model
state whenever it needs to segment a new image.

## What a stroke looks like

A stroke is a continuous mouse-down → drag → mouse-up motion. The OS
delivers mouse position events at a fixed polling rate (display refresh
rate on macOS — measured at ~120Hz on a 120Hz ProMotion display). Between
consecutive mouse events, the painting toolkit (Qt's QPainter) draws a
capsule shape — a line segment with round caps connecting the two cursor
positions, filled at the brush radius.

**Key properties:**
- The cursor position is sampled at the display refresh rate (~120Hz)
- If the user moves fast, consecutive samples are farther apart but the
  capsule stroke still produces continuous coverage
- If they move slowly, samples are closer together (more precision on
  curves near boundaries)
- The user cannot physically move the cursor faster than their hand allows
  (~400 px/s for corrective annotation)

**Brush size:** Fixed within a stroke (may vary between strokes).

**Efficiency principle:** The user is efficient. Complex stroke patterns
represent MORE work, not better annotation. The simulated user should not
do unnecessarily complex strokes unless the geometry of the region
requires it.

## Simulated time model

Each mouse event corresponds to one OS polling interval:

- **Painting events:** dt = 1 / MOUSE_POLL_HZ (≈8.3ms at 120Hz). The
  spacing between consecutive waypoints determines the implied cursor
  speed — closer waypoints = slower movement (curves near boundaries),
  wider spacing = faster movement (straight sections).
- **Mouse-up travel:** Decomposed into visual search time (200ms to
  find the next target and decide to go there) plus physical travel
  time (distance / 800 px/s). Short repositions are fast (~225ms for
  20px), long ones take proportionally longer.
- **Assessment pause:** After all painting is complete, a brief pause
  (200ms) for the user to scan the image and confirm no errors remain.

### Mouse polling rate

macOS coalesces mouse events to the display refresh rate regardless of
mouse hardware polling rate (even gaming mice at 500–1000Hz). Measured
~119Hz on a 120Hz ProMotion display using `measure_mouse_rate.py`. Qt
receives one `mouseMoveEvent` per display frame.

### Assessment pause timing

The 200ms assessment pause is grounded in scene gist research: humans
recognise the gist of a scene at >80% accuracy after just 36ms of
processing (Larson & Loschky, K-State Visual Cognition Lab). Esports
players (CS:GO) make two-choice decisions ~89ms faster than novices
(Hyde & von Bastian, University of Sheffield; https://osf.io/4kuqc).
For a trained annotator scanning a familiar image they just finished
painting, 200ms provides comfortable margin for a confirmatory visual
sweep plus the decision to move on.

## Implementation: A* pathfinding annotator

The corrective annotator uses A* pathfinding to navigate error regions.
The ground truth class mask defines walkable terrain — the brush path
cannot spill onto the wrong class.

### Algorithm

1. **Pick brush radius** from the GT class distance transform (50% of
   max depth), refined by how far errors extend within the class region.
2. **Erode GT mask** by brush radius so the full brush disk stays inside
   the correct class at every point along the path.
3. **A* to nearest uncovered error pixel**, paint along the path using
   waypoints, repeat until all errors are covered.
4. **Halve brush radius** and repeat for errors the big brush couldn't
   reach (too close to class boundary for the large brush to fit).

### Safe waypoints

The A* path is subsampled to waypoints that are safe to connect with
straight capsule strokes (matching Qt's QPainter.drawLine with round
caps). This happens in two passes:

1. **Coarse pass:** Space waypoints at max mouse speed
   (MOUSE_PAINT_SPEED / MOUSE_POLL_HZ ≈ 3.3px at 400px/s, 120Hz).
   This is the widest spacing the user's hand physically allows.
2. **Safety refinement:** For each pair of consecutive waypoints, check
   if the straight line between them stays on the walkable (eroded) mask.
   If not, bisect using the original A* path and check both halves.
   Repeat until all straight connections are safe.

This models a user who moves fast on straight sections and slows down
on curves near class boundaries — more mouse samples where precision
matters, fewer where it doesn't.

### Why waypoints guarantee no spill

The walkable mask is the GT class eroded by brush radius. If the straight
line between two waypoints stays on this mask, then at every point along
that line, the full brush disk (radius R centered on the cursor) is
contained within the GT class region. The capsule stroke between waypoints
fills exactly this swept area — so no pixels land on the wrong class.

## Video / visualization

Four panels side by side for every frame:
- Panel 1: input image
- Panel 2: annotation built up so far (red = FG, green = BG) with cursor
- Panel 3: model segmentation (cyan overlay)
- Panel 4: corrected segmentation (prediction + annotation applied)

Charts below the panels track confidence, val F1, seg F1, and corrected
F1 over time, with phase transition markers (initial → corrective) and
best model save markers.

## Current issues (to resolve)

_None currently open._

## Resolved issues

- **BG/FG annotation landing on wrong class.** The brush could spill
  across the GT class boundary. Fixed with safe-zone erosion: erode
  gt_class_mask by disk(brush_radius) so the full brush disk stays
  inside the class at every point along the path.

- **Straight capsule strokes cutting corners on curves.** When waypoints
  were spaced only by mouse speed, the straight capsule between them
  could cut across non-walkable area on tight curves near class
  boundaries. Fixed with safety refinement: bisect the A* path and add
  waypoints wherever the straight connection leaves the walkable mask.
  This models the user slowing down (more mouse samples) on curves.

## References

- Larson, A. M. & Loschky, L. C. "The Spatiotemporal Dynamics of Scene
  Gist Recognition." Journal of Experimental Psychology: Human Perception
  and Performance, 2014. Scene gist recognised at >80% accuracy after
  36ms of processing. https://www.k-state.edu/psych/vcl/basic-research/scene-gist.html

- Hyde, E. & von Bastian, C. University of Sheffield, 2024. CS:GO
  experts outperform novices by ~89ms in two-choice decision tasks;
  faster stimulus encoding and evidence accumulation.
  https://osf.io/4kuqc
