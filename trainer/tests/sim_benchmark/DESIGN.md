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
clear error region, they paint a corrective stroke in the appropriate
channel (FG where the model missed foreground, BG where the model
hallucinated foreground).

**What gets corrected:** All clear errors. Tiny ambiguous border issues
(1–3 pixels next to a boundary) may be left alone, since boundaries are
often genuinely unclear. For now, the simulation corrects all errors above
a small size threshold. Boundary ambiguity could be added later.

**Brush size trade-off:** The user picks the largest brush they can use
confidently at speed. The brush should be large enough to cover the area
efficiently (not many thin strokes in a big space), but small enough that
comfortable margin exists on both sides of the stroke path — the user
shouldn't need pixel-precise placement. Roughly: brush diameter is a
fraction of the safe zone width, leaving enough room for fast confident
strokes.

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
  interest, the user switches to corrective annotation. The exact
  threshold for "approximately predicting" may need experimentation.
- In corrective mode: the user sees the prediction, corrects all clear
  errors, or skips if no clear errors exist. For now, skipping is strict
  — only skip when there are truly no errors.

**Note for simulation:** Training runs asynchronously in a background
thread, matching the real tool's architecture. The trainer runs
continuously once started, and the annotation loop grabs the best model
state whenever it needs to segment a new image.

## What a stroke looks like

A stroke is a continuous line from point A to point B. It is NOT a
sequence of dabs or blobs. The cursor position is sampled at a fixed tick
interval — if the user moves fast, the line has fewer sampled x,y points
but is still a connected line that can be long. If they move slowly, more
points are sampled but the line is shorter.

**Stroke characteristics:**
- Roughly straight. Not zigzag, not squiggly.
- Fills empty space efficiently. The user does not squiggle around empty
  space for no reason. Draw a line that gets the job done and move on.
- A stroke should not go over itself in strange ways. If a single
  straight stroke covers the area, that's what the user does.
- In some cases a non-straight path may be needed (e.g. following a
  curved boundary), but the default is: simple, efficient, straight.

**Brush size:** Fixed within a stroke (may vary between strokes).

**Efficiency principle:** The user is efficient. Complex stroke patterns
represent MORE work, not better annotation. The simulated user should not
do unnecessarily complex strokes unless the geometry of the region
requires it.

## Simulated time model

Each mouse action has a simulated time cost:
- **Mouse-up travel** (repositioning between strokes): fast, straight line
- **Painting** (mouse-down drag): speed depends on how much margin the
  user has — fast in open areas, slow near edges
- **Direction changes** have a cost. A gentle curve or loop around a blob
  is low cost. Lots of back-and-forth or sharp precise angle changes is
  high cost — harder than just drawing a straight line. This naturally
  pushes the simulation toward long straight strokes, which is realistic.

**Duration-based model:** Rather than pixel-per-second speeds (which vary
with image resolution and zoom level), the simulation uses stroke
durations:
- FG stroke duration: ~1.4s (log-normal variation, spread 0.3)
- BG stroke duration: ~0.75s
- Inter-stroke gap: ~1.5s
- Jitter derived from effective speed (distance / duration) × jitter rate

## Video / visualization

Three panels side by side for every image:
- Panel 1: raw image
- Panel 2: FG/BG annotation shown over image
- Panel 3: segmentation (prediction) shown over image

Training transition frames between images showing F1 progress.

## Implementation approach

Current approach is rules-based: compute regions, pick brush size, plan
stroke paths geometrically. This has proven brittle — each edge case
needs a new rule, and rules interact in unexpected ways.

If the rules-based approach continues to struggle, alternatives to
consider:

- **Vector field / cost-benefit optimization:** Define a field over the
  image where each point has an annotation value (how much the model
  needs correction there) and a cost (proximity to boundaries, direction
  change penalty). Stroke paths follow high benefit-to-cost gradients.
  Naturally produces efficient straight strokes through valuable regions.
- **RL:** Agent learns a stroke placement policy given current state
  (image, GT, existing annotation). Reward = coverage per unit time.
  Could be combined with constraints learned from real user data.
- **Learned from real data:** If real annotation traces (mouse
  trajectories from actual RootPainter sessions) become available, use
  them to learn realistic stroke patterns or to constrain/validate the
  simulation.

For now: keep rules-based, but informed by this design document to avoid
the cycle of fix → break → fix.

## Current issues (to resolve)

_To be filled in as we identify specific problems._
