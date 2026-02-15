"""
Verification video for simulated corrective annotation benchmark.

Creates a GIF showing each image with overlays for ground truth,
annotation, and prediction, so you can visually verify the sim user.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Overlay colours (RGBA)
GT_CONTOUR = (255, 255, 0, 160)    # yellow
FG_ANNOT = (255, 0, 0, 160)       # red
BG_ANNOT = (0, 255, 0, 160)       # green
PRED_FG = (0, 100, 255, 120)      # blue
ERROR_FN = (255, 0, 0, 140)       # red  (missed FG)
ERROR_FP = (0, 0, 255, 140)       # blue (false FG)


def gt_contour(gt):
    """Return boolean mask of GT boundary pixels (1px thick)."""
    from scipy.ndimage import binary_dilation
    dilated = binary_dilation(gt, iterations=1)
    return dilated & ~gt.astype(bool) | (gt.astype(bool) & ~binary_dilation(
        ~gt.astype(bool), iterations=1).astype(bool))


def overlay(base_rgb, mask, colour):
    """Composite a coloured mask onto an RGB image, return RGB array."""
    base = Image.fromarray(base_rgb).convert('RGBA')
    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    overlay_arr = np.array(overlay)
    overlay_arr[mask] = colour
    overlay = Image.fromarray(overlay_arr)
    return np.array(Image.alpha_composite(base, overlay).convert('RGB'))


def add_label(img_arr, text):
    """Burn a text label into the top-left of an image array."""
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()
    # black outline for readability
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            draw.text((5 + dx, 5 + dy), text, fill=(0, 0, 0), font=font)
    draw.text((5, 5), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def make_frame(rgb_image, ground_truth, annot, prediction, stem, phase,
               val_f1):
    """Create a composite frame for one benchmark image.

    Returns an RGB numpy array showing panels side-by-side:
      [Image + GT contour] [Image + Annotation] [Image + Prediction/Errors]

    The third panel is only shown for corrective phase.
    """
    h, w = ground_truth.shape
    # Ensure rgb is same size as GT (may differ if padded during segment)
    rgb = np.array(Image.fromarray(rgb_image).resize((w, h)))

    # Panel 1: image + GT contour
    contour = gt_contour(ground_truth)
    panel1 = overlay(rgb, contour, GT_CONTOUR)
    panel1 = add_label(panel1, f"#{stem}  GT contour")

    # Panel 2: image + annotation overlay
    panel2 = rgb.copy()
    if annot is not None:
        fg_mask = annot[:, :, 0] > 0
        bg_mask = annot[:, :, 1] > 0
        panel2 = overlay(panel2, fg_mask, FG_ANNOT)
        panel2 = overlay(panel2, bg_mask, BG_ANNOT)
    label2 = f"{phase}"
    if annot is None:
        label2 += " (skipped)"
    panel2 = add_label(panel2, label2)

    panels = [panel1, panel2]

    # Panel 3: prediction + errors (corrective phase only)
    if prediction is not None:
        fn_mask = (ground_truth == 1) & (prediction == 0)
        fp_mask = (ground_truth == 0) & (prediction == 1)
        panel3 = overlay(rgb, prediction.astype(bool), PRED_FG)
        panel3 = overlay(panel3, fn_mask, ERROR_FN)
        panel3 = overlay(panel3, fp_mask, ERROR_FP)
        panel3 = add_label(panel3, f"prediction  val_f1={val_f1:.3f}")
        panels.append(panel3)

    # Add a thin white separator between panels
    sep = np.full((h, 2, 3), 255, dtype=np.uint8)
    parts = []
    for j, p in enumerate(panels):
        if j > 0:
            parts.append(sep)
        parts.append(p)

    return np.concatenate(parts, axis=1)


CURSOR_COLOR = (255, 255, 255)
CURSOR_OUTLINE = (0, 0, 0)


def draw_cursor(img_arr, r, c, painting, brush_radius):
    """Draw a cursor circle on the image. Filled when painting, outline when not."""
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    radius = max(3, brush_radius) if painting else 4
    bbox = [c - radius, r - radius, c + radius, r + radius]
    if painting:
        draw.ellipse(bbox, outline=CURSOR_COLOR, width=2)
    else:
        # Small crosshair for mouse-up
        draw.ellipse(bbox, outline=CURSOR_OUTLINE, width=1)
        draw.line([c - 5, r, c + 5, r], fill=CURSOR_COLOR, width=1)
        draw.line([c, r - 5, c, r + 5], fill=CURSOR_COLOR, width=1)
    return np.array(img)


def trajectory_duration(trajectory):
    """Return total simulated time (seconds) of a trajectory."""
    return sum(e.get('dt', 0.0) for e in trajectory)


def render_trajectory_frames(rgb_image, ground_truth, trajectory,
                             prediction, stem, phase, val_f1,
                             image_index=0, time_offset=0.0, fps=6,
                             max_time=None):
    """Render trajectory playback frames at a fixed FPS based on simulated time.

    Every trajectory entry has a 'dt' field (seconds). Frames are rendered
    at fixed intervals (1/fps seconds of simulated time), so the video
    plays back at real speed when viewed at the given FPS.

    Args:
        rgb_image: (H, W, 3) uint8 RGB image
        ground_truth: (H, W) binary ground truth
        trajectory: list of mouse event dicts with 'dt' field
        prediction: (H, W) binary prediction or None
        stem: image name
        phase: 'initial' or 'corrective'
        val_f1: current best validation F1
        image_index: which image in the sequence (0-based)
        time_offset: simulated time (seconds) at the start of this image
        fps: frames per second to render
        max_time: stop rendering after this simulated time (seconds), or
                  None for no limit

    Returns:
        (frames, end_time) — list of frame arrays, and final simulated time
    """
    from sim_benchmark.brush import paint, new_annot

    if not trajectory:
        return [], time_offset

    h, w = ground_truth.shape
    rgb = np.array(Image.fromarray(rgb_image).resize((w, h)))
    contour = gt_contour(ground_truth)

    annot = new_annot(h, w)
    frames = []
    sep = np.full((h, 2, 3), 255, dtype=np.uint8)

    frame_interval = 1.0 / fps
    sim_time = time_offset
    next_frame_time = time_offset  # render first frame immediately

    last_r, last_c = trajectory[0]['r'], trajectory[0]['c']
    last_painting = trajectory[0]['painting']
    last_brush_radius = trajectory[0].get('brush_radius', 0)

    for entry in trajectory:
        r, c = entry['r'], entry['c']
        dt = entry.get('dt', 0.0)
        sim_time += dt

        if max_time is not None and sim_time > max_time:
            break

        if entry['painting']:
            paint(annot, (r, c), entry['brush_radius'], entry['channel'])

        last_r, last_c = r, c
        last_painting = entry['painting']
        last_brush_radius = entry.get('brush_radius', 0)

        if sim_time < next_frame_time:
            continue
        next_frame_time = sim_time + frame_interval

        # Panel 1: image + GT contour + time info
        panel1 = overlay(rgb, contour, GT_CONTOUR)
        panel1 = add_label(
            panel1,
            f"image {image_index + 1}  t={sim_time:.1f}s  {phase}")

        # Panel 2: current annotation + cursor
        panel2 = rgb.copy()
        fg_mask = annot[:, :, 0] > 0
        bg_mask = annot[:, :, 1] > 0
        if np.any(fg_mask):
            panel2 = overlay(panel2, fg_mask, FG_ANNOT)
        if np.any(bg_mask):
            panel2 = overlay(panel2, bg_mask, BG_ANNOT)
        panel2 = draw_cursor(panel2, last_r, last_c, last_painting,
                              last_brush_radius)
        panel2 = add_label(panel2, stem)

        panels = [panel1, sep, panel2]

        if prediction is not None:
            fn_mask = (ground_truth == 1) & (prediction == 0)
            fp_mask = (ground_truth == 0) & (prediction == 1)
            panel3 = overlay(rgb, prediction.astype(bool), PRED_FG)
            panel3 = overlay(panel3, fn_mask, ERROR_FN)
            panel3 = overlay(panel3, fp_mask, ERROR_FP)
            panel3 = add_label(panel3, f"prediction  val_f1={val_f1:.3f}")
            panels.extend([sep, panel3])

        frames.append(np.concatenate(panels, axis=1))

    return frames, sim_time


def save_frames(frames, output_dir):
    """Save frames as individual PNGs for stepping through in a viewer.

    Args:
        frames: list of (H, W, 3) uint8 arrays
        output_dir: directory to save PNGs (created if needed)

    Returns:
        list of saved file paths
    """
    import os
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        path = os.path.join(frames_dir, f'frame_{i:03d}.png')
        Image.fromarray(frame).save(path)
        paths.append(path)
    return paths
