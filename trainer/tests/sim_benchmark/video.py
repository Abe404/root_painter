"""
Verification video for simulated corrective annotation benchmark.

Creates a GIF showing each image with overlays for ground truth,
annotation, and prediction, so you can visually verify the sim user.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Overlay colours (RGBA) — matching RootPainter's actual colours
FG_ANNOT = (255, 0, 0, 180)       # red, same as painter
BG_ANNOT = (0, 255, 0, 180)       # green, same as painter
PRED_FG = (0, 255, 255, 179)      # cyan, same as trainer seg output


def _get_font(size=14):
    """Load a good font, falling back to default."""
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def gt_contour(gt, thickness=2):
    """Return boolean mask of GT boundary pixels."""
    from scipy.ndimage import binary_dilation, binary_erosion
    outer = binary_dilation(gt, iterations=thickness) & ~gt.astype(bool)
    inner = gt.astype(bool) & ~binary_erosion(gt, iterations=thickness).astype(bool)
    return outer | inner


def clean_prediction(pred):
    """Remove noise from prediction for cleaner display."""
    from scipy.ndimage import binary_opening, label
    from sim_benchmark.brush import disk
    # Morphological opening removes small noise blobs
    opened = binary_opening(pred, structure=disk(3))
    # Then keep only large connected components
    labeled, num = label(opened)
    clean = np.zeros_like(pred)
    for i in range(1, num + 1):
        if np.sum(labeled == i) >= 50:
            clean[labeled == i] = 1
    return clean


def darken(rgb, factor=0.4):
    """Darken an RGB image by a factor."""
    return (rgb.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)


def overlay(base_rgb, mask, colour):
    """Composite a coloured mask onto an RGB image, return RGB array."""
    base = Image.fromarray(base_rgb).convert('RGBA')
    ov = Image.new('RGBA', base.size, (0, 0, 0, 0))
    ov_arr = np.array(ov)
    ov_arr[mask] = colour
    ov = Image.fromarray(ov_arr)
    return np.array(Image.alpha_composite(base, ov).convert('RGB'))


def add_label(img_arr, text, position=(8, 8), size=14, color=(255, 255, 255)):
    """Burn a text label with outline onto an image array."""
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    font = _get_font(size)
    x, y = position
    # black outline for readability
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=color, font=font)
    return np.array(img)


def draw_f1_bar(img_arr, f1, y_offset=None):
    """Draw a horizontal F1 progress bar at the bottom of the image."""
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    h, w = img_arr.shape[:2]
    bar_h = 6
    if y_offset is None:
        y_offset = h - bar_h - 4
    bar_w = w - 16
    x0 = 8

    # Background bar (dark)
    draw.rectangle([x0, y_offset, x0 + bar_w, y_offset + bar_h],
                   fill=(40, 40, 40))
    # Fill bar — color shifts from red to green as F1 improves
    fill_w = max(1, int(bar_w * min(1.0, f1)))
    r = int(255 * (1 - f1))
    g = int(255 * f1)
    draw.rectangle([x0, y_offset, x0 + fill_w, y_offset + bar_h],
                   fill=(r, g, 80))
    return np.array(img)


def make_frame(rgb_image, ground_truth, annot, prediction, stem, phase,
               val_f1):
    """Create a composite frame for one benchmark image.

    Always 3 panels: [Image] [Image + Annotation] [Image + Segmentation]
    """
    h, w = ground_truth.shape
    rgb = np.array(Image.fromarray(rgb_image).resize((w, h)))

    # Panel 1: plain image
    panel1 = rgb.copy()
    panel1 = add_label(panel1, stem)

    # Panel 2: image + annotation overlay
    panel2 = rgb.copy()
    if annot is not None:
        fg_mask = annot[:, :, 0] > 0
        bg_mask = annot[:, :, 1] > 0
        panel2 = overlay(panel2, fg_mask, FG_ANNOT)
        panel2 = overlay(panel2, bg_mask, BG_ANNOT)
    label2 = phase
    if annot is None:
        label2 += " (skipped)"
    panel2 = add_label(panel2, label2)

    # Panel 3: image + segmentation overlay
    panel3 = rgb.copy()
    if prediction is not None:
        panel3 = overlay(panel3, prediction.astype(bool), PRED_FG)
        panel3 = add_label(panel3, f"seg  F1={val_f1:.3f}")
    else:
        panel3 = add_label(panel3, "no seg yet")

    # Dark separator between panels
    sep = np.full((h, 3, 3), 30, dtype=np.uint8)
    return np.concatenate([panel1, sep, panel2, sep, panel3], axis=1)


def make_training_frame(num_annotated, f1_before, f1_after, panel_height,
                        panel_width, num_panels=2):
    """Create a dark transition frame showing training status between images.

    Args:
        num_annotated: total annotations so far
        f1_before: val F1 before this training round
        f1_after: val F1 after this training round
        panel_height: height of annotation panels (for matching)
        panel_width: width of a single panel
        num_panels: 2 for initial phase, 3 for corrective
    """
    # Match total width: panels + separators
    total_w = panel_width * num_panels + 3 * (num_panels - 1)
    frame = np.zeros((panel_height, total_w, 3), dtype=np.uint8)
    frame[:] = 20  # dark background

    # Center text vertically
    cy = panel_height // 2
    cx = total_w // 2

    # Training status
    frame = add_label(frame, f"Training on {num_annotated} annotations...",
                      position=(cx - 120, cy - 30), size=16)

    # F1 change
    improved = f1_after > f1_before
    f1_color = (80, 255, 80) if improved else (255, 255, 255)
    f1_text = f"F1: {f1_before:.3f} -> {f1_after:.3f}"
    frame = add_label(frame, f1_text,
                      position=(cx - 80, cy + 5), size=16, color=f1_color)

    # F1 bar at bottom
    frame = draw_f1_bar(frame, f1_after)

    return frame


CURSOR_COLOR = (255, 255, 255)
CURSOR_OUTLINE = (0, 0, 0)
CURSOR_FG = (255, 80, 80)
CURSOR_BG = (80, 255, 140)


def draw_cursor(img_arr, r, c, painting, brush_radius, channel=-1):
    """Draw a cursor circle on the image. Colored when painting, crosshair when not."""
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    if painting:
        radius = max(3, brush_radius)
        bbox = [c - radius, r - radius, c + radius, r + radius]
        color = CURSOR_FG if channel == 0 else CURSOR_BG
        draw.ellipse(bbox, outline=color, width=2)
        # Inner glow
        if radius > 4:
            inner = [c - radius + 2, r - radius + 2,
                     c + radius - 2, r + radius - 2]
            draw.ellipse(inner, outline=(*color[:3], 100), width=1)
    else:
        # Crosshair for mouse-up
        draw.ellipse([c - 4, r - 4, c + 4, r + 4],
                     outline=CURSOR_OUTLINE, width=1)
        draw.line([c - 6, r, c + 6, r], fill=CURSOR_COLOR, width=1)
        draw.line([c, r - 6, c, r + 6], fill=CURSOR_COLOR, width=1)
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

    Returns:
        (frames, end_time) — list of frame arrays, and final simulated time
    """
    from sim_benchmark.brush import paint, new_annot

    if not trajectory:
        return [], time_offset

    h, w = ground_truth.shape
    rgb = np.array(Image.fromarray(rgb_image).resize((w, h)))

    annot = new_annot(h, w)
    frames = []
    sep = np.full((h, 3, 3), 30, dtype=np.uint8)

    frame_interval = 1.0 / fps
    sim_time = time_offset
    next_frame_time = time_offset  # render first frame immediately

    last_r, last_c = trajectory[0]['r'], trajectory[0]['c']
    last_painting = trajectory[0]['painting']
    last_brush_radius = trajectory[0].get('brush_radius', 0)
    last_channel = trajectory[0].get('channel', -1)

    for entry in trajectory:
        r, c = entry['r'], entry['c']
        dt = entry.get('dt', 0.0)
        sim_time += dt

        if max_time is not None and sim_time > max_time:
            break

        if entry.get('painted', entry['painting']):
            paint(annot, (r, c), entry['brush_radius'], entry['channel'])

        last_r, last_c = r, c
        last_painting = entry['painting']
        last_brush_radius = entry.get('brush_radius', 0)
        last_channel = entry.get('channel', -1)

        if sim_time < next_frame_time:
            continue
        next_frame_time = sim_time + frame_interval

        # Panel 1: plain image
        panel1 = rgb.copy()
        panel1 = add_label(
            panel1,
            f"image {image_index + 1}  t={sim_time:.1f}s  {phase}")

        # Panel 2: image + current annotation + cursor
        panel2 = rgb.copy()
        fg_mask = annot[:, :, 0] > 0
        bg_mask = annot[:, :, 1] > 0
        if np.any(fg_mask):
            panel2 = overlay(panel2, fg_mask, FG_ANNOT)
        if np.any(bg_mask):
            panel2 = overlay(panel2, bg_mask, BG_ANNOT)
        panel2 = draw_cursor(panel2, last_r, last_c, last_painting,
                              last_brush_radius, last_channel)
        panel2 = add_label(panel2, stem)

        # Panel 3: image + segmentation overlay
        panel3 = rgb.copy()
        if prediction is not None:
            panel3 = overlay(panel3, prediction.astype(bool), PRED_FG)
            panel3 = add_label(panel3, f"seg  F1={val_f1:.3f}")
        else:
            panel3 = add_label(panel3, "no seg yet")

        frames.append(np.concatenate([panel1, sep, panel2, sep, panel3], axis=1))

    return frames, sim_time


def save_frames(frames, output_dir):
    """Save frames as individual PNGs for stepping through in a viewer.

    Returns:
        list of saved file paths
    """
    import os
    import shutil
    frames_dir = os.path.join(output_dir, 'frames')
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)
    paths = []
    for i, frame in enumerate(frames):
        path = os.path.join(frames_dir, f'frame_{i:04d}.png')
        Image.fromarray(frame).save(path)
        paths.append(path)
    return paths


def save_gif(frames, output_path, fps=6, scale=2):
    """Save frames as an animated GIF.

    Args:
        frames: list of (H, W, 3) uint8 arrays
        output_path: path for the .gif file
        fps: playback speed
        scale: upscale factor for crisp pixels
    """
    if not frames:
        return
    pil_frames = []
    for f in frames:
        img = Image.fromarray(f)
        if scale != 1:
            img = img.resize((img.width * scale, img.height * scale),
                             Image.NEAREST)
        pil_frames.append(img)

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=False,
    )
