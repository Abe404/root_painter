"""Render sim benchmark frames + chart strips into an mp4.

Standalone — reads existing frame PNGs and stats.csv, draws chart
strips below each frame, pipes to ffmpeg.

Usage:
    python -m sim_benchmark.make_video                    # default 18fps (3x)
    python -m sim_benchmark.make_video --fps 6            # real-time
    python -m sim_benchmark.make_video --fps 30            # 5x speed
"""
import csv
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _get_font(size=11):
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


FONT = _get_font()
BG = (26, 26, 46)  # matches viewer #1a1a2e


def load_stats(frames_dir):
    """Load stats.csv, return dict keyed by filename."""
    stats = {}
    path = os.path.join(frames_dir, 'stats.csv')
    if not os.path.exists(path):
        return stats
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            stats[row['filename']] = row
    return stats


def extract_series(frame_fnames, stats):
    """Extract chart series from stats aligned to frame list."""
    confidence = []
    val_f1 = []
    seg_f1 = []
    corrected_f1 = []
    phase_transitions = []
    best_markers = []
    best_so_far = -1.0
    prev_phase = None

    for i, fname in enumerate(frame_fnames):
        s = stats.get(fname)
        if not s:
            confidence.append(None)
            val_f1.append(None)
            seg_f1.append(None)
            corrected_f1.append(None)
            continue

        phase = s.get('phase')
        if phase != prev_phase and phase == 'corrective':
            phase_transitions.append(i)
        prev_phase = phase

        try:
            confidence.append(float(s['confidence']))
        except (KeyError, ValueError):
            confidence.append(None)
        try:
            v = float(s.get('val_f1', s.get('f1', '')))
            val_f1.append(v)
            if v > best_so_far:
                best_so_far = v
                best_markers.append(i)
        except (KeyError, ValueError):
            val_f1.append(None)
        try:
            seg_f1.append(float(s['seg_f1']))
        except (KeyError, ValueError):
            seg_f1.append(None)
        try:
            corrected_f1.append(float(s['corrected_f1']))
        except (KeyError, ValueError):
            corrected_f1.append(None)

    return {
        'confidence': confidence,
        'val_f1': val_f1,
        'seg_f1': seg_f1,
        'corrected_f1': corrected_f1,
        'phase_transitions': phase_transitions,
        'best_markers': best_markers,
    }


def draw_chart(width, height, values, cursor, label, color,
               markers=None, phase_markers=None):
    """Render a single chart strip as an RGB numpy array."""
    img = Image.new('RGB', (width, height), BG)
    draw = ImageDraw.Draw(img)

    ml, mr, mt, mb = 36, 4, 3, 12
    cw = width - ml - mr
    ch = height - mt - mb
    if cw < 2 or ch < 2:
        return np.array(img)

    n = len(values)
    valid = [v for v in values if v is not None]
    if not valid:
        return np.array(img)

    y_min, y_max = min(valid), max(valid)
    if y_max - y_min < 1e-6:
        y_min -= 0.05
        y_max += 0.05

    def to_x(i):
        return ml + int(i / max(1, n - 1) * cw) if n > 1 else ml + cw // 2

    def to_y(v):
        return mt + ch - int((v - y_min) / (y_max - y_min) * ch)

    # Grid lines + labels
    for frac in [0.0, 0.5, 1.0]:
        yv = y_min + frac * (y_max - y_min)
        yp = to_y(yv)
        draw.line([(ml, yp), (width - mr, yp)], fill=(51, 51, 51))
        draw.text((2, yp - 5), f"{yv:.2f}", fill=(136, 136, 136), font=FONT)

    # Best model markers
    if markers:
        for mi in markers:
            if 0 <= mi < n:
                mx = to_x(mi)
                for y in range(mt, height - mb, 4):
                    draw.line([(mx, y), (mx, min(y + 2, height - mb))],
                              fill=(85, 85, 85))

    # Phase transition markers
    if phase_markers:
        for mi in phase_markers:
            if 0 <= mi < n:
                mx = to_x(mi)
                for y in range(mt, height - mb, 6):
                    draw.line([(mx, y), (mx, min(y + 3, height - mb))],
                              fill=(255, 255, 0))

    # Data line
    prev = None
    for i, v in enumerate(values):
        if v is None:
            prev = None
            continue
        x1, y1 = to_x(i), to_y(v)
        if prev is not None:
            draw.line([prev, (x1, y1)], fill=color, width=1)
        prev = (x1, y1)

    # Cursor
    if 0 <= cursor < n:
        cx = to_x(cursor)
        draw.line([(cx, mt), (cx, height - mb)], fill=(255, 255, 255))

    # Label
    draw.text((ml + 4, mt), label, fill=color, font=FONT)

    return np.array(img)


def render_charts_strip(width, cursor, series, chart_height=50):
    """Render all charts stacked, return RGB numpy array."""
    charts = [
        ('confidence', series['confidence'], (0, 255, 255), None),
        ('val F1', series['val_f1'], (0, 255, 0), series['best_markers']),
        ('seg F1', series['seg_f1'], (136, 136, 255), None),
        ('corrected F1', series['corrected_f1'], (255, 136, 0), None),
    ]
    strips = []
    for label, values, color, markers in charts:
        strip = draw_chart(width, chart_height, values, cursor, label, color,
                           markers=markers,
                           phase_markers=series['phase_transitions'])
        strips.append(strip)
    return np.concatenate(strips, axis=0)


def main():
    fps = 18
    for i, arg in enumerate(sys.argv):
        if arg == '--fps' and i + 1 < len(sys.argv):
            fps = int(sys.argv[i + 1])

    this_dir = os.path.dirname(os.path.abspath(__file__))
    frames_dir = os.path.join(this_dir, 'quick_output', 'frames')
    output_path = os.path.join(this_dir, 'quick_output', 'output.mp4')

    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        frames_dir = sys.argv[1]
        output_path = os.path.join(os.path.dirname(frames_dir), 'output.mp4')

    # Load frames and stats
    frame_paths = sorted(
        f for f in os.listdir(frames_dir) if f.endswith('.png'))
    if not frame_paths:
        print(f"No frames in {frames_dir}")
        return

    stats = load_stats(frames_dir)
    series = extract_series(frame_paths, stats)

    # Probe frame size from first image
    first = np.array(Image.open(os.path.join(frames_dir, frame_paths[0])))
    frame_h, frame_w = first.shape[:2]
    chart_height = 50
    total_h = frame_h + chart_height * 4
    # Ensure even dimensions for h264
    out_w = frame_w + (frame_w % 2)
    out_h = total_h + (total_h % 2)

    print(f"{len(frame_paths)} frames, {fps}fps, "
          f"{out_w}x{out_h}px -> {output_path}")

    # Pipe raw frames to ffmpeg
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{out_w}x{out_h}', '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    for i, fname in enumerate(frame_paths):
        frame = np.array(Image.open(os.path.join(frames_dir, fname)))
        charts = render_charts_strip(frame_w, i, series, chart_height)
        composite = np.concatenate([frame, charts], axis=0)

        # Pad to even dimensions if needed
        if composite.shape[1] < out_w or composite.shape[0] < out_h:
            padded = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            padded[:composite.shape[0], :composite.shape[1]] = composite
            composite = padded

        proc.stdin.write(composite.tobytes())

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(frame_paths)}")

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        print(f"ffmpeg error (exit {proc.returncode})")
    else:
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        duration = len(frame_paths) / fps
        print(f"Done: {duration:.0f}s, {size_mb:.1f}MB")


if __name__ == '__main__':
    main()
