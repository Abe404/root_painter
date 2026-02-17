"""Fast annotation test harness — no model training required.

Loads saved test cases (image + GT + prediction PNGs) and runs the
annotator on each one, checking coverage and spill. Much faster
iteration than running the full quick_check pipeline.

Usage:
    python -m sim_benchmark.test_annotator               # test all cases
    python -m sim_benchmark.test_annotator --astar       # test A* annotator
    python -m sim_benchmark.test_annotator --cam         # test CAM annotator
    python -m sim_benchmark.test_annotator --save-frames # save trajectory frames

Test cases are PNG triples ({name}_image.png, _gt.png, _pred.png) in
test_cases/. They can be saved from the viewer (S key) during a
quick_check run, or created with save_test_case().
"""
import sys
import os
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.morphology import disk

this_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.dirname(this_dir)
src_dir = os.path.join(os.path.dirname(tests_dir), 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, tests_dir)

CASES_DIR = os.path.join(this_dir, 'test_cases')


def save_test_case(name, image, gt, pred):
    """Save a test case as PNGs for easy viewing."""
    os.makedirs(CASES_DIR, exist_ok=True)
    Image.fromarray(image).save(os.path.join(CASES_DIR, f'{name}_image.png'))
    Image.fromarray((gt * 255).astype(np.uint8)).save(
        os.path.join(CASES_DIR, f'{name}_gt.png'))
    Image.fromarray((pred * 255).astype(np.uint8)).save(
        os.path.join(CASES_DIR, f'{name}_pred.png'))
    print(f"  Saved test case: {CASES_DIR}/{name}_*.png")


def load_test_cases():
    """Load test cases from PNGs or legacy .npz files."""
    cases = []
    if not os.path.exists(CASES_DIR):
        return cases

    # Find PNG test cases: look for *_gt.png files
    seen_names = set()
    for fname in sorted(os.listdir(CASES_DIR)):
        if not fname.endswith('_gt.png'):
            continue
        name = fname.replace('_gt.png', '')
        image_path = os.path.join(CASES_DIR, f'{name}_image.png')
        gt_path = os.path.join(CASES_DIR, f'{name}_gt.png')
        pred_path = os.path.join(CASES_DIR, f'{name}_pred.png')
        if not (os.path.exists(image_path) and os.path.exists(pred_path)):
            continue
        image = np.array(Image.open(image_path))
        gt = (np.array(Image.open(gt_path)) > 127).astype(np.uint8)
        pred = (np.array(Image.open(pred_path)) > 127).astype(np.uint8)
        cases.append({'name': name, 'image': image, 'gt': gt, 'pred': pred})
        seen_names.add(name)

    # Legacy .npz test cases
    for fname in sorted(os.listdir(CASES_DIR)):
        if not fname.endswith('.npz'):
            continue
        name = fname.replace('.npz', '')
        if name in seen_names:
            continue
        path = os.path.join(CASES_DIR, fname)
        data = np.load(path)
        cases.append({
            'name': name,
            'image': data['image'],
            'gt': data['gt'],
            'pred': data['pred'],
        })
    return cases


def evaluate_annotation(annot, gt, pred):
    """Evaluate annotation quality — coverage, spill, corrected F1."""
    fn_mask = (gt == 1) & (pred == 0)
    fp_mask = (gt == 0) & (pred == 1)

    fg_annot = annot[:, :, 0] > 0
    bg_annot = annot[:, :, 1] > 0

    # Spill: annotation on wrong class
    fg_spill = int(np.sum(fg_annot & (gt == 0)))
    bg_spill = int(np.sum(bg_annot & (gt == 1)))

    # Coverage: what fraction of interior errors are covered?
    near_bg = binary_dilation(gt == 0, structure=disk(2))
    near_fg = binary_dilation(gt == 1, structure=disk(2))
    fn_interior = fn_mask & ~near_bg
    fp_interior = fp_mask & ~near_fg
    fn_covered = int(np.sum(fn_interior & fg_annot))
    fp_covered = int(np.sum(fp_interior & bg_annot))
    fn_total = int(np.sum(fn_interior))
    fp_total = int(np.sum(fp_interior))

    # Corrected prediction
    corrected = pred.copy()
    corrected[fg_annot] = 1
    corrected[bg_annot] = 0
    tp = int(np.sum((corrected == 1) & (gt == 1)))
    fp_c = int(np.sum((corrected == 1) & (gt == 0)))
    fn_c = int(np.sum((corrected == 0) & (gt == 1)))
    corrected_f1 = 2 * tp / max(1, 2 * tp + fp_c + fn_c)

    return {
        'fg_spill': fg_spill,
        'bg_spill': bg_spill,
        'fn_covered': fn_covered,
        'fn_total': fn_total,
        'fp_covered': fp_covered,
        'fp_total': fp_total,
        'corrected_f1': corrected_f1,
    }


def render_result(image, gt, pred, annot, trajectory):
    """Render a 3-panel visualization."""
    h, w = gt.shape
    panel_w = w
    canvas = np.zeros((h, panel_w * 3, 3), dtype=np.uint8)

    # Panel 1: image with GT contour
    canvas[:, :panel_w] = image
    from skimage.segmentation import find_boundaries
    boundary = find_boundaries(gt, mode='thick')
    canvas[:, :panel_w][boundary] = [255, 255, 0]

    # Panel 2: annotation overlay
    canvas[:, panel_w:2*panel_w] = image // 2
    fg = annot[:, :, 0] > 0
    bg = annot[:, :, 1] > 0
    canvas[:, panel_w:2*panel_w][fg] = [255, 60, 60]
    canvas[:, panel_w:2*panel_w][bg] = [60, 255, 60]

    # Panel 3: prediction with errors
    canvas[:, 2*panel_w:3*panel_w] = image // 2
    canvas[:, 2*panel_w:3*panel_w][pred == 1] = [100, 100, 255]
    fn = (gt == 1) & (pred == 0)
    fp = (gt == 0) & (pred == 1)
    canvas[:, 2*panel_w:3*panel_w][fn] = [255, 0, 0]
    canvas[:, 2*panel_w:3*panel_w][fp] = [0, 0, 255]

    return canvas


def main():
    use_cam = '--cam' in sys.argv
    use_astar = '--astar' in sys.argv
    save_frames = '--save-frames' in sys.argv

    if use_astar:
        from astar_annotator.annotator import corrective_annotation
        print("Testing A*-based annotator\n")
    elif use_cam:
        from sim_benchmark.sim_user_cam import corrective_annotation
        print("Testing CAM-based annotator\n")
    else:
        from sim_benchmark.sim_user import corrective_annotation
        print("Testing original annotator\n")

    cases = load_test_cases()
    if not cases:
        print(f"No test cases found in {CASES_DIR}/")
        print("Run quick_check first, or create cases with save_test_case()")
        # Generate synthetic cases for quick testing
        print("\nGenerating synthetic test cases...")
        from sim_benchmark.quick_check import make_images
        images = make_images(10)
        for i, (rgb, gt, name) in enumerate(images[:3]):
            # Simulate a shifted prediction
            pred = np.roll(gt, 15 + i * 5, axis=1)
            save_test_case(f'synthetic_{name}', rgb, gt, pred)
        cases = load_test_cases()

    out_dir = os.path.join(this_dir, 'test_output')
    frames_dir = os.path.join(out_dir, 'frames')
    if save_frames:
        import shutil
        from sim_benchmark.video import render_trajectory_frames
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir)
        frame_idx = 0
        sim_time_acc = 0.0

    for case in cases:
        name = case['name']
        gt = case['gt']
        pred = case['pred']
        image = case['image']

        annot, traj = corrective_annotation(gt, pred)
        if annot is None:
            print(f"{name}: no errors detected")
            continue

        n_strokes = sum(1 for j, e in enumerate(traj)
                        if e['painting'] and (j == 0 or not traj[j-1]['painting']))
        sim_time = sum(e.get('dt', 0) for e in traj)

        result = evaluate_annotation(annot, gt, pred)

        fn_pct = (100 * result['fn_covered'] / result['fn_total']
                  if result['fn_total'] > 0 else 100)
        fp_pct = (100 * result['fp_covered'] / result['fp_total']
                  if result['fp_total'] > 0 else 100)

        status = 'OK' if (result['fg_spill'] == 0 and result['bg_spill'] == 0
                          and fn_pct > 90 and fp_pct > 90) else 'FAIL'

        print(f"{name}: {n_strokes} strokes, {sim_time:.1f}s  "
              f"FN={result['fn_covered']}/{result['fn_total']} ({fn_pct:.0f}%)  "
              f"FP={result['fp_covered']}/{result['fp_total']} ({fp_pct:.0f}%)  "
              f"spill={result['fg_spill']}+{result['bg_spill']}  "
              f"corrF1={result['corrected_f1']:.3f}  [{status}]")

        if save_frames:
            # Seg F1: prediction vs ground truth
            tp = int(np.sum((pred == 1) & (gt == 1)))
            fp_c = int(np.sum((pred == 1) & (gt == 0)))
            fn_c = int(np.sum((pred == 0) & (gt == 1)))
            seg_f1 = 2 * tp / max(1, 2 * tp + fp_c + fn_c)

            frames, frame_times, frame_cf1s, sim_time_acc = \
                render_trajectory_frames(
                    image, gt, traj, pred, name, 'corrective', seg_f1,
                    time_offset=sim_time_acc, fps=6)
            for frame in frames:
                fname = f'frame_{frame_idx:04d}_{name}.png'
                Image.fromarray(frame).save(
                    os.path.join(frames_dir, fname))
                frame_idx += 1

    if save_frames:
        print(f"\n{frame_idx} frames saved to {frames_dir}/")


if __name__ == '__main__':
    main()
