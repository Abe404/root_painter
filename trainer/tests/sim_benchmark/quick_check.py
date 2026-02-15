"""Quick visual check — 30 synthetic images with async training.

Run:          python -m sim_benchmark.quick_check
Resume:       python -m sim_benchmark.quick_check --resume
"""
import sys, os
import copy
import shutil
import tempfile
import threading
import numpy as np
import torch
from PIL import Image

this_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.dirname(this_dir)
src_dir = os.path.join(os.path.dirname(tests_dir), 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, tests_dir)

import model_utils
import im_utils
from unet import UNetGNRes
from datasets import TrainDataset
from multi_epoch.multi_epoch_loader import MultiEpochsDataLoader
from metrics import get_metrics
from skimage.io import imsave

from sim_benchmark.sim_user import initial_annotation, corrective_annotation
from sim_benchmark.video import render_trajectory_frames
from sim_benchmark.benchmark import get_annot_target_dir


IN_W = 172
OUT_W = 100
BATCH_SIZE = 2
MIN_EPOCH_TILES = 10
CORRECTIVE_CONFIDENCE_THRESHOLD = 0.8
MIN_INITIAL_IMAGES = 2

CHECKPOINT_PATH = os.path.join(this_dir, 'quick_output', 'checkpoint.pt')


def make_images(num_images=30, size=200):
    """Random ellipses on noisy backgrounds."""
    rng = np.random.RandomState(42)
    y, x = np.ogrid[:size, :size]
    images = []
    for i in range(num_images):
        cx, cy = rng.randint(40, 160, 2)
        rx, ry = rng.randint(20, 60, 2)
        gt = (((x - cx) / rx)**2 + ((y - cy) / ry)**2 <= 1).astype(np.uint8)
        rgb = rng.randint(20, 50, (size, size, 3), dtype=np.uint8)
        rgb[gt == 1] = rng.randint(160, 240, (np.sum(gt), 3), dtype=np.uint8)
        images.append((rgb, gt, f'img_{i:02d}'))
    return images


def save_frame(frame, frames_dir, idx):
    """Save a single frame immediately."""
    path = os.path.join(frames_dir, f'frame_{idx:04d}.png')
    Image.fromarray(frame).save(path)


class Trainer(threading.Thread):
    """Background trainer that runs continuously, just like the real trainer."""

    def __init__(self, model, optimizer, train_annot_dir, val_annot_dir,
                 dataset_dir):
        super().__init__(daemon=True)
        self.model = model
        self.optimizer = optimizer
        self.train_annot_dir = train_annot_dir
        self.val_annot_dir = val_annot_dir
        self.dataset_dir = dataset_dir
        self.lock = threading.Lock()
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.best_f1 = 0.0
        self.epoch_count = 0
        self.running = True

    def get_best_model_state(self):
        with self.lock:
            return copy.deepcopy(self.best_model_state), self.best_f1

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            has_train = any(f.endswith('.png')
                           for f in os.listdir(self.train_annot_dir))
            has_val = any(f.endswith('.png')
                         for f in os.listdir(self.val_annot_dir))
            if not (has_train and has_val):
                continue

            train_set = TrainDataset(self.train_annot_dir, self.dataset_dir,
                                     IN_W, OUT_W,
                                     min_epoch_tiles=MIN_EPOCH_TILES)
            train_loader = MultiEpochsDataLoader(
                train_set, BATCH_SIZE, shuffle=True,
                num_workers=0, drop_last=False, pin_memory=False)

            train_result = model_utils.epoch(
                self.model, train_loader, BATCH_SIZE,
                optimizer=self.optimizer, step_callback=None, stop_fn=None)
            if train_result is None:
                continue

            self.epoch_count += 1
            self.model.eval()
            val_metrics = model_utils.get_val_metrics(
                self.model, self.val_annot_dir, self.dataset_dir,
                IN_W, OUT_W, bs=BATCH_SIZE)
            val_f1 = val_metrics['f1']

            with self.lock:
                if not np.isnan(val_f1) and val_f1 > self.best_f1:
                    self.best_f1 = val_f1
                    self.best_model_state = copy.deepcopy(
                        self.model.state_dict())
                    print(f"  [trainer] epoch {self.epoch_count} "
                          f"new best f1={val_f1:.3f}")


def segment(model_state, rgb):
    """Segment an image. Returns (binary_mask, confidence).

    Confidence = fraction of pixels where the model is sure (prob > 0.8
    or prob < 0.2), i.e. not fuzzy.
    """
    model = UNetGNRes()
    model.load_state_dict(model_state)
    model.to(model_utils.device)
    model.eval()
    image, pad_settings = im_utils.pad_to_min(rgb, min_w=IN_W, min_h=IN_W)
    with torch.no_grad():
        probs = model_utils.unet_segment(
            model, image, 2, IN_W, OUT_W, threshold=None)
    probs = im_utils.crop_from_pad_settings(probs, pad_settings)
    pred = (probs > 0.5).astype(np.uint8)
    confident = (probs > 0.8) | (probs < 0.2)
    confidence = float(np.mean(confident))
    return pred, confidence


def save_checkpoint(trainer, model, optimizer, dataset_dir,
                    train_annot_dir, val_annot_dir,
                    sim_time, frame_idx, num_annotated, start_image):
    """Save state so we can resume from the corrective phase."""
    best_state, best_f1 = trainer.get_best_model_state()
    ckpt = {
        'best_model_state': best_state,
        'best_f1': best_f1,
        'model_state': copy.deepcopy(model.state_dict()),
        'optimizer_state': copy.deepcopy(optimizer.state_dict()),
        'epoch_count': trainer.epoch_count,
        'sim_time': sim_time,
        'frame_idx': frame_idx,
        'num_annotated': num_annotated,
        'start_image': start_image,
        # Save annotation + dataset file contents
        'dataset_files': {},
        'train_annot_files': {},
        'val_annot_files': {},
    }
    for d, key in [(dataset_dir, 'dataset_files'),
                   (train_annot_dir, 'train_annot_files'),
                   (val_annot_dir, 'val_annot_files')]:
        for fname in os.listdir(d):
            path = os.path.join(d, fname)
            with open(path, 'rb') as f:
                ckpt[key][fname] = f.read()

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(ckpt, CHECKPOINT_PATH)
    print(f"  Checkpoint saved: {CHECKPOINT_PATH}")


def load_checkpoint(dataset_dir, train_annot_dir, val_annot_dir):
    """Restore state from checkpoint."""
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
    for d, key in [(dataset_dir, 'dataset_files'),
                   (train_annot_dir, 'train_annot_files'),
                   (val_annot_dir, 'val_annot_files')]:
        for fname, data in ckpt[key].items():
            with open(os.path.join(d, fname), 'wb') as f:
                f.write(data)
    return ckpt


def main():
    resume = '--resume' in sys.argv

    out_dir = os.path.join(this_dir, 'quick_output')
    frames_dir = os.path.join(out_dir, 'frames')
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)

    print(f"Frames dir: {frames_dir}")
    print("Launch viewer to watch live:  python sim_benchmark/viewer.py\n")

    tmpdir = tempfile.mkdtemp()
    dataset_dir = os.path.join(tmpdir, 'dataset')
    train_annot_dir = os.path.join(tmpdir, 'annot', 'train')
    val_annot_dir = os.path.join(tmpdir, 'annot', 'val')
    os.makedirs(dataset_dir)
    os.makedirs(train_annot_dir)
    os.makedirs(val_annot_dir)

    model = UNetGNRes()
    model.to(model_utils.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                                momentum=0.99, nesterov=True)

    images = make_images()

    if resume and os.path.exists(CHECKPOINT_PATH):
        ckpt = load_checkpoint(dataset_dir, train_annot_dir, val_annot_dir)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        trainer = Trainer(model, optimizer, train_annot_dir, val_annot_dir,
                          dataset_dir)
        trainer.best_model_state = ckpt['best_model_state']
        trainer.best_f1 = ckpt['best_f1']
        trainer.epoch_count = ckpt['epoch_count']
        sim_time = ckpt['sim_time']
        frame_idx = 0  # fresh frames for viewer
        num_annotated = ckpt['num_annotated']
        start_image = ckpt['start_image']
        corrective_mode = True
        print(f"Resumed from checkpoint: image {start_image}, "
              f"f1={ckpt['best_f1']:.3f}, {ckpt['epoch_count']} epochs, "
              f"{num_annotated} annotated\n")
        trainer.start()
    else:
        trainer = Trainer(model, optimizer, train_annot_dir, val_annot_dir,
                          dataset_dir)
        sim_time = 0.0
        frame_idx = 0
        num_annotated = 0
        start_image = 0
        corrective_mode = False

    training_started = resume

    for i, (rgb, gt, name) in enumerate(images):
        if i < start_image:
            continue

        # Save image to disk for TrainDataset
        img_path = os.path.join(dataset_dir, name + '.jpg')
        if not os.path.exists(img_path):
            Image.fromarray(rgb).save(img_path)

        # Segment with best model so far
        best_state, best_f1 = trainer.get_best_model_state()
        pred, confidence = segment(best_state, rgb)

        # Decide phase — switch to corrective when the model is confident
        if num_annotated < MIN_INITIAL_IMAGES:
            phase = 'initial'
        elif not corrective_mode and (best_f1 == 0
                                      or confidence < CORRECTIVE_CONFIDENCE_THRESHOLD):
            phase = 'initial'
        else:
            if not corrective_mode:
                print(f"  Switching to corrective "
                      f"(f1={best_f1:.3f}, confidence={confidence:.2f})")
                save_checkpoint(trainer, model, optimizer, dataset_dir,
                                train_annot_dir, val_annot_dir,
                                sim_time, frame_idx, num_annotated, i)
            corrective_mode = True
            phase = 'corrective'

        # Annotate
        if phase == 'initial':
            annot, traj = initial_annotation(gt, seed=i + 1)
        else:
            annot, traj = corrective_annotation(gt, pred)
            if annot is None:
                print(f"{name}: no errors found, skipping")
                continue

        # BG annotation must never land on foreground
        bg_on_fg = np.any((annot[:, :, 1] > 0) & (gt == 1))
        assert not bg_on_fg, f"{name}: BG annotation on foreground!"

        total_time = sum(e.get('dt', 0) for e in traj)
        n_strokes = sum(1 for j, e in enumerate(traj)
                        if e['painting'] and (j == 0 or not traj[j-1]['painting']))

        # Render annotation frames
        frames, sim_time = render_trajectory_frames(
            rgb, gt, traj, pred, name, phase, best_f1,
            image_index=i, time_offset=sim_time, fps=6)
        for frame in frames:
            save_frame(frame, frames_dir, frame_idx)
            frame_idx += 1

        # Save annotation to train/val split
        target_dir = get_annot_target_dir(train_annot_dir, val_annot_dir)
        imsave(os.path.join(target_dir, name + '.png'), annot,
               check_contrast=False)
        split = 'val' if target_dir == val_annot_dir else 'train'
        num_annotated += 1

        print(f"{name}: {total_time:.1f}s sim, {n_strokes} strokes  "
              f"[{split}] {phase}  f1={best_f1:.3f}  conf={confidence:.2f}  "
              f"epochs={trainer.epoch_count}")

        # User clicks 'start training' after 2nd annotation
        if num_annotated >= 2 and not training_started:
            print("  [trainer] starting")
            trainer.start()
            training_started = True

    trainer.stop()
    if training_started:
        trainer.join(timeout=30)
    shutil.rmtree(tmpdir)

    _, final_f1 = trainer.get_best_model_state()
    print(f"\n{frame_idx} frames saved. Final f1={final_f1:.3f} "
          f"after {trainer.epoch_count} epochs.")


if __name__ == '__main__':
    main()
