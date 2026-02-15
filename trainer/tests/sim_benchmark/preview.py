"""Quick preview of annotation on a single synthetic image."""
import sys
import os
import numpy as np
from PIL import Image

this_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.dirname(this_dir)
src_dir = os.path.join(os.path.dirname(tests_dir), 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, tests_dir)

from sim_benchmark.sim_user import initial_annotation
from sim_benchmark.video import make_frame

# Synthetic image: circle on noisy background
size = 300
rng = np.random.RandomState(42)
y, x = np.ogrid[:size, :size]
cx, cy, r = 150, 150, 50
gt = ((x - cx) ** 2 + (y - cy) ** 2 <= r ** 2).astype(np.uint8)
rgb = rng.randint(20, 60, (size, size, 3), dtype=np.uint8)
rgb[gt == 1] = rng.randint(180, 240, (np.sum(gt), 3), dtype=np.uint8)

annot, traj = initial_annotation(gt, coverage=0.05, seed=1)
annot_time = sum(e.get('dt', 0.0) for e in traj)
dabs = sum(1 for e in traj if e['painting'])
print(f"Annotation: {annot_time:.1f}s, {len(traj)} events, {dabs} dabs")

frame = make_frame(rgb, gt, annot, None, 'synth_000', 'initial', 0.0)
out = os.path.join(this_dir, 'preview.png')
Image.fromarray(frame).save(out)
os.system(f'open {out}')
