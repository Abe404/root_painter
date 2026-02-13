"""
Test that training works without error and a reasonable validation accuracy is obtained.
This test is a bit more of a benchmark than a test.

Copyright (C) 2023 Abraham George Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import sys

# Add trainer/src to sys.path so we can import the project modules
test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(test_dir), 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, test_dir)

import json
import shutil
import subprocess
import tempfile
import torch
import numpy as np
import time
from datasets import TrainDataset
from multi_epoch.multi_epoch_loader import MultiEpochsDataLoader
from test_utils import log_metrics, dl_dir_from_zip
from metrics import get_metrics


# First 6 annotations (in project order) that contain both FG and BG.
# These are the initial "clear" annotations before the model made predictions.
# From zenodo.org/records/11236258 corrective roots project file.
ROOTS_CORRECTIVE_CLEAR_ANNOTS = [
    '16_06_21_12E10c_P6210535_000.png',   # val
    '16_08_22_12W10a_P8222612_000.png',   # train
    '16_07_18_11E10d_P7181739_000.png',   # train
    '16_06_21_10E8b_P6210600_000.png',    # train
    '16_07_18_11W2a_P7181921_000.png',    # train
    '16_08_22_10W8c_P8222514_000.png',    # train
]

# All paths relative to the test file location, not CWD
sync_dir = os.path.join(test_dir, 'test_rp_sync')
bp_annot_dir = os.path.join(sync_dir, 'projects', 'biopores_corrective_a', 'annotations')
dense_root_annot_dir = os.path.join(sync_dir, 'projects', 'roots_dense_a', 'annotations')
corrective_root_annot_dir = os.path.join(sync_dir, 'projects', 'roots_corrective_a', 'annotations')
dense_nodules_annot_dir = os.path.join(sync_dir, 'projects', 'nodules_dense_a', 'annotations')
datasets_dir = os.path.join(sync_dir, 'datasets')
bp_dataset_dir = os.path.join(datasets_dir, 'biopores_750_training')
root_dataset_dir = os.path.join(datasets_dir, 'towers_750_training')
nodules_dataset_dir = os.path.join(datasets_dir, 'nodules_750_training')
metrics_dir = os.path.join(test_dir, 'metrics')


def setup_function():
    print('running setup')
    if not os.path.isdir(datasets_dir):
        os.makedirs(datasets_dir)
    if not os.path.isdir(metrics_dir):
        os.makedirs(metrics_dir)

    biopore_url = 'https://zenodo.org/records/3754046/files/biopores_750_training.zip'
    dl_dir_from_zip(biopore_url, bp_dataset_dir)

    biopore_annot_url = 'https://zenodo.org/records/8041842/files/user_a_corrective_biopores_750_training_annotation.zip'
    dl_dir_from_zip(biopore_annot_url, bp_annot_dir)

    root_url = 'https://zenodo.org/records/3754046/files/towers_750_training.zip'
    dl_dir_from_zip(root_url, root_dataset_dir)

    dense_root_annot_url = 'https://zenodo.org/records/11236258/files/user_a_dense_roots_750_training_annotation.zip'
    dl_dir_from_zip(dense_root_annot_url, dense_root_annot_dir)

    corrective_root_annot_url = 'https://zenodo.org/records/11236258/files/user_a_corrective_roots_750_training_annotation.zip'
    dl_dir_from_zip(corrective_root_annot_url, corrective_root_annot_dir)

    nodules_url = 'https://zenodo.org/records/3754046/files/nodules_750_training.zip'
    dl_dir_from_zip(nodules_url, nodules_dataset_dir)

    dense_nodules_annot_url = 'https://zenodo.org/records/11236258/files/user_a_dense_nodules_750_training_annotation.zip'
    dl_dir_from_zip(dense_nodules_annot_url, dense_nodules_annot_dir)


def get_gpu_name():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True)
        return result.stdout.strip()
    except FileNotFoundError:
        if torch.backends.mps.is_available():
            return 'Apple MPS'
        return 'CPU'


def get_git_commit():
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True,
            cwd=os.path.join(test_dir, '..'))
        return result.stdout.strip()
    except FileNotFoundError:
        return 'unknown'


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def make_run_info(run, seed, model_name, optimizer_str,
                  in_w, out_w, batch_size, epochs, dataset_name):
    return {
        'run': run,
        'seed': seed,
        'model': model_name,
        'optimizer': optimizer_str,
        'in_w': in_w,
        'out_w': out_w,
        'batch_size': batch_size,
        'epochs': epochs,
        'dataset': dataset_name,
        'gpu': get_gpu_name(),
        'torch_version': torch.__version__,
        'git_commit': get_git_commit(),
    }


def run_is_complete(run_dir, expected_info):
    """Check if a run already completed with matching settings."""
    info_path = os.path.join(run_dir, 'run_info.json')
    val_path = os.path.join(run_dir, 'val.csv')
    if not os.path.isfile(info_path) or not os.path.isfile(val_path):
        return False
    with open(info_path) as f:
        existing_info = json.load(f)
    # check that all settings match (ignore timestamp)
    for key in expected_info:
        if existing_info.get(key) != expected_info[key]:
            return False
    # check that we have the expected number of epoch rows
    with open(val_path) as f:
        # subtract 1 for header row
        num_rows = sum(1 for _ in f) - 1
    if num_rows < expected_info['epochs']:
        return False
    return True


def save_run_info(run_dir, info):
    info_with_timestamp = dict(info)
    info_with_timestamp['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    info_path = os.path.join(run_dir, 'run_info.json')
    with open(info_path, 'w') as f:
        json.dump(info_with_timestamp, f, indent=2)


def make_filtered_annot_dir(annot_dir, filenames):
    """Create a temp directory with symlinks to only the specified annotation files."""
    tmp_dir = tempfile.mkdtemp(prefix='clear_annots_')
    for fname in filenames:
        src = os.path.join(annot_dir, fname)
        if os.path.exists(src):
            os.symlink(src, os.path.join(tmp_dir, fname))
    return tmp_dir


def training(dataset_dir, annot_dir, name, clear_annots=None,
             stage1_epochs=4, runs=5, epochs=60):
    """Run training benchmark: multiple runs, logging train and val metrics per epoch.

    If clear_annots is provided, runs two-stage training:
      Stage 1: train on only the clear annotations for stage1_epochs
      Stage 2: train on all annotations for remaining epochs
    """
    import model_utils
    from unet import UNetGNRes
    in_w = 572
    out_w = in_w - 72
    batch_size = 6
    lr = 0.01

    train_annot_dir = os.path.join(annot_dir, 'train')
    val_annot_dir = os.path.join(annot_dir, 'val')

    # For two-stage: figure out which clear annots are in train vs val
    clear_train_dir = None
    clear_val_dir = None
    if clear_annots:
        train_files = set(os.listdir(train_annot_dir))
        val_files = set(os.listdir(val_annot_dir))
        clear_train = [f for f in clear_annots if f in train_files]
        clear_val = [f for f in clear_annots if f in val_files]
        print(f'two-stage: {len(clear_train)} clear in train, '
              f'{len(clear_val)} clear in val, '
              f'stage1={stage1_epochs} epochs')
        clear_train_dir = make_filtered_annot_dir(train_annot_dir, clear_train)
        clear_val_dir = make_filtered_annot_dir(val_annot_dir, clear_val)

    try:
        for run in range(1, runs + 1):
            seed = run
            run_dir = os.path.join(metrics_dir, name + '_run' + str(run))

            run_info = make_run_info(run, seed, 'UNetGNRes',
                                     f'SGD(lr={lr}, momentum=0.99, nesterov=True)',
                                     in_w, out_w, batch_size, epochs, name)

            if run_is_complete(run_dir, run_info):
                print(f'skipping run {run} for {name} (already complete)')
                continue

            set_seed(seed)

            model = UNetGNRes()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                        momentum=0.99, nesterov=True)

            os.makedirs(run_dir, exist_ok=True)
            val_metrics_path = os.path.join(run_dir, 'val.csv')
            train_metrics_path = os.path.join(run_dir, 'train.csv')
            save_run_info(run_dir, run_info)

            for i in range(epochs):
                # Two-stage: use clear-only annotations for stage 1
                if clear_annots and i < stage1_epochs:
                    stage = 1
                    cur_train_dir = clear_train_dir
                    cur_val_dir = clear_val_dir
                else:
                    stage = 2 if clear_annots else 0
                    cur_train_dir = train_annot_dir
                    cur_val_dir = val_annot_dir

                print(f'starting epoch {i+1} run {run}',
                      f'(stage {stage})' if clear_annots else '')

                train_set = TrainDataset(cur_train_dir, dataset_dir, in_w, out_w)
                train_loader = MultiEpochsDataLoader(
                    train_set, batch_size, shuffle=True,
                    num_workers=6, drop_last=False, pin_memory=True)

                start_time = time.time()
                train_result = model_utils.epoch(model, train_loader, batch_size,
                                                 optimizer=optimizer,
                                                 step_callback=None, stop_fn=None)

                duration = time.time() - start_time
                print('train epoch complete time', duration)
                tps, fps, tns, fns, defined_sum = train_result
                total = tps + fps + tns + fns
                assert total > 0
                train_metrics = get_metrics(tps, fps, tns, fns, defined_sum, duration)
                # Always validate on full val set
                val_metrics = model_utils.get_val_metrics(
                    model, val_annot_dir, dataset_dir, in_w, out_w, bs=batch_size)
                print('val epoch complete time', time.time() - start_time)
                print('val_metrics', val_metrics)

                log_metrics(val_metrics, val_metrics_path)
                log_metrics(train_metrics, train_metrics_path)
    finally:
        if clear_train_dir:
            shutil.rmtree(clear_train_dir, ignore_errors=True)
        if clear_val_dir:
            shutil.rmtree(clear_val_dir, ignore_errors=True)


def test_corrective_biopore_training():
    training(name='bp_cor', annot_dir=bp_annot_dir, dataset_dir=bp_dataset_dir)


def test_dense_roots_training():
    training(annot_dir=dense_root_annot_dir, name='root_dense', dataset_dir=root_dataset_dir)


def test_corrective_roots_training():
    training(annot_dir=corrective_root_annot_dir, name='root_corrective',
             dataset_dir=root_dataset_dir,
             clear_annots=ROOTS_CORRECTIVE_CLEAR_ANNOTS,
             stage1_epochs=6, runs=10, epochs=26)


def test_dense_nodules_training():
    training(annot_dir=dense_nodules_annot_dir, name='nodules_dense', dataset_dir=nodules_dataset_dir)


if __name__ == '__main__':
    setup_function()
    test_corrective_biopore_training()
