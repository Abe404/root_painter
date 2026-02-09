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
import subprocess
import torch
import numpy as np
import time
from datasets import TrainDataset
from torch.utils.data import DataLoader
from multi_epoch.multi_epoch_loader import MultiEpochsDataLoader
from test_utils import get_acc, log_metrics, dl_dir_from_zip
from metrics import get_metrics

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


def training(dataset_dir, annot_dir, name):
    """Run training benchmark: multiple runs, logging train and val metrics per epoch."""
    import model_utils
    from unet import UNetGNRes
    runs = 5
    in_w = 572
    out_w = in_w - 72
    batch_size = 6
    epochs = 60
    lr = 0.01

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

        train_annot_dir = os.path.join(annot_dir, 'train')
        val_annot_dir = os.path.join(annot_dir, 'val')
        start_time = time.time()

        train_set = TrainDataset(train_annot_dir,
                                 dataset_dir,
                                 in_w, out_w)

        os.makedirs(run_dir, exist_ok=True)

        val_metrics_path = os.path.join(run_dir, 'val.csv')
        train_metrics_path = os.path.join(run_dir, 'train.csv')

        save_run_info(run_dir, run_info)

        train_loader = MultiEpochsDataLoader(train_set, batch_size, shuffle=False,
                                             num_workers=6,
                                             drop_last=False, pin_memory=True)

        print('loader setup time', time.time() - start_time)
        for i in range(epochs):
            print('starting epoch', i + 1, 'run', run)

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
            val_metrics = model_utils.get_val_metrics(model, val_annot_dir, dataset_dir,
                                                      in_w, out_w, bs=batch_size)
            print('val epoch complete time', time.time() - start_time)
            print('val_metrics', val_metrics)

            log_metrics(val_metrics, val_metrics_path)
            log_metrics(train_metrics, train_metrics_path)


def test_corrective_biopore_training():
    training(name='bp_cor', annot_dir=bp_annot_dir, dataset_dir=bp_dataset_dir)


def test_dense_roots_training():
    training(annot_dir=dense_root_annot_dir, name='root_dense', dataset_dir=root_dataset_dir)


def test_corrective_roots_training():
    training(annot_dir=corrective_root_annot_dir, name='root_corrective', dataset_dir=root_dataset_dir)


def test_dense_nodules_training():
    training(annot_dir=dense_nodules_annot_dir, name='nodules_dense', dataset_dir=nodules_dataset_dir)


if __name__ == '__main__':
    setup_function()
    test_corrective_biopore_training()
