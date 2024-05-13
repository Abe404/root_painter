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

# Add the parent directory to sys.path
parent_dir = os.path.abspath('../')
sys.path.insert(0, parent_dir)


import torch
import time
from datasets import TrainDataset
from torch.utils.data import DataLoader
from multi_epoch.multi_epoch_loader import MultiEpochsDataLoader


# sync directory for use with tests
sync_dir = os.path.join(os.getcwd(), 'tests', 'test_rp_sync')
annot_dir = os.path.join(sync_dir, 'projects', 'biopores_corrective_a', 'annotations')
datasets_dir = os.path.join(sync_dir, 'datasets')
bp_dataset_dir = os.path.join(datasets_dir, 'biopores_750_training')

timeout_ms = 20000


def setup_function():
    import urllib.request
    import zipfile
    import shutil
    from test_utils import dl_dir_from_zip
    print('running setup')
    # prepare biopores training dataset
    if not os.path.isdir(datasets_dir):
        os.makedirs(datasets_dir)
    biopore_url = 'https://zenodo.org/record/3754046/files/biopores_750_training.zip'
    # TODO: import copy function accross and import.
    dl_dir_from_zip(biopore_url, bp_dataset_dir)
    
    # download some annotations that can be used for training.
    biopore_annot_url = 'https://zenodo.org/record/8041842/files/user_a_corrective_biopores_750_training_annotation.zip'
    dl_dir_from_zip(biopore_annot_url, annot_dir)


def test_corrective_biopore_training():
    # a specific training set f1 score can be obtained in a specific number of update steps
    # and wall clock time?
    import model_utils
    from unet import UNetGNRes
    model = UNetGNRes()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                                momentum=0.99, nesterov=True)
    in_w = 572
    out_w = in_w - 72
    batch_size = 6
    train_annot_dir = os.path.join(annot_dir, 'train')
    val_annot_dir = os.path.join(annot_dir, 'val')
    start_time = time.time()


    train_set = TrainDataset(train_annot_dir,
                             bp_dataset_dir,
                             in_w, out_w)
    
    train_loader = MultiEpochsDataLoader(train_set, batch_size, shuffle=False,
        # 12 workers is good for performance
        # on 2 RTX2080 Tis (but depends on CPU also)
        # 0 workers is good for debugging
        # don't go above max_workers (user specified but default 12) 
        # and don't go above the number of cpus, provided by cpu_count.
        num_workers=6,
        drop_last=False, pin_memory=True)

    print('loader setup time',  time.time() - start_time)
    train_result = model_utils.epoch(model, train_loader, batch_size,
                                     optimizer=optimizer, step_callback=None, stop_fn=None)

    print('train epoch complete time', time.time() - start_time)
    val_metrics = model_utils.get_val_metrics(model, val_annot_dir, bp_dataset_dir,
                                              in_w, out_w, bs=batch_size)

    print('val epoch complete time', time.time() - start_time)
    # pass - epoch runs without error.


def dense_roots_training():
    pass


def corrective_biopore_training():
    # a specific validation set f1 score can be obtained in a specific number of update steps
    # and wall clock time?
    pass

def corrective_nodules_training():
    # a specific validation set f1 score can be obtained in a specific number of update steps
    # and wall clock time?
    pass

def corrective_roots_training():
    # a specific validation set f1 score can be obtained in a specific number of update steps
    # and wall clock time?
    pass
