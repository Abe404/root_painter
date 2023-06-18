"""
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
import shutil
from PyQt6 import QtCore
from test_utils import dl_dir_from_zip

sync_dir = os.path.join(os.getcwd(), 'test_rp_sync')
datasets_dir = os.path.join(sync_dir, 'datasets')
bp_full_dataset_dir = os.path.join(datasets_dir, 'biopores_750_training')
bp_train_dataset_dir = os.path.join(datasets_dir, 'biopores_750_training')

timeout_ms = 20000


def setup_function():
    import urllib.request
    import zipfile
    import shutil
    print('running setup')

    projects_location = os.path.join(sync_dir, 'projects')
    shutil.rmtree(projects_location)
    os.makedirs(projects_location)

    # prepare biopores training dataset
    if not os.path.isdir(datasets_dir):
        os.makedirs(datasets_dir)
    bp_full_dataset_url = 'https://zenodo.org/record/3753969/files/BP_uncounted.zip'

    dl_dir_from_zip(bp_full_dataset_url, bp_full_dataset_dir)


def test_create_training_dataset(qtbot):
    from create_dataset import CreateDatasetWidget
    # initialise the widget
    widget = CreateDatasetWidget(sync_dir)
    widget.show()
    # test that we can click the dataset specification btn
    qtbot.mouseClick(widget.specify_image_dir_btn, QtCore.Qt.MouseButton.LeftButton)
    # can't actually reproduce user interaction with the QFileDialog so just set the directory
    widget.source_dir = bp_full_dataset_dir
    name = 'TestDataset9000'
    widget.name_edit_widget.name = name
    widget.validate()  
    widget.create_btn.click()
      
    def check_output():
        import json
        out_dataset_location = os.path.join(sync_dir, 'datasets', name)
        if not os.path.isdir(out_dataset_location):
            return False
        
        return len(os.listdir(out_dataset_location)) == len(os.listdir(bp_full_dataset_dir))

    qtbot.waitUntil(check_output, timeout=20000)
