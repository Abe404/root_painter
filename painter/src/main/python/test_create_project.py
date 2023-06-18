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
bp_dataset_dir = os.path.join(datasets_dir, 'biopores_750_training')

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
    biopore_url = 'https://zenodo.org/record/3754046/files/biopores_750_training.zip'
    dl_dir_from_zip(biopore_url, bp_dataset_dir)


def test_specify_seg_for_mask_widget(qtbot):
    from create_project import CreateProjectWidget
    # initialise the widget
    widget = CreateProjectWidget(sync_dir)
    widget.show()
    # test that we can click the dataset specification btn
    qtbot.mouseClick(widget.specify_image_dir_btn, QtCore.Qt.MouseButton.LeftButton)
    # can't actually reproduce user interaction with the QFileDialog so just set the directory
    widget.selected_dir = bp_dataset_dir
    proj_name = 'TestProject9000'
    widget.name_edit_widget.name = proj_name
    widget.validate()  
    widget.create_project_btn.click()
      
    def check_output():
        import json
        proj_location = os.path.join(sync_dir, 'projects', proj_name)
        if not os.path.isdir(proj_location):
            return False

        proj_file_path = os.path.join(proj_location, (proj_name + '.seg_proj'))
        if not os.path.isfile(proj_file_path):
            return False

        # load the project file
        # read file
        with open(proj_file_path, 'r') as proj_file:
            proj_file_data = proj_file.read()
            proj_file_obj = json.loads(proj_file_data)
        return len(proj_file_obj['file_names']) == len(os.listdir(bp_dataset_dir))

    qtbot.waitUntil(check_output, timeout=5000)
