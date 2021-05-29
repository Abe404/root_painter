"""
Copyright (C) 2020 Abraham George Smith
Copyright (C) 2021 Abraham George Smith

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

#pylint: disable=I1101,C0111,W0201,R0903,E0611, R0902, R0914
import os
import random
import time
import shutil
import json

from pathlib import Path, PurePosixPath
from PyQt5 import QtWidgets
from PyQt5 import QtCore

from im_utils import is_image
from name_edit_widget import NameEditWidget
from palette import PaletteEditWidget

class CreateProjectWidget(QtWidgets.QWidget):

    created = QtCore.pyqtSignal(Path)

    def __init__(self, sync_dir):
        super().__init__()
        self.selected_dir = None
        self.proj_name = None
        self.selected_model = None
        self.use_random_weights = True
        self.sync_dir = sync_dir
        self.initUI()

    def initUI(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.name_edit_widget = NameEditWidget('Project')
        self.name_edit_widget.changed.connect(self.validate)
        self.layout.addWidget(self.name_edit_widget)

        self.add_im_dir_widget()
        self.add_radio_widget()
        self.add_model_btn()
        self.add_palette_widget() 
        self.add_info_label()
        self.add_create_btn()

    def add_im_dir_widget(self):
        # Add specify image directory button
        directory_label = QtWidgets.QLabel()
        directory_label.setText("Image directory: Not yet specified")
        self.layout.addWidget(directory_label)
        self.directory_label = directory_label

        specify_image_dir_btn = QtWidgets.QPushButton('Specify image directory')
        specify_image_dir_btn.clicked.connect(self.select_photo_dir)
        self.layout.addWidget(specify_image_dir_btn)

    def add_radio_widget(self):
        radio_widget = QtWidgets.QWidget()
        radio_layout = QtWidgets.QHBoxLayout()
        radio_widget.setLayout(radio_layout)
        self.layout.addWidget(radio_widget)

        # Add radio, use random weight or specify model file.
        radio = QtWidgets.QRadioButton("Random Weights")
        radio.setChecked(True)
        radio.name = "random"
        radio.toggled.connect(self.on_radio_clicked)
        radio_layout.addWidget(radio)

        radio = QtWidgets.QRadioButton("Specify Model")
        radio.name = "specify"
        radio.toggled.connect(self.on_radio_clicked)
        radio_layout.addWidget(radio)

    def add_model_btn(self):
        model_label = QtWidgets.QLabel()
        model_label.setText("Model: Please specify model file")
        self.layout.addWidget(model_label)
        self.model_label = model_label
        specify_model_btn = QtWidgets.QPushButton('Specify model file')
        specify_model_btn.clicked.connect(self.select_model)
        self.specify_model_btn = specify_model_btn
        self.layout.addWidget(specify_model_btn)

        self.model_label.setVisible(False)
        self.specify_model_btn.setVisible(False)

    def add_palette_widget(self):
        self.palette_edit_widget = PaletteEditWidget()
        self.palette_edit_widget.changed.connect(self.validate)
        self.layout.addWidget(self.palette_edit_widget)

    def add_info_label(self):
        info_label = QtWidgets.QLabel()
        info_label.setText("Name, directory and model must be specified"
                           " to create project.")
        self.layout.addWidget(info_label)
        self.info_label = info_label

    def add_create_btn(self):
        # Add create button
        create_project_btn = QtWidgets.QPushButton('Create project')
        create_project_btn.clicked.connect(self.create_project)
        self.layout.addWidget(create_project_btn)
        create_project_btn.setEnabled(False)
        self.create_project_btn = create_project_btn

    def on_radio_clicked(self):
        radio = self.sender()
        if radio.isChecked():
            print("Radio is %s" % (radio.name))
            specify = (radio.name == 'specify')
            self.model_label.setVisible(specify)
            self.specify_model_btn.setVisible(specify)
            self.use_random_weights = not specify
            self.validate()

    def validate(self):
        self.proj_name = self.name_edit_widget.name
        if not self.proj_name:
            self.info_label.setText("Name must be specified to create project")
            self.create_project_btn.setEnabled(False)
            return

        if not self.selected_dir:
            self.info_label.setText("Directory must be specified to create project")
            self.create_project_btn.setEnabled(False)
            return

        if not self.use_random_weights and not self.selected_model:
            self.info_label.setText("Starting model must be specified to create project")
            self.create_project_btn.setEnabled(False)
            return

        cur_files = os.listdir(self.selected_dir)
        cur_files = [is_image(f) for f in cur_files]
        if not cur_files:
            message = "Folder contains no images."
            self.info_label.setText(message)
            self.create_project_btn.setEnabled(False)
            return

        if len(self.palette_edit_widget.get_brush_data()) < 2:
            self.info_label.setText('At least one foreground class must be specified')
            self.create_project_btn.setEnabled(False)
            return

        self.project_location = os.path.join('projects', self.proj_name)
        if os.path.exists(os.path.join(self.sync_dir, self.project_location)):
            self.info_label.setText(f"Project with name {self.proj_name} already exists")
            self.create_project_btn.setEnabled(False)
        else:
            self.info_label.setText(f"Project location: {self.project_location}")
            self.create_project_btn.setEnabled(True)


    def select_photo_dir(self):
        self.photo_dialog = QtWidgets.QFileDialog(self)
        self.photo_dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        def output_selected():
            self.selected_dir = self.photo_dialog.selectedFiles()[0]
            self.directory_label.setText('Image directory: ' + self.selected_dir)
            self.validate()

        self.photo_dialog.fileSelected.connect(output_selected)
        self.photo_dialog.open()


    def select_model(self):
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                             "Specify model file", "",
                                                             "Pickle Files (*.pkl)",
                                                             options=options)
        if file_path:
            file_path = os.path.abspath(file_path)
            self.selected_model = file_path
            self.model_label.setText('Model file: ' + self.selected_model)
            self.validate()

    def create_project(self):
        project_name = self.proj_name
        project_location = Path(self.project_location)

        dataset_path = os.path.abspath(self.selected_dir)
        datasets_dir = str(self.sync_dir / 'datasets')
    
        if not dataset_path.startswith(datasets_dir):
            message = ("When creating a project the selected dataset must be in "
                       "the datasets folder. The selected dataset is "
                       f"{dataset_path} and the datasets folder is "
                       f"{datasets_dir}.")
            QtWidgets.QMessageBox.about(self, 'Project Creation Error', message)
            return

        os.makedirs(self.sync_dir / project_location)
        proj_file_path = (self.sync_dir / project_location /
                          (project_name + '.seg_proj'))
        os.makedirs(self.sync_dir / project_location / 'annotations' / 'train')
        os.makedirs(self.sync_dir / project_location / 'annotations' / 'val')
        os.makedirs(self.sync_dir / project_location / 'segmentations')
        os.makedirs(self.sync_dir / project_location / 'models')
        os.makedirs(self.sync_dir / project_location / 'messages')
        os.makedirs(self.sync_dir / project_location / 'logs')

        if self.use_random_weights:
            original_model_file = 'random weights'
        else:
            model_num = 1
            model_name = str(model_num).zfill(6)
            model_name += '_' + str(int(round(time.time()))) + '.pkl'
            shutil.copyfile(self.selected_model,
                            self.sync_dir / project_location /
                            'models' / model_name)
            original_model_file = self.selected_model


        dataset = os.path.basename(dataset_path)
        # get files in random order for training.
        all_fnames = os.listdir(dataset_path)
        # images only
        all_fnames = [a for a in all_fnames if is_image(a)]

        all_fnames = sorted(all_fnames)
        random.shuffle(all_fnames)

        # create project file.
        project_info = {
            'name': project_name,
            'dataset': dataset,
            'original_model_file': original_model_file,
            'location': str(PurePosixPath(project_location)),
            'file_names': all_fnames,
            'classes': self.palette_edit_widget.get_brush_data()
        }
        with open(proj_file_path, 'w') as json_file:
            json.dump(project_info, json_file, indent=4)
        self.created.emit(proj_file_path)
        self.close()
