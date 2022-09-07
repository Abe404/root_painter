"""
Copyright (C) 2020 Abraham George Smith

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

# pylint: disable=I1101,C0111,W0201,R0903,E0611, R0902, R0914

# too many statements
# pylint: disable=R0915

# catching too general exception
# pylint: disable=W0703

# too many public methods

import sys
import os
from pathlib import PurePath, Path
import json
from functools import partial

from skimage.io import use_plugin
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PIL import Image

from about import AboutWindow, LicenseWindow
from create_project import CreateProjectWidget
from create_dataset import CreateDatasetWidget, check_extend_dataset
from segment_folder import SegmentFolderWidget
from extract_count import ExtractCountWidget
from extract_regions import ExtractRegionsWidget
from extract_length import ExtractLengthWidget
from extract_comp import ExtractCompWidget
from convert_seg import ConvertSegForRVEWidget
from graphics_scene import GraphicsScene
from graphics_view import CustomGraphicsView
from nav import NavWidget
from visibility_widget import VisibilityWidget
from file_utils import last_fname_with_annotations
from file_utils import get_annot_path
from file_utils import maybe_save_annotation
from instructions import send_instruction
from plot_seg_metrics import MetricsPlot, ExtractMetricsWidget
from im_viewer import ContextViewer
use_plugin("pil")

Image.MAX_IMAGE_PIXELS = None

class RootPainter(QtWidgets.QMainWindow):

    closed = QtCore.pyqtSignal()

    def __init__(self, sync_dir):
        super().__init__()
        # give the main RootPainter window an Icon.
        app_dir = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QtGui.QIcon(os.path.join(app_dir, 'icons/linux/128.png')))

        self.assign_sync_directory(sync_dir)
        self.tracking = False
        self.image_pixmap_holder = None
        self.seg_pixmap_holder = None
        self.annot_pixmap_holder = None

        self.image_visible = True
        self.seg_visible = False
        self.annot_visible = True
        self.pre_segment_count = 0
        self.im_width = None
        self.im_height = None
        self.metrics_plot = None

        self.initUI()

    def assign_sync_directory(self, sync_dir):
        self.sync_dir = sync_dir
        self.instruction_dir = sync_dir / 'instructions'
        self.send_instruction = partial(send_instruction,
                                        instruction_dir=self.instruction_dir,
                                        sync_dir=sync_dir)

    def mouse_scroll(self, event):
        scroll_up = event.angleDelta().y() > 0
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        alt_down = (modifiers & QtCore.Qt.AltModifier)
        shift_down = (modifiers & QtCore.Qt.ShiftModifier)

        if alt_down or shift_down:
            # change by 10% (nearest int) or 1 (min)
            increment = max(1, int(round(self.scene.brush_size / 10)))
            if scroll_up:
                self.scene.brush_size += increment
            else:
                self.scene.brush_size -= increment
            self.scene.brush_size = max(1, self.scene.brush_size)
            self.update_cursor()
        else:
            if scroll_up:
                self.graphics_view.zoom *= 1.1
            else:
                self.graphics_view.zoom /= 1.1
            self.graphics_view.update_zoom()

    def initUI(self):
        if len(sys.argv) < 2:
            self.init_missing_project_ui()
            return

        fname = sys.argv[1]
        if os.path.splitext(fname)[1] == '.seg_proj':
            proj_file_path = os.path.abspath(sys.argv[1])
            self.open_project(proj_file_path)
        else:
            # only warn if -psn not in the args. -psn is in the args when
            # user opened app in a normal way by clicking on the Application icon.
            if not '-psn' in sys.argv[1]:
                QtWidgets.QMessageBox.about(self, 'Error', sys.argv[1] +
                                            ' is not a valid '
                                            'segmentation project (.seg_proj) file')
            self.init_missing_project_ui()

    def open_project(self, proj_file_path):
        # extract json
        with open(proj_file_path, 'r') as json_file:
            settings = json.load(json_file)
            self.dataset_dir = self.sync_dir / 'datasets' / PurePath(settings['dataset'])

            self.proj_location = self.sync_dir / PurePath(settings['location'])
            self.image_fnames = settings['file_names']
            self.seg_dir = self.proj_location / 'segmentations'
            self.log_dir = self.proj_location / 'logs'
            self.train_annot_dir = self.proj_location / 'annotations' / 'train'
            self.val_annot_dir = self.proj_location / 'annotations' / 'val'

            self.model_dir = self.proj_location / 'models'

            self.message_dir = self.proj_location / 'messages'

            self.proj_file_path = proj_file_path

            # If there are any annotations which have already been saved
            # then go through the annotations in the order specified
            # by self.image_fnames
            # and set fname (current image) to be the last image with annotation
            last_with_annot = last_fname_with_annotations(self.image_fnames,
                                                          self.train_annot_dir,
                                                          self.val_annot_dir)
            if last_with_annot:
                fname = last_with_annot
            else:
                fname = self.image_fnames[0]

            # manual override for the image to show
            if 'image_index' in settings:
                fname = self.image_fnames[settings['image_index']]

            # set first image from project to be current image
            self.image_path = os.path.join(self.dataset_dir, fname)
            self.update_window_title()
            self.seg_path = os.path.join(self.seg_dir, fname)
            self.annot_path = get_annot_path(fname, self.train_annot_dir,
                                             self.val_annot_dir)
            self.init_active_project_ui()
            self.track_changes()


    def update_file(self, fpath):
        
        fname = os.path.basename(fpath)

        # update selected point in the plot.
        # do first to give fast user feedback. 
        if self.metrics_plot and self.metrics_plot.plot_window:
            self.metrics_plot.plot_window.set_highlight_point(fname)


        # Save current annotation (if it exists) before moving on
        self.save_annotation()

        # set first image from project to be current image
        self.image_path = os.path.join(self.dataset_dir, fname)
        self.png_fname = os.path.splitext(fname)[0] + '.png'
        self.seg_path = os.path.join(self.seg_dir, self.png_fname)
        self.annot_path = get_annot_path(self.png_fname,
                                         self.train_annot_dir,
                                         self.val_annot_dir)
        self.update_image()


        self.scene.history = []
        self.scene.redo_list = []

        self.update_seg()
        self.update_annot()

        self.segment_current_image()
        self.update_window_title()



    def update_image(self):
        # Will also update self.im_width and self.im_height
        assert os.path.isfile(self.image_path), f"Cannot find file {self.image_path}"
        image_pixmap = QtGui.QPixmap(self.image_path)
        im_size = image_pixmap.size()
        im_width, im_height = im_size.width(), im_size.height()
        assert im_width > 0, self.image_path
        assert im_height > 0, self.image_path
        self.graphics_view.image = image_pixmap # for resize later
        self.im_width = im_width
        self.im_height = im_height

        self.scene.setSceneRect(-15, -15, im_width+30, im_height+30)

        # Used to replace the segmentation or annotation when they are not visible.
        self.blank_pixmap = QtGui.QPixmap(self.im_width, self.im_height)
        self.blank_pixmap.fill(Qt.transparent)

        self.black_pixmap = QtGui.QPixmap(self.im_width, self.im_height)
        self.black_pixmap.fill(Qt.black)

        if self.image_pixmap_holder:
            self.image_pixmap_holder.setPixmap(image_pixmap)
        else:
            self.image_pixmap_holder = self.scene.addPixmap(image_pixmap)
        if not self.image_visible:
            self.image_pixmap_holder.setPixmap(self.black_pixmap)


    def update_seg(self):
        # if seg file is present then load.
        if os.path.isfile(self.seg_path):
            self.seg_mtime = os.path.getmtime(self.seg_path)
            self.seg_pixmap = QtGui.QPixmap(self.seg_path)
            self.nav.next_image_button.setText('Save && Next >')
            if hasattr(self, 'vis_widget'):
                self.vis_widget.seg_checkbox.setText('Segmentation (S)')
            self.nav.next_image_button.setEnabled(True)
        else:
            self.seg_mtime = None
            # otherwise use blank
            self.seg_pixmap = QtGui.QPixmap(self.im_width, self.im_height)
            self.seg_pixmap.fill(Qt.transparent)
            painter = QtGui.QPainter()
            painter.begin(self.seg_pixmap)
            font = QtGui.QFont()
            font.setPointSize(48)
            painter.setFont(font)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255), Qt.SolidPattern))
            if sys.platform == 'win32':
                # For some reason the text has a different size
                # and position on windows
                # so change the background rectangle also.
                painter.drawRect(0, 0, 657, 75)
            else:
                painter.drawRect(10, 10, 465, 55)
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 150)))
            painter.drawText(16, 51, 'Loading segmentation')
            painter.end()
            self.nav.next_image_button.setText('Loading Segmentation...')
            if hasattr(self, 'vis_widget'):
                self.vis_widget.seg_checkbox.setText('Segmentation (Loading)')
            self.nav.next_image_button.setEnabled(False)

        if self.seg_pixmap_holder:
            self.seg_pixmap_holder.setPixmap(self.seg_pixmap)
        else:
            self.seg_pixmap_holder = self.scene.addPixmap(self.seg_pixmap)
        if not self.seg_visible:
            self.seg_pixmap_holder.setPixmap(self.blank_pixmap)

    def update_annot(self):
        # if annot file is present then load
        if self.annot_path and os.path.isfile(self.annot_path):
            self.annot_pixmap = QtGui.QPixmap(self.annot_path)
        else:
            # otherwise use blank
            self.annot_pixmap = QtGui.QPixmap(self.im_width, self.im_height)
            self.annot_pixmap.fill(Qt.transparent)
        if self.annot_pixmap_holder:
            self.annot_pixmap_holder.setPixmap(self.annot_pixmap)
        else:
            self.annot_pixmap_holder = self.scene.addPixmap(self.annot_pixmap)
        self.scene.annot_pixmap_holder = self.annot_pixmap_holder
        self.scene.annot_pixmap = self.annot_pixmap
        self.scene.history.append(self.scene.annot_pixmap.copy())

        if not self.annot_visible:
            self.annot_pixmap_holder.setPixmap(self.blank_pixmap)


    def segment_image(self, image_fnames):
        # send instruction to segment the new image.
        content = {
            "dataset_dir": self.dataset_dir,
            "seg_dir": self.seg_dir,
            "file_names": image_fnames,
            "message_dir": self.message_dir,
            "model_dir": self.model_dir
        }
        self.send_instruction('segment', content)

    def segment_current_image(self):
        dir_path, _ = os.path.split(self.image_path)
        path_list = self.nav.get_path_list(dir_path)
        cur_index = path_list.index(self.image_path)
        to_segment_paths = path_list[cur_index:1+cur_index+self.pre_segment_count]
        to_segment_paths = [f for f in to_segment_paths if
                            os.path.isfile(os.path.join(self.seg_dir, f))]
        to_segment_fnames = [os.path.basename(p) for p in to_segment_paths]
        self.segment_image(to_segment_fnames)

    def show_open_project_widget(self):
        options = QtWidgets.QFileDialog.Options()
        default_loc = self.sync_dir / 'projects'
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load project file",
            str(default_loc),
            "Segmentation project file (*.seg_proj)",
            options=options)

        if file_path:
            self.open_project(file_path)

    def show_create_project_widget(self):
        print("Open the create project widget..")
        self.create_project_widget = CreateProjectWidget(self.sync_dir)
        self.create_project_widget.show()
        self.create_project_widget.created.connect(self.open_project)

    def init_missing_project_ui(self):
        ## Create project menu
        # project has not yet been selected or created
        # need to open minimal interface which allows users
        # to open or create a project.

        menu_bar = self.menuBar()
        self.menu_bar = menu_bar
        self.menu_bar.clear()
        self.project_menu = menu_bar.addMenu("Project")

        # Open project
        self.open_project_action = QtWidgets.QAction(QtGui.QIcon(""), "Open project", self)
        self.open_project_action.setShortcut("Ctrl+O")

        self.project_menu.addAction(self.open_project_action)
        self.open_project_action.triggered.connect(self.show_open_project_widget)

        # Create project
        self.create_project_action = QtWidgets.QAction(QtGui.QIcon(""), "Create project", self)
        self.create_project_action.setShortcut("Ctrl+C")
        self.project_menu.addAction(self.create_project_action)
        self.create_project_action.triggered.connect(self.show_create_project_widget)

        # Network Menu
        self.network_menu = menu_bar.addMenu('Network')
        # # segment folder
        self.segment_folder_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                                    'Segment folder', self)

        def show_segment_folder():
            self.segment_folder_widget = SegmentFolderWidget(self.sync_dir,
                                                             self.instruction_dir)
            self.segment_folder_widget.show()
        self.segment_folder_btn.triggered.connect(show_segment_folder)
        self.network_menu.addAction(self.segment_folder_btn)

        self.add_measurements_menu(menu_bar)
        self.add_extras_menu(menu_bar)
        self.add_about_menu(menu_bar)

        ### Add project btns to open window (so it shows something useful)
        project_btn_widget = QtWidgets.QWidget()
        self.setCentralWidget(project_btn_widget)

        layout = QtWidgets.QHBoxLayout()
        project_btn_widget.setLayout(layout)
        open_project_btn = QtWidgets.QPushButton('Open existing project')
        open_project_btn.clicked.connect(self.show_open_project_widget)
        layout.addWidget(open_project_btn)

        create_project_btn = QtWidgets.QPushButton('Create new project')
        create_project_btn.clicked.connect(self.show_create_project_widget)
        layout.addWidget(create_project_btn)

        create_dataset_btn = QtWidgets.QPushButton('Create training dataset')
        def show_create_dataset():
            self.create_dataset_widget = CreateDatasetWidget(self.sync_dir)
            self.create_dataset_widget.show()
        create_dataset_btn.clicked.connect(show_create_dataset)
        layout.addWidget(create_dataset_btn)

        self.setWindowTitle("RootPainter")
        self.resize(layout.sizeHint())

    def specify_sync_directory(self):
        """ User may choose to update the sync directory.
            This may happen if they initially specified the wrong
            sync directory.

        """
        settings_path = os.path.join(Path.home(), 'root_painter_settings.json')
        dir_path = QtWidgets.QFileDialog.getExistingDirectory()
        if dir_path:
            with open(settings_path, 'w') as json_file:
                content = {
                    "sync_dir": os.path.abspath(dir_path)
                }
                json.dump(content, json_file, indent=4)
            self.sync_dir = Path(json.load(open(settings_path, 'r'))['sync_dir'])
            self.assign_sync_directory(self.sync_dir)


    def add_extras_menu(self, menu_bar, project_open=False):
        extras_menu = menu_bar.addMenu('Extras')
        comp_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Extract composites', self)
        comp_btn.triggered.connect(self.show_extract_comp)
        extras_menu.addAction(comp_btn)

        conv_to_rve_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                            'Convert segmentations for RhizoVision Explorer',
                                             self)
        conv_to_rve_btn.triggered.connect(self.show_conv_to_rve)
        extras_menu.addAction(conv_to_rve_btn)

        specify_sync_dir_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                                 'Specify sync directory',
                                                 self)
        specify_sync_dir_btn.triggered.connect(self.specify_sync_directory)
        extras_menu.addAction(specify_sync_dir_btn)


        def view_metric_csv():
            # select a csv file and then show it in the plots
            options = QtWidgets.QFileDialog.Options()
            default_loc = self.sync_dir / 'projects'
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "View Metrics CSV data",
                str(default_loc),
                "Metric CSV file (*.csv)",
                options=options)
            if file_path:
                self.metrics_plot.view_plot_from_csv(file_path)

        view_metrics_csv_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                                 'View Metrics Plot from CSV',
                                                  self)
        self.metrics_plot = MetricsPlot()


        view_metrics_csv_btn.triggered.connect(view_metric_csv)
        extras_menu.addAction(view_metrics_csv_btn)



        if project_open:
            metrics_plot_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                                             'Show metrics plot',
                                                              self)
            self.metrics_plot = MetricsPlot()
            def navigate_to_file(fname):
                fpath = os.path.join(self.dataset_dir, fname)
                self.nav.image_path = fpath
                self.nav.update_nav_label()
                self.update_file(fpath)
            def open_metric_plot():
                self.metrics_plot.create_metrics_plot(
                    self.proj_file_path,
                    navigate_to_file,
                    self.image_path)
            metrics_plot_btn.triggered.connect(open_metric_plot)
            extras_menu.addAction(metrics_plot_btn)

            metrics_csv_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                                            'Export metrics CSV',
                                                             self)
            self.metrics_plot = MetricsPlot()
            def open_metric_export():
                self.extract_metrics_widget = ExtractMetricsWidget(self.proj_file_path)
                self.extract_metrics_widget.show()
            metrics_csv_btn.triggered.connect(open_metric_export)
            extras_menu.addAction(metrics_csv_btn)





            extend_dataset_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Extend dataset', self)
            def update_dataset_after_check():
                was_extended, file_names = check_extend_dataset(self,
                                                                self.dataset_dir,
                                                                self.image_fnames,
                                                                self.proj_file_path)
                if was_extended:
                    self.image_fnames = file_names
                    self.nav.all_fnames = file_names
                    self.nav.update_nav_label()
            extend_dataset_btn.triggered.connect(update_dataset_after_check)
            extras_menu.addAction(extend_dataset_btn)
    

    def add_about_menu(self, menu_bar):
        about_menu = menu_bar.addMenu('About')
        license_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'License', self)
        license_btn.triggered.connect(self.show_license_window)
        about_menu.addAction(license_btn)

        about_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'RootPainter', self)
        about_btn.triggered.connect(self.show_about_window)
        about_menu.addAction(about_btn)

    def show_license_window(self):
        self.license_window = LicenseWindow()
        self.license_window.show()

    def show_about_window(self):
        self.about_window = AboutWindow()
        self.about_window.show()

    def update_window_title(self):
        proj_dirname = os.path.basename(self.proj_location)
        self.setWindowTitle(f"RootPainter {proj_dirname}"
                            f" {os.path.basename(self.image_path)}")

    def init_active_project_ui(self):
        # container for both nav and graphics view.
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container.setLayout(container_layout)
        self.setCentralWidget(container)

        self.graphics_view = CustomGraphicsView()
        self.graphics_view.zoom_change.connect(self.update_cursor)

        container_layout.addWidget(self.graphics_view)
        scene = GraphicsScene()
        scene.parent = self
        self.graphics_view.setScene(scene)
        self.graphics_view.mouse_scroll_event.connect(self.mouse_scroll)

        # Required so graphics scene can track mouse up when mouse is not pressed
        self.graphics_view.setMouseTracking(True)
        self.scene = scene
        self.nav = NavWidget(self.image_fnames, [self.train_annot_dir, self.val_annot_dir])
        self.update_file(self.image_path)

        # bottom bar
        bottom_bar = QtWidgets.QWidget()
        bottom_bar_layout = QtWidgets.QHBoxLayout()
        # left, top, right, bottom
        bottom_bar_layout.setContentsMargins(20, 0, 20, 20)
        bottom_bar_layout.setSpacing(0)
        bottom_bar.setLayout(bottom_bar_layout)

        container_layout.addWidget(bottom_bar)

        # Bottom bar left
        self.vis_widget = VisibilityWidget()
        self.vis_widget.setMaximumWidth(200)
        self.vis_widget.setMinimumWidth(200)
        self.vis_widget.seg_checkbox.stateChanged.connect(self.seg_checkbox_change)
        self.vis_widget.annot_checkbox.stateChanged.connect(self.annot_checkbox_change)
        self.vis_widget.im_checkbox.stateChanged.connect(self.im_checkbox_change)
        bottom_bar_layout.addWidget(self.vis_widget)

        # bottom bar right
        bottom_bar_r = QtWidgets.QWidget()
        bottom_bar_r_layout = QtWidgets.QVBoxLayout()
        bottom_bar_r.setLayout(bottom_bar_r_layout)
        bottom_bar_layout.addWidget(bottom_bar_r)

        # Nav
        self.nav.file_change.connect(self.update_file)

        self.nav.image_path = self.image_path
        self.nav.update_nav_label()

        # info label
        info_container = QtWidgets.QWidget()
        info_container_layout = QtWidgets.QHBoxLayout()
        info_container_layout.setAlignment(Qt.AlignCenter)
        info_label = QtWidgets.QLabel()
        info_label.setText("")
        info_container_layout.addWidget(info_label)
        # left, top, right, bottom
        info_container_layout.setContentsMargins(0, 0, 0, 0)
        info_container.setLayout(info_container_layout)
        self.info_label = info_label

        bottom_bar_r_layout.addWidget(info_container)
        bottom_bar_r_layout.addWidget(self.nav)

        self.add_menu()

        self.resize(container_layout.sizeHint())

        self.update_cursor()

        def view_fix():
            """ hack for linux bug """
            self.update_cursor()
            self.graphics_view.fit_to_view()
        QtCore.QTimer.singleShot(100, view_fix)

    def track_changes(self):
        if self.tracking:
            return
        print('Starting watch for changes')
        self.tracking = True
        def check():
            # check for any messages
            messages = os.listdir(str(self.message_dir))
            for m in messages:
                if hasattr(self, 'info_label'):
                    self.info_label.setText(m)
                try:
                    # Added try catch because this error happened (very rarely)
                    # PermissionError: [WinError 32]
                    # The process cannot access the file because it is
                    # being used by another process
                    os.remove(os.path.join(self.message_dir, m))
                except Exception as e:
                    print('Caught exception when trying to detele msg', e)
            if hasattr(self, 'seg_path') and os.path.isfile(self.seg_path):
                try:
                    # seg mtime is not actually used any more.
                    new_mtime = os.path.getmtime(self.seg_path)
                    # seg_mtime is None before the seg is loaded.
                    if not self.seg_mtime:
                        print('load seg from file.')
                        self.seg_pixmap = QtGui.QPixmap(self.seg_path)
                        self.seg_mtime = new_mtime
                        self.nav.next_image_button.setText('Save && Next >')
                        self.nav.next_image_button.setEnabled(True)
                        if self.seg_visible:
                            self.seg_pixmap_holder.setPixmap(self.seg_pixmap)
                        if hasattr(self, 'vis_widget'):
                            self.vis_widget.seg_checkbox.setText('Segmentation (S)')
                except Exception as e:
                    print('Error: when trying to load segmention ' + str(e))
                    # sometimes problems reading file.
                    # don't worry about this exception
            else:
                print('no seg found', end=",")
            QtCore.QTimer.singleShot(500, check)
        QtCore.QTimer.singleShot(500, check)


    def close_project_window(self):
        self.close()
        self.closed.emit()

    def add_menu(self):
        menu_bar = self.menuBar()
        menu_bar.clear()

        self.project_menu = menu_bar.addMenu("Project")

        self.close_project_action = QtWidgets.QAction(QtGui.QIcon(""), "Close project", self)
        self.project_menu.addAction(self.close_project_action)
        self.close_project_action.triggered.connect(self.close_project_window)

        edit_menu = menu_bar.addMenu("Edit")
        # Undo
        undo_action = QtWidgets.QAction(QtGui.QIcon(""), "Undo", self)
        undo_action.setShortcut("Z")
        edit_menu.addAction(undo_action)
        undo_action.triggered.connect(self.scene.undo)

        # Redo
        redo_action = QtWidgets.QAction(QtGui.QIcon(""), "Redo", self)
        redo_action.setShortcut("Ctrl+Shift+Z")
        edit_menu.addAction(redo_action)
        redo_action.triggered.connect(self.scene.redo)

        options_menu = menu_bar.addMenu("Options")
        # pre segment count
        pre_segment_count_action = QtWidgets.QAction(QtGui.QIcon(""), "Pre-Segment", self)
        options_menu.addAction(pre_segment_count_action)
        pre_segment_count_action.triggered.connect(self.open_pre_segment_count_dialog)

        brush_edit_action = QtWidgets.QAction(QtGui.QIcon(""), "Change brush size", self)
        options_menu.addAction(brush_edit_action)
        brush_edit_action.triggered.connect(self.show_brush_size_edit)

        # change brush colors
        change_foreground_color_action = QtWidgets.QAction(QtGui.QIcon(""),
                                                           "Foreground brush colour",
                                                           self)
        options_menu.addAction(change_foreground_color_action)
        change_foreground_color_action.triggered.connect(self.change_foreground_color)
        change_background_color_action = QtWidgets.QAction(QtGui.QIcon(""),
                                                           "Background brush colour",
                                                           self)
        options_menu.addAction(change_background_color_action)
        change_background_color_action.triggered.connect(self.change_background_color)

        brush_menu = menu_bar.addMenu("Brushes")
        foreground_color_action = QtWidgets.QAction(QtGui.QIcon(""), "Foreground", self)
        foreground_color_action.setShortcut("Q")
        brush_menu.addAction(foreground_color_action)
        foreground_color_action.triggered.connect(self.set_foreground_color)

        background_color_action = QtWidgets.QAction(QtGui.QIcon(""), "Background", self)
        background_color_action.setShortcut("W")
        brush_menu.addAction(background_color_action)
        background_color_action.triggered.connect(self.set_background_color)

        eraser_color_action = QtWidgets.QAction(QtGui.QIcon(""), "Eraser", self)
        eraser_color_action.setShortcut("E")
        brush_menu.addAction(eraser_color_action)
        eraser_color_action.triggered.connect(self.set_eraser_color)

        ## View menu
        # Fit to view
        view_menu = menu_bar.addMenu('View')
        fit_to_view_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Fit to View', self)
        fit_to_view_btn.setShortcut('Ctrl+F')
        fit_to_view_btn.setStatusTip('Fit image to view')
        fit_to_view_btn.triggered.connect(self.graphics_view.fit_to_view)
        view_menu.addAction(fit_to_view_btn)

        # Actual size
        actual_size_view_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Actual size', self)
        actual_size_view_btn.setShortcut('Ctrl+A')
        actual_size_view_btn.setStatusTip('Show image at actual size')
        actual_size_view_btn.triggered.connect(self.graphics_view.show_actual_size)
        view_menu.addAction(actual_size_view_btn)

        toggle_seg_visibility_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                                      'Toggle segmentation visibility', self)
        toggle_seg_visibility_btn.setShortcut('S')
        toggle_seg_visibility_btn.setStatusTip('Show or hide segmentation')
        toggle_seg_visibility_btn.triggered.connect(self.show_hide_seg)
        view_menu.addAction(toggle_seg_visibility_btn)

        toggle_annot_visibility_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                                        'Toggle annotation visibility', self)
        toggle_annot_visibility_btn.setShortcut('A')
        toggle_annot_visibility_btn.setStatusTip('Show or hide annotation')
        toggle_annot_visibility_btn.triggered.connect(self.show_hide_annot)
        view_menu.addAction(toggle_annot_visibility_btn)

        toggle_image_visibility_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                                        'Toggle image visibility', self)
        toggle_image_visibility_btn.setShortcut('I')
        toggle_image_visibility_btn.setStatusTip('Show or hide image')
        toggle_image_visibility_btn.triggered.connect(self.show_hide_image)
        view_menu.addAction(toggle_image_visibility_btn)


        show_image_context_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                                   'View image context',
                                                    self)

        show_image_context_btn.setShortcut('Ctrl+C')

        def show_image_context():
            fname = os.path.splitext(self.png_fname)[0]
            tile_num_str = fname.split('_')[-1]
            fname = fname[:len(fname) - len('_' + tile_num_str)]
            proj_settings = json.load(open(self.proj_file_path))

            if 'original_image_dir' in proj_settings:
                original_image_dir = proj_settings['original_image_dir']
            else:
                # if the project doesn't yet have the path for the original
                # images then ask the user for it.
                msg = QtWidgets.QMessageBox()
                output = ("Original image directory not yet specified. "
                         "Please specify the original image directory.")
                msg.setText(output)
                msg.exec_()
                original_image_dir = QtWidgets.QFileDialog.getExistingDirectory()
                if not original_image_dir:
                    return
                else:
                    proj_settings['original_image_dir'] = original_image_dir
                    with open(self.proj_file_path, 'w') as file:
                        json.dump(proj_settings, file, indent=4)

            original_images = os.listdir(original_image_dir)
            original_fname = None
            for fname_with_ext in original_images:
                if os.path.splitext(fname_with_ext)[0] == fname:
                    original_fname = fname_with_ext
                    break 
            original_fpath = os.path.join(original_image_dir, original_fname)
            self.context_viewer = ContextViewer(original_fpath, self.graphics_view.image)
            self.context_viewer.show()

        show_image_context_btn.triggered.connect(show_image_context)
        print('view menu add action')
        view_menu.addAction(show_image_context_btn)


        def zoom_in():
            self.graphics_view.zoom *= 1.1
            self.graphics_view.update_zoom()

        def zoom_out():
            self.graphics_view.zoom /= 1.1
            self.graphics_view.update_zoom()

        zoom_in_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Zoom in', self)
        zoom_in_btn.setShortcut('+')
        zoom_in_btn.setStatusTip('Zoom in')
        zoom_in_btn.triggered.connect(zoom_in)
        view_menu.addAction(zoom_in_btn)

        zoom_out_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Zoom out', self)
        zoom_out_btn.setShortcut('-')
        zoom_out_btn.setStatusTip('Zoom out')
        zoom_out_btn.triggered.connect(zoom_out)
        view_menu.addAction(zoom_out_btn)

        # Network Menu
        network_menu = menu_bar.addMenu('Network')

        # start training
        start_training_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Start training', self)
        start_training_btn.triggered.connect(self.start_training)
        network_menu.addAction(start_training_btn)

        # stop training
        stop_training_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Stop training', self)
        stop_training_btn.triggered.connect(self.stop_training)
        network_menu.addAction(stop_training_btn)

        # # segment folder
        segment_folder_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Segment folder', self)

        def show_segment_folder():
            self.segment_folder_widget = SegmentFolderWidget(self.sync_dir,
                                                             self.instruction_dir)
            self.segment_folder_widget.show()
        segment_folder_btn.triggered.connect(show_segment_folder)
        network_menu.addAction(segment_folder_btn)

        # segment current image
        # segment_image_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
        #                                       'Segment current image', self)
        # segment_image_btn.triggered.connect(self.segment_current_image)
        # network_menu.addAction(segment_image_btn)

        self.add_measurements_menu(menu_bar)
        self.add_extras_menu(menu_bar, project_open=True)


    def add_measurements_menu(self, menu_bar):
        # Measurements
        measurements_menu = menu_bar.addMenu('Measurements')
        # object count
        object_count_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                             'Extract count', self)
        def show_extract_count():
            self.extract_count_widget = ExtractCountWidget()
            self.extract_count_widget.show()
        object_count_btn.triggered.connect(show_extract_count)
        measurements_menu.addAction(object_count_btn)

        # length
        length_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                       'Extract length', self)
        def show_extract_length():
            self.extract_length_widget = ExtractLengthWidget()
            self.extract_length_widget.show()
        length_btn.triggered.connect(show_extract_length)
        measurements_menu.addAction(length_btn)

        # region props
        region_props_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                             'Extract region properties', self)
        def show_extract_region_props():
            self.extract_regions_widget = ExtractRegionsWidget()
            self.extract_regions_widget.show()
        region_props_btn.triggered.connect(show_extract_region_props)
        measurements_menu.addAction(region_props_btn)


    def show_extract_comp(self):
        self.extract_comp_widget = ExtractCompWidget()
        self.extract_comp_widget.show()

    def show_conv_to_rve(self):
        """ show window to convert segmentations
            to RhizoVision Explorer compatible format """
        self.convert_to_rve_widget = ConvertSegForRVEWidget()
        self.convert_to_rve_widget.show()

    def stop_training(self):
        self.info_label.setText("Stopping training...")
        content = {"message_dir": self.message_dir}
        self.send_instruction('stop_training', content)

    def start_training(self):
        self.info_label.setText("Starting training...")
        content = {
            "model_dir": self.model_dir,
            "dataset_dir": self.dataset_dir,
            "train_annot_dir": self.train_annot_dir,
            "val_annot_dir": self.val_annot_dir,
            "seg_dir": self.seg_dir,
            "log_dir": self.log_dir,
            "message_dir": self.message_dir
        }
        self.send_instruction('start_training', content)

    def seg_checkbox_change(self, state):
        checked = (state == QtCore.Qt.Checked)
        if checked is not self.seg_visible:
            self.show_hide_seg()

    def annot_checkbox_change(self, state):
        checked = (state == QtCore.Qt.Checked)
        if checked is not self.annot_visible:
            self.show_hide_annot()

    def im_checkbox_change(self, state):
        checked = (state == QtCore.Qt.Checked)
        if checked is not self.image_visible:
            self.show_hide_image()

    def show_hide_seg(self):
        # show or hide the current segmentation.
        if self.seg_visible:
            self.seg_pixmap_holder.setPixmap(self.blank_pixmap)
            self.seg_visible = False
        else:
            self.seg_pixmap_holder.setPixmap(self.seg_pixmap)
            self.seg_visible = True
        self.vis_widget.seg_checkbox.setChecked(self.seg_visible)

    def show_hide_image(self):
        # show or hide the current image.
        # Could be useful to help inspect the segmentation or annotation
        if self.image_visible:
            self.image_pixmap_holder.setPixmap(self.black_pixmap)
            self.image_visible = False
        else:
            self.image_pixmap_holder.setPixmap(self.graphics_view.image)
            self.image_visible = True
        self.vis_widget.im_checkbox.setChecked(self.image_visible)

    def show_hide_annot(self):
        # show or hide the current annotations.
        # Could be useful to help inspect the background image
        if self.annot_visible:
            self.annot_pixmap_holder.setPixmap(self.blank_pixmap)
            self.annot_visible = False
        else:
            self.scene.annot_pixmap_holder.setPixmap(self.scene.annot_pixmap)
            self.annot_visible = True
        self.vis_widget.annot_checkbox.setChecked(self.annot_visible)

    def set_foreground_color(self, _event):
        self.scene.brush_color = self.scene.foreground_color
        self.update_cursor()

    def change_foreground_color(self, _event):
        foreground_set = (self.scene.brush_color == self.scene.foreground_color)
        show_alpha_option = QtWidgets.QColorDialog.ColorDialogOption(1)
        new_color = QtWidgets.QColorDialog.getColor(
            self.scene.foreground_color,
            options=show_alpha_option)

        if new_color.isValid():
            self.scene.foreground_color = new_color

        if foreground_set:
            self.scene.brush_color = self.scene.foreground_color
            self.update_cursor()

    def change_background_color(self, _event):
        background_set = (self.scene.brush_color == self.scene.background_color)
        show_alpha_option = QtWidgets.QColorDialog.ColorDialogOption(1)
        new_color = QtWidgets.QColorDialog.getColor(
            self.scene.background_color,
            options=show_alpha_option)

        if new_color.isValid():
            self.scene.background_color = new_color

        if background_set:
            self.scene.brush_color = self.scene.background_color
            self.update_cursor()

    def set_background_color(self, _event):
        self.scene.brush_color = self.scene.background_color
        self.update_cursor()

    def set_eraser_color(self, _event):
        self.scene.brush_color = self.scene.eraser_color
        self.update_cursor()

    def show_brush_size_edit(self):
         new_size, ok = QtWidgets.QInputDialog.getInt(self, "",
                 "Bursh size can also be altered by holding shift and moving the cursor. \n \n \n Select brush size", self.scene.brush_size, 1, 300, 1)
         if ok:
             self.scene.brush_size = new_size
             self.update_cursor()

    def update_cursor(self):
        brush_w = self.scene.brush_size * self.graphics_view.zoom * 0.93
        brush_w = max(brush_w, 3)

        canvas_w = max(brush_w, 30)
        pm = QtGui.QPixmap(round(canvas_w), round(canvas_w))
        pm.fill(Qt.transparent)
        painter = QtGui.QPainter(pm)

        painter.drawPixmap(round(canvas_w), round(canvas_w), pm)

        brush_rgb = self.scene.brush_color.toRgb()
        r, g, b = brush_rgb.red(), brush_rgb.green(), brush_rgb.blue()
        cursor_color = QtGui.QColor(r, g, b, 120)

        painter.setPen(QtGui.QPen(cursor_color, 3, Qt.SolidLine,
                                  Qt.RoundCap, Qt.RoundJoin))
        ellipse_x = int(round(canvas_w/2 - (brush_w)/2))
        ellipse_y = int(round(canvas_w/2 - (brush_w)/2))
        ellipse_w = int(round(brush_w))
        ellipse_h = int(round(brush_w))

        
        painter.drawEllipse(ellipse_x, ellipse_y, ellipse_w, ellipse_h)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 180), 2,
                                  Qt.SolidLine, Qt.FlatCap))

        # Draw black to show where cursor is even when brush is small
        painter.drawLine(0, round(canvas_w/2), round(canvas_w*2), round(canvas_w/2))
        painter.drawLine(round(canvas_w/2), 0, round(canvas_w/2), round(canvas_w*2))
        painter.end()

        cursor = QtGui.QCursor(pm)
        self.setCursor(cursor)

    def open_pre_segment_count_dialog(self):
        new_count, ok = QtWidgets.QInputDialog.getInt(self, "",
                                                      "Select Pre-Segment count",
                                                      self.pre_segment_count,
                                                      0, 100, 1)
        if ok:
            self.pre_segment_count = new_count
        # For some reason the events get confused and
        # scroll+pan gets switched on here.
        # Check if control key is up to disble it.
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if not modifiers & QtCore.Qt.ControlModifier:
            self.graphics_view.setDragMode(QtWidgets.QGraphicsView.NoDrag)

    def save_annotation(self):
        if self.scene.annot_pixmap:
            self.annot_path = maybe_save_annotation(self.proj_location,
                                                    self.scene.annot_pixmap,
                                                    self.annot_path,
                                                    self.png_fname,
                                                    self.train_annot_dir,
                                                    self.val_annot_dir)

            self.metrics_plot.add_file_metrics(os.path.basename(self.image_path))



