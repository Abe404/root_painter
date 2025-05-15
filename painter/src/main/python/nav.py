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

# pylint: disable=C0111, I1101, E0611
import os

from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtCore import Qt


class NavWidget(QtWidgets.QWidget):
    """ Shows next and previous buttons as well as image position in folder.
    """
    file_change = QtCore.pyqtSignal(str)

    def __init__(self, all_fnames, annot_dirs):
        super().__init__()
        self.image_path = None
        self.all_fnames = all_fnames
        self.annot_dirs = annot_dirs
        self.initUI()

    def initUI(self):
        # container goes full width to allow contents to be center aligned within it.
        nav = QtWidgets.QWidget()
        nav_layout = QtWidgets.QHBoxLayout()

        # && to escape it and show single &
        self.prev_image_button = QtWidgets.QPushButton('< Previous')
        self.prev_image_button.setFocusPolicy(Qt.NoFocus)
        self.prev_image_button.clicked.connect(self.show_prev_image)
        nav_layout.addWidget(self.prev_image_button)
        self.nav_label = QtWidgets.QLabel()
        nav_layout.addWidget(self.nav_label)

        # && to escape it and show single &
        self.next_image_button = QtWidgets.QPushButton('Save && Next >')
        self.next_image_button.setFocusPolicy(Qt.NoFocus)
        self.next_image_button.clicked.connect(self.show_next_image)
        nav_layout.addWidget(self.next_image_button)

        # left, top, right, bottom
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(2)
        nav.setLayout(nav_layout)
        nav.setMaximumWidth(600)

        container_layout = QtWidgets.QHBoxLayout()
        container_layout.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(nav)
        self.setLayout(container_layout)
        container_layout.setContentsMargins(0, 0, 0, 0)

    def get_path_list(self, dir_path):
        all_files = self.all_fnames
        all_paths = [os.path.abspath(os.path.join(os.path.abspath(dir_path), a))
                     for a in all_files]
        return all_paths


    def select_highest_entropy_image(self):
        """
        Return filename of active image with highest entropy among
        active images *after* the current image, or None if unavailable.
        """
        active_images_path = os.path.join(self.parent.proj_location, "active_images.txt")
        uncertainty_dir = os.path.join(self.parent.proj_location, "uncertainty")

        if not (os.path.isfile(active_images_path) and os.path.isdir(uncertainty_dir)):
            return None

        # Get active images set
        with open(active_images_path, "r") as f:
            active_images = {line.strip() for line in f if line.strip()}

        # Determine current image position
        current_fname = os.path.basename(self.image_path)
        try:
            current_idx = self.all_fnames.index(current_fname)
        except ValueError:
            return None  # safety fallback

        # Only consider images after current
        future_images = set(self.all_fnames[current_idx+1:])
        candidate_images = active_images.intersection(future_images)

        # Gather entropy scores for candidates
        entropy_scores = {}
        for fname in candidate_images:
            score_file = os.path.join(uncertainty_dir, fname + ".txt")
            if os.path.isfile(score_file):
                try:
                    with open(score_file, "r") as f_score:
                        entropy = float(f_score.read().strip())
                        entropy_scores[fname] = entropy
                except Exception:
                    continue

        if not entropy_scores:
            return None

        return max(entropy_scores, key=entropy_scores.get)


    def prefill_next_active_image(self):
        """
        Check if next even-index image (after current) is active and replace with highest entropy image.
        """
        current_idx = self.all_fnames.index(os.path.basename(self.image_path))
        next_idx = current_idx + 1
        if next_idx >= len(self.all_fnames):
            return  # no next image

        # Check if next image is in active list
        active_images_path = os.path.join(self.parent.proj_location, "active_images.txt")
        if not os.path.isfile(active_images_path):
            return

        with open(active_images_path, "r") as f:
            active_images = {line.strip() for line in f if line.strip()}

        next_image_name = self.all_fnames[next_idx]
        if next_image_name not in active_images:
            return  # next image is not active → no replacement

        # Select best future active image
        next_entropy_fname = self.select_highest_entropy_image()
        if not next_entropy_fname:
            return

        # Replace only if different
        if self.all_fnames[next_idx] != next_entropy_fname:
            print(f"Prefill active: replacing {self.all_fnames[next_idx]} → {next_entropy_fname}")
            self.all_fnames[next_idx] = next_entropy_fname

            # Save updated list to project file
            import json
            project_file = self.parent.proj_file_path
            with open(project_file, 'r') as f:
                project_data = json.load(f)
            project_data['file_names'] = self.all_fnames
            with open(project_file, 'w') as f:
                json.dump(project_data, f, indent=4)


    def show_next_image(self):
        self.next_image_button.setEnabled(False)
        self.next_image_button.setText('Loading..')
        QtWidgets.QApplication.processEvents()

        # normal next
        dir_path, _ = os.path.split(self.image_path)
        all_paths = self.get_path_list(dir_path)
        cur_idx = all_paths.index(self.image_path)
        next_idx = (cur_idx + 1) % len(all_paths)
        next_fname = os.path.basename(all_paths[next_idx])

        # if next image is active, use highest entropy image instead
        next_entropy_fname = None
        active_images_path = os.path.join(self.parent.proj_location, "active_images.txt")
        if os.path.isfile(active_images_path):
            with open(active_images_path, "r") as f:
                active_images = {line.strip() for line in f if line.strip()}
            if next_fname in active_images:
                next_entropy_fname = self.select_highest_entropy_image()

        if next_entropy_fname:
            next_image_path = os.path.join(dir_path, next_entropy_fname)
        else:
            next_image_path = all_paths[next_idx]

        # update current image
        self.image_path = next_image_path
        self.file_change.emit(self.image_path)
        self.update_nav_label()

        # prefill future active slot
        self.prefill_next_active_image()


    def show_prev_image(self):
        dir_path, _ = os.path.split(self.image_path)
        all_paths = self.get_path_list(dir_path)
        cur_idx = all_paths.index(self.image_path)
        next_idx = cur_idx - 1
        if next_idx <= 0:
            next_idx = 0
        self.image_path = all_paths[next_idx]
        self.file_change.emit(self.image_path)
        self.update_nav_label()

    def update_nav_label(self):
        dir_path, _ = os.path.split(self.image_path)
        all_paths = self.get_path_list(dir_path)
        cur_idx = all_paths.index(os.path.abspath(self.image_path))
        
        annotation_count = 0
        for annot_dir in self.annot_dirs:
            annotation_count += len(os.listdir(annot_dir))
        self.nav_label.setText(
            f'{cur_idx + 1} / {len(all_paths)} ({annotation_count} Annotated)')
