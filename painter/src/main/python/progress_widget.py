"""
Copyright (C) 2020-2023 Abraham George Smith

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

# disable docstring warnings
#pylint: disable=C0111, C0111

# Module has no member
#pylint: disable=I1101


import time
from PyQt6 import QtWidgets
from humanfriendly import format_timespan


class DoneMessageWindow(QtWidgets.QWidget):

    def __init__(self, parent, task, errors=[]):
        super().__init__()
        self.layout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel(self)

        self.layout.addWidget(self.label)
        label_msg = f'{task} complete \n'
        if len(errors):
            # if there are errors then show then in a text box.
            label_msg += f'\n There were {len(errors)} error(s):'
            error_msg = ''
            for e in errors:
                error_msg += '\n' + e
            self.text_area = QtWidgets.QPlainTextEdit(self)
            self.layout.addWidget(self.text_area)
            self.text_area.insertPlainText(error_msg)
            self.setGeometry(200, 200, 500, 400)

        self.label.setText(label_msg)
        self.ok_button = QtWidgets.QPushButton("ok")
        self.layout.addWidget(self.ok_button)
        self.ok_button.clicked.connect(self.close)
        self.setLayout(self.layout)




class BaseProgressWidget(QtWidgets.QWidget):
    """
    Once a process starts this widget displays progress
    """
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.start_time = None
        self.initUI()

    def get_seconds_remaining(self, processed_so_far, total):
        seconds_so_far = time.time() - self.start_time
        seconds_per_image = seconds_so_far / processed_so_far
        remaining_images = total - processed_so_far
        estimated_remaining_seconds = seconds_per_image * remaining_images
        return estimated_remaining_seconds

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()
        self.layout = layout # to add progress bar later.
        self.setLayout(layout)
        # info label for user feedback
        info_label = QtWidgets.QLabel()
        info_label.setText("")
        layout.addWidget(info_label)
        self.info_label = info_label
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.layout.addWidget(self.progress_bar)
        self.setWindowTitle(self.task)

    def onCountChanged(self, value, total):
        # 'not self.start_time' because value could be greater than 2
        # (and self.start_time could be None, causing bugs)
        # if segmenting the remaining images from a partially segmented dataset
        if value < 2 or not self.start_time:
            # first image could take a while due to the initial delay in syncing
            # so start estimating remaining time from second image onwards.
            self.start_time = time.time() 
            self.info_label.setText(f'{self.task} {value}/{total}. '
                                    'Estimating time remaining..')
        else:
            # value-1 because start_time is once the first image has completed.
            seconds_remaining = self.get_seconds_remaining(value-1, total)
            self.info_label.setText(f'{self.task} {value}/{total}. '
                                    'Estimated time remaining: '
                                    f'{format_timespan(seconds_remaining)}')
        self.progress_bar.setValue(value)

    def done(self, errors=[]):
        self.done_window = DoneMessageWindow(self, self.task, errors)
        self.done_window.show()
        self.close()
