
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
print('Initialising RootPainter')
#pylint: disable=I1101,C0111,W0201,R0903,E0611,R0902,R0914,W0703
import sys
import os
from pathlib import Path
import json
import traceback
from PyQt5 import QtWidgets

from root_painter import RootPainter


def ensure_sync_dir_ready(sync_dir):
    """Create sync dir and required subfolders if they don't exist.
    Mirrors trainer/src/startup.py ensure_required_folders_exist()."""
    os.makedirs(sync_dir, exist_ok=True)
    for subfolder in ['projects', 'datasets', 'instructions',
                      'executed_instructions', 'failed_instructions']:
        os.makedirs(os.path.join(sync_dir, subfolder), exist_ok=True)


def init_root_painter():
    settings_path = os.path.join(Path.home(), 'root_painter_settings.json')
    try:
        app = QtWidgets.QApplication([])
        if not os.path.isfile(settings_path):
            default_sync = os.path.join(Path.home(), 'root_painter_sync')
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle('RootPainter')
            msg.setText(f'Create sync directory at {default_sync}?')
            msg.setInformativeText(
                'RootPainter needs a sync directory to store projects, '
                'datasets and communicate with the training server.')
            msg.setStandardButtons(
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            msg.setDefaultButton(QtWidgets.QMessageBox.Yes)
            if msg.exec_() == QtWidgets.QMessageBox.Yes:
                dir_path = default_sync
            else:
                dir_path = QtWidgets.QFileDialog.getExistingDirectory()
                if not dir_path:
                    exit()
            ensure_sync_dir_ready(dir_path)
            with open(settings_path, 'w') as json_file:
                content = {
                    "sync_dir": os.path.abspath(dir_path)
                }
                json.dump(content, json_file, indent=4)
        sync_dir = Path(json.load(open(settings_path, 'r'))['sync_dir'])

        def reopen():
            main_window = RootPainter(sync_dir)
            main_window.closed.connect(reopen)
            main_window.show()

        main_window = RootPainter(sync_dir)
        # close project causes reopen with missing project UI
        main_window.closed.connect(reopen)
        main_window.show()

        exit_code = app.exec_()
    except Exception as e:
        msg = QtWidgets.QMessageBox()
        output = f"""
        repr(e): {repr(e)}
        traceback.format_exc(): {traceback.format_exc()}
        sys.exec_info()[0]: {sys.exc_info()[0]}
        """
        msg.setText(output)
        msg.exec_()
    else:
        sys.exit(exit_code)

if __name__ == '__main__':
    init_root_painter()
