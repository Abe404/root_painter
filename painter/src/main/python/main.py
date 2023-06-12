
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
from PyQt6 import QtWidgets

from root_painter import RootPainter

def init_root_painter():
    settings_path = os.path.join(Path.home(), 'root_painter_settings.json')
    try:
        app = QtWidgets.QApplication([])
        # if the settings file does not exist then create it with
        # a user specified sync_dir
        if not os.path.isfile(settings_path):
            msg = QtWidgets.QMessageBox()
            output = "Sync directory not specified. Please specify a sync directory."
            msg.setText(output)
            msg.exec_()
            dir_path = QtWidgets.QFileDialog.getExistingDirectory()
            if not dir_path:
                exit()
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
        msg.exec()
    else:
        sys.exit(exit_code)

if __name__ == '__main__':
    init_root_painter()
