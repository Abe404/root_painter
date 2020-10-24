"""
Check for required folders on startup and create if required

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


import os
import glob
from pathlib import Path
import json

def startup_setup(settings_path):
    """
    1. if the settings file doesn't exist
       then ask the user for a sync dir and create the settings file
       else if it does exist then read the sync dir from it.
    """
    if settings_path is not None:
        # Get sync dir from settings file.
        if os.path.isfile(settings_path):
            sync_dir = Path(json.load(open(settings_path, 'r'))['sync_dir'])
            sync_dir_abs = os.path.abspath(sync_dir)
        else:
            # Or if the settings file doesn't exist get a sync_dir
            # from the user and save it to a settings file.
            sync_dir = input("Please specify RootPainter sync directory")
            sync_dir = os.path.expanduser(sync_dir)
            sync_dir_abs = os.path.abspath(sync_dir)
            with open(settings_path, 'w') as json_file:
                content = {
                    "sync_dir": sync_dir_abs
                }
                print(f'Writing {sync_dir_abs} to {settings_path}')
                json.dump(content, json_file, indent=4)


def ensure_required_folders_exist(sync_dir):
    """
    1. If the sync dir doesn't exist then create it.
    2. If the required sync dir subfolders don't exist then create them.
    """

    sync_dir_abs = os.path.abspath(sync_dir)
    # If sync_dir doesn't exist then create it.
    if not os.path.isdir(sync_dir_abs):
        print('Creating', sync_dir_abs)
        os.mkdir(sync_dir_abs)

    # RootPainter requires some folders to run. If they aren't already
    # in the sync_dir then create them
    required_subfolders = ['projects', 'datasets', 'instructions']
    for subfolder in required_subfolders:
        subfolder_path = os.path.join(sync_dir_abs, subfolder)
        if not os.path.isdir(subfolder_path):
            print('Creating', subfolder_path)
            os.mkdir(subfolder_path)

    # clear the instructions at the start
    # executing old instructions can get confusing for users
    instructions_dir = os.path.join(sync_dir_abs, 'instructions')
    files = glob.glob(instructions_dir + '/*')
    for f in files:
        os.remove(f)
