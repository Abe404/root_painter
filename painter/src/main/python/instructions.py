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

#pylint: disable=C0111
import os
import json
from pathlib import Path

def fix_path(path, sync_dir):
    """ fix path by removing everything that might
        be different on another machine """
    path_str = str(path)
    path_str = path_str.replace(str(sync_dir), '')
    path_str = path_str.replace('\\', '/') # server is unix.

    # sometimes the path has forward slash so the sync dir doesn't
    # get removed. In this case get the sync dir with forward
    # slash and then try to remove that.
    sync_dir_forward = str(sync_dir).replace('\\', '/')
    path_str = path_str.replace(sync_dir_forward, '')

    # only remove first character if it is a forward slash
    if path_str[0] == '/':
        return path_str[1:]
    return path_str


def fix_instruction_paths(old_config, sync_dir):
    # remove part of path that might be different on server.
    new_config = {}
    for k, v in old_config.items():
        if k == 'file_names':
            # names dont need anything removing
            new_config[k] = v
        elif isinstance(v, list):
            # if its a list fix each string in the list.
            new_list = []
            for e in v:
                new_val = fix_path(e, sync_dir)
                new_list.append(new_val)
            new_config[k] = new_list
        elif isinstance(v, str):
            new_config[k] = fix_path(v, sync_dir)
        elif isinstance(v, Path):
            new_config[k] = fix_path(v, sync_dir)
        else:
            new_config[k] = v
    return new_config


def send_instruction(name, content, instruction_dir, sync_dir):
    if not os.path.isdir(instruction_dir):
        raise FileNotFoundError(
            f"Could not connect to the sync directory. "
            f"The instructions folder was not found at:\n\n"
            f"{instruction_dir}\n\n"
            f"Please check that the trainer has been started.\n\n"
            f"You can verify your sync directory from the Extras menu:\n"
            f"  Extras > Open sync directory\n"
            f"  Extras > Specify sync directory")
    content = fix_instruction_paths(content, sync_dir)
    # append a hash to avoid over writing older instructions that
    # have not yet finished.
    hash_str = '_' + str(hash(json.dumps(content)))
    fpath = os.path.join(instruction_dir, name + hash_str)
    # if this instruction already exists then don't send again
    if not os.path.isfile(fpath):
        with open(fpath, 'w') as json_file:
            json.dump(content, json_file, indent=4)
