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
import os

def ls(dir_path):
    # Don't show hidden files
    # These can happen due to issues like file system 
    # synchonisation technology. RootPainter doesn't use them anywhere
    fnames = os.listdir(dir_path)
    fnames = [f for f in fnames if f[0] != '.']
    return fnames
