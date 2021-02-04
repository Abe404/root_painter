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
import shutil
from os.path import join

def fix_app():
    """
    Unfortunately fbs/pyInstaller is no longer capable of building
    RootPainter on it's own.

    It may be that I upgraded library versions to ones which are no longer
    supported i.e scikit-image.
    
    We will need to copy accross some dependencies to get it working again.
    """
    
    # If you are using Mac, then you will have the following folder after
    # running the fbs freeze command
    # ./target/RootPainter.app
    is_mac = os.path.isdir('./target/RootPainter.app')
    is_linux = os.path.exists('./target/RootPainter/RootPainter')
    
    # or the following folder on windows
    is_windows = os.path.exists('target\RootPainter\RootPainter.exe')
   
    print('is_windows', is_windows)
    print('is_mac', is_mac)
    print('is_linux', is_linux)

    # If you try to run RootPainter on the command line like so:
    # ./target/RootPainter.app/Contents/MacOS/RootPainter 
    # Then you may receive the following error:
    # File "skimage/feature/orb_cy.pyx", line 12, in init skimage.feature.orb_cy
    # ModuleNotFoundError: No module named 'skimage.feature._orb_descriptor_positions'
    
    # It seems the built application is missing some crucial files from skimage.
    # To copy these accross we will assume you have an environment created with venv (virtual env)
    # in the current working directory call 'env'
    env_dir = './env'
    assert os.path.isdir(env_dir), f'Could not find env folder {env_dir}'
    if is_mac:
        site_packages_dir = os.path.join(env_dir, 'lib/python3.6/site-packages')
        build_dir = './target/RootPainter.app/Contents/MacOS/'
    elif is_windows:
        site_packages_dir = os.path.join(env_dir, 'Lib', 'site-packages')
        build_dir = './target/RootPainter'
    elif is_linux:
        site_packages_dir = os.path.join(env_dir, 'lib/python3.7/site-packages')
        build_dir = './target/RootPainter'


    # Copy missing orb files
    skimage_dir = join(site_packages_dir, 'skimage')

    orbpy_src = join(skimage_dir, 'feature/_orb_descriptor_positions.py')
    orbpy_target = join(build_dir, 'skimage/feature/_orb_descriptor_positions.py')
    shutil.copyfile(orbpy_src, orbpy_target)

    # copy missing orb plugin file
    orbtxt_src = join(skimage_dir, 'feature/orb_descriptor_positions.txt')
    orbtxt_target = join(build_dir, 'skimage/io/_plugins/orb_descriptor_positions.txt')
    shutil.copyfile(orbtxt_src, orbtxt_target)


    # Copy missing tiffile plugin
    tif_src = join(skimage_dir, 'io/_plugins/tifffile_plugin.py')
    tif_target = join(build_dir, 'skimage/io/_plugins/tifffile_plugin.py')
    shutil.copyfile(tif_src, tif_target)


if __name__ == '__main__':
    fix_app()

