import PyInstaller.__main__


# to run this script, first activate the environment.
# if you don't have an environment.
# create an environment with
# python3 -m venv env
# activate with
# source env/bin/activate
# and install environment requirements with 
# pip install -r requirements.txt


# Q: Should we use many arguments passed to pyinstaller, or should we use a spec file?
"""
A:
From: https://github.com/pyinstaller/pyinstaller/blob/fbf7948be85177dd44b41217e9f039e1d176de6b/doc/spec-files.rst

`For many uses of PyInstaller you do not need to examine or modify the spec
file. It is usually enough to give all the needed information (such as
hidden imports) as options to the pyinstaller command and let it run.

There are four cases where it is useful to modify the spec file:

1. When you want to bundle data files with the app.
2. When you want to include run-time libraries (.dll or .so files) that
   PyInstaller does not know about from any other source.
3. When you want to add Python run-time options to the executable.
4. When you want to create a multiprogram bundle with merged common modules.'


So the plan for now is to go with more options passed to PyInstaller, and then
only move onto using a custom spec file when this is recommended by the above
advice.
"""
import shutil
import os
from sys import platform

# pyinstaller expects icon file to be in the dist folder
if not os.path.isdir('dist'):
    os.makedirs('dist')
    os.makedirs('dist/tmp_files')

# Icon for used for mac or windows (linux has different mechanism, see make_deb_file.sh)
icon_fname = 'favicon2.ico' # ico for windows.
if platform == "darwin":
    icon_fname = 'Icon.icns' # icns for mac
    # icon path should be relative to the dist folder
    shutil.copyfile(os.path.join('src/main/icons', icon_fname),
                    os.path.join('dist', icon_fname))

else:
    # icon path should be relative to the dist folder
    shutil.copyfile(os.path.join('src/main/icons', icon_fname), icon_fname)

    shutil.copyfile(os.path.join('src/main/icons', icon_fname),
                    os.path.join('dist/tmp_files', icon_fname))


# pyinstaller command line argument documentation is available from:
# https://pyinstaller.org/en/stable/usage.html

PyInstaller.__main__.run([
    # --noconfirm: don't ask user to confirm when deleting existing files in dist folder.
    '--noconfirm',

    # --clean: Clean PyInstaller cache and remove temporary files before building.
    '--clean',

    # hidden imports added based on this advice:    
    # https://github.com/pyqtgraph/pyqtgraph/issues/2179
    '--hidden-import', 'pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_pyqt5',
    '--hidden-import', 'pyqtgraph.graphicsItems.PlotItem.plotConfigTemplate_pyqt5',
    '--hidden-import', 'pyqtgraph.imageview.ImageViewTemplate_pyqt5',

    # Where to put all the temporary work files .log, .pyz and etc. (default: ./build)
    '--workpath', 'dist/tmp_files',
    
    # --debug==all provides a significant amount of diagnostic information.
    # This can be useful during development of a complex package, or when your
    # app doesn’t seem to be starting, or just to learn how the runtime works.
    # '--debug', 'all', 

    # Name to assign to the bundled app and spec file (default: first script’s basename)
    '--name', 'RootPainter',

    # I dont think this makes a difference for ubuntu, but I think it will help on OSX.
    # see https://pyinstaller.org/en/stable/usage.html?highlight=icon#cmdoption-i
    # -i FILE.exe,ID or FILE.icns or "NONE"> FILE.ico: apply the icon to a Windows
    # executable. FILE.exe,ID: extract the icon with ID from an exe. FILE.icns: apply
    # the icon to the .app bundle on Mac OS. Use "NONE" to not apply any icon,
    # thereby making the OS to show some default (default: apply PyInstaller's icon)
    # thereby making the OS to show some default (default: apply PyInstaller's icon)
    #'-i', './src/main/icons/Icon.ico',  # windows
    #'--icon', os.path.join('dist', 'src', 'main', 'icons', icon_fname),  # should be relative to the dist directory
    '--icon', icon_fname,  # should be relative to the dist directory
    # I dont actually use the spec file yet, so put the auto-generated one in dist to avoid cluttering the repo
    '--specpath', 'dist', 

    # Windows and Mac OS X: do not provide a console window for standard i/o.
    # On Mac OS this also triggers building a Mac OS .app bundle. On Windows
    # this option is automatically set if the first script is a ‘.pyw’ file.
    # This option is ignored on *NIX systems.
    '--windowed',

    # Mac OS .app bundle identifier is used as the default unique program name
    # for code signing purposes. The usual form is a hierarchical name in reverse DNS
    # notation. For example: com.mycompany.department.appname (default: first
    # script’s basename)
    '--osx-bundle-identifier', 'com.rootpainter',

    # scriptname: Name of scriptfile to be processed.
    'src/main/python/main.py'
])
