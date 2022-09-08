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
    '--icon', 'src/main/icons/Icon.ico',

    # I dont actually use the spec file yet, so put the auto-generated one in dist to avoid cluttering the repo
    '--specpath', 'dist', 

    # Windows and Mac OS X: do not provide a console window for standard i/o.
    # On Mac OS this also triggers building a Mac OS .app bundle. On Windows
    # this option is automatically set if the first script is a ‘.pyw’ file.
    # This option is ignored on *NIX systems.
    '--windowed',

    # scriptname: Name of scriptfile to be processed.
    'src/main/python/main.py'
])
