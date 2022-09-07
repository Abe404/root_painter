
Installers for OSX, Windows and Debian based linux distrubtions are avilable from the following URL which is recommended for most users:
https://github.com/Abe404/root_painter/releases
Alternatively the painter source code can be ran directly using the python 3 interpreter as outlined in the following instructions.

The server (trainer) must be running for the client to function.

## Install dependencies 
I typically suggest using a virtual environment for this.

    > pip install -r requirements.txt

### Mac

For generating the installer on MacOSX, the `create-dmg` command is required. It is recommended to install using homebrew to install:

    > brew install create-dmg

### Windows

For generating executable on Windows, ensure that the NSIS tools are installed and available in path: https://nsis.sourceforge.io/Main_Page

Also ensure that C++ Redistributable for Visual Studio 2012 is installed: https://www.microsoft.com/en-us/download/details.aspx?id=30679"

And the Windows 10 SDK from https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk.

## to run

    > python src/main/python/main

Or alternatively 

    > python src/main/python/main.py

## to build the application and installer

    > python src/build/clean.py
    > python src/build/freeze.py

And then create the installer. Installers must be created on the target platform i.e windows installer must be created on windows and osx on osx etc.

    > python src/build/installer


### Ubuntu (deb file)

## to build the application

    > python src/build/run_pyinstaller.py

## to build the linux package (deb file)

    > bash src/build/make_installer.sh

The output installer will be located at dist/RootPainter.deb

## Installer the installer
> sudo dpkg -i dist/RootPainter.deb
