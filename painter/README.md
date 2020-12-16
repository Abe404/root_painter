
Installers for OSX, Windows and Debian based linux distrubtions are avilable from the following URL which is recommended for most users:
https://github.com/Abe404/root_painter/releases
Alternatively the painter source code can be ran directly using the python 3 interpreter.

The server (trainer) must be running for the client to function.

## Install dependencies 
I typically suggest using a virtual environment for this.

    > pip install -r requirements.txt

## to run
    > fbs run

Or alternatively 

    > python src/main/python/main.py

## to build the application and installer

    > fbs clean

fbs only supports Python 3.6 so Python 3.6 must be used for the freeze and build steps.

    > fbs freeze

See install_fixes.py which will likely need to be ran to fix issues with scikit-image in the built progam prepared by the freeze command.

    > python install_fixes.py
And then create the installer. Installers must be created on the target platform i.e windows installer must be created on windows and osx on osx etc.
    
    > fbs installer
