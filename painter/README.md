
**Note** server must be running for the client to function.

Installers for OSX, Windows and Debian based linux distrubtions are avilable from the following URL (recommended for most users):
https://github.com/Abe404/root_painter/tags

Alternatively the painter source code can be ran directly using the python 3 interpreter.

# Install dependencies 
**Note:** I typically suggest using a virtual environment for this.

> pip install -r requirements.txt

# to run
> fbs run
Or alternatively 
> python src/main/python/main.py

# to build the application and installer

> fbs clean

**Note:** fbs only supports Python 3.6 so Python 3.6 must be used for the freeze and build steps.

> fbs freeze

**Note:** See install_fixes.py which will likely need to be ran to fix issues with scikit-image in the built progam prepared by the freeze command.

> python install_fixes.py

> fbs installer
