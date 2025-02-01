# Run Painter from source
It is recommended that you run the pre-compilled executable for your OS instead of running the painter software from source. However, if you'd like to help improve the software, the latest version has not been compilled, or for whatever reason you would rather execute the python source code instead of using the pre-compilled executables, this guide is for you. 

### Step by step guide:

1. [Make sure that you have python installed](https://www.python.org/downloads/) and your [PATH enviroment variable contains the path to the python executable](#python-missing-from-path)

2.  Install the python modules required to run the painter software on a virtual enviroment:

    Start venv
    ```python
    python -m venv root_painter_venv 
    ```
    activate venv On macOS/Linux
    ```python
    source root_painter_venv/bin/activate
    ```

    Activate venv On Windows
    ```python
    root_painter_venv\Scripts\activate
    ```

    Install modules with pip
    ```python
    pip install -r painter\requirements.txt
    ```


3. Launch painter

    ```
    python painter\src\main\python\main.py
    ```


## Python missing from PATH

On windows you might get this error if Python is not in your PATH

```
Python was not found; run without arguments to install from the Microsoft Store, or disable this shortcut from Settings > Manage App Execution Aliases.
```

to fix it, first locate your python installation

try typing 'python' into the windows search bar, click 'open file location', locate the python shortcut, then right click and click 'open file location'

you might find it here 
```
C:\Users\YourUsername\AppData\Local\Programs\Python
```
or here 
```
C:\Program Files\PythonXX
```

Note down the folder path (e.g., C:\Users\YourUsername\AppData\Local\Programs\Python\PythonXX).



3. Add Python to System PATH
If Python is installed but not recognized, add it to the PATH environment variable:

Open Environment Variables:
Press Win + R, type sysdm.cpl, and hit Enter.
Go to Advanced → Click Environment Variables.

Edit PATH:
In **System variables**, find **Path** and click **Edit**.

In the **Edit enviroment variable** window:
Click New and add:

C:\Users\YourUsername\AppData\Local\Programs\Python\PythonXX
C:\Users\YourUsername\AppData\Local\Programs\Python\PythonXX\Scripts

Save & Restart
Click OK → Restart your PC.
Try running python --version again.
