### RootPainter Server Setup with sshfs

This is is a guide for getting started with the RootPainter software
using sshfs to connect the client and server components. RootPainter is described in [this paper.](https://www.biorxiv.org/content/10.1101/2020.04.16.044461v1)

This page includes instructions for setting up the server. After completing these instructions, you can follow the [lung tutorial](cxr_lung_tutorial.md) for an example of how to use RootPainter to train a model using the graphical client.

I assume you have ssh access to a linux server with a suitable GPU and CUDA installed.

1. SSH into your server to set up the server component of RootPainter.
```
ssh username@xxx.xxx.xxx.xxx
```

2. Clone the RootPainter code from the repository and then cd into the trainer directory (the server component).
```
git clone https://github.com/Abe404/root_painter.git
cd root_painter/trainer
```

3. To avoid alterating global packages. I suggest using a virtual environment. Create a virtual environment and activate it. This tutorial has been tested using python 3.7
```
python -m venv env
source ./env/bin/activate
```

4. Install dependencies in the virtual environment. (takes over 3 minutes)
```
pip install torch
pip install -r requirements.txt
```

5. Run root painter. This will first create the sync directory.
```
python main.py
```
You will be prompted to input a location for the sync directory. This is the folder where files are shared between the client and server. I will use ~/root_painter_sync.
RootPainter will then create some folders inside ~/root_painter_sync.
The server should print the automatically selected batch size, which should be greater than 0. It will then start watching for instructions from the client.

If you see a batch size above 0 and 'Started main loop. Checking for instructions..' in the console output then this means you have been successful in starting the server and can move onto the next step.

6. To mount the sync directory from your local machine you will need to install sshfs locally (SSH Filesystem client).

**If you are using Debian / Ubuntu**
```
sudo apt-get install sshfs
```

**If you are using mac / OSX:**

Download the [macFUSE dmg](https://github.com/osxfuse/osxfuse/releases).
Open the OSXFuse dmg and run the installer.
Similarly, download the [SSHFS pkg](http://osxfuse.github.io)
and run the downloaded SSHFS installer.


**If you are using Windows:**
See [sshfs-win](https://github.com/billziss-gh/sshfs-win)


7. Create the directory that you will mount.

You will need to run this command on your local machine.

```
mkdir ~/Desktop/root_painter_sync
```


8.   mount the drive locally using sshfs. 

You will need to run the following command on your local machine. You should replace username with the username of the corresponding user on the server (the one you use when sshing into your server). You should also replace xxx.xxx.xxx.xxx with the ip address of your server.
```
sudo sshfs -o allow_other,default_permissions username@xxx.xxx.xxx.xxx:/home/username/root_painter_sync ~/Desktop/root_painter_sync
```

You should now be able to see the folders created by RootPainter (datasets, instructions and projects) inside ~/Desktop/root_painter_sync on your local machine 
See [lung tutorial](cxr_lung_tutorial.md) for an example of how to use RootPainter to train a model.

