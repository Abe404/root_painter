### RootPainter Server Setup with sshfs

This is is a guide for getting started with the RootPainter software
using sshfs to connect the client and server components. RootPainter is described in [this paper.](https://www.biorxiv.org/content/10.1101/2020.04.16.044461v1)

I assume you have ssh access to a linux server with a suitable GPU and CUDA installed.

1. SSH into your server to set up the server component of RootPainter.
```
ssh username@xxx.xxx.xxx.xxx
```

3. Clone the RootPainter code from the repository and then cd into the trainer directory (the server component).
```
git clone --branch 0.2.3 https://github.com/Abe404/root_painter
cd root_painter/trainer
```

4. To avoid alterating any global packages. I suggest using a virtual environment. Create a virtual environment and activate it.
```
python -m venv env
source ./env/bin/activate
```

5. Install dependencies in the virtual environment. (takes ~3 minutes)
```
pip install -r requirements.txt
```

6. Run root painter. This will first create the sync directory.
```
python main.py
```
You will be prompted to input a location for the sync directory. This is the folder where files are shared between the client and server. I will use ~/root_painter_sync.
RootPainter will then create some folders inside ~/root_painter_sync
The server should print the automatically selected batch size, which should be greater than 0.
```
---

7. To mount the sync directory from your local machine you will need to install sshfs locally (SSH Filesystem client).

Debian / Ubuntu:
```
sudo apt-get install sshfs
```

OSX:
```
brew cask install osxfuse
```

Windows:
[Digital ocean has a guide](https://www.digitalocean.com/community/tutorials/how-to-use-sshfs-to-mount-remote-file-systems-over-ssh)

8. Create the directory and mount the drive locally using sshfs. 
```
mkdir ~/Desktop/root_painter_sync
sudo sshfs -o allow_other,default_permissions username@xxx.xxx.xxx.xxx:/home/username/root_painter_sync ~/Desktop/root_painter_sync
```

You should now be able to see the folders created by RootPainter (datasets, instructions and projects) inside ~/root_painter_sync on your local machine 
See [lung tutorial](cxr_lung_tutorial.md) for an example of how to use RootPainter to train a model.

