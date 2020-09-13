## RootPainter

Described in the paper "RootPainter: Deep Learning Segmentation of Biological Images with Corrective Annotation"

https://www.biorxiv.org/content/10.1101/2020.04.16.044461v2

RootPainter is a GUI-based software tool for the rapid training of deep neural networks for use in biological image analysis. 
RootPainter uses a client-server architecture, allowing users with a typical laptop to utilise a GPU on a more computationally powerful server.   

#### Server setup 

This is for a local server. If you want to run the server component of RootPainter on a remote machine then please consider [the sshfs server setup tutorial](https://github.com/Abe404/root_painter/blob/master/docs/server_setup_sshfs.md). You can also use Dropbox instead of sshfs or [Google Drive with Google Colab](https://colab.research.google.com/drive/104narYAvTBt-X4QEDrBSOZm_DRaAKHtA?usp=sharing) if you don't have access to a suitable GPU.

For the next steps I assume you have a suitable GPU and CUDA installed.


1. Clone the RootPainter code from the repository and then cd into the trainer directory (the server component).
```
git clone https://github.com/Abe404/root_painter.git
cd root_painter/trainer
```

2. To avoid alterating global packages. I suggest using a virtual environment. Create a virtual environment and activate it.
```
python -m venv env
source ./env/bin/activate
```

3. Install dependencies in the virtual environment. (takes over 3 minutes)
```
pip install -r requirements.txt
```

4. Run root painter. This will first create the sync directory.
```
python main.py
```
You will be prompted to input a location for the sync directory. This is the folder where files are shared between the client and server. I will use ~/root_painter_sync.
RootPainter will then create some folders inside ~/root_painter_sync.
The server should print the automatically selected batch size, which should be greater than 0. It will then start watching for instructions from the client.

You should now be able to see the folders created by RootPainter (datasets, instructions and projects) inside ~/Desktop/root_painter_sync on your local machine 
See [lung tutorial](docs/cxr_lung_tutorial.md) for an example of how to use RootPainter to train a model.

