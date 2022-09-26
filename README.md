## RootPainter

Described in the paper "RootPainter: Deep Learning Segmentation of Biological Images with Corrective Annotation"

Published peer-reviewed paper available in the New Phytologist at:
[https://doi.org/10.1111/nph.18387](https://doi.org/10.1111/nph.18387)

BioRxiv Pre-print available at:
[https://www.biorxiv.org/content/10.1101/2020.04.16.044461v2](https://www.biorxiv.org/content/10.1101/2020.04.16.044461v2)

RootPainter is a GUI-based software tool for the rapid training of deep neural networks for use in biological image analysis. 
RootPainter uses a client-server architecture, allowing users with a typical laptop to utilise a GPU on a more computationally powerful server.   

### Getting started quickly

 I suggest the [colab tutorial](https://colab.research.google.com/drive/104narYAvTBt-X4QEDrBSOZm_DRaAKHtA?usp=sharing).
 
### A video demonstrating how to train and use a model is also available to [download](https://nph.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1111%2Fnph.18387&file=nph18387-sup-0002-VideoS1.mp4)

### Client Downloads

See [releases](https://github.com/Abe404/root_painter/releases) 

If you are not confident with the linux administration then to get started quickly I suggest the [colab tutorial](https://colab.research.google.com/drive/104narYAvTBt-X4QEDrBSOZm_DRaAKHtA?usp=sharing).

#### Server setup 

The following instructions are for a local server. If you do not have linux running a suitable NVIDIA GPU with at least 8GB of GPU memory then my current recommendation is to run via Google colab. A publicly available notebook is available at [Google Drive with Google Colab](https://colab.research.google.com/drive/104narYAvTBt-X4QEDrBSOZm_DRaAKHtA?usp=sharing).

Other options to run the server component of RootPainter on a remote machine include the [the sshfs server setup tutorial](https://github.com/Abe404/root_painter/blob/master/docs/server_setup_sshfs.md). You can also use Dropbox instead of sshfs.


For the next steps I assume you have a suitable GPU and CUDA installed.

1. To install the RootPainter trainer:

```
pip install root-painter-trainer
```

2. To run the trainer.  This will first create the sync directory.

```
start-trainer
```

You will be prompted to input a location for the sync directory. This is the folder where files are shared between the client and server. I will use ~/root_painter_sync.
RootPainter will then create some folders inside ~/root_painter_sync.
The server should print the automatically selected batch size, which should be greater than 0. It will then start watching for instructions from the client.

You should now be able to see the folders created by RootPainter (datasets, instructions and projects) inside ~/Desktop/root_painter_sync on your local machine 
See [lung tutorial](docs/cxr_lung_tutorial.md) for an example of how to use RootPainter to train a model.

