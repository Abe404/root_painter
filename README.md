## RootPainter

RootPainter is a GUI-based software tool for the rapid training of deep neural networks for use in biological image analysis. 
RootPainter uses a client-server architecture, allowing users with a typical laptop to utilise a GPU on a more computationally powerful server.  

A detailed description is available in the paper published in the New Phytologist  [RootPainter: Deep Learning Segmentation of Biological Images with Corrective Annotation](https://doi.org/10.1111/nph.18387)

![RootPainter Interface](https://user-images.githubusercontent.com/376295/224013411-cb44c7c2-5c72-4819-98a3-6c0ab8b9ea4d.png)

To see a list of work using (or citing) the RootPainter paper, please see the [google scholar page](https://scholar.google.com/scholar?cites=12740268016453642124)

A BioRxiv Pre-print (earlier version of the paper) is available at:
[https://www.biorxiv.org/content/10.1101/2020.04.16.044461v2](https://www.biorxiv.org/content/10.1101/2020.04.16.044461v2)


### Getting started quickly

 I suggest the [colab tutorial](https://colab.research.google.com/drive/104narYAvTBt-X4QEDrBSOZm_DRaAKHtA?usp=sharing).
 
 A  shorter [mini guide](https://github.com/Abe404/root_painter/blob/master/docs/mini_guide.md) is available including more concise instruction, that could be used as reference. I suggest the paper, videos and then colab tutorial to get an idea of how the software interface could be used and then this mini guide for reference to help remember each of the key steps to get from raw data to final measurements. 
 
 
 

 
### Videos
A video demonstrating how to train and use a model is available to [download](https://nph.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1111%2Fnph.18387&file=nph18387-sup-0002-VideoS1.mp4)

There is a [youtube video](https://www.youtube.com/watch?v=73u73tBvRO4) of a workshop explaining the background behind the software and covering using the colab notebook to train and use a root segmentation model.


### Client Downloads

See [releases](https://github.com/Abe404/root_painter/releases) 

If you are not confident installing and running python applications on the command line then to get started quickly I suggest the [colab tutorial](https://colab.research.google.com/drive/104narYAvTBt-X4QEDrBSOZm_DRaAKHtA?usp=sharing).

#### Server setup 

The following instructions are for a local server. If you do not have a suitable NVIDIA GPU with at least 8GB of GPU memory then my current recommendation is to run via Google colab. A publicly available notebook is available at [Google Drive with Google Colab](https://colab.research.google.com/drive/104narYAvTBt-X4QEDrBSOZm_DRaAKHtA?usp=sharing).

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
See [lung tutorial](docs/cxr_lung_tutorial.md) for an example of how to use RootPainter to train a model. I now actually suggest following the [colab tutorial](https://colab.research.google.com/drive/104narYAvTBt-X4QEDrBSOZm_DRaAKHtA?usp=sharing) instructions but using your local setup instead of the colab server, as these are easier to follow than the lung tutorial.


 ### Questions and Problems
 
The [FAQ](https://github.com/Abe404/root_painter/blob/master/docs/FAQ.md) may  be worth checking before reaching out with any questions you have. If you do have a question you can either email me or post in the [discussions](https://github.com/Abe404/root_painter/discussions). If you have an issue/ have identified a problem with the software then you can [post an issue](https://github.com/Abe404/root_painter/issues).
