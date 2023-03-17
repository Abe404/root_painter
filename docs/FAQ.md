### Frequently asked questions


##### Table of contents
  * [Question - How do I skip to earlier images?](#question---how-do-i-skip-to-images)
  * [Question - Should I let training finish?](#question----should-i-let-training-finish)
  * [Question - How do I decide when the model is good enough?](#question---how-do-i-decide-when-the-model-is-good-enough)
  * [Question - Why is the segmentation not loading?](#question---why-is-the-segmentation-not-loading)
  * [Question - On ubuntu I get an error related to xcb](#question---on-ubuntu-i-get-an-error-related-to-xcb)
  * [Question - How can I use RootPainter for a multiclass segmentation task?](#question---how-can-i-use-rootpainter-for-a-multiclass-segmentation-task)
  * [Question - I already have a trained model. Do I still need a GPU for segmentation?](#question---i-already-have-a-trained-model-do-i-still-need-a-gpu-for-segmentation)

<!---
<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>
-->


#### Question - [How do I skip to images?](https://github.com/Abe404/root_painter/issues/59)

If you want to skip back to the first few images it is possible to do this with the back/previous button but for large projects this can take a while as you will need to wait for each image to load. A more efficient method is possible using the metrics plot.

With the project open, go to the extras menu and click on view metrics plot. Then click on the image point in the metrics plot and it will take you to the corresponding image in the viewer.

#### Question -  Should I let training finish?

I you stop annotating and let the training continue it will eventually reach 60 epochs out of 60 with no progress. It may not be easy for you to do this with your hardware (colab, for example, has some time contraints). You may be wondering if leaving training to finish is essential and if it makes the model more robust.

We experimented with this in the original study. See [Figure 8](https://nph.onlinelibrary.wiley.com/doi/full/10.1111/nph.18387#nph18387-fig-0008). In short, we found that letting the model train to completion can provide some marginal benefits in some cases.

I suspect this is hardware specific. If you have slow hardware (such as google colab) then it's more likely that the hardware is a bottleneck and it is training time (rather than amount of annotation) that is the main bottleneck preventing performance improvements. In this case letting the model train for a bit longer may provide more benefits.

RootPainter provides an interactive-machine-learning experience where what you see is what you get. Meaning the segmentations you see in the interface are an accurate representation of the models accuracy on the data you wish to process. It is OK to stop training and the model for segmentation at any time, ideally once you are happy with what you see in terms of segmentation quality. The decisions of when to stop annotation/interactive-training may also be informed by the corrective-metrics plot which is available from the extras menu. 


#### Question - How do I decide when the model is good enough?


My personal recommendation is to use the metrics plot that is available from the extras menu. When you annotate images in the RootPainter interface, if you annotate all the error (or approximately all the error) in each image then you have a measure of the model performance/generalisation to new data.

The metrics plot is created by computing the difference between the initial segmentation provided by RootPainter and the segmentation with the corrections assigned. This difference gives us a measure of error for each image annotated correctively.

The metrics plot allows you to compute the error in terms of accuracy, dice, precision, recall, area etc. 

In the RootPainter paper we [have examples](https://nph.onlinelibrary.wiley.com/doi/full/10.1111/nph.18387#nph18387-fig-0010) of some corrective-metrics plots for each of the datasets investigated 

You can see from the first two rows in the [linked figure](https://nph.onlinelibrary.wiley.com/doi/full/10.1111/nph.18387#nph18387-fig-0010) (labelled a) that the dice goes above 0.8 for the roots datasets. If you are also training a model for root segmentation and the running mean of your corrective dice as computed using RootPainter's integrated metrics plot is above 0.8 then you could report this (and perhaps include the plot itself in the supplementary material) as a measurement that shows your model is accurate. Exporting the plot as an image is possible by right clicking on it.

The plot relies on you correcting all the error in each image. If you skip a large part of the error in each image (don't annotate it correctively) then I would not say that the corrective-metrics are an accurate measure of error.

The other indicator is qualitative. You should in general have some idea that the model is doing the job from looking at the segmentations in the interface. Reporting these qualitative results in your paper is best done using the 'extract composites' feature available from the extras menu. These composites are image files that show the segmentation in combination with the input image and can be used as figures in presentations, your paper or supplement.

Knowing exactly how accurate a model needs to be for your specific research question is currently out of the scope of the RootPainter software. This depends on the size of the effect you are measuring, the number of samples and the variance present in your dataset. It may be that a dice or accuracy of just 0.4 (for example) may be enough for your research question and object of interest. The metrics provided in the corrective-metrics plot may be expanded on [upon request](https://github.com/Abe404/root_painter/issues/new). The area metric, which allow you to see if a model is over- or underestimating your structure of interest based on your corrections was added recently.

#### Question - Why is the segmentation not loading?

#### Answer:

My best guess is that there is a delay in sync or the sync software is not set up and/or working properly. There might be another problem (such as a bug with RootPainter) but here are a few steps to help you isolate the problem:

1. **Check the sync directory is speciﬁed correctly**. In your home directory i.e the user directory, there is a ﬁle called root_painter_settings.json. You can open this ﬁle in a text editor to see which directory is speciﬁed as the sync directory. You can also just manually set a new sync directory (to make sure it is correct) using the option from the extras menu. The sync directory should be set as a path to the drive_rp_sync folder if inside your google drive on your local computer, if you are following the colab tutorial. Otherwise it should be the folder setup to share data between the RootPainter client and server (which was also specified when you started the RootPainter server).

2. **Check segmentation instructions are being created**. 
Without starting the trainer (server) , open the client (GUI) and go to an image that doesn’t already have a segmentation. You should see ‘loading segmentation’ in the client. At this point the client should also create an instruction (just a text ﬁle) in the instructions folder which is inside the sync folder. If you are following the colab tutorial then your sync folder is a folder in your google drive called drive_rp_sync. This instruction ﬁle is a text ﬁle that tells the server which image to segment. You can conﬁrm this ﬁle is being created correctly by checking the instruction folder on your computer and checking if the instruction ﬁle exists. You need to do this step without the server running, because if the server is running it will delete the instruction as soon as it has read it. I suggest checking your local google drive folder ﬁrst. This should exist on your local machine. If the instructions are never being created in the correct location then there may be a problem with the sync directory or perhaps some permissions issues.


3. (if you are using colab) **Check segmentation instruction is being synced to google drive**. The colab tutorial assumes you have set up and installed google drive for desktop. If you have google drive for desktop up and running and then the sync folder speciﬁed correctly then instructions should get created in your local instructions folder. These instructions should also get synced to google drive. If you have ﬁrst conﬁrmed the instructions are created correctly then you can check your online google drive to see if they are synced with google's servers. Go to the following url: https://drive.google.com/drive/u/0/my-drive (or just navigate to your google drive online using a web browser) and then go to the drive_rp_sync folder (that was created as part of the colab tutorial). Inside you should see an instructions folder and in the instructions folder should be the instruction ﬁle that was created locally on your computer using the RootPainter client (the GUI application). If you do see the instruction ﬁle locally but you don’t see it online then it indicates there is a problem with synchronization of ﬁles from your computer to google drive.

4. (if you are using colab). **Inspect the status of google drive for desktop** (that should be running on your computer). It could be delayed, busy syncing many ﬁles or not set up to sync the correct locations. In some rare cases it can be buggy and you may want to try uninstalling it and reinstalling again to get sync working properly.

5. **Check for errors in the server output**. When
running both client and server together, inspect the console output for the server in colab. You should see output after you start the server (which should be left running whilst using RootPainter). For example, if you view an image in the client named CHNCXR_0068_0.jpg that does not have a segmentation yet, it then you should see the following output from the server console:


  > execute_instruction segment
  
  > ensemble segment CHNCXR_0068_0.jpg, dur 0.2
  
  > Seconds to segment 1 images:  0.237

Note: Sometimes due to slow sync time it will appear after a delay, so wait a couple of minutes just to be sure there isn’t some delay in sync.


6. **Inspect the console output for the client**. The most recent versions of RootPainter (https://github.com/Abe404/root_painter/releases/) should open a black console that displays error messages from time to time. Perhaps something is going wrong and the output will give us a clue.


Hopefully, even if it doesn’t solve the problem, these instructions will help you get more information that will help us figure out what is going wrong. If you think you have found a bug in the software then feel free to report an issue (https://github.com/Abe404/root_painter/issues) 



#### Question - On ubuntu I get an error related to xcb

The error message may be similar to the following:
```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, wayland-egl, wayland, wayland-xcomposite-egl, wayland-xcomposite-glx, webgl, xcb.
```

As outlined in this [forum discussion answer](https://forum.qt.io/topic/93247/qt-qpa-plugin-could-not-load-the-qt-platform-plugin-xcb-in-even-though-it-was-found/20?_=1678962734314&lang=en-GB) installing libxcb-xinerama0 appears to fix the problem. Which can be done with the following command:

```
sudo apt-get install libxcb-xinerama0
```


#### Question - How can I use RootPainter for a multiclass segmentation task? 

It's possible to train a binary single class model for each of your classes. A more experimental (developer friendly) multiclass version of RootPainter is also availale in the branch named 'multiclass'. When more testing has been done, I will make it available in a more user-friendly client installer.

A [colab notebook](https://colab.research.google.com/drive/1n1Iku3FwoLI0ImLTRQmMGawRyUU4YEJN) is available that runs the multiclass version of RootPainter.
Classes can be specified when creating a project. Each class that is specified implicity has it's own background, thus a backround class does not need to be explicitly specified. Foreground and background annotation should be assigned correctively for each class for each image.

The multiclass client can be ran from source by using git to clone the repo and swith to the multiclass branch.
```
git clone --single-branch --branch multiclass https://github.com/Abe404/root_painter.git
```


#### Question - I already have a trained model. Do I still need a GPU for segmentation?

Yes, a GPU is required for both training and segmentation. Other functionality, such as generating composites, converting segmentations for Rhizivision explorer and extracting measurements does not require a powerful GPU and can be computed using the client only.


