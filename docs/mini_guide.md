### A concise guide to using RootPainter

This is a short guide to help users remember the key steps to obtain measurements using RootPainter.  See [main page](https://github.com/Abe404/root_painter) for additional documentation including links to videos, papers and a colab notebook.

Note: RootPainter is primarily used for root segmentation but works for many other objects. Modify instructions accordingly.

1. Open RootPainter and select 'Create training dataset'. Create a training dataset using 1000 images with a target size of 750 and 1 tile per image.
2. Create a new project referencing the new training dataset as the source dataset.
3. Annotate 2 images that contain both roots and soil. Don’t annotate more than (roughly estimated) 10x more soil than roots. Don’t annotate images that don’t contain roots.
4. After 2 images are annotated click start training (from network menu) to start the network training on your annotations and annotated images.
5. Annotate 4 more images. At this point, you are not interested in the model segmentation. The point is just to provide some clear annotations to get training started. 
    - Note: if using colab or a slow server, I suggest to annotate 12-18 initial clear examples instead of 6. (A general rule is that you should not switch to corrective annotation until your model is approximately predicting the object of interest. There is not much point in correcting random noise).
7. When 6 images are annotated (clear examples, including both root and soil), then on image 7 switch to corrective annotation. Inspect the model prediction (segmentation) for the image and assign foreground and background annotation to all of the errors in each image. Corrections don’t need to be 100% perfect in terms of capturing absolutely every incorrect pixel but aim to avoid annotating foreground on the soil and background on the root as these annotatoins errors are particularly problematic.
8. Continue to annotate correctively, annotating the errors in each image. If root is missing from the segmentation, annotate the root or missing region of the root in red. If soil/background is predicted as root then annotate this region in green. Use a big brush to annotate the background because it is quicker. Use a small brush to annotate the roots to avoid annotating roots on the soil.
9. After a while (30 images) check the metrics plot. This is available from the extras menu. Show the metrics plot to track progress. Metrics of interest include:
    - Dice - harmonized mean of precision and recall.
    - Recall - the number of root pixels predicted to be root as ratio of total root pixels in the image.
    - Precision - number of predicted pixels that were actually root out of total predicted root pixels.
    - Accuracy - % of image predicted correctly
    - False positives - number of soil pixels predicted as root.
    - False negatives - number of root pixels predicted as soil.
    - Area including predicted-correct - This measure of area error gives an indication if area is being over or under estimated by the model. See ‘corrected area’ to get an idea of absolute values of area. You can then observe rolling averages and see how big area error is in relation to the correct area (i.e does area error matter for your downstream analysis and overarching/primary research question).
10. You can plot the metrics over time and compute rolling average within RootPainter. If metrics get very high and stay high i.e plateau, then you may gain confidence that further annotation is not required. You can click on points in the metrics plot to inspect outliers that may be related to artifacts in the data or perhaps errors in annotation. The metrics plot is only a reliable indicator of model error if you annotated approximately all the error in each image as the metrics are computed by taking the difference between the predictions and corrected segmentation (segmentation with corrections assigned).
11. As you progress through the images, you should be able to annotate faster and faster as the images should require less annotation as the model improves. This means you can likely skip many of the easier images and begin to target your corrective annotation towards anomalies or outliers.
12. If you are using colab or another slow server or have a slow connection to your GPU server, you may wish to segment some images in advance of viewing them to speed up the annotation process. To enable this select 'pre-sement' from the options menu and input a value greater than 0. I suggest starting 1 and gradually insteasing as you start to annotate faster, evantually you may wish to increase this to a value between 5-10 to avoid spending too much time waiting for segmentations to correct.
13. Once happy with the model you can segment the full folder of the original images by going to the network menu and clicking ‘segment folder’. Put the output segmentations in the results folder of the project i.e. project_name/results/model_20_seg. You can name these output folder whatever you like. I like to use the model name so I know which model was used to generate the output segmentations.
14. Extract measurements using the extract length option from the measurements menu in RootPainter to get a CSV you can analyze in other software.
15. Get diameter and diameter classes or other traits using RhizoVisionExplorer (RVE) (downloaded separately). To prepare RootPainter output segmentation for analysis in RVE, use the convert segmentations for RVE option available from the RootPainter extras menu.
16. To inspect final segmentation output (not the segmentation generated during the interactive training process but the segmentation generated from the original images using the segment folder option) use the ‘generate composite’ functionality. This is useful for generating figures to use in presentations and in manuscripts. 
17. To communicate that your model was accurate you can use both composites of random images and the metrics plot. The metrics plot can be right-clicked giving an export option. You may export to various formats including a PNG that can be inserted into manuscript (supplementary material etc).
18. If you generate the composites or extract measurements and then decide later that you want to further train the model to get even better or more accurate results then you should reopen the project and annotate more images (correcting the errors on new images) whilst training. Then you should repeat steps 7-15. But give the output segmentation and results files new names, assuming a new more accurate model has been generated.
19. Note: RootPainter displays a message stating X epochs out of 60 (or similar) when it is training. You don’t have to wait until 60 epochs without progress to use the model trained, this is optional and may only be marginally beneficial.






