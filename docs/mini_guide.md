### A concise guide to using RootPainter

This is a short guide to help users remember the key steps to obtaining measurements using RootPainter.  See [main page](https://github.com/Abe404/root_painter) for additional documentation including links to videos and a colab notebook.

Note: RootPainter is primarily used for root semgentation but works for many other objects. Modify instructions accordingly.

1. Open RootPainter and select 'Create training dataset'. Create a training dataset using 1000 images with a target size of 900 and 1 tile per image.
2. Create a new project referencing the new training dataset as the source dataset.
3. Annotate 2 images that contain both roots and soil. Don’t annotate more than (roughly estimated) 10x more soil than roots. Don’t annotate images that don’t contain roots.
4. After 2 images are annotated click start training (from network menu) to start the network training on your annotations and annotated images.
5. Annotate 4 more images. At this point, you are not interested in the model segmentation. The point is just to provide some clear annotations to get training started. 
    - Note: if using colab or a slow server, I suggest to annotate 12-18 initial clear examples instead of 6. (A general rule is that you should not switch to corrective annotation until your model is approximately predicting the object of interest. There is not much point in correcting random noise).
7. When 6 images are annotated (clear examples, including both root and soil), then on image 7 I suggest you switch to corrective annotation. That means that you inspect the model prediction (segmentation) for the image and assign foreground and background annotation to all of the errors in each image (The corrections don’t need to be 100% perfect in terms of capturing absolutely every incorrect pixel with your annotation but try hard to avoid annotating foreground on the soil and background on the root).
8. From image 7 onwards, annotate correctively, annotating the errors in each image. If the root is missing from the segmentation, annotate the root or missing region of the root in red. If soil/background is predicted as root then annotate this region in green. Use a big brush to annotate the background because it is quicker. Use a small brush to annotate the roots to avoid annotating roots on the soil (be careful to avoid annotating the soil/background red).
9. After a while (30 images) check the metrics plot. This is available from the extras menu. Show the metrics plot to track progress. Metrics of interest are:
    - Dice (harmonized mean of precision and recall)
    - Recall (the number of root pixels predicted to be root as ratio of total root pixels in the image
    - Precision (number of predicted pixels that were actually rooted out of total predicted pixels)
    - Accuracy (% of image predicted correctly)
    - False positives (number of soil pixels predicted as root).
    - False negatives (number of root pixels predicted as soil).
    - Area (predicted-correct, gives you indication if area is being over or under estimated by the model. See ‘corrected area’ to get an idea of absolute values of area. You can then take rolling averages and see how big area error is in relation to the correct area (i.e does area error matter).
10. You can plot the metrics over time and compute rolling average within RootPainter. If metrics get very high and stay high i.e plateau, then you may gain confidence that further annotation is not required. You can click on points in the metrics plot to inspect outliers that may be related to artifacts in the data or annotation. The metrics plot is only a reliable indicator of model error if you annotated approximately all the error in each image as the metrics are computed by taking the difference between the predictions and corrected segmentation (segmentation with corrections assigned).
11. As you progress through the images, you should be able to annotate faster and faster as the images should require less annotation as the model improves. This means you can likely skip many of the easier images and begin to target your corrective annotation towards anomalies or outliers.
12. Once happy with the model you can segment the full folder of the original images by going to the network menu and clicking ‘segment folder’. Put the output segmentations in the results folder of the project i.e. project_name/results/model_20_seg (name it whatever you like, I like to use the model name itself so I know which model was used to generate the output segmentations).
13. Extract measurements using the extract length option from the measurements menu in RootPainter to get a CSV you can analyze in other software.
14. Get diameter and diameter classes or other traits using RhizoVisionExplorer (RVE) (downloaded separately). To prepare Rootpainter output segmentation for analysis in RVE, use the convert segmentations for RVE option available from the RootPainter extras menu.
15. To inspect final segmentation output (not the segmentation generated during the interactive training process but the segmentation generated from the original images using the segment folder option) use the ‘generate composite’ functionality. This is useful for generating figures to use in presentations and in manuscripts. 
16. To communicate that your model was accurate you can use both composites of random images and the metrics plot. The metrics plot can be right-clicked giving an export option. You may export to various formats including a PNG that can be inserted into manuscript (supplementary material etc).
17. If you generate the composites or extract measurements and then decide actually you want to further train the model to get even better or more accurate results then you should reopen the project and annotate more images (correcting the errors on new images) whilst training. Then you should repeat steps 7-15. But give the output segmentation and results files new names, assuming a new more accurate model has been generated.
18. Note: RootPainter says X epochs remaining out of 60 (or similar) when it is training. You don’t have to wait until 60 epochs without progress to use the model trained, this is optional and may only be marginally beneficial.






