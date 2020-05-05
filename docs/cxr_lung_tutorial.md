### RootPainter Chest X-ray Segmentation Tutorial

This is a guide for using [RootPainter](https://www.biorxiv.org/content/10.1101/2020.04.16.044461v1) to train a model to segment lungs in chest X-ray images

In this guide I assume you have the server component of RootPainter set up and running. I also assume the sync directory location to be ~/Desktop/root_painter_sync. Please modify accordingly.

1. Download and extract the [Shenzhen Hospital X-ray Set](https://lhncbc.nlm.nih.gov/publication/pub9931):
```
  wget http://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip
  unzip ChinaSet_AllFiles.zip
```

2. This step is optional but highly recommended. The images are big and more than we need for this tutorial. I suggest resizing the data to speed up the training process and data transfer. We will also use only 300 of the images.
  
Add the following to a file named resize_cxr.py:
```
import os
import sys
from multiprocessing import Pool
import random

import tqdm
from skimage.io import imread, imsave
from skimage import img_as_ubyte
from skimage.transform import resize

src_dir = sys.argv[1]
out_dir = sys.argv[2]
assert os.path.isdir(src_dir)
assert not os.path.isdir(out_dir) and not os.path.isfile(out_dir)
fnames = [a for a in os.listdir(src_dir) if os.path.splitext(a)[1] == '.png']

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

def resize_file(f):
    out_path = os.path.join(out_dir, os.path.splitext(f)[0] + '.jpg')
    if not os.path.isfile(out_path):
        im = imread(os.path.join(src_dir, f))
        h, w = im.shape[:2]
        # resize so smallest dimension is 600
        if w < h:
            new_w = 600
            ratio = new_w/w
            new_h = round(h * ratio)
        else:
            new_h = 600
            ratio = new_h/h
            new_w = round(w * ratio)
        im = resize(im, (new_h, new_w))
        imsave(out_path, img_as_ubyte(im))

if __name__ == '__main__':
    fnames = random.sample(fnames, k=300)
    with Pool(processes=16) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(resize_file, fnames), total=len(fnames)):
            pass
```

3. Run the resize script. This will add the resized images to a new RootPainter dataset in the
datasets folder.
```
  python resize_cxr.py ChinaSet_AllFiles/CXR_png/ ~/Desktop/root_painter_sync/datasets/cxr
```
  
4. If you haven't already installed the RootPainter client then download the
  client installer for your operating system.
  DMG (OSX) EXE (Windows) and DEB (Debian/Ubuntu) files are available from
  [this link](https://github.com/Abe404/root_painter/releases/tag/0.2.0).
  Then run the downloaded installer.

5. When you open the installed application for the first time. You will be promped to specify a sync directory. Specify ~/Desktop/root_painter_sync. The specified path will be added to a settings file in your home folder named root_painter_settings.json and the value found in that file will be used from then on.

6. Click the 'Create new project' button which is visible when opening RootPainter.
    * Specify ~/Desktop/root_painter_sync/datasets/cxr as the image directory.
    * Specify random weights
    * name your project 'cxr_tutorial' (or whatever you want).

7. Annotate and train.
    - Label two images. Use the red (foreground) brush for the lungs and the green (background) brush for everything else. Leave ambiguous regions as undefined.
    - Click start training from the network menu.
    - Label a third image.
    - From the fourth image onwards, view the segmentation first and then target the annotations towards area where the segentation is inadequate.
    - If waiting for the segmentation is slowing you down then set Pre-Segment from 0 to 1. Pre-Segment is in the options menu.
    - Keep progressing through the images until you are happy with the quality of the model
