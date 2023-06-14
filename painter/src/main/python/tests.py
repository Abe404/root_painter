
import os
from PyQt6 import QtCore


def test_specify_seg(qtbot):
    """ test we can click the specify_seg_btn without an error """
    from mask_images import MaskImWidget
    # initialise the mask im widget
    mask_im_widget = MaskImWidget()
    mask_im_widget.show()
    qtbot.mouseClick(mask_im_widget.specify_seg_btn, QtCore.Qt.MouseButton.LeftButton)


def setup_function():
    import urllib.request
    import zipfile
    import shutil
    sync_dir = os.path.join(os.getcwd(), 'test_rp_sync')
    datasets_dir = os.path.join(sync_dir, 'datasets')
    if not os.path.isdir(datasets_dir):
        os.makedirs(datasets_dir)
    bp_dataset_dir = os.path.join(datasets_dir, 'biopores_750_training')
    # if the directory does not exist, assume it needs downloading
    if not os.path.isdir(bp_dataset_dir):
        biopore_url = 'https://zenodo.org/record/3754046/files/biopores_750_training.zip'
        print('downloading', biopore_url)
        urllib.request.urlretrieve(biopore_url, os.path.join(os.getcwd(), 'bp.zip'))
        with zipfile.ZipFile("bp.zip", "r") as zip_ref:
            zip_ref.extractall(datasets_dir)
        # remove the junk osx metadata that was in the zip file
        shutil.rmtree(os.path.join(datasets_dir, '__MACOSX'))
        os.remove(os.path.join(os.getcwd(), 'bp.zip'))


def test_mask_operation(qtbot):
    from mask_images import MaskImWidget
    import time
    mask_widget = MaskImWidget()
    mask_widget.show()
    proj_dir = os.path.join(os.getcwd(), 'test_rp_sync/projects/biopores')
    dataset_dir = os.path.join(os.getcwd(), 'test_rp_sync/datasets/biopores_750_training')
    mask_widget.seg_dir = os.path.join(proj_dir, 'results/seg')
    mask_widget.im_dir = dataset_dir
    mask_widget.out_dir = os.path.join(proj_dir, 'results/masked')
    mask_widget.validate()
    mask_widget.submit_btn.click()

    def check_output():
        return len(os.listdir(mask_widget.out_dir)) == len(os.listdir(mask_widget.seg_dir))

    qtbot.waitUntil(check_output, timeout=20000)
