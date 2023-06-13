
import os
from PyQt6 import QtCore

def test_specify_seg(qtbot):
    """ test we can click the specify_seg_btn without an error """
    from mask_images import MaskImWidget
    # initialise the mask im widget
    mask_im_widget = MaskImWidget()
    mask_im_widget.show()
    qtbot.mouseClick(mask_im_widget.specify_seg_btn, QtCore.Qt.MouseButton.LeftButton)


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

