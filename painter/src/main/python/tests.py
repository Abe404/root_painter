from PyQt6 import QtCore

def test_mask_function(qtbot):
    """ test we can click the specify_seg_btn without an error """
    from mask_images import MaskImWidget
    # initialise the mask im widget
    mask_im_widget = MaskImWidget()
    mask_im_widget.show()
    qtbot.mouseClick(mask_im_widget.specify_seg_btn, QtCore.Qt.MouseButton.LeftButton)
