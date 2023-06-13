from PyQt6 import QtCore

def test_mask_function(qtbot):
    from mask_images import MaskImWidget
    # initialise the mask im widget
    mask_im_widget = MaskImWidget()
    mask_im_widget.show()
    qtbot.mouseClick(mask_im_widget.specify_seg_btn, QtCore.Qt.MouseButton.LeftButton)


# def test_hello(qtbot):
#     widget = HelloWidget()
#     qtbot.addWidget(widget)
# 
#     # click in the Greet button and make sure it updates the appropriate label
#     qtbot.mouseClick(widget.button_greet, QtCore.Qt.LeftButton)
# 
#     assert widget.greet_label.text() == "Hello!"
# 
