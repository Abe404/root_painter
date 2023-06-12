

def test_full_im_seg():
    """ test a full image can be segmented with output having correct shape """
    import numpy as np
    from unet import UNetGNRes
    from model_utils import unet_segment, get_device
    model = UNetGNRes().to(get_device())
    test_input = np.zeros((900, 900, 3))
    in_w = 572
    out_w = 500
    seg = unet_segment(model, test_input, in_w, out_w)
    assert seg.shape[0] == test_input.shape[0]
    assert seg.shape[1] == test_input.shape[1]
