from unet import UNetGNRes


def setup_function():
    """setup any state tied to the execution of the given function.
    Invoked for every test function in the module.
    """
    if not os.path.isdir('test_temp_output'):
        os.makedirs('test_temp_output')


def test_inference():
    """ should not raise an exception """
    import torch
    from model_utils import get_device
    from PIL import Image
    device = get_device()
    from torch.nn.functional import softmax
    from skimage.io import imsave
    import numpy as np
    from skimage import img_as_uint
    unet = UNetGNRes()
    unet.eval()
    unet.to(device)
    test_input = np.zeros((1, 3, 572, 572))
    test_input = torch.from_numpy(test_input)
    test_input = test_input.float().to(device)
    output = unet(test_input)
    output = output.detach().cpu()
    softmaxed = softmax(output, 1)[:, 1, :] # just fg probability
    softmaxed = softmaxed[0] # single image.
    softmaxed = softmaxed.numpy()
    im = img_as_uint(softmaxed)
    imsave('test_temp_output/out.png', im)


def test_training():
    """ test that network can be trained,
        and can approximate a square """
    import torch
    from model_utils import get_device
    import numpy as np
    from torch.nn.functional import softmax
    from loss import combined_loss as criterion
    from torch.nn.functional import cross_entropy
    from skimage.io import imsave
    from skimage import img_as_uint

    loss_fn = torch.nn.CrossEntropyLoss()
    device = get_device()
    unet = UNetGNRes()
    unet.to(device)
    optimizer = torch.optim.SGD(unet.parameters(), lr=0.01,
                                momentum=0.99, nesterov=True)

    test_input = np.zeros((1, 3, 572, 572))
    test_input[:, :, 100:-100,100:-100] = 1.0
    test_input = torch.from_numpy(test_input)
    test_input = test_input.float().to(device)
    target = (test_input[:, 0, 36:-36, 36:-36] > 0.5)
    im = img_as_uint(target.cpu().numpy())
    target = target.int()
    for step in range(300):
        optimizer.zero_grad()
        output = unet(test_input)
        loss = loss_fn(output, target) # all zeros output
        # print('loss', loss.item())
        loss.backward()
        optimizer.step()
        softmaxed = softmax(output, 1)[:, 1] # just fg probability
        im = softmaxed.detach().cpu().numpy()[0]
        im = img_as_uint(im)
        imsave('test_temp_output/out_' + str(step).zfill(3) + '.png', im,
               check_contrast=False)
    assert loss < 1e-7