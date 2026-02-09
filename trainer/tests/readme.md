Tests for the server (trainer) component of RootPainter

To run the unit tests (from this directory):

    ../env/bin/python -m pytest test_loss.py test_unet.py -v

To run the training benchmarks (downloads data from Zenodo on first run):

    ../env/bin/python -m pytest test_training.py -v -s
