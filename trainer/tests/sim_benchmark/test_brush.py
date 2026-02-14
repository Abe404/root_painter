"""
Unit tests for brush annotation utilities.
"""
import numpy as np
from sim_benchmark.brush import disk, paint, new_annot


def test_disk_shape():
    d = disk(5)
    assert d.shape == (11, 11)
    # symmetric
    assert np.array_equal(d, d[::-1, :])
    assert np.array_equal(d, d[:, ::-1])


def test_disk_values():
    d = disk(5)
    # center is True
    assert d[5, 5]
    # corners are False (distance from center to corner > radius)
    assert not d[0, 0]
    assert not d[0, 10]
    assert not d[10, 0]
    assert not d[10, 10]


def test_paint_sets_channel():
    annot = new_annot(100, 100)
    paint(annot, (50, 50), 5, channel=0)
    # foreground channel should have non-zero pixels
    assert np.sum(annot[:, :, 0]) > 0
    # background channel should be untouched
    assert np.sum(annot[:, :, 1]) == 0


def test_paint_clips_at_edge():
    annot = new_annot(50, 50)
    # paint near top-left corner — should not crash
    paint(annot, (2, 2), 10, channel=0)
    assert np.sum(annot[:, :, 0]) > 0
    # paint near bottom-right corner
    paint(annot, (48, 48), 10, channel=1)
    assert np.sum(annot[:, :, 1]) > 0


def test_new_annot_shape():
    a = new_annot(200, 300)
    assert a.shape == (200, 300, 4)
    assert a.dtype == np.uint8
    assert np.sum(a) == 0
