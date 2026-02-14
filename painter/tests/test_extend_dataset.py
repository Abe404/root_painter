"""
Test that extend dataset excludes hidden files.
See https://github.com/Abe404/root_painter/issues/173
"""
import os
import sys
import tempfile

test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(test_dir), 'src', 'main', 'python')
sys.path.insert(0, src_dir)

from im_utils import is_image
from create_dataset import get_image_names


def test_hidden_files_excluded_from_extend_dataset():
    """Hidden files (starting with '.') should not appear in dataset listing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create normal image files
        open(os.path.join(tmpdir, 'image1.png'), 'w').close()
        open(os.path.join(tmpdir, 'image2.jpg'), 'w').close()
        # Create hidden files that macOS filesystem sync can produce
        open(os.path.join(tmpdir, '._image1.png'), 'w').close()
        open(os.path.join(tmpdir, '.DS_Store'), 'w').close()
        open(os.path.join(tmpdir, '._2017-11-14_14_53_16-0002.jpg'), 'w').close()

        names = get_image_names(tmpdir)
        assert len(names) == 2, f"Expected 2 images, got {len(names)}: {names}"
        assert all(not f.startswith('.') for f in names), f"Hidden files found: {names}"
