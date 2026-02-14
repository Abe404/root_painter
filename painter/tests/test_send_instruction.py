"""
Test that send_instruction handles missing instructions directory gracefully.
See https://github.com/Abe404/root_painter/issues/156
"""
import os
import sys
import tempfile
import pytest

test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(test_dir), 'src', 'main', 'python')
sys.path.insert(0, src_dir)

from instructions import send_instruction


def test_send_instruction_missing_dir_gives_clear_error():
    """When instructions dir doesn't exist, raise an error with a helpful
    message about the sync directory, not a raw FileNotFoundError from open()."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sync_dir = tmpdir
        missing_dir = os.path.join(tmpdir, 'instructions')
        content = {'dataset_dir': '/some/path', 'file_names': ['a.png']}

        with pytest.raises(FileNotFoundError, match='sync directory'):
            send_instruction('segment', content, missing_dir, sync_dir)


def test_send_instruction_works_when_dir_exists():
    """Normal case: instruction file is created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sync_dir = tmpdir
        instruction_dir = os.path.join(tmpdir, 'instructions')
        os.makedirs(instruction_dir)
        content = {'dataset_dir': '/some/path', 'file_names': ['a.png']}

        send_instruction('segment', content, instruction_dir, sync_dir)

        files = os.listdir(instruction_dir)
        assert len(files) == 1
        assert files[0].startswith('segment_')
