"""
Tests for instruction retry and failure handling

Copyright (C) 2026 Abraham George Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import sys
import json
import tempfile
import shutil

import pytest

# Add trainer/src to sys.path so we can import the project modules
test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(test_dir), 'src')
sys.path.insert(0, src_dir)

from trainer import Trainer


@pytest.fixture
def sync_dir(tmp_path):
    """Create a temporary sync dir with required subfolders."""
    for folder in ['instructions', 'executed_instructions',
                   'failed_instructions', 'projects', 'datasets']:
        (tmp_path / folder).mkdir()
    # trainer.log() writes to sync_dir/server_log.txt
    return str(tmp_path)


def create_trainer(sync_dir):
    """Create a Trainer pointing at the given sync_dir without full init."""
    t = object.__new__(Trainer)
    t.sync_dir = sync_dir
    t.instruction_dir = os.path.join(sync_dir, 'instructions')
    t.executed_dir = os.path.join(sync_dir, 'executed_instructions')
    t.failed_dir = os.path.join(sync_dir, 'failed_instructions')
    t.retry_counts = {}
    t.max_retries = 3  # low for fast tests
    t.training = False
    t.valid_instructions = [t.stop_training]
    t.msg_dir = None
    return t


def write_instruction(sync_dir, fname, content):
    fpath = os.path.join(sync_dir, 'instructions', fname)
    with open(fpath, 'w') as f:
        json.dump(content, f)
    return fpath


def test_successful_instruction_moves_to_executed(sync_dir):
    t = create_trainer(sync_dir)
    # stop_training when not training is a no-op — should still succeed
    t.training = True
    t.epochs_without_progress = 0
    # stop_training calls write_message which needs msg_dir
    msg_dir = os.path.join(sync_dir, 'projects', 'test_proj', 'messages')
    os.makedirs(msg_dir, exist_ok=True)
    t.msg_dir = msg_dir

    fname = 'stop_training_12345'
    write_instruction(sync_dir, fname, {})

    t.check_for_instructions()

    assert not os.path.exists(os.path.join(sync_dir, 'instructions', fname))
    assert os.path.exists(os.path.join(sync_dir, 'executed_instructions', fname))


def test_failed_instruction_retries_then_moves_to_failed(sync_dir):
    t = create_trainer(sync_dir)
    t.max_retries = 3

    # segment instruction referencing a non-existent model_dir will fail
    t.valid_instructions = [t.segment]
    fname = 'segment_99999'
    write_instruction(sync_dir, fname, {
        'dataset_dir': 'datasets/nonexistent',
        'seg_dir': 'projects/nonexistent/segmentations',
        'model_dir': 'projects/nonexistent/models',
        'file_names': ['test.tiff']
    })

    instruction_path = os.path.join(sync_dir, 'instructions', fname)

    # First two retries — file should stay in instructions
    t.check_for_instructions()
    assert t.retry_counts[fname] == 1
    assert os.path.exists(instruction_path)

    t.check_for_instructions()
    assert t.retry_counts[fname] == 2
    assert os.path.exists(instruction_path)

    # Third retry — should move to failed
    t.check_for_instructions()
    assert fname not in t.retry_counts
    assert not os.path.exists(instruction_path)
    assert os.path.exists(os.path.join(sync_dir, 'failed_instructions', fname))

    # Check exception file was written
    error_path = os.path.join(sync_dir, 'failed_instructions',
                              fname + '_exception.txt')
    assert os.path.exists(error_path)
    error_content = open(error_path).read()
    assert len(error_content) > 0


def test_retry_count_resets_after_success(sync_dir):
    t = create_trainer(sync_dir)
    t.max_retries = 5

    # Manually set a retry count for a file
    t.retry_counts['segment_99999'] = 4

    # Now process a successful instruction
    t.training = True
    t.epochs_without_progress = 0
    msg_dir = os.path.join(sync_dir, 'projects', 'test_proj', 'messages')
    os.makedirs(msg_dir, exist_ok=True)
    t.msg_dir = msg_dir

    fname = 'stop_training_11111'
    write_instruction(sync_dir, fname, {})
    t.valid_instructions = [t.stop_training]

    t.check_for_instructions()

    # The successful one should be cleared, the other should remain
    assert fname not in t.retry_counts
    assert t.retry_counts['segment_99999'] == 4


def test_empty_instruction_does_not_move(sync_dir):
    t = create_trainer(sync_dir)
    t.valid_instructions = [t.stop_training]

    fname = 'stop_training_00000'
    # Write empty content
    fpath = os.path.join(sync_dir, 'instructions', fname)
    with open(fpath, 'w') as f:
        f.write('   ')

    t.check_for_instructions()

    # Empty instruction returns False — should stay and increment retry
    assert os.path.exists(fpath)
    assert t.retry_counts.get(fname, 0) == 1
