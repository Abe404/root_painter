"""
Tests for client interaction-log corruption detection and recovery.

The client log (logs/client.csv) can be corrupted by a syncing network
filesystem: blank lines (which crash the parser) and records that are out of
chronological order (which silently bias the interaction-time metric). These
tests cover detection, recovery, and that the parser no longer crashes on a
blank line.
"""
import os
import sys
import tempfile

test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(test_dir), 'src', 'main', 'python')
sys.path.insert(0, src_dir)

from log_recovery import detect_log_corruption, recover_log, timestamp_from_line


def mouse_line(epoch, name='mouse_press', fname='img_000.png'):
    """ Build a well-formed 9-field mouse event line. """
    return (f'2026-01-01 00:00:00.000000,{epoch},{name},fname:{fname},'
            f'x:1.0,y:2.0,brush_size:3.0,brush_color:#00ff00,drawing:True\n')


def write_log(tmpdir, lines):
    log_dir = os.path.join(tmpdir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    fpath = os.path.join(log_dir, 'client.csv')
    with open(fpath, 'w') as f:
        f.writelines(lines)
    return fpath


def test_timestamp_from_line_parses_and_rejects():
    assert timestamp_from_line(mouse_line(1777451407.5)) == 1777451407.5
    assert timestamp_from_line('\n') is None
    assert timestamp_from_line('') is None
    assert timestamp_from_line('not,a,valid,number_in_field_two') is None


def test_detect_clean_log_is_not_corrupt():
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = write_log(tmpdir, [
            mouse_line(100.0, 'mouse_press'),
            mouse_line(101.0, 'mouse_release'),
            mouse_line(102.0, 'mouse_press'),
        ])
        assert detect_log_corruption(fpath) is False


def test_detect_blank_line_is_corrupt():
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = write_log(tmpdir, [
            mouse_line(100.0),
            '\n',
            mouse_line(101.0),
        ])
        assert detect_log_corruption(fpath) is True


def test_detect_out_of_order_is_corrupt():
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = write_log(tmpdir, [
            mouse_line(100.0),
            mouse_line(101.0),
            mouse_line(99.5),  # ~1.5s in the past, like a sync merge seam
            mouse_line(102.0),
        ])
        assert detect_log_corruption(fpath) is True


def test_detect_missing_file_is_not_corrupt():
    assert detect_log_corruption('/no/such/path/client.csv') is False


def test_recover_removes_blanks_and_sorts():
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = write_log(tmpdir, [
            mouse_line(100.0, 'mouse_press'),
            mouse_line(101.0, 'mouse_release'),
            '\n',
            mouse_line(99.5, 'mouse_press'),   # out of order
            mouse_line(102.0, 'mouse_release'),
        ])
        backup_fpath = recover_log(fpath)

        # backup preserves the original, including the blank line.
        assert os.path.isfile(backup_fpath)
        assert os.path.basename(backup_fpath).startswith('corrupt_client_backup_')
        with open(backup_fpath) as f:
            assert '\n' in f.readlines()  # the bare blank line is still there

        # recovered file: no blanks, chronological, all real events kept.
        with open(fpath) as f:
            recovered = f.readlines()
        assert all(l.strip() != '' for l in recovered)
        times = [timestamp_from_line(l) for l in recovered]
        assert times == [99.5, 100.0, 101.0, 102.0]
        assert len(recovered) == 4  # all four real events preserved

        # recovered file is now clean.
        assert detect_log_corruption(fpath) is False


def test_recover_preserves_four_field_events():
    """ save_annotation / update_file_end lines have only 4 fields and must
        survive recovery. """
    with tempfile.TemporaryDirectory() as tmpdir:
        save_line = '2026-01-01 00:00:00.000000,101.0,save_annotation,fname:img_000.png\n'
        fpath = write_log(tmpdir, [
            mouse_line(100.0),
            '\n',
            save_line,
        ])
        recover_log(fpath)
        with open(fpath) as f:
            recovered = f.readlines()
        assert save_line in recovered
        assert len(recovered) == 2


def test_recover_status_callback_reports_steps():
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = write_log(tmpdir, [mouse_line(100.0), '\n', mouse_line(101.0)])
        messages = []
        recover_log(fpath, status_callback=messages.append)
        assert messages  # at least one step reported
        assert all(m.startswith('Log file recovery:') for m in messages)
