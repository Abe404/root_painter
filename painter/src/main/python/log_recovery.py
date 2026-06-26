"""
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


Recovery for corrupted client interaction logs (logs/client.csv).

The client log is written one whole line at a time from a single (GUI) thread,
so the writer itself cannot produce blank lines or out-of-order records. When
the sync directory is a network/cloud-synced mount (sshfs, Dropbox, Google
Drive, ERDA, ...) appends are not guaranteed to be atomic and the file may be
replaced wholesale during sync. Two symptoms have been observed in real logs:

  1. Blank lines ('\\n') - a sync artifact carrying no event data. These crash
     the parser (interaction_time.events_from_client_log does parts[1]).
  2. Records out of chronological order (typically ~1s, consistent with two
     sessions whose clocks differ by ~1s being merged by the sync layer). The
     duration estimate iterates events in file order and assumes they are
     chronological, so reordering silently biases annot_duration_s.

This module detects those conditions and, when found, repairs the log:
back up the original, drop blank/unparseable lines, stable-sort by timestamp,
and write the cleaned log back. Lines that carry a valid timestamp are always
preserved (never dropped), so no event data is lost.
"""

import os
import shutil
from datetime import datetime


def timestamp_from_line(line):
    """ Return the epoch timestamp (float) recorded in a client.csv line,
        or None if the line is blank/malformed.

        A valid line is: '<datetime>,<epoch_time>,<event_name>,...,fname:...'
        so the epoch time is the second comma-separated field. """
    parts = line.rstrip('\n').split(',')
    if len(parts) < 3:
        return None
    try:
        return float(parts[1])
    except ValueError:
        return None


def detect_log_corruption(client_log_fpath):
    """ Return True if the client log contains corruption that warrants
        recovery: blank/unparseable lines, or records that are out of
        chronological order. Read-only - never modifies the file. """
    if not os.path.isfile(client_log_fpath):
        return False
    prev_time = None
    with open(client_log_fpath) as log_file:
        for line in log_file:
            t = timestamp_from_line(line)
            if t is None:
                # blank or otherwise unparseable line.
                return True
            if prev_time is not None and t < prev_time:
                # an earlier event appears after a later one.
                return True
            prev_time = t
    return False


def recover_log(client_log_fpath, status_callback=None):
    """ Repair a corrupted client log in place and return the path of the
        backup that was taken of the original (corrupt) file.

        Steps (each announced via status_callback if provided):
          1. back up the original as corrupt_client_backup_<datetime>.csv
          2. drop blank/unparseable lines and stable-sort by timestamp
          3. atomically replace client.csv with the recovered content

        status_callback, if given, is called with a short human-readable
        message for each step so the GUI can show progress. """

    def report(message):
        if status_callback is not None:
            status_callback(message)

    log_dir = os.path.dirname(client_log_fpath)

    # 1. back up the original corrupt log alongside it.
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    backup_name = f'corrupt_client_backup_{stamp}.csv'
    backup_fpath = os.path.join(log_dir, backup_name)
    report(f'Log file recovery: backing up corrupt log to {backup_name}')
    shutil.copy2(client_log_fpath, backup_fpath)

    # 2. keep only lines with a valid timestamp, then stable-sort by it.
    #    stable sort keeps the original order of records sharing a timestamp.
    report('Log file recovery: de-corrupting log file')
    with open(client_log_fpath) as log_file:
        lines = log_file.readlines()
    valid = [(timestamp_from_line(l), l) for l in lines]
    valid = [(t, l) for (t, l) in valid if t is not None]
    valid.sort(key=lambda pair: pair[0])
    recovered_lines = [l for (_, l) in valid]

    # 3. write the recovered log via a temp file + replace so a failure
    #    midway cannot leave a half-written client.csv (the original is
    #    already safe in the backup regardless).
    report('Log file recovery: saving recovered log')
    tmp_fpath = client_log_fpath + '.recovered'
    with open(tmp_fpath, 'w') as tmp_file:
        tmp_file.writelines(recovered_lines)
    os.replace(tmp_fpath, client_log_fpath)

    return backup_fpath
