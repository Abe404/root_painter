"""
Copyright (C) 2023 Abraham George Smith

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

from dataclasses import dataclass
import os

@dataclass
class Event:
    """Class for keeping track of an interaction event."""
    name: str # event type i.e mouse_press
    time: float
    fname: int = 0


def is_pause(prev_event, period_len_s):
    # if the previous event is not mouse_press 
    # (maybe the user us just taking their time to draw something without lifting the mouse)
    if prev_event.name != 'mouse_press':
        # and the duration is over 20 seconds
        if period_len_s > 20:
            # then don't include the interaction - we consider this a pause in annotation
            return True
    return False

def get_annot_duration_s(events, fname):
    fname = os.path.splitext(fname)[0] # event object doesn't include file extension
    # we consider interaction as mouseup and mouse down events
    fname_events = [e for e in events if e.fname == fname and e.name in [
                    'mouse_press', 'mouse_release',
                    'save_annotation', 'update_file_end']]

    if len(fname_events) < 2: # must have both mouse_press and mouse_release
        return 0, 0
    click_count = 0
    # fname_events are already filtered by filename
    total_duration = 0
    most_recent_event = fname_events[0]
    for e in fname_events[1:]:
        period_between_events = e.time - most_recent_event.time
        
        # if the current event is mouse_up then we count this as a click.
        if e.name == 'mouse_release':
            click_count += 1
        if not is_pause(most_recent_event, period_between_events):
            total_duration += period_between_events
        most_recent_event = e
    return total_duration, click_count


def events_from_client_log(client_log_fpath):
    lines = open(client_log_fpath).readlines()
    events = []
    for l in lines:
        parts = l.strip().split(',')
        event_time = parts[1]
        event_name = parts[2]
        fname = [p for p in parts if 'fname:' in p][0].replace('fname:', '')
        # because load image has .jpg name (perhaps) but save annotation has png
        # so let's just work with the file name without the extension.
        fname = os.path.splitext(fname)[0] 
        events.append(Event(name=event_name, fname=fname, time=float(event_time)))

    return events


def interaction_time_per_fname_s(client_log_fpath):
    """ estimate interaction time for each file based
        on the timing of mouse_press and mouse_release events
        logged whilst the user was interacting with each file 

        Example usage:
        interaction_times = interaction_time_per_fname_s(client_log_fpath='logs/client.csv')
     """
    events = events_from_client_log(client_log_fpath)     
    unique_fnames = list(set(e.fname for e in events))
    fname_times = {}
    for fname in unique_fnames:
        fname_times[fname] = get_annot_duration_s(events, fname)
    return fname_times
