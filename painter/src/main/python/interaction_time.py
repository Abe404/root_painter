from dataclasses import dataclass

@dataclass
class Event:
    """Class for keeping track of an interaction event."""
    name: str # event type i.e mouse_press
    time: float
    fname: int = 0


def is_pause(prev_event, period_len_s):
    # if the previous event is mouse_release
    if prev_event.name == 'mouse_release':
        # and the duration is over 20 seconds
        if period_len_s > 20:
            # then don't include the interaction - we consider this a pause in annotation
            return True
    return False



def get_interaction_time_for_events(events):
    assert len(events) >= 2, f'at least two events are required, events={events}'
    # events are already filtered by filename
    total_duration = 0
    most_recent_event = events[0]
    inactivity_limit_s = 20
    for e in events[1:]:
        assert e.name != most_recent_event.name
        period_between_events = e.time - most_recent_event.time
        if not is_pause(most_recent_event, period_between_events):
            total_duration += period_between_events
            print('include', e, period_between_events)
        else:
            print('exclude', e, period_between_events)
        most_recent_event = e
    return total_duration

def interaction_time_per_fname_s(client_log_fpath):
    """ estimate interaction time for each file based
        on the timing of mouse_press and mouse_release events
        logged whilst the user was interacting with each file 

        Example usage:
        interaction_times = interaction_time_per_fname_s(client_log_fpath='logs/client.csv')
     """
    
    fname_times = {}
    lines = open(client_log_fpath).readlines()
    events = []

    for l in lines:
        parts = l.strip().split(',')
        event_time = parts[1]
        event_name = parts[2]
        fname = [p for p in parts if 'fname:' in p][0].replace('fname:', '')
        events.append(Event(name=event_name, fname=fname, time=float(event_time)))

    unique_fnames = list(set([e.fname for e in events]))
    
    for fname in unique_fnames:
        # we consider interaction as mouseup and mouse down events
        fname_events = [e for e in events if e.fname == fname and e.name in [
                        'mouse_press', 'mouse_release']]
        if len(fname_events) >= 2: # must have both mouse_press and mouse_release
            fname_times[fname] = get_interaction_time_for_events(fname_events)
    return fname_times
