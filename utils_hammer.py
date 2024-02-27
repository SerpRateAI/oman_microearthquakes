# Functions and classes for analyzing the hammer signals
from re import search

## Get the time window of the hammer shot from its name
## All hammer shots happenned on Jan 15th!

def get_timewin_from_hammer_name(name):
    pattern = r"Hammer(\d{2})"
    match = search(pattern, name)

    if match:
        hour = match.group(1)
        timewin = f"2020-01-25-{hour}-00-00"

        return timewin
    else:
        raise ValueError("The name of the hammer shot is not in the right format")
    


