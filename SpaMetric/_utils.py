# -*- coding: utf-8 -*-

# from https://github.com/tqdm/tqdm/blob/master/tqdm/std.py
def format_interval(t):
    """
    Formats a number of seconds as a clock time, [H:]MM:SS
    Parameters
    ----------
    t  : int
        Number of seconds.
    Returns
    -------
    out  : str
        [H:]MM:SS
    """
    mins, s = divmod(int(t), 60)
    h, m = divmod(mins, 60)
    if h:
        return '{0:d}:{1:02d}:{2:02d}'.format(h, m, s)
    else:
        return '{0:02d}:{1:02d}'.format(m, s)



















