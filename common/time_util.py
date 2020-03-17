# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

from contextlib import contextmanager
from datetime import datetime


@contextmanager
def timer(name):
    """
    Timer for program.
    Cited from <Qianghua Xuexi Jingyao> (Chao Feng)
    Example:
        with timer("Task..."):
            do something
            ...

    :param name: str.
    """
    dt_start = datetime.now()
    print(name, ": started at", dt_start)
    yield
    dt_end = datetime.now()
    print(name, ": ended at", dt_end,
          ", consume = ", (dt_end - dt_start).total_seconds(), "(s)")


def pretty_eta(seconds_left):
    """Print the number of seconds in human readable format.
    Cited from baselines (OpenAI)

    Examples:
    2 days
    2 hours and 37 minutes
    less than a minute

    Paramters
    ---------
    seconds_left: int
        Number of seconds to be converted to the ETA
    Returns
    -------
    eta: str
        String representing the pretty ETA.
    """
    minutes_left = seconds_left // 60
    seconds_left %= 60
    hours_left = minutes_left // 60
    minutes_left %= 60
    days_left = hours_left // 24
    hours_left %= 24

    def helper(cnt, name):
        return "{} {}{}".format(str(cnt), name, ('s' if cnt > 1 else ''))

    if days_left > 0:
        msg = helper(days_left, 'day')
        if hours_left > 0:
            msg += ' and ' + helper(hours_left, 'hour')
        return msg
    if hours_left > 0:
        msg = helper(hours_left, 'hour')
        if minutes_left > 0:
            msg += ' and ' + helper(minutes_left, 'minute')
        return msg
    if minutes_left > 0:
        return helper(minutes_left, 'minute')
    return 'less than a minute'
