"""Environment-variable helpers. Deliberately dependency-free."""

import os

_FALSEY = {"", "0", "false", "no", "off", "none"}


def env_flag(name, default=False):
    """
    Parse a boolean environment variable.

    The obvious `bool(os.environ.get(name, 0))` is wrong: every non-empty
    string is truthy, so SMOKE_TEST=0 and SMOKE_TEST=false both enable
    the flag. That is not a harmless quirk here - SMOKE_TEST truncates
    the dataset to a couple of batches, so a run started with
    SMOKE_TEST=0 in the belief it was disabled would train on 64 clips
    and report perfectly ordinary-looking losses.
    """
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in _FALSEY
