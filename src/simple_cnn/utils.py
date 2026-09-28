from contextlib import contextmanager
import io
import os
import sys
from typing import Generator


@contextmanager
def quiet() -> Generator[tuple[io.StringIO, io.StringIO], None, None]:
    """Context manager that redirects stdout and stderr to StringIOs.

    The suppressed output is provided as a StringIO as the return value of the
    context manager::

        with quiet() as (stdout, stderr):
            ...
        output = stream.getvalue()
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield (sys.stdout, sys.stderr)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


@contextmanager
def loud() -> Generator[None, None, None]:
    """Turn stdout and stderr back on within a context manager."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
