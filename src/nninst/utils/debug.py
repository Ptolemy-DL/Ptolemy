import sys

from IPython.core.debugger import Pdb as CorePdb

__all__ = ["breakpoint"]


def breakpoint(condition=True):
    """
    Set a breakpoint at the location the function is called if `condition == True`.
    """
    if condition:
        debugger = CorePdb()
        frame = sys._getframe()
        debugger.set_trace(frame.f_back)
        return debugger
