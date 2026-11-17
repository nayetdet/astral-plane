import ctypes
from collections.abc import Callable
from typing import TypeAlias

FrameHandler: TypeAlias = Callable[[int, int, int], None]


class FrameMode(ctypes.Structure):
    _fields_ = [
        ("reserved", ctypes.c_uint32),
        ("resolution", ctypes.c_int32),
        ("format", ctypes.c_int32),
        ("bytes", ctypes.c_int32),
        ("width", ctypes.c_int16),
        ("height", ctypes.c_int16),
        ("data_bits_per_pixel", ctypes.c_int8),
        ("padding_bits_per_pixel", ctypes.c_int8),
        ("framerate", ctypes.c_int8),
        ("is_valid", ctypes.c_int8),
    ]


class Timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


FrameCallback = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32
)
