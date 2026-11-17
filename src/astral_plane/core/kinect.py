from __future__ import annotations

import ctypes
import ctypes.util
import time
from collections import deque
from threading import Condition, Event, Thread
from typing import Final

import numpy as np
from numpy.typing import NDArray

from astral_plane.bindings.libfreenect import (
    FrameCallback,
    FrameHandler,
    FrameMode,
    Timeval,
)
from astral_plane.schemas.rgbd_frame import RGBDFrame


class Kinect:
    _RESOLUTION_MEDIUM: Final[int] = 1
    _DEPTH_REGISTERED: Final[int] = 4
    _VIDEO_RGB: Final[int] = 0
    _CLOCK_HZ: Final[int] = 60_000_000
    _SYNC_WINDOW_TICKS: Final[int] = _CLOCK_HZ // 50
    _TIMESTAMP_WRAP: Final[int] = 1 << 32
    _EVENT_TIMEOUT_US: Final[int] = 10_000

    def __enter__(self) -> Kinect:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __init__(self, device_index: int = 0) -> None:
        self._lib: ctypes.CDLL = self._load_library()
        self._context: ctypes.c_void_p = ctypes.c_void_p()
        self._device: ctypes.c_void_p = ctypes.c_void_p()
        self._depth_streaming: bool = False
        self._rgb_streaming: bool = False
        self._frame_condition: Condition = Condition()
        self._depth_queue: deque[tuple[int, NDArray[np.uint16]]] = deque(maxlen=3)
        self._rgb_queue: deque[tuple[int, NDArray[np.uint8]]] = deque(maxlen=3)
        self._stream_error: RuntimeError | None = None
        self._last_timestamp: int | None = None
        self._depth_timestamp: int | None = None
        self._rgb_timestamp: int | None = None
        self._depth_epoch: int = 0
        self._rgb_epoch: int = 0
        self._stop: Event = Event()
        self._event_thread: Thread | None = None
        self._is_closed: bool = False
        self._configure_api()
        self._open_device(device_index)
        self._configure_streams()
        self._start_streams()
        self._start_event_thread()

    @staticmethod
    def _load_library() -> ctypes.CDLL:
        library_path: str | None = ctypes.util.find_library("freenect")
        if library_path is None:
            raise RuntimeError("libfreenect was not found in the system library path")
        try:
            return ctypes.CDLL(library_path)
        except OSError as exc:
            raise RuntimeError(f"Could not load {library_path}: {exc}") from exc

    def _open_device(self, device_index: int) -> None:
        if self._lib.freenect_init(ctypes.byref(self._context), None) < 0:
            raise RuntimeError("libfreenect could not initialize USB")
        if self._lib.freenect_num_devices(self._context) <= device_index:
            self.close()
            raise RuntimeError("No Kinect device found")

        self._lib.freenect_select_subdevices(self._context, 0x02)
        if (
            self._lib.freenect_open_device(
                self._context, ctypes.byref(self._device), device_index
            )
            < 0
        ):
            self.close()
            raise RuntimeError(
                "Could not open Kinect camera; check the USB connection and power"
            )

    def _configure_streams(self) -> None:
        depth_mode: FrameMode = self._lib.freenect_find_depth_mode(
            self._RESOLUTION_MEDIUM, self._DEPTH_REGISTERED
        )

        rgb_mode: FrameMode = self._lib.freenect_find_video_mode(
            self._RESOLUTION_MEDIUM, self._VIDEO_RGB
        )

        if not depth_mode.is_valid or not rgb_mode.is_valid:
            self.close()
            raise RuntimeError("libfreenect does not support the requested RGB-D modes")
        if depth_mode.width != rgb_mode.width or depth_mode.height != rgb_mode.height:
            self.close()
            raise RuntimeError("Kinect depth and color modes have different dimensions")

        self._width: int = depth_mode.width
        self._height: int = depth_mode.height
        self._depth_callback: FrameHandler = FrameCallback(self._on_depth_frame)
        self._rgb_callback: FrameHandler = FrameCallback(self._on_rgb_frame)
        self._lib.freenect_set_depth_callback(self._device, self._depth_callback)
        self._lib.freenect_set_video_callback(self._device, self._rgb_callback)
        if self._lib.freenect_set_depth_mode(self._device, depth_mode) < 0:
            self.close()
            raise RuntimeError("Could not set registered Kinect depth mode")
        if self._lib.freenect_set_video_mode(self._device, rgb_mode) < 0:
            self.close()
            raise RuntimeError("Could not set Kinect RGB video mode")

    def _start_streams(self) -> None:
        if self._lib.freenect_start_depth(self._device) < 0:
            self.close()
            raise RuntimeError("Could not start Kinect depth stream")

        self._depth_streaming = True
        if self._lib.freenect_start_video(self._device) < 0:
            self.close()
            raise RuntimeError("Could not start Kinect video stream")
        self._rgb_streaming = True

    def _start_event_thread(self) -> None:
        self._event_thread = Thread(
            target=self._pump_events,
            name="kinect-usb-events",
            daemon=True,
        )
        self._event_thread.start()

    def _configure_api(self) -> None:
        lib: ctypes.CDLL = self._lib
        lib.freenect_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        lib.freenect_init.restype = ctypes.c_int
        lib.freenect_num_devices.argtypes = [ctypes.c_void_p]
        lib.freenect_num_devices.restype = ctypes.c_int
        lib.freenect_select_subdevices.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.freenect_select_subdevices.restype = None
        lib.freenect_open_device.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
        ]

        lib.freenect_open_device.restype = ctypes.c_int
        lib.freenect_find_depth_mode.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.freenect_find_depth_mode.restype = FrameMode
        lib.freenect_find_video_mode.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.freenect_find_video_mode.restype = FrameMode
        lib.freenect_set_depth_mode.argtypes = [ctypes.c_void_p, FrameMode]
        lib.freenect_set_depth_mode.restype = ctypes.c_int
        lib.freenect_set_video_mode.argtypes = [ctypes.c_void_p, FrameMode]
        lib.freenect_set_video_mode.restype = ctypes.c_int

        for name in (
            "freenect_start_depth",
            "freenect_start_video",
            "freenect_stop_depth",
            "freenect_stop_video",
        ):
            function = getattr(lib, name)
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_int

        lib.freenect_set_depth_callback.argtypes = [ctypes.c_void_p, FrameCallback]
        lib.freenect_set_video_callback.argtypes = [ctypes.c_void_p, FrameCallback]
        lib.freenect_process_events_timeout.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(Timeval),
        ]

        lib.freenect_process_events_timeout.restype = ctypes.c_int
        lib.freenect_close_device.argtypes = [ctypes.c_void_p]
        lib.freenect_close_device.restype = ctypes.c_int
        lib.freenect_shutdown.argtypes = [ctypes.c_void_p]
        lib.freenect_shutdown.restype = ctypes.c_int

    def _on_depth_frame(self, _device: int, buffer: int, timestamp: int) -> None:
        depth_mm: NDArray[np.uint16] = np.ctypeslib.as_array(
            ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint16)),
            shape=(self._height, self._width),
        ).copy()

        with self._frame_condition:
            timestamp, self._depth_timestamp, self._depth_epoch = (
                self._unwrap_timestamp(
                    timestamp, self._depth_timestamp, self._depth_epoch
                )
            )
            self._depth_queue.append((timestamp, depth_mm))
            self._frame_condition.notify()

    def _on_rgb_frame(self, _device: int, buffer: int, timestamp: int) -> None:
        rgb: NDArray[np.uint8] = np.ctypeslib.as_array(
            ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8)),
            shape=(self._height, self._width, 3),
        ).copy()

        with self._frame_condition:
            timestamp, self._rgb_timestamp, self._rgb_epoch = self._unwrap_timestamp(
                timestamp, self._rgb_timestamp, self._rgb_epoch
            )
            self._rgb_queue.append((timestamp, rgb))
            self._frame_condition.notify()

    @classmethod
    def _unwrap_timestamp(
        cls, timestamp: int, previous: int | None, epoch: int
    ) -> tuple[int, int, int]:
        if previous is not None and timestamp < previous:
            epoch += cls._TIMESTAMP_WRAP
        return timestamp + epoch, timestamp, epoch

    def _pump_events(self) -> None:
        while not self._stop.is_set():
            timeout: Timeval = Timeval(0, self._EVENT_TIMEOUT_US)
            event_result: int = self._lib.freenect_process_events_timeout(
                self._context, ctypes.byref(timeout)
            )

            if event_result < 0 and not self._stop.is_set():
                with self._frame_condition:
                    self._stream_error = RuntimeError(
                        "Kinect USB stream stopped while receiving frames"
                    )
                    self._frame_condition.notify_all()

                return

    def read(self, timeout: float = 30.0) -> RGBDFrame | None:
        if self._is_closed:
            raise RuntimeError("Kinect reader is closed")

        deadline: float = time.monotonic() + timeout
        with self._frame_condition:
            while True:
                if self._stream_error is not None:
                    raise self._stream_error

                frame = self._next_frame()
                if frame is not None:
                    return frame

                remaining: float = deadline - time.monotonic()
                if remaining <= 0:
                    return None

                self._frame_condition.wait(timeout=remaining)

    def _next_frame(self) -> RGBDFrame | None:
        pairs = [
            (
                abs(depth_timestamp - rgb_timestamp),
                max(depth_timestamp, rgb_timestamp),
                depth_index,
                rgb_index,
            )
            for depth_index, (depth_timestamp, _) in enumerate(self._depth_queue)
            for rgb_index, (rgb_timestamp, _) in enumerate(self._rgb_queue)
            if abs(depth_timestamp - rgb_timestamp) <= self._SYNC_WINDOW_TICKS
        ]

        if self._last_timestamp is not None:
            pairs = [pair for pair in pairs if pair[1] > self._last_timestamp]

        if not pairs:
            if self._depth_queue and self._rgb_queue:
                if self._depth_queue[0][0] < self._rgb_queue[0][0]:
                    self._depth_queue.popleft()
                else:
                    self._rgb_queue.popleft()
            return None

        _, _, depth_index, rgb_index = min(pairs, key=lambda pair: (-pair[1], pair[0]))

        depth_timestamp, depth_mm = self._depth_queue[depth_index]
        rgb_timestamp, rgb = self._rgb_queue[rgb_index]

        for _ in range(depth_index + 1):
            self._depth_queue.popleft()

        for _ in range(rgb_index + 1):
            self._rgb_queue.popleft()

        self._last_timestamp = max(depth_timestamp, rgb_timestamp)
        depth_m: NDArray[np.float32] = depth_mm.astype(np.float32) / 1000.0
        depth_m[depth_mm == 0] = 0.0
        return RGBDFrame(
            rgb=rgb,
            depth_m=depth_m,
            timestamp=(depth_timestamp + rgb_timestamp) / (2 * self._CLOCK_HZ),
        )

    def close(self) -> None:
        if self._is_closed:
            return

        self._is_closed = True
        self._stop.set()

        if self._event_thread is not None:
            self._event_thread.join(timeout=1.0)
        if self._device.value and self._rgb_streaming:
            self._lib.freenect_stop_video(self._device)
            self._rgb_streaming = False
        if self._device.value and self._depth_streaming:
            self._lib.freenect_stop_depth(self._device)
            self._depth_streaming = False
        if self._device.value:
            self._lib.freenect_close_device(self._device)
            self._device = ctypes.c_void_p()
        if self._context.value:
            self._lib.freenect_shutdown(self._context)
            self._context = ctypes.c_void_p()
