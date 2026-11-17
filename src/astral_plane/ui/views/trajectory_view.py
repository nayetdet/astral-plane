from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Final

import numpy as np
import pygame
from numpy.typing import NDArray


class TrajectoryView:
    _SIZE: Final[tuple[int, int]] = (720, 520)
    _AREA: Final[pygame.Rect] = pygame.Rect(20, 20, 680, 480)
    _PATH_COLOR: Final[tuple[int, int, int]] = (30, 100, 180)
    _CURRENT_COLOR: Final[tuple[int, int, int]] = (220, 70, 50)
    _MIN_POINTS: Final[int] = 2
    _MAX_SPEED_METERS_PER_SECOND: Final[float] = 2.0
    _POSITION_TOLERANCE_METERS: Final[float] = 0.04

    def __init__(self) -> None:
        self._trajectory: deque[NDArray[np.float32] | None] = deque(maxlen=3000)
        self._last_position: NDArray[np.float32] | None = None
        self._last_timestamp: float | None = None
        self._zoom: float = 1.0
        self._pan: NDArray[np.float32] = np.zeros(2, dtype=np.float32)
        self._dragging: bool = False

    def handle_events(
        self, events: Iterable[pygame.event.Event], area: pygame.Rect
    ) -> None:
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._dragging = area.collidepoint(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._dragging = False
            elif event.type == pygame.MOUSEMOTION and self._dragging:
                self._pan += np.array(
                    (
                        event.rel[0] * self._SIZE[0] / area.width,
                        event.rel[1] * self._SIZE[1] / area.height,
                    ),
                    dtype=np.float32,
                )
            elif event.type == pygame.MOUSEWHEEL and area.collidepoint(
                pygame.mouse.get_pos()
            ):
                self._zoom = float(np.clip(self._zoom * 1.1**event.y, 0.2, 8.0))

    def update(
        self,
        pose: NDArray[np.float32],
        *,
        timestamp: float,
        tracking: bool,
        updated: bool,
    ) -> None:
        if tracking and updated:
            pos = (-pose[:3, :3].T @ pose[:3, 3]).astype(np.float32)
            if self._is_continuous(pos, timestamp):
                self._trajectory.append(pos)
            else:
                self._trajectory.append(None)
                self._trajectory.append(pos)
            self._last_position = pos
            self._last_timestamp = timestamp

    def render(self) -> pygame.Surface:
        surface = pygame.Surface(self._SIZE)
        surface.fill((255, 255, 255))
        if len(self._trajectory) < self._MIN_POINTS:
            return surface

        path = [point for point in self._trajectory if point is not None]
        points = np.array([(point[0], point[2]) for point in path], dtype=np.float32)

        points -= points[0]
        span = np.maximum(np.ptp(points, axis=0), 0.25)
        scale = (
            min(
                (self._AREA.width - 48) / span[0],
                (self._AREA.height - 64) / span[1],
            )
            * self._zoom
        )

        screen = np.empty_like(points)
        screen[:, 0] = self._AREA.centerx + points[:, 0] * scale + self._pan[0]
        screen[:, 1] = self._AREA.centery - points[:, 1] * scale + self._pan[1]
        segment: list[list[int]] = []
        point_index = 0

        for point in self._trajectory:
            if point is None:
                if len(segment) >= self._MIN_POINTS:
                    pygame.draw.lines(surface, self._PATH_COLOR, False, segment, 2)
                segment = []
                continue
            segment.append(screen[point_index].astype(int).tolist())
            point_index += 1

        if len(segment) >= self._MIN_POINTS:
            pygame.draw.lines(surface, self._PATH_COLOR, False, segment, 2)

        pygame.draw.circle(
            surface,
            self._CURRENT_COLOR,
            screen[-1].astype(int),
            5,
        )

        return surface

    def _is_continuous(self, position: NDArray[np.float32], timestamp: float) -> bool:
        if self._last_position is None or self._last_timestamp is None:
            return True

        elapsed = timestamp - self._last_timestamp
        if elapsed <= 0:
            return False

        maximum_distance = (
            self._MAX_SPEED_METERS_PER_SECOND * elapsed
            + self._POSITION_TOLERANCE_METERS
        )

        return bool(np.linalg.norm(position - self._last_position) <= maximum_distance)
