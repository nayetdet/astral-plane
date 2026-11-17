from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import numpy as np
import pygame
from numpy.typing import NDArray

from astral_plane.schemas.rgbd_frame import RGBDFrame


class PointCloudView:
    _SAMPLE_STEP: Final[int] = 6
    _VOXEL_SIZE: Final[float] = 0.015
    _MAX_POINTS: Final[int] = 180_000
    _UPDATE_INTERVAL: Final[float] = 0.12
    _RENDER_INTERVAL: Final[float] = 0.1
    _MIN_DEPTH_MM: Final[int] = 150
    _MAX_DEPTH_MM: Final[int] = 6000
    _NEAR_PLANE: Final[float] = 0.1
    _SEGMENT_GAP: Final[float] = 0.5
    _FX: Final[float] = 517.306408
    _FY: Final[float] = 516.469215
    _CX: Final[float] = 318.643040
    _CY: Final[float] = 255.313989

    def __init__(self) -> None:
        self._points: NDArray[np.float32] = np.empty((0, 3), dtype=np.float32)
        self._colors: NDArray[np.uint8] = np.empty((0, 3), dtype=np.uint8)
        self._segments: NDArray[np.int32] = np.empty(0, dtype=np.int32)
        self._voxel_size: float = self._VOXEL_SIZE
        self._voxels: set[tuple[int, int, int, int]] = set()
        self._last_update: float = float("-inf")
        self._last_render: float = float("-inf")
        self._surface: pygame.Surface | None = None
        self._dirty: bool = True
        self._yaw: float = 0.0
        self._pitch: float = 0.45
        self._zoom: float = 1.0
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
                self._yaw += event.rel[0] * 0.01
                self._pitch = float(
                    np.clip(self._pitch + event.rel[1] * 0.01, -1.45, 1.45)
                )
                self._dirty = True
            elif event.type == pygame.MOUSEWHEEL and area.collidepoint(
                pygame.mouse.get_pos()
            ):
                self._zoom = float(np.clip(self._zoom * 1.1**event.y, 0.15, 12.0))
                self._dirty = True

    def clear(self) -> None:
        self._points = np.empty((0, 3), dtype=np.float32)
        self._colors = np.empty((0, 3), dtype=np.uint8)
        self._segments = np.empty(0, dtype=np.int32)
        self._voxels.clear()
        self._last_update = float("-inf")
        self._surface = None
        self._dirty = True

    def update(
        self,
        frame: RGBDFrame,
        pose: NDArray[np.float32],
        *,
        tracking: bool,
        updated: bool,
        map_segment: int,
    ) -> None:
        if (
            tracking
            and updated
            and frame.timestamp - self._last_update >= self._UPDATE_INTERVAL
        ):
            self._add(frame, pose, map_segment)
            self._last_update = frame.timestamp

    def render(self, timestamp: float, size: tuple[int, int]) -> pygame.Surface:
        if self._surface is None or (
            self._dirty and timestamp - self._last_render >= self._RENDER_INTERVAL
        ):
            self._surface = pygame.Surface(size)
            self._surface.fill((36, 39, 44))
            if len(self._points):
                self._draw(self._surface)
            self._last_render = timestamp
            self._dirty = False
        return self._surface

    def _add(
        self, frame: RGBDFrame, pose: NDArray[np.float32], map_segment: int
    ) -> None:
        step = self._SAMPLE_STEP
        depth = frame.depth_mm[::step, ::step]
        rows, columns = np.nonzero(
            (depth > self._MIN_DEPTH_MM) & (depth < self._MAX_DEPTH_MM)
        )

        if not len(rows):
            return

        distance = depth[rows, columns].astype(np.float32) / 1000.0
        camera = np.column_stack(
            (
                (columns * step - self._CX) * distance / self._FX,
                (rows * step - self._CY) * distance / self._FY,
                distance,
            )
        ).astype(np.float32)

        rotation = pose[:3, :3]
        origin = -rotation.T @ pose[:3, 3]
        world = camera @ rotation + origin
        voxels = np.floor(world / self._voxel_size).astype(np.int32)
        _, unique = np.unique(voxels, axis=0, return_index=True)
        unique.sort()

        colors = frame.rgb[rows * step, columns * step]
        new_indices = [
            index
            for index in unique
            if (map_segment, *voxels[index]) not in self._voxels
        ]

        if not new_indices:
            return

        self._voxels.update((map_segment, *voxels[index]) for index in new_indices)
        self._points = np.concatenate((self._points, world[new_indices]))
        self._colors = np.concatenate((self._colors, colors[new_indices]))
        self._segments = np.concatenate(
            (
                self._segments,
                np.full(len(new_indices), map_segment, dtype=np.int32),
            )
        )
        if len(self._points) > self._MAX_POINTS:
            self._compact()
        self._dirty = True

    def _draw(self, surface: pygame.Surface) -> None:
        points = np.empty_like(self._points)
        offset = 0.0
        for segment in np.unique(self._segments):
            mask = self._segments == segment
            group = self._rotate(self._points[mask] - self._points[mask].mean(axis=0))
            minimum = float(group[:, 0].min())
            maximum = float(group[:, 0].max())
            group[:, 0] += offset - minimum
            points[mask] = group
            offset += maximum - minimum + self._SEGMENT_GAP

        points -= points.mean(axis=0)
        radius = float(np.linalg.norm(points, axis=1).max())
        distance = max(radius * 2.8 / self._zoom, 0.8)
        depth = distance - points[:, 2]
        visible = depth > self._NEAR_PLANE
        focal = min(surface.get_size()) * 1.15
        safe_depth = np.maximum(depth, self._NEAR_PLANE)
        x = (surface.get_width() / 2 - focal * points[:, 0] / safe_depth).astype(int)
        y = (surface.get_height() / 2 + focal * points[:, 1] / safe_depth).astype(int)
        visible &= (x >= 1) & (x < surface.get_width() - 1)
        visible &= (y >= 1) & (y < surface.get_height() - 1)
        if not visible.any():
            return

        order = np.argsort(depth[visible])[::-1]
        x, y = x[visible][order], y[visible][order]
        colors = self._colors[visible][order]
        pixels = pygame.surfarray.pixels3d(surface)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            pixels[x + dx, y + dy] = colors
        del pixels

    def _compact(self) -> None:
        voxel_size = self._voxel_size * 2
        while True:
            voxels = np.floor(self._points / voxel_size).astype(np.int32)
            keys = np.column_stack((self._segments, voxels))
            unique, inverse = np.unique(keys, axis=0, return_inverse=True)
            if len(unique) <= self._MAX_POINTS:
                break
            voxel_size *= 2

        counts = np.bincount(inverse, minlength=len(unique)).astype(np.float32)
        points = np.zeros((len(unique), 3), dtype=np.float32)
        colors = np.zeros((len(unique), 3), dtype=np.float32)
        np.add.at(points, inverse, self._points)
        np.add.at(colors, inverse, self._colors)
        self._points = points / counts[:, None]
        self._colors = (colors / counts[:, None]).astype(np.uint8)
        self._segments = unique[:, 0].astype(np.int32)
        self._voxel_size = voxel_size
        self._voxels.clear()
        self._voxels = {tuple(voxel) for voxel in unique}

    def _rotate(self, points: NDArray[np.float32]) -> NDArray[np.float32]:
        cy, sy = np.cos(self._yaw), np.sin(self._yaw)
        cp, sp = np.cos(self._pitch), np.sin(self._pitch)
        rotated = points.copy()
        rotated[:, 0] = cy * points[:, 0] + sy * points[:, 2]
        rotated[:, 2] = -sy * points[:, 0] + cy * points[:, 2]
        rotated[:, 1] = cp * points[:, 1] - sp * rotated[:, 2]
        rotated[:, 2] = sp * points[:, 1] + cp * rotated[:, 2]
        return rotated
