from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import numpy as np
import pygame
from numpy.typing import NDArray

from astral_plane.schemas.rgbd_frame import RGBDFrame


class PointCloudView:
    _SAMPLE_STEP: Final[int] = 12
    _VOXEL_SIZE: Final[float] = 0.03
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
        self._voxels: set[tuple[int, int, int]] = set()
        self._last_update: float = float("-inf")
        self._last_render: float = float("-inf")
        self._surface: pygame.Surface | None = None
        self._dirty: bool = True
        self._map_segment: int | None = None
        self._segment_offset_pending: bool = False
        self._yaw: float = 0.0
        self._pitch: float = 0.45
        self._zoom: float = 1.5
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
                self._zoom = float(np.clip(self._zoom * 1.1**event.y, 0.3, 4.0))
                self._dirty = True

    def render(
        self,
        frame: RGBDFrame,
        pose: NDArray[np.float32],
        *,
        tracking: bool,
        updated: bool,
        size: tuple[int, int],
    ) -> pygame.Surface:
        if (
            tracking
            and updated
            and frame.timestamp - self._last_update >= self._UPDATE_INTERVAL
        ):
            self._add(frame, pose)
            self._last_update = frame.timestamp

        if self._surface is None or (
            self._dirty and frame.timestamp - self._last_render >= self._RENDER_INTERVAL
        ):
            self._surface = pygame.Surface(size)
            self._surface.fill((255, 255, 255))
            if len(self._points):
                self._draw(self._surface)
            self._last_render = frame.timestamp
            self._dirty = False
        return self._surface

    def update_map_segment(self, map_segment: int) -> None:
        if self._map_segment != map_segment:
            self._segment_offset_pending = bool(len(self._points))
            self._map_segment = map_segment

    def _add(self, frame: RGBDFrame, pose: NDArray[np.float32]) -> None:
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
        if self._segment_offset_pending:
            segment_offset = (
                self._points[:, 0].max() + self._SEGMENT_GAP - world[:, 0].min()
            )
            world[:, 0] += segment_offset
            self._segment_offset_pending = False

        voxels = np.floor(world / self._VOXEL_SIZE).astype(np.int32)
        _, unique = np.unique(voxels, axis=0, return_index=True)
        unique.sort()

        colors = frame.rgb[rows * step, columns * step]
        new_indices = [
            index for index in unique if tuple(voxels[index]) not in self._voxels
        ]

        if not new_indices:
            return

        self._voxels.update(tuple(voxels[index]) for index in new_indices)
        self._points = np.concatenate((self._points, world[new_indices]))
        self._colors = np.concatenate((self._colors, colors[new_indices]))
        if len(self._points) > self._MAX_POINTS:
            self._compact()
        self._dirty = True

    def _draw(self, surface: pygame.Surface) -> None:
        center = self._points.mean(axis=0)
        points = self._rotate(self._points - center)
        radius = float(np.linalg.norm(points, axis=1).max())
        distance = max(radius * 2.0 / self._zoom, 0.8)
        depth = points[:, 2] + distance
        visible = depth > self._NEAR_PLANE
        focal = min(surface.get_size()) * 1.15
        x = (surface.get_width() / 2 + focal * points[:, 0] / depth).astype(int)
        y = (surface.get_height() / 2 + focal * points[:, 1] / depth).astype(int)
        visible &= (x >= 1) & (x < surface.get_width() - 2)
        visible &= (y >= 1) & (y < surface.get_height() - 2)
        order = np.argsort(depth[visible])[::-1]
        x, y = x[visible][order], y[visible][order]
        shade = np.clip(1.25 - depth[visible][order] / distance, 0.72, 1.0)
        colors = (self._colors[visible][order] * shade[:, None]).astype(np.uint8)
        pixels = pygame.surfarray.pixels3d(surface)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            pixels[x + dx, y + dy] = colors
        del pixels

    def _compact(self) -> None:
        self._points = self._points[::2]
        self._colors = self._colors[::2]
        voxels = np.floor(self._points / self._VOXEL_SIZE).astype(np.int32)
        self._voxels = {tuple(voxel) for voxel in voxels}

    def _rotate(self, points: NDArray[np.float32]) -> NDArray[np.float32]:
        cy, sy = np.cos(self._yaw), np.sin(self._yaw)
        cp, sp = np.cos(self._pitch), np.sin(self._pitch)
        rotated = points.copy()
        rotated[:, 0] = cy * points[:, 0] + sy * points[:, 2]
        rotated[:, 2] = -sy * points[:, 0] + cy * points[:, 2]
        rotated[:, 1] = cp * points[:, 1] - sp * rotated[:, 2]
        rotated[:, 2] = sp * points[:, 1] + cp * rotated[:, 2]
        return rotated
