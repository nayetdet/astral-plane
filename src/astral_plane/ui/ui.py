from __future__ import annotations

from collections.abc import Callable
from typing import Final

import numpy as np
import pygame
from numpy.typing import NDArray

from astral_plane.schemas.rgbd_frame import RGBDFrame
from astral_plane.ui.views.depth_view import DepthView
from astral_plane.ui.views.point_cloud_view import PointCloudView
from astral_plane.ui.views.slam_state_view import SlamStateView
from astral_plane.ui.views.trajectory_view import TrajectoryView


class UI:
    _SIZE: Final[tuple[int, int]] = (877, 720)
    _PADDING: Final[int] = 16
    _BUTTON_HEIGHT: Final[int] = 38
    _TAB_GAP: Final[int] = 6
    _ACTION_GAP: Final[int] = 8
    _RESET_BUTTON_WIDTH: Final[int] = 82
    _EXPORT_BUTTON_WIDTH: Final[int] = 84
    _TAB_WIDTH: Final[int] = 136
    _STATUS_DURATION_MS: Final[int] = 3000
    _STATE: Final[int] = 0
    _DEPTH: Final[int] = 1
    _TRAJECTORY: Final[int] = 2
    _POINT_CLOUD: Final[int] = 3
    _LABELS: Final[tuple[str, ...]] = (
        "SLAM State",
        "Depth",
        "Trajectory",
        "Point Cloud",
    )

    def __init__(self, reset_map: Callable[[], None]) -> None:
        pygame.init()
        self._screen: pygame.Surface = pygame.display.set_mode(self._SIZE)
        pygame.display.set_caption("Astral Plane")
        self._font: pygame.font.Font = pygame.font.Font(None, 26)
        self._slam_state: SlamStateView = SlamStateView()
        self._depth_view: DepthView = DepthView()
        self._point_cloud: PointCloudView = PointCloudView()
        self._trajectory_view: TrajectoryView = TrajectoryView()
        self._active_tab: int = self._STATE
        self._reset_map: Callable[[], None] = reset_map

        self._buttons: tuple[pygame.Rect, ...] = tuple(
            pygame.Rect(
                self._PADDING + index * (self._TAB_WIDTH + self._TAB_GAP),
                self._PADDING,
                self._TAB_WIDTH,
                self._BUTTON_HEIGHT,
            )
            for index in range(len(self._LABELS))
        )

        self._reset_button: pygame.Rect = pygame.Rect(
            self._SIZE[0] - self._PADDING - self._RESET_BUTTON_WIDTH,
            self._PADDING,
            self._RESET_BUTTON_WIDTH,
            self._BUTTON_HEIGHT,
        )

        self._export_button: pygame.Rect = pygame.Rect(
            self._reset_button.left - self._ACTION_GAP - self._EXPORT_BUTTON_WIDTH,
            self._PADDING,
            self._EXPORT_BUTTON_WIDTH,
            self._BUTTON_HEIGHT,
        )

        self._status_message: str = ""
        self._status_until: int = 0
        self._content: pygame.Rect = pygame.Rect(
            self._PADDING,
            self._PADDING * 2 + self._BUTTON_HEIGHT,
            self._SIZE[0] - self._PADDING * 2,
            self._SIZE[1] - self._PADDING * 3 - self._BUTTON_HEIGHT,
        )

    def update(
        self,
        frame: RGBDFrame,
        pose: NDArray[np.float32],
        *,
        tracking: bool,
        frame_updated: bool,
        map_segment: int,
    ) -> bool:
        events = pygame.event.get()
        keep_running, map_reset = self._handle_events(events)
        if not keep_running:
            return False

        if not map_reset:
            if self._active_tab == self._POINT_CLOUD:
                self._point_cloud.handle_events(events, self._content)
            elif self._active_tab == self._TRAJECTORY:
                self._trajectory_view.handle_events(events, self._content)

            self._trajectory_view.update(
                pose,
                timestamp=frame.timestamp,
                tracking=tracking,
                frame_updated=frame_updated,
                map_segment=map_segment,
            )

            self._point_cloud.update(
                frame,
                pose,
                tracking=tracking,
                frame_updated=frame_updated,
                map_segment=map_segment,
            )

        view = self._render_view(frame)
        self._screen.fill((250, 250, 248))
        for index, button in enumerate(self._buttons):
            active = index == self._active_tab
            color = (35, 35, 35) if active else (255, 255, 255)
            text_color = (255, 255, 255) if active else (35, 35, 35)
            pygame.draw.rect(self._screen, color, button, border_radius=4)
            pygame.draw.rect(self._screen, (25, 25, 25), button, 1, border_radius=4)
            label = self._font.render(self._LABELS[index], True, text_color)
            self._screen.blit(label, label.get_rect(center=button.center))

        pygame.draw.rect(
            self._screen, (185, 55, 55), self._reset_button, border_radius=4
        )
        pygame.draw.rect(
            self._screen,
            (145, 35, 35),
            self._reset_button,
            1,
            border_radius=4,
        )
        label = self._font.render("Reset", True, (255, 255, 255))
        self._screen.blit(label, label.get_rect(center=self._reset_button.center))
        pygame.draw.rect(
            self._screen, (45, 90, 185), self._export_button, border_radius=4
        )
        pygame.draw.rect(
            self._screen,
            (30, 65, 140),
            self._export_button,
            1,
            border_radius=4,
        )
        label = self._font.render("Export", True, (255, 255, 255))
        self._screen.blit(label, label.get_rect(center=self._export_button.center))

        scale = min(
            self._content.width / view.get_width(),
            self._content.height / view.get_height(),
        )

        size = (round(view.get_width() * scale), round(view.get_height() * scale))
        image = pygame.transform.smoothscale(view, size)
        rect = image.get_rect(center=self._content.center)
        self._screen.blit(image, rect)
        pygame.draw.rect(self._screen, (35, 35, 35), self._content, 1)
        if self._status_message and pygame.time.get_ticks() < self._status_until:
            status = self._font.render(self._status_message, True, (0, 0, 0))
            self._screen.blit(
                status, (self._PADDING, self._SIZE[1] - status.get_height() - 4)
            )

        pygame.display.flip()
        return True

    def _handle_events(self, events: list[pygame.event.Event]) -> tuple[bool, bool]:
        map_reset = False
        for event in events:
            if event.type in (pygame.QUIT, pygame.WINDOWCLOSE) or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                return False, map_reset

            if event.type == pygame.KEYDOWN and pygame.K_1 <= event.key <= pygame.K_4:
                self._active_tab = event.key - pygame.K_1
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self._reset_button.collidepoint(event.pos):
                    self._reset_map()
                    self._point_cloud.clear()
                    self._trajectory_view.clear()
                    map_reset = True
                    self._show_status("Map reset")
                elif self._export_button.collidepoint(event.pos):
                    self._export_point_cloud()
                else:
                    for index, button in enumerate(self._buttons):
                        if button.collidepoint(event.pos):
                            self._active_tab = index
        return True, map_reset

    def _export_point_cloud(self) -> None:
        try:
            path = self._point_cloud.export_ply()
        except (OSError, ValueError) as error:
            self._show_status(f"Export failed: {error}")
        else:
            self._show_status(f"Exported to {path}")

    def _show_status(self, message: str) -> None:
        self._status_message = message
        self._status_until = pygame.time.get_ticks() + self._STATUS_DURATION_MS

    def _render_view(self, frame: RGBDFrame) -> pygame.Surface:
        if self._active_tab == self._STATE:
            return self._slam_state.render(frame.rgb)
        if self._active_tab == self._DEPTH:
            return self._depth_view.render(frame.depth_mm)
        if self._active_tab == self._TRAJECTORY:
            return self._trajectory_view.render()
        return self._point_cloud.render(frame.timestamp, self._content.size)

    def close(self) -> None:
        pygame.quit()
