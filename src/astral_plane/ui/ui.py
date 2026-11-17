from __future__ import annotations

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

    def __init__(self) -> None:
        pygame.init()
        self._screen: pygame.Surface = pygame.display.set_mode(self._SIZE)
        pygame.display.set_caption("Astral Plane")
        self._font: pygame.font.Font = pygame.font.Font(None, 26)
        self._state: SlamStateView = SlamStateView()
        self._depth: DepthView = DepthView()
        self._cloud: PointCloudView = PointCloudView()
        self._trajectory: TrajectoryView = TrajectoryView()
        self._mode: int = self._STATE

        button_width = 140
        total_width = button_width * len(self._LABELS) + self._PADDING * (
            len(self._LABELS) - 1
        )

        left = (self._SIZE[0] - total_width) // 2
        self._buttons: tuple[pygame.Rect, ...] = tuple(
            pygame.Rect(
                left + index * (button_width + self._PADDING),
                self._PADDING,
                button_width,
                self._BUTTON_HEIGHT,
            )
            for index in range(len(self._LABELS))
        )

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
        updated: bool,
        map_segment: int,
    ) -> bool:
        events = pygame.event.get()
        for event in events:
            if event.type in (pygame.QUIT, pygame.WINDOWCLOSE) or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                return False

            if event.type == pygame.KEYDOWN and pygame.K_1 <= event.key <= pygame.K_4:
                self._mode = event.key - pygame.K_1
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for index, button in enumerate(self._buttons):
                    if button.collidepoint(event.pos):
                        self._mode = index

        if self._mode == self._POINT_CLOUD:
            self._cloud.handle_events(events, self._content)
        elif self._mode == self._TRAJECTORY:
            self._trajectory.handle_events(events, self._content)

        self._trajectory.update(
            pose,
            timestamp=frame.timestamp,
            tracking=tracking,
            updated=updated,
            map_segment=map_segment,
        )

        self._cloud.update(
            frame,
            pose,
            tracking=tracking,
            updated=updated,
            map_segment=map_segment,
        )

        view = self._render_view(frame)
        self._screen.fill((250, 250, 248))
        for index, button in enumerate(self._buttons):
            active = index == self._mode
            color = (35, 35, 35) if active else (255, 255, 255)
            text_color = (255, 255, 255) if active else (35, 35, 35)
            pygame.draw.rect(self._screen, color, button, border_radius=4)
            pygame.draw.rect(self._screen, (25, 25, 25), button, 1, border_radius=4)
            label = self._font.render(self._LABELS[index], True, text_color)
            self._screen.blit(label, label.get_rect(center=button.center))

        scale = min(
            self._content.width / view.get_width(),
            self._content.height / view.get_height(),
        )

        size = (round(view.get_width() * scale), round(view.get_height() * scale))
        image = pygame.transform.smoothscale(view, size)
        rect = image.get_rect(center=self._content.center)
        self._screen.blit(image, rect)
        pygame.draw.rect(self._screen, (35, 35, 35), self._content, 1)
        pygame.display.flip()
        return True

    def _render_view(self, frame: RGBDFrame) -> pygame.Surface:
        if self._mode == self._STATE:
            return self._state.render(frame.rgb)
        if self._mode == self._DEPTH:
            return self._depth.render(frame.depth_mm)
        if self._mode == self._TRAJECTORY:
            return self._trajectory.render()
        return self._cloud.render(frame.timestamp, self._content.size)

    def close(self) -> None:
        pygame.quit()
