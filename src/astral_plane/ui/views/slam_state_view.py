from __future__ import annotations

from typing import Final

import numpy as np
import pygame
from numpy.typing import NDArray


class SlamStateView:
    _CELL_SIZE: Final[int] = 32
    _MARGIN: Final[int] = 8
    _MIN_RESPONSE: Final[int] = 900

    def render(self, rgb: NDArray[np.uint8]) -> pygame.Surface:
        surface = pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))
        gray = rgb.astype(np.int16).mean(axis=2)
        horizontal = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        vertical = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        response = horizontal * vertical

        for top in range(
            self._MARGIN,
            response.shape[0] - self._MARGIN,
            self._CELL_SIZE,
        ):
            for left in range(
                self._MARGIN,
                response.shape[1] - self._MARGIN,
                self._CELL_SIZE,
            ):
                cell = response[
                    top : top + self._CELL_SIZE,
                    left : left + self._CELL_SIZE,
                ]

                row, column = np.unravel_index(np.argmax(cell), cell.shape)
                if cell[row, column] > self._MIN_RESPONSE:
                    pygame.draw.rect(
                        surface,
                        (0, 255, 70),
                        (int(left + column - 4), int(top + row - 4), 9, 9),
                        2,
                    )

        return surface
