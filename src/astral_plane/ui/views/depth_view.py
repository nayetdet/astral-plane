from __future__ import annotations

import numpy as np
import pygame
from numpy.typing import NDArray


class DepthView:
    def render(self, depth_mm: NDArray[np.uint16]) -> pygame.Surface:
        normalized: NDArray[np.float32] = np.clip(
            depth_mm.astype(np.float32) / 5000.0, 0, 1
        )

        depth_rgb: NDArray[np.uint8] = np.stack(
            (
                255 * (1 - normalized),
                255 * (1 - np.abs(2 * normalized - 1)),
                255 * normalized,
            ),
            axis=-1,
        ).astype(np.uint8)

        depth_rgb[depth_mm == 0] = 0
        return pygame.surfarray.make_surface(np.swapaxes(depth_rgb, 0, 1))
