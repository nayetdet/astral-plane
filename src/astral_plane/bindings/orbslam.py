from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class OrbSlamBinding(Protocol):
    def TrackRGBD(
        self,
        rgb: NDArray[np.uint8],
        depth_m: NDArray[np.float32],
        timestamp: float,
    ) -> NDArray[np.float32]: ...

    def Shutdown(self) -> None: ...
