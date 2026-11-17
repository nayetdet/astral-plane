from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RGBDFrame:
    rgb: NDArray[np.uint8]
    depth_mm: NDArray[np.uint16]
    timestamp: float
