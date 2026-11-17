from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RGBDFrame:
    rgb: NDArray[np.uint8]
    depth_m: NDArray[np.float32]
    timestamp: float
