from __future__ import annotations

import numpy as np
import orbslam3

from .settings import settings


def main() -> None:
    yy, xx = np.indices((480, 640))
    pattern = ((xx // 24 + yy // 24) % 2 * 180 + 40).astype(np.uint8)
    image = np.stack((pattern, np.roll(pattern, 7, axis=0), pattern), axis=-1)
    depth = np.full((480, 640), 1.0, dtype=np.float32)
    system = orbslam3.ORB_SLAM3(
        str(settings.vocabulary_path),
        str(settings.settings_path),
        "rgbd",
        visualize=False,
    )

    try:
        pose = system.TrackRGBD(image, depth, settings.timestamp)
        print(pose)
    finally:
        system.Shutdown()


if __name__ == "__main__":
    main()
