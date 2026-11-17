from __future__ import annotations

import sys
from contextlib import closing

import numpy as np
from numpy.typing import NDArray

from astral_plane.core.kinect import Kinect
from astral_plane.core.orbslam import OrbSlam
from astral_plane.schemas.rgbd_frame import RGBDFrame
from astral_plane.settings import settings
from astral_plane.ui.ui import UI


def main() -> None:
    try:
        with Kinect() as camera:
            frame: RGBDFrame | None = camera.read(timeout=15.0)
            if frame is None:
                print(
                    "Kinect did not produce an RGB-D frame within 15 seconds; "
                    "check its USB connection and power",
                    file=sys.stderr,
                )

                raise SystemExit(1)

            with closing(OrbSlam(settings)) as slam, closing(UI(slam.reset_map)) as ui:
                pose: NDArray[np.float32] = slam.track(frame)
                tracking: bool = slam.tracking
                frame_updated: bool = True
                while True:
                    next_frame: RGBDFrame | None = camera.read(timeout=0.03)
                    if next_frame is not None:
                        frame = next_frame
                        pose = slam.track(frame)
                        tracking = slam.tracking
                        frame_updated = True

                    if not ui.update(
                        frame,
                        pose,
                        tracking=tracking,
                        frame_updated=frame_updated,
                        map_segment=slam.map_segment,
                    ):
                        break
                    frame_updated = False
    except KeyboardInterrupt:
        print("Kinect capture stopped.")
    except RuntimeError as exc:
        print(f"Kinect startup or capture failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
