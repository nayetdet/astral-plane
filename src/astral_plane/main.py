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
            first: RGBDFrame | None = camera.read(timeout=15.0)
            if first is None:
                print(
                    "Kinect did not produce an RGB-D frame within 15 seconds; "
                    "check its USB connection and power",
                    file=sys.stderr,
                )
                raise SystemExit(1)

            frame: RGBDFrame = first
            with closing(OrbSlam(settings)) as slam, closing(UI()) as ui:
                pose: NDArray[np.float32] = slam.track(frame)
                tracked: bool = slam.is_tracking()
                updated: bool = True
                while True:
                    new: RGBDFrame | None = camera.read(timeout=0.03)
                    if new is not None:
                        frame = new
                        pose = slam.track(frame)
                        tracked = slam.is_tracking()
                        updated = True

                    if not ui.update(
                        frame,
                        pose,
                        tracking=tracked,
                        updated=updated,
                        map_segment=slam.map_segment(),
                    ):
                        break
                    updated = False
    except KeyboardInterrupt:
        print("Kinect capture stopped.")
    except RuntimeError as exc:
        print(f"Kinect startup or capture failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
