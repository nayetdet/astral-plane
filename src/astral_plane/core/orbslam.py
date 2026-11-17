from __future__ import annotations

import numpy as np
import orbslam3
from numpy.typing import NDArray

from astral_plane.bindings.orbslam import OrbSlamBinding
from astral_plane.schemas.rgbd_frame import RGBDFrame
from astral_plane.settings import Settings


class OrbSlam:
    def __init__(self, settings: Settings) -> None:
        self._slam: OrbSlamBinding = orbslam3.ORB_SLAM3(
            str(settings.vocabulary_path),
            str(settings.slam_config_path),
            "rgbd",
            visualize=False,
        )

        self._tracking: bool = False

    def track(self, frame: RGBDFrame) -> NDArray[np.float32]:
        pose = self._slam.TrackRGBD(frame.rgb, frame.depth_m, frame.timestamp)
        self._tracking = self._is_tracked_pose(pose)
        return pose

    def is_tracking(self) -> bool:
        return self._tracking

    def close(self) -> None:
        self._slam.Shutdown()

    @staticmethod
    def _is_tracked_pose(pose: NDArray[np.float32]) -> bool:
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            return False
        rotation = pose[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3):
            return False
        return not np.allclose(pose, np.eye(4, dtype=np.float32), atol=1e-6)
