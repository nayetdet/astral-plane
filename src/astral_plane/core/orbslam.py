from __future__ import annotations

import numpy as np
import orbslam3
from numpy.typing import NDArray

from astral_plane.bindings.orbslam import OrbSlamBinding
from astral_plane.schemas.rgbd_frame import RGBDFrame
from astral_plane.settings import Settings


class OrbSlam:
    _NEW_SEGMENT_AFTER: float = 1.0

    def __init__(self, settings: Settings) -> None:
        self._slam: OrbSlamBinding = orbslam3.ORB_SLAM3(
            str(settings.vocabulary_path),
            str(settings.slam_config_path),
            "rgbd",
            visualize=False,
        )

        self._tracking: bool = False
        self._tracking_lost_at: float | None = None
        self._map_segment: int = 0

    def track(self, frame: RGBDFrame) -> NDArray[np.float32]:
        pose = self._slam.TrackRGBD(frame.rgb, frame.depth_mm, frame.timestamp)
        self._tracking = self._is_tracked_pose(pose)
        if not self._tracking:
            if self._tracking_lost_at is None:
                self._tracking_lost_at = frame.timestamp

            return pose

        if (
            self._tracking_lost_at is not None
            and frame.timestamp - self._tracking_lost_at >= self._NEW_SEGMENT_AFTER
        ):
            self._map_segment += 1

        self._tracking_lost_at = None
        return pose

    def is_tracking(self) -> bool:
        return self._tracking

    def map_segment(self) -> int:
        return self._map_segment

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
