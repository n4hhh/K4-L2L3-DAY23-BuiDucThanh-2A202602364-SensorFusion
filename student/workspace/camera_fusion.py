"""Part G: Camera FOV, pinhole projection and measurement model."""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from fusion_lab.workspace_support import get_tracking_params

Matrix = np.matrix | np.ndarray


def is_in_field_of_view(x: Matrix, sensor: Any) -> bool:
    """Check whether a state is inside the sensor horizontal FOV."""
    position = np.asarray(x, dtype=float).reshape(-1)[:3]
    transform = np.asarray(sensor.veh_to_sens, dtype=float)

    # Vehicle frame -> sensor frame
    p_s = transform[:3, :3] @ position + transform[:3, 3]

    if not np.isfinite(p_s).all():
        return False

    depth, left, up = p_s

    # A camera cannot observe points behind its image plane.
    if sensor.name == "camera" and depth <= 1e-6:
        return False

    angle = np.arctan2(left, depth)
    fov_min, fov_max = sensor.fov

    return bool(fov_min <= angle <= fov_max)


def camera_measurement_prediction(
    x: Matrix,
    sensor: Any,
) -> Matrix:
    """Project 3D vehicle-frame position into 2D image pixels."""
    position = np.asarray(x, dtype=float).reshape(-1)[:3]
    transform = np.asarray(sensor.veh_to_sens, dtype=float)

    p_s = transform[:3, :3] @ position + transform[:3, 3]

    if not np.isfinite(p_s).all() or p_s[0] <= 1e-6:
        raise ValueError(
            "Camera projection requires finite coordinates and "
            f"positive depth > 1e-6; sensor position={p_s.tolist()}"
        )

    depth, left, up = p_s

    u = sensor.c_i - sensor.f_i * left / depth
    v = sensor.c_j - sensor.f_j * up / depth

    return np.asmatrix([[u], [v]])


def build_camera_measurement(
    z: Sequence[float],
    sensor: Any,
) -> dict[str, Any]:
    """Build camera observation vector and noise covariance."""
    params = get_tracking_params()

    z_mat = np.asmatrix(np.asarray(z, dtype=float).reshape(2, 1))

    R = np.asmatrix(
        np.diag(
            [
                params.sigma_cam_i**2,
                params.sigma_cam_j**2,
            ]
        )
    )

    return {
        "z": z_mat,
        "R": R,
        "sensor": sensor,
    }
