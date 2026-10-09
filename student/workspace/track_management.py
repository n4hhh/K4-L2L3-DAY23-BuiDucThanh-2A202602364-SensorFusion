"""Part H: LiDAR-driven track initialization and lifecycle."""

from __future__ import annotations

from typing import Any

import numpy as np

from fusion_lab.workspace_support import get_tracking_params


def init_track_state_from_meas(meas: Any) -> dict[str, Any]:
    """Initialize a 6D track from a LiDAR measurement."""
    params = get_tracking_params()

    # Transform measured position from sensor to vehicle frame.
    transform = np.asarray(meas.sensor.sens_to_veh, dtype=float)
    rotation = transform[:3, :3]
    translation = transform[:3, 3].reshape(3, 1)

    position_sensor = np.asarray(meas.z, dtype=float).reshape(3, 1)

    position_vehicle = rotation @ position_sensor + translation

    # State: [px, py, pz, vx, vy, vz]^T
    # Initial velocity is unknown, so start from zero.
    x = np.asmatrix(
        np.vstack(
            [
                position_vehicle,
                np.zeros((3, 1)),
            ]
        )
    )

    # Transform measurement uncertainty into vehicle frame.
    measurement_R = np.asarray(meas.R, dtype=float)
    position_cov = rotation @ measurement_R @ rotation.T

    P = np.zeros((6, 6), dtype=float)
    P[:3, :3] = position_cov

    # Initial velocity uncertainty.
    P[3, 3] = params.sigma_p44**2
    P[4, 4] = params.sigma_p55**2
    P[5, 5] = params.sigma_p66**2

    return {
        "x": x,
        "P": np.asmatrix(P),
        "state": "initialized",
        "score": 1.0 / params.window,
    }


def update_track_score(
    track: dict[str, Any],
    associated: bool,
) -> dict[str, Any]:
    """Update track existence from a LiDAR hit or visible miss."""
    params = get_tracking_params()

    step = 1.0 / params.window
    score = float(track["score"])

    if associated:
        score = min(1.0, score + step)
    else:
        score -= step

    track["score"] = score

    # Once confirmed, a track stays confirmed until deleted.
    if track["state"] == "confirmed":
        return track

    if score > params.confirmed_threshold:
        track["state"] = "confirmed"
    elif associated:
        track["state"] = "tentative"

    return track


def should_delete_track(track: dict[str, Any]) -> bool:
    """Decide whether a track should be removed on the LiDAR pass."""
    params = get_tracking_params()

    P = np.asarray(track["P"], dtype=float)
    score = float(track["score"])
    state = track["state"]

    # Delete if horizontal position uncertainty is too high.
    if P[0, 0] > params.max_P or P[1, 1] > params.max_P:
        return True

    if state == "confirmed":
        return score < params.delete_threshold

    # Unconfirmed tracks expire when their score reaches zero.
    return score <= 0.0
