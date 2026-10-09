"""Part F: Mahalanobis gating and greedy sensor association."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.stats import chi2

from fusion_lab.workspace_support import get_tracking_params
from fusion_lab.workspace_loader import load_workspace_module


def mahalanobis_distance(track: Any, meas: Any) -> float:
    """Compute squared Mahalanobis distance for a track and measurement."""
    kalman = load_workspace_module("kalman")

    H = meas.sensor.get_H(track.x)
    gamma = kalman.innovation(track.x, meas)
    S = kalman.innovation_covariance(track.P, meas, H)

    # Solve S * y = gamma instead of explicitly inverting S.
    solved = np.linalg.solve(
        np.asarray(S, dtype=float),
        np.asarray(gamma, dtype=float),
    )

    d_squared = np.asarray(gamma).T @ solved
    return float(d_squared.item())


def chi2_gate(mhd_sq: float, sensor: Any) -> bool:
    """Accept measurements within the chi-square confidence gate."""
    params = get_tracking_params()

    threshold = chi2.ppf(
        params.gating_threshold,
        df=sensor.dim_meas,
    )

    return bool(np.isfinite(mhd_sq) and 0.0 <= mhd_sq <= threshold)


def association_cost_matrix(
    track_list: Sequence[Any],
    meas_list: Sequence[Any],
) -> np.matrix:
    """Build the gated Mahalanobis cost matrix."""
    n_tracks = len(track_list)
    n_meas = len(meas_list)

    costs = np.full(
        (n_tracks, n_meas),
        np.inf,
        dtype=float,
    )

    for i, track in enumerate(track_list):
        for j, meas in enumerate(meas_list):
            sensor = meas.sensor

            # Reject invisible pairs BEFORE projection or distance.
            if not sensor.in_fov(track.x):
                continue

            distance = mahalanobis_distance(track, meas)

            if chi2_gate(distance, sensor):
                costs[i, j] = distance

    return np.asmatrix(costs)


def pick_next_pair(
    association_matrix: np.matrix,
    unassigned_tracks: Sequence[Any],
    unassigned_meas: Sequence[Any],
) -> tuple[Any, Any, np.matrix, list[Any], list[Any]]:
    """Pick minimum finite cost and remove its row and column."""
    costs = np.asarray(association_matrix, dtype=float)

    tracks = list(unassigned_tracks)
    measurements = list(unassigned_meas)

    # Nothing can be matched.
    if costs.size == 0 or not np.isfinite(costs).any():
        return (
            np.nan,
            np.nan,
            np.asmatrix(costs),
            tracks,
            measurements,
        )

    valid_costs = np.where(np.isfinite(costs), costs, np.inf)

    i, j = np.unravel_index(
        np.argmin(valid_costs),
        valid_costs.shape,
    )

    track = tracks.pop(i)
    meas = measurements.pop(j)

    # Remove the assigned track row and measurement column.
    remaining_costs = np.delete(costs, i, axis=0)
    remaining_costs = np.delete(remaining_costs, j, axis=1)

    return (
        track,
        meas,
        np.asmatrix(remaining_costs),
        tracks,
        measurements,
    )


def associate_and_update(
    manager: Any,
    meas_list: Sequence[Any],
    filter_obj: Any,
    sensor: Any,
) -> None:
    """Match measurements, update EKF, and manage unmatched objects."""
    unassigned_tracks = list(manager.track_list)
    unassigned_meas = list(meas_list)

    costs = association_cost_matrix(
        unassigned_tracks,
        unassigned_meas,
    )

    while unassigned_tracks and unassigned_meas:
        if not np.isfinite(np.asarray(costs)).any():
            break

        (
            track,
            meas,
            costs,
            unassigned_tracks,
            unassigned_meas,
        ) = pick_next_pair(
            costs,
            unassigned_tracks,
            unassigned_meas,
        )

        # Visibility was already validated in the cost matrix.
        filter_obj.update(track, meas)
        manager.handle_updated_track(track, sensor)

    # Always run track management, even on empty sensor passes.
    manager.manage_tracks(
        unassigned_tracks,
        unassigned_meas,
        sensor,
    )
