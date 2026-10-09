"""Part E: 6D EKF with constant-velocity motion model."""

from __future__ import annotations

from typing import Any, Optional
import numpy as np

from fusion_lab.workspace_support import get_tracking_params

Matrix = np.matrix | np.ndarray


def build_F(dt: Optional[float] = None) -> Matrix:
    """Build 6x6 constant-velocity transition matrix."""
    if dt is None:
        dt = get_tracking_params().dt

    F = np.asmatrix(np.eye(6))
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return F


def build_Q(
    dt: Optional[float] = None,
    q: Optional[float] = None,
) -> Matrix:
    """Build the lab's diagonal process-noise covariance."""
    params = get_tracking_params()

    if dt is None:
        dt = params.dt
    if q is None:
        q = params.q

    return np.asmatrix(np.eye(6) * dt * q)


def ekf_predict(
    x: Matrix,
    P: Matrix,
    F: Optional[Matrix] = None,
    Q: Optional[Matrix] = None,
) -> tuple[Matrix, Matrix]:
    """Predict state and covariance."""
    if F is None:
        F = build_F()
    if Q is None:
        Q = build_Q()

    x_pred = F @ x
    P_pred = F @ P @ F.T + Q

    return x_pred, P_pred


def innovation(x: Matrix, meas: Any) -> Matrix:
    """Measurement residual: z - h(x)."""
    return meas.z - meas.sensor.get_hx(x)


def innovation_covariance(
    P: Matrix,
    meas: Any,
    H: Matrix,
) -> Matrix:
    """Innovation covariance: S = H P H.T + R."""
    return H @ P @ H.T + meas.R


def ekf_update(
    x: Matrix,
    P: Matrix,
    meas: Any,
) -> tuple[Matrix, Matrix]:
    """Correct state and covariance using a measurement."""
    H = meas.sensor.get_H(x)

    gamma = innovation(x, meas)
    S = innovation_covariance(P, meas, H)

    K = P @ H.T @ np.linalg.inv(S)

    x_upd = x + K @ gamma

    I = np.asmatrix(np.eye(6))
    P_upd = (I - K @ H) @ P

    return x_upd, P_upd
