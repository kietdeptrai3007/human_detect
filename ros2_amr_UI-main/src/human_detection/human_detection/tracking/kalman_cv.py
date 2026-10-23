"""Bộ lọc Kalman vận tốc không đổi (constant velocity) cho một điểm 2D. Chỉ dùng numpy."""

from __future__ import annotations

import numpy as np


class ConstantVelocityKalman:
    """
    Trạng thái [x, y, vx, vy]. Mô hình nhiễu: gia tốc ngẫu nhiên có độ lệch chuẩn accel_std
    (đơn vị / s^2), nên người đổi hướng đột ngột chỉ làm tăng độ bất định chứ không làm lệch bộ lọc.
    """

    def __init__(self, x: float, y: float, pos_std: float, vel_std: float, accel_std: float):
        self.x = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([pos_std ** 2, pos_std ** 2, vel_std ** 2, vel_std ** 2]).astype(np.float64)
        self.accel_std = float(accel_std)

    @staticmethod
    def _F(dt: float) -> np.ndarray:
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        q = self.accel_std ** 2
        a, b, c = dt ** 4 / 4, dt ** 3 / 2, dt ** 2
        Q = np.zeros((4, 4))
        Q[0, 0] = Q[1, 1] = a * q
        Q[0, 2] = Q[2, 0] = Q[1, 3] = Q[3, 1] = b * q
        Q[2, 2] = Q[3, 3] = c * q
        return Q

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

    def update(self, zx: float, zy: float, std: float) -> None:
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        R = np.eye(2) * std ** 2
        y = np.array([zx, zy]) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T          # dạng Joseph, giữ P đối xứng

    def pos_sigma(self) -> float:
        return float(np.sqrt((self.P[0, 0] + self.P[1, 1]) / 2))

    def predict_ahead(self, dt: float):
        """Trạng thái và sigma vị trí sau dt giây, KHÔNG làm thay đổi bộ lọc."""
        F = self._F(dt)
        x = F @ self.x
        P = F @ self.P @ F.T + self._Q(dt)
        return x, float(np.sqrt((P[0, 0] + P[1, 1]) / 2))
