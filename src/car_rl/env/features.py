from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import numpy as np


class PolicyInputAdapter:
    """Convert env observations into fixed-size float32 vectors for policy networks."""

    def __init__(
        self,
        observation_mode: str,
        boundary_ray_key: str = "ray_distances_norm",
        boundary_num_rays: Optional[int] = None,
        dtype: Any = np.float32,
    ) -> None:
        if observation_mode not in ("state", "boundary"):
            raise ValueError("observation_mode must be 'state' or 'boundary'")
        self.observation_mode = observation_mode
        self.boundary_ray_key = boundary_ray_key
        self.dtype = dtype
        self._boundary_num_rays = boundary_num_rays

    def transform(self, obs: Mapping[str, Any]) -> np.ndarray:
        if self.observation_mode == "state":
            return self._transform_state(obs)
        return self._transform_boundary(obs)

    def transform_batch(self, observations: Iterable[Mapping[str, Any]]) -> np.ndarray:
        rows = [self.transform(obs) for obs in observations]
        if not rows:
            return np.zeros((0, self.feature_dim), dtype=self.dtype)
        return np.stack(rows, axis=0)

    @property
    def feature_dim(self) -> int:
        if self.observation_mode == "state":
            return 6
        if self._boundary_num_rays is None:
            raise ValueError("boundary_num_rays is unknown; call transform once or set it in constructor")
        return 2 + self._boundary_num_rays

    def feature_names(self) -> list[str]:
        if self.observation_mode == "state":
            return ["x", "y", "sin_yaw", "cos_yaw", "v", "delta"]

        if self._boundary_num_rays is None:
            raise ValueError("boundary_num_rays is unknown; call transform once or set it in constructor")

        names = ["v_norm", "delta_norm"]
        for i in range(self._boundary_num_rays):
            names.append(f"ray_{i:02d}_norm")
        return names

    def _transform_state(self, obs: Mapping[str, Any]) -> np.ndarray:
        yaw = float(obs["yaw"])
        return np.asarray(
            [
                float(obs["x"]),
                float(obs["y"]),
                np.sin(yaw),
                np.cos(yaw),
                float(obs["v"]),
                float(obs["delta"]),
            ],
            dtype=self.dtype,
        )

    def _transform_boundary(self, obs: Mapping[str, Any]) -> np.ndarray:
        rays_raw = obs[self.boundary_ray_key]
        rays = [float(v) for v in rays_raw]

        if self._boundary_num_rays is None:
            self._boundary_num_rays = len(rays)
        if len(rays) != self._boundary_num_rays:
            raise ValueError(
                f"expected {self._boundary_num_rays} rays in '{self.boundary_ray_key}', got {len(rays)}"
            )

        vec = np.empty((2 + self._boundary_num_rays,), dtype=self.dtype)
        vec[0] = float(obs["v_norm"])
        vec[1] = float(obs["delta_norm"])
        vec[2:] = np.asarray(rays, dtype=self.dtype)
        return vec
