import unittest

import numpy as np

from tests import _bootstrap  # noqa: F401
from depth_collector.geometry import EquirectangularCameraModel, PinholeCameraModel
from depth_collector.geometry import generate_equirectangular_rays, generate_pinhole_rays


class GeometryTest(unittest.TestCase):
    def test_pinhole_rays_are_normalized(self) -> None:
        camera = PinholeCameraModel(width=8, height=6, fx=4.0, fy=4.0, cx=4.0, cy=3.0)
        rays = generate_pinhole_rays(camera)
        norms = np.linalg.norm(rays, axis=-1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-6))

    def test_equirectangular_rays_are_normalized(self) -> None:
        camera = EquirectangularCameraModel(width=16, height=8)
        rays = generate_equirectangular_rays(camera)
        norms = np.linalg.norm(rays, axis=-1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
