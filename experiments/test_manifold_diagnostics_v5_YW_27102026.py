"""Small regression tests; run directly with Python (no pytest dependency)."""
import unittest
import numpy as np
from scipy.linalg import expm
import manifold_diagnostics_v5_YW_27102026 as d


class ManifoldDiagnosticsTests(unittest.TestCase):
    def test_algebra_norms(self):
        v = np.array([0.6, -0.4, 0.7])
        U = d.core.hat(v)
        self.assertAlmostEqual(np.linalg.norm(U, 'fro'), np.sqrt(2)*np.linalg.norm(v))
        self.assertAlmostEqual(np.linalg.norm(U, 2), np.linalg.norm(v))
        w = np.array([-0.1, 0.4, 0.8])
        np.testing.assert_allclose(U@d.core.hat(w)-d.core.hat(w)@U,
                                   d.core.hat(np.cross(v, w)), atol=1e-15)

    def test_closed_form_exponential(self):
        rng = np.random.default_rng(301)
        for scale in [1e-9, 1e-4, 0.5]:
            v = scale*rng.normal(size=3)
            R = np.column_stack([d.core.rotate(v, e) for e in np.eye(3)])
            np.testing.assert_allclose(R, expm(d.core.hat(v)), atol=1e-14, rtol=1e-14)

    def test_cayley_group_and_local_order(self):
        A = d.core.hat(np.array([0.6, -0.4, 0.7]))
        errors = []
        for h in [0.1, 0.05, 0.025]:
            U = h*A
            C = np.linalg.solve(np.eye(3)-U/2, np.eye(3)+U/2)
            np.testing.assert_allclose(C.T@C, np.eye(3), atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(C), 1.)
            errors.append(np.linalg.norm(C-expm(U)))
        self.assertTrue(all(7.9 < errors[i]/errors[i+1] < 8.1 for i in (0, 1)))

    def test_recording_matches_v4(self):
        self.assertLess(d.verify_recording(), 1e-12)

    def test_preservation_without_projection(self):
        nodes = np.random.default_rng(302).random((4, 16))
        y = d.sphere_paths(nodes, 0.5, 5., 4.)
        radial = abs(np.linalg.norm(y, axis=-1)-1)
        self.assertLess(radial[:, :, :5].max(), 1e-12)
        self.assertGreater(radial[:, :, 5].max(), 1e-6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
