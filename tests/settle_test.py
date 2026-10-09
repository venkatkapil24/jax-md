"""Regression for the in-plane SETTLE frame and its rotation branch."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jax_md import space
from jax_md.mm_forcefields.amber.constraints import SettleData, settle_apply_positions


class SettleTest(unittest.TestCase):
    def test_rigid_prediction_is_unchanged_across_periodic_boundary(self):
        with jax.enable_x64():
            box = 10.
            theta = 1.91061193216
            water = np.array([[0., 0., 0.], [1., 0., 0.],
                              [np.cos(theta), np.sin(theta), 0.]])
            rotation, _ = np.linalg.qr(np.random.default_rng(24).normal(size=(3, 3)))
            original = (water @ rotation + [9.999, 4.2, 2.3]) % box
            predicted = (water @ rotation + [10.004, 4.203, 2.301]) % box
            masses = jnp.array([15.99943, 1.007947, 1.007947])
            clusters = SettleData(jnp.array([0]), jnp.array([1]), jnp.array([2]),
                                  jnp.array([1.]), jnp.array([2*np.sin(theta/2)]))
            delta, shift = space.periodic(box)
            apply = jax.jit(lambda old, new: settle_apply_positions(
                old, new, clusters, masses, delta, shift))
            corrected = apply(jnp.asarray(original), jnp.asarray(predicted))
            difference = np.asarray(corrected) - predicted
            difference -= box*np.rint(difference/box)
            np.testing.assert_allclose(difference, 0., atol=1e-12, rtol=0.)


if __name__ == '__main__':
    unittest.main()
