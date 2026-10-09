"""SETTLE rotation regression and numerical comparison with OpenMM Reference.

These checks establish numerical agreement, not literal byte identity.
"""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jax_md import space
from jax_md.mm_forcefields.amber.constraints import (
    SettleData, settle_apply_positions, settle_apply_velocities,
)

try:
    import openmm
    from openmm import unit
except ImportError:
    openmm = None


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

    @unittest.skipIf(openmm is None, 'OpenMM is required for the reference check')
    def test_matches_openmm_reference_with_matching_lengths(self):
        # Units are nm and nm/ps on both sides, avoiding unit conversion and
        # periodic wrapping as possible sources of floating-point differences.
        with jax.enable_x64():
            rng = np.random.default_rng(42)
            n = 256
            oh = .1
            angle = 1.91061193216
            hh = 2*oh*np.sin(angle/2)
            water = np.array([[0., 0., 0.], [oh, 0., 0.],
                              [oh*np.cos(angle), oh*np.sin(angle), 0.]])
            original = np.array([
                water @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
                + rng.uniform(-3, 3, 3) for _ in range(n)
            ]).reshape(-1, 3)
            predicted = original + rng.normal(scale=.0001, size=original.shape)
            velocity = rng.normal(size=original.shape)
            masses = np.tile([15.99943, 1.007947, 1.007947], n)
            system = openmm.System()
            for mass in masses:
                system.addParticle(float(mass))
            for oxygen in range(0, len(original), 3):
                system.addConstraint(oxygen, oxygen+1, oh)
                system.addConstraint(oxygen, oxygen+2, oh)
                system.addConstraint(oxygen+1, oxygen+2, hh)
            integrator = openmm.CustomIntegrator(.0005)
            integrator.addPerDofVariable('predicted', 0.)
            integrator.addComputePerDof('x', 'predicted')
            integrator.addConstrainPositions()
            context = openmm.Context(
                system, integrator, openmm.Platform.getPlatformByName('Reference'))
            context.setPositions(original)
            integrator.setPerDofVariableByName('predicted', predicted)
            integrator.step(1)
            reference_position = context.getState(positions=True).getPositions(
                asNumpy=True).value_in_unit(unit.nanometer)
            context.setPositions(original)
            context.setVelocities(velocity)
            context.applyVelocityConstraints(1e-12)
            reference_velocity = context.getState(velocities=True).getVelocities(
                asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
            del context

            # ReferenceConstraints in OpenMM 8.6.1 first stores SETTLE lengths
            # as float32 in nm, then converts them back to float64. The JAX
            # projection retains the supplied precision; match preprocessing
            # here without changing production geometry or loosening the test.
            rounded_oh, rounded_hh = np.array([oh, hh], dtype=np.float32).astype(
                np.float64)
            oxygen = jnp.arange(n)*3
            clusters = SettleData(oxygen, oxygen+1, oxygen+2,
                                  jnp.full(n, rounded_oh), jnp.full(n, rounded_hh))
            delta, shift = space.free()
            mass = jnp.asarray(masses)
            positions = jax.jit(lambda r, p: settle_apply_positions(
                r, p, clusters, mass, delta, shift))
            velocities = jax.jit(lambda r, v: settle_apply_velocities(
                r, v, clusters, mass, delta))
            for name, apply, values, reference in (
                ('positions', positions, predicted, reference_position),
                ('velocities', velocities, velocity, reference_velocity),
            ):
                with self.subTest(projection=name):
                    result = np.asarray(apply(jnp.asarray(original),
                                              jnp.asarray(values)))
                    self.assertTrue(np.isfinite(result).all())
                    np.testing.assert_allclose(result, reference, atol=1e-13,
                                               rtol=0.)


if __name__ == '__main__':
    unittest.main()
