"""Focused tests for explicit-image molecular-mechanics nonbonded terms."""

import itertools
import math

from absl.testing import absltest
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from jax_md import partition
from jax_md import test_util
from jax_md.mm_forcefields.amber import energy as amber_energy
from jax_md.mm_forcefields.base import NonbondedOptions, Topology
from jax_md.mm_forcefields.nonbonded.electrostatics import CutoffCoulomb
from jax_md.mm_forcefields.oplsaa.params import create_parameters


def _two_atom_system(with_exception=False):
  empty_float = jnp.zeros((0,), dtype=jnp.float64)
  empty_int = jnp.zeros((0,), dtype=jnp.int32)
  if with_exception:
    exception_pairs = jnp.array([[0, 1]], dtype=jnp.int32)
  else:
    exception_pairs = jnp.zeros((0, 2), dtype=jnp.int32)

  topology = Topology(
    n_atoms=2,
    bonds=jnp.zeros((0, 2), dtype=jnp.int32),
    angles=jnp.zeros((0, 3), dtype=jnp.int32),
    torsions=jnp.zeros((0, 4), dtype=jnp.int32),
    impropers=jnp.zeros((0, 4), dtype=jnp.int32),
    exclusion_mask=None,
    pair_14_mask=None,
    molecule_id=jnp.array([0, 0], dtype=jnp.int32),
    cmap_atoms=None,
    cmap_map_idx=None,
    exc_pairs=exception_pairs,
    nbfix_atom_type=None,
  )
  params = create_parameters(
    bond_k=empty_float,
    bond_r0=empty_float,
    angle_k=empty_float,
    angle_theta0=empty_float,
    torsion_k=empty_float,
    torsion_n=empty_int,
    torsion_gamma=empty_float,
    improper_k=empty_float,
    improper_n=empty_int,
    improper_gamma=empty_float,
    charges=jnp.zeros((2,), dtype=jnp.float64),
    sigma=jnp.array([1.0, 1.2], dtype=jnp.float64),
    epsilon=jnp.array([0.2, 0.4], dtype=jnp.float64),
    exc_charge_prod=jnp.zeros((len(exception_pairs),), dtype=jnp.float64),
    exc_sigma=jnp.ones((len(exception_pairs),), dtype=jnp.float64),
    exc_epsilon=jnp.zeros((len(exception_pairs),), dtype=jnp.float64),
  )
  return params, topology


def _options(
  cutoff, neighbor_format=partition.NeighborListFormat.OrderedSparse
):
  return NonbondedOptions(
    r_cut=cutoff,
    dr_threshold=0.0,
    nb_format=neighbor_format,
    use_pbc=True,
    use_periodic_general=False,
    fractional_coordinates=False,
    r_switch=None,
  )


class AmberMultiImageTest(test_util.JAXMDTestCase):
  def test_matches_minimum_image_path_when_cutoff_is_valid(self):
    params, topology = _two_atom_system()
    positions = jnp.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=jnp.float64)
    box = jnp.array([8.0, 8.0, 8.0], dtype=jnp.float64)
    options = _options(2.5)
    energies = []
    forces = []

    for multi_image in (False, True):
      energy_fn, neighbor_fn, _, _ = amber_energy(
        params,
        topology,
        box,
        coulomb_options=CutoffCoulomb(2.5),
        nb_options=options,
        multi_image=multi_image,
      )
      neighbors = neighbor_fn.allocate(positions)
      total_fn = lambda pos: energy_fn(pos, neighbors)['etotal']
      energies.append(total_fn(positions))
      forces.append(-jax.grad(total_fn)(positions))

    self.assertAllClose(energies[0], energies[1], rtol=1e-12, atol=1e-12)
    self.assertAllClose(forces[0], forces[1], rtol=1e-12, atol=1e-12)

  def test_exception_removes_only_the_central_image(self):
    params, topology = _two_atom_system(with_exception=True)
    positions = jnp.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=jnp.float64)
    box_length = 2.0
    cutoff = 2.5
    box = jnp.array([box_length] * 3, dtype=jnp.float64)
    positions_np = np.asarray(positions)
    sigma = np.asarray(params.nonbonded.sigma)
    epsilon = np.asarray(params.nonbonded.epsilon)
    expected = 0.0
    for i, j in itertools.product(range(2), repeat=2):
      for shift in itertools.product(range(-2, 3), repeat=3):
        if i == j and shift == (0, 0, 0):
          continue
        if i != j and shift == (0, 0, 0):
          continue
        delta = (
          positions_np[j] + box_length * np.asarray(shift) - positions_np[i]
        )
        distance = float(np.linalg.norm(delta))
        if distance >= cutoff:
          continue
        sigma_ij = 0.5 * (sigma[i] + sigma[j])
        epsilon_ij = math.sqrt(epsilon[i] * epsilon[j])
        ratio6 = (sigma_ij / distance) ** 6
        expected += 4.0 * epsilon_ij * (ratio6 * ratio6 - ratio6)
    expected *= 0.5

    for neighbor_format in (
      partition.NeighborListFormat.Dense,
      partition.NeighborListFormat.Sparse,
      partition.NeighborListFormat.OrderedSparse,
    ):
      with self.subTest(neighbor_format=neighbor_format):
        energy_fn, neighbor_fn, _, _ = amber_energy(
          params,
          topology,
          box,
          coulomb_options=CutoffCoulomb(cutoff),
          nb_options=_options(cutoff, neighbor_format),
          multi_image=True,
        )
        neighbors = neighbor_fn.allocate(positions)
        actual = energy_fn(positions, neighbors)['lj_pot']

        self.assertAllClose(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertNotEqual(float(actual), 0.0)


if __name__ == '__main__':
  absltest.main()
