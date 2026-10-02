# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for google3.third_party.py.jax_md.simulate."""

import functools
import io
from typing import Any

import numpy as onp

from absl.testing import absltest
from absl.testing import parameterized

from jax import jit
from jax import vmap
from jax import random
from jax import lax

import jax
import jax.numpy as np

from jax_md import quantity
from jax_md import simulate
from jax_md import space
from jax_md import energy
from jax_md import util
from jax_md import test_util
from jax_md import partition
from jax_md import smap
from jax_md import dataclasses
from jax_md import replica_exchange
from jax_md.util import *

from functools import partial

jax.config.parse_flags_with_absl()


PARTICLE_COUNT = 1000
DYNAMICS_STEPS = 800
SHORT_DYNAMICS_STEPS = 20
STOCHASTIC_SAMPLES = 5
SPATIAL_DIMENSION = [2, 3]
COORDS = ['fractional', 'real']

LANGEVIN_PARTICLE_COUNT = 8000
LANGEVIN_DYNAMICS_STEPS = 8000

BROWNIAN_PARTICLE_COUNT = 24000
BROWNIAN_DYNAMICS_STEPS = 8000

DTYPE = [f32]
if util.x64_enabled():
  DTYPE += [f64]


ke_fn = lambda p, m: quantity.kinetic_energy(momentum=p, mass=m)
kT_fn = lambda p, m: quantity.temperature(momentum=p, mass=m)


# pylint: disable=invalid-name
class SimulateTest(test_util.JAXMDTestCase):
  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
        'spatial_dimension': dim,
        'dtype': dtype,
      }
      for dim in SPATIAL_DIMENSION
      for dtype in DTYPE
    )
  )
  def test_nve_ensemble(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)
    pos_key, center_key, vel_key, mass_key = random.split(key, 4)
    R = random.normal(pos_key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype)
    R0 = random.normal(
      center_key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype
    )
    mass = random.uniform(
      mass_key, (PARTICLE_COUNT,), minval=0.1, maxval=5.0, dtype=dtype
    )
    _, shift = space.free()

    E = lambda R, **kwargs: np.sum((R - R0) ** 2)

    init_fn, apply_fn = simulate.nve(E, shift, 1e-3)
    apply_fn = jit(apply_fn)

    state = init_fn(vel_key, R, kT=0.5, mass=mass)

    E_T = lambda state: E(state.position) + ke_fn(state.momentum, state.mass)
    E_initial = E_T(state)

    for _ in range(DYNAMICS_STEPS):
      state = apply_fn(state)
      E_total = E_T(state)
      assert np.abs(E_total - E_initial) < E_initial * 0.01
      assert state.position.dtype == dtype

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
        'spatial_dimension': dim,
        'dtype': dtype,
      }
      for dim in SPATIAL_DIMENSION
      for dtype in DTYPE
    )
  )
  def test_nve_jammed(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    state = test_util.load_jammed_state('simulation_test_state.npy', dtype)
    displacement_fn, shift_fn = space.periodic(state.box[0, 0])

    E = energy.soft_sphere_pair(displacement_fn, state.species, state.sigma)

    init_fn, apply_fn = simulate.nve(E, shift_fn, 1e-3)
    apply_fn = jit(apply_fn)

    state = init_fn(key, state.real_position, kT=1e-3)

    E_T = lambda state: E(state.position) + ke_fn(state.momentum, state.mass)
    E_initial = E_T(state) * np.ones((DYNAMICS_STEPS,))

    def step_fn(i, state_and_energy):
      state, energy = state_and_energy
      state = apply_fn(state)
      energy = energy.at[i].set(E_T(state))
      return state, energy

    Es = np.zeros((DYNAMICS_STEPS,))
    state, Es = lax.fori_loop(0, DYNAMICS_STEPS, step_fn, (state, Es))

    tol = 1e-3 if dtype is f32 else 1e-7
    self.assertEqual(state.position.dtype, dtype)
    self.assertAllClose(Es, E_initial, rtol=tol, atol=tol)

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': f'_dtype={dtype.__name__}_coordinates={coords}',
        'dtype': dtype,
        'coords': coords,
      }
      for dtype in DTYPE
      for coords in COORDS
    )
  )
  def test_nve_jammed_periodic_general(self, dtype, coords):
    key = random.PRNGKey(0)

    state = test_util.load_jammed_state('simulation_test_state.npy', dtype)
    displacement_fn, shift_fn = space.periodic_general(
      state.box, coords == 'fractional'
    )

    E = energy.soft_sphere_pair(displacement_fn, state.species, state.sigma)

    init_fn, apply_fn = simulate.nve(E, shift_fn, 1e-3)
    apply_fn = jit(apply_fn)

    state = init_fn(key, getattr(state, coords + '_position'), kT=1e-3)

    E_T = lambda state: E(state.position) + ke_fn(state.momentum, state.mass)
    E_initial = E_T(state) * np.ones((DYNAMICS_STEPS,))

    def step_fn(i, state_and_energy):
      state, energy = state_and_energy
      state = apply_fn(state)
      energy = energy.at[i].set(E_T(state))
      return state, energy

    Es = np.zeros((DYNAMICS_STEPS,))
    state, Es = lax.fori_loop(0, DYNAMICS_STEPS, step_fn, (state, Es))

    tol = 1e-3 if dtype is f32 else 1e-7
    self.assertEqual(state.position.dtype, dtype)
    self.assertAllClose(Es, E_initial, rtol=tol, atol=tol)

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
        'spatial_dimension': dim,
        'dtype': dtype,
      }
      for dim in SPATIAL_DIMENSION
      for dtype in DTYPE
    )
  )
  def test_nve_neighbor_list(self, spatial_dimension, dtype):
    Nx = particles_per_side = 8
    spacing = f32(1.25)

    tol = 5e-12 if dtype == np.float64 else 5e-3

    L = Nx * spacing
    if spatial_dimension == 2:
      R = np.stack([np.array(r) for r in onp.ndindex(Nx, Nx)]) * spacing
    elif spatial_dimension == 3:
      R = np.stack([np.array(r) for r in onp.ndindex(Nx, Nx, Nx)]) * spacing

    R = np.array(R, dtype)

    displacement, shift = space.periodic(L)

    neighbor_fn, energy_fn = energy.lennard_jones_neighbor_list(displacement, L)
    exact_energy_fn = energy.lennard_jones_pair(displacement)

    init_fn, apply_fn = simulate.nve(energy_fn, shift, 1e-3)
    exact_init_fn, exact_apply_fn = simulate.nve(exact_energy_fn, shift, 1e-3)

    nbrs = neighbor_fn(R)
    state = init_fn(random.PRNGKey(0), R, kT=0.5, neighbor=nbrs)
    exact_state = exact_init_fn(random.PRNGKey(0), R, kT=0.5)

    def body_fn(i, state):
      state, nbrs, exact_state = state
      nbrs = neighbor_fn(state.position, nbrs)
      state = apply_fn(state, neighbor=nbrs)
      return state, nbrs, exact_apply_fn(exact_state)

    step = 0
    for i in range(20):
      new_state, nbrs, new_exact_state = lax.fori_loop(
        0, 100, body_fn, (state, nbrs, exact_state)
      )
      if nbrs.did_buffer_overflow:
        nbrs = neighbor_fn(state.position)
      else:
        state = new_state
        exact_state = new_exact_state
        step += 1
    assert state.position.dtype == dtype
    self.assertAllClose(
      state.position, exact_state.position, atol=tol, rtol=tol
    )

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': f'_dim={dim}_dtype={dtype.__name__}_sy_steps={sy_steps}',
        'spatial_dimension': dim,
        'dtype': dtype,
        'sy_steps': sy_steps,
      }
      for dim in SPATIAL_DIMENSION
      for dtype in DTYPE
      for sy_steps in [1, 3, 5, 7]
    )
  )
  def test_nvt_nose_hoover(self, spatial_dimension, dtype, sy_steps):
    key = random.PRNGKey(0)

    box_size = quantity.box_size_at_number_density(
      PARTICLE_COUNT, f32(1.2), spatial_dimension
    )
    displacement_fn, shift_fn = space.periodic(box_size)

    bonds_i = np.arange(PARTICLE_COUNT)
    bonds_j = np.roll(bonds_i, 1)
    bonds = np.stack([bonds_i, bonds_j])

    E = energy.simple_spring_bond(displacement_fn, bonds)

    invariant = partial(simulate.nvt_nose_hoover_invariant, E)

    for _ in range(STOCHASTIC_SAMPLES):
      key, pos_key, vel_key, T_key, masses_key = random.split(key, 5)

      R = box_size * random.uniform(
        pos_key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype
      )
      T = random.uniform(T_key, (), minval=0.3, maxval=1.4, dtype=dtype)
      mass = 1 + random.uniform(masses_key, (PARTICLE_COUNT,), dtype=dtype)
      init_fn, apply_fn = simulate.nvt_nose_hoover(
        E, shift_fn, 1e-3, T, sy_steps=sy_steps
      )
      apply_fn = jit(apply_fn)

      state = init_fn(vel_key, R, mass=mass)

      initial = invariant(state, T)

      for _ in range(DYNAMICS_STEPS):
        state = apply_fn(state)

      T_final = kT_fn(state.momentum, state.mass)
      assert np.abs(T_final - T) / T < 0.1
      tol = 5e-4 if dtype is f32 else 1e-6
      self.assertAllClose(invariant(state, T), initial, rtol=tol)
      self.assertEqual(state.position.dtype, dtype)

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': f'dtype={dtype.__name__}_sy_steps={sy_steps}',
        'dtype': dtype,
        'sy_steps': sy_steps,
      }
      for dtype in DTYPE
      for sy_steps in [1, 3, 5, 7]
    )
  )
  def test_nvt_nose_hoover_jammed(self, dtype, sy_steps):
    key = random.PRNGKey(0)

    state = test_util.load_jammed_state('simulation_test_state.npy', dtype)
    displacement_fn, shift_fn = space.periodic(state.box[0, 0])

    E = energy.soft_sphere_pair(displacement_fn, state.species, state.sigma)
    invariant = partial(simulate.nvt_nose_hoover_invariant, E)

    kT = 1e-3
    init_fn, apply_fn = simulate.nvt_nose_hoover(
      E, shift_fn, 1e-3, kT=kT, sy_steps=sy_steps
    )
    apply_fn = jit(apply_fn)

    state = init_fn(key, state.real_position)

    E_initial = invariant(state, kT) * np.ones((DYNAMICS_STEPS,))

    def step_fn(i, state_and_energy):
      state, energy = state_and_energy
      state = apply_fn(state)
      energy = energy.at[i].set(invariant(state, kT))
      return state, energy

    Es = np.zeros((DYNAMICS_STEPS,))
    state, Es = lax.fori_loop(0, DYNAMICS_STEPS, step_fn, (state, Es))

    tol = 1e-3 if dtype is f32 else 1e-7
    self.assertEqual(state.position.dtype, dtype)
    self.assertAllClose(Es, E_initial, rtol=tol, atol=tol)

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': f'dtype={dtype.__name__}_sy_steps={sy_steps}',
        'dtype': dtype,
        'sy_steps': sy_steps,
      }
      for dtype in DTYPE
      for sy_steps in [1, 3, 5, 7]
    )
  )
  def test_npt_nose_hoover_jammed(self, dtype, sy_steps):
    key = random.PRNGKey(0)

    state = test_util.load_jammed_state('simulation_test_state.npy', dtype)
    displacement_fn, shift_fn = space.periodic_general(state.box)

    E = energy.soft_sphere_pair(displacement_fn, state.species, state.sigma)
    invariant = partial(simulate.npt_nose_hoover_invariant, E)
    pressure_fn = partial(quantity.pressure, E)

    nhc_kwargs = {sy_steps: sy_steps}
    kT = 1e-3
    P = state.pressure
    init_fn, apply_fn = simulate.npt_nose_hoover(
      E, shift_fn, 1e-3, P, kT, nhc_kwargs, nhc_kwargs
    )
    apply_fn = jit(apply_fn)

    state = init_fn(key, state.fractional_position, state.box)

    E_initial = invariant(state, P, kT) * np.ones((DYNAMICS_STEPS,))
    P_target = P * np.ones((DYNAMICS_STEPS,))

    def step_fn(i, state_energy_pressure):
      state, energy, pressure = state_energy_pressure
      state = apply_fn(state)
      energy = energy.at[i].set(invariant(state, P, kT))
      box = simulate.npt_box(state)
      KE = ke_fn(state.momentum, state.mass)
      p = pressure_fn(state.position, box, KE)
      pressure = pressure.at[i].set(p)
      return state, energy, pressure

    Es = np.zeros((DYNAMICS_STEPS,))
    Ps = np.zeros((DYNAMICS_STEPS,))
    state, Es, Ps = lax.fori_loop(0, DYNAMICS_STEPS, step_fn, (state, Es, Ps))

    tol = 1e-3 if dtype is f32 else 1.2e-4
    self.assertEqual(state.position.dtype, dtype)
    self.assertAllClose(Es, E_initial, rtol=tol, atol=tol)
    self.assertAllClose(Ps, P_target, rtol=0.05, atol=0.05)

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': f'_dtype={dtype.__name__}',
        'dtype': dtype,
      }
      for dtype in DTYPE
    )
  )
  def test_npt_csvr(self, dtype):
    particle_count = 64
    box = dtype((particle_count / 0.7) ** (1 / 3))
    grid = np.stack(
      np.meshgrid(*[np.arange(4, dtype=dtype)] * 3, indexing='ij'), axis=-1
    )
    position = (grid.reshape(-1, 3) + dtype(0.5)) / dtype(4)
    displacement, shift = space.periodic_general(box)
    energy_fn = energy.lennard_jones_pair(
      displacement,
      sigma=dtype(1.0),
      epsilon=dtype(1.0),
      r_onset=dtype(1.8),
      r_cutoff=dtype(2.0),
    )
    init_fn, apply_fn = simulate.npt_csvr(
      energy_fn,
      shift,
      dt=dtype(2e-3),
      pressure=dtype(0.4),
      kT=dtype(1.0),
      tau_p=dtype(1.0),
      tau_t=dtype(0.1),
    )
    apply_fn = jit(apply_fn)
    state = init_fn(random.PRNGKey(0), position, box=box)
    self.assertFalse(np.allclose(np.sum(state.momentum, axis=0), 0))
    initial_box = state.box
    initial_conserved = state.conserved_quantity

    def step_fn(i, state_and_conserved):
      state, conserved = state_and_conserved
      state = apply_fn(state)
      return state, conserved.at[i].set(state.conserved_quantity)

    conserved = np.zeros((SHORT_DYNAMICS_STEPS,), dtype=dtype)
    state, conserved = lax.fori_loop(
      0, SHORT_DYNAMICS_STEPS, step_fn, (state, conserved)
    )

    kinetic = quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)
    volume = quantity.volume(3, state.box)
    pressure = (2 * kinetic - state.dUdV) / (3 * volume)
    expected_pressure = quantity.pressure(
      energy_fn, state.position, state.box, kinetic
    )
    expected_force = quantity.force(energy_fn)(state.position, box=state.box)
    expected_energy = energy.lennard_jones_pair(
      space.periodic_general(state.box)[0],
      sigma=dtype(1.0),
      epsilon=dtype(1.0),
      r_onset=dtype(1.8),
      r_cutoff=dtype(2.0),
    )(state.position)
    tol = 2e-4 if dtype is f32 else 1e-10

    self.assertEqual(state.position.dtype, dtype)
    self.assertEqual(state.box.dtype, dtype)
    self.assertFalse(np.allclose(state.box, initial_box))
    self.assertAllClose(
      state.potential_energy,
      expected_energy,
      rtol=tol,
      atol=tol,
    )
    self.assertAllClose(state.force, expected_force, rtol=tol, atol=tol)
    self.assertAllClose(pressure, expected_pressure, rtol=tol, atol=tol)
    self.assertAllClose(
      conserved,
      np.full_like(conserved, initial_conserved),
      rtol=2e-4,
      atol=2e-4,
    )
    for value in (
      state.position,
      state.momentum,
      state.force,
      state.potential_energy,
      state.dUdV,
      state.box,
      pressure,
    ):
      self.assertTrue(np.all(np.isfinite(value)))

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': f'_dtype={dtype.__name__}',
        'dtype': dtype,
      }
      for dtype in DTYPE
    )
  )
  def test_npt_csvr_initialization(self, dtype):
    particle_count = 8
    position = (
      np.stack(
        np.meshgrid(*[np.arange(2, dtype=dtype)] * 3, indexing='ij'),
        axis=-1,
      ).reshape(-1, 3)
      + dtype(0.5)
    ) / dtype(2)
    mass = np.arange(1, particle_count + 1, dtype=dtype)
    momenta = np.arange(1, position.size + 1, dtype=dtype).reshape(
      position.shape
    ) / dtype(10)
    temperature = dtype(1.2)
    tau_p = dtype(0.7)
    box_velocity = dtype(0.2)

    for box in (dtype(5.0), np.eye(3, dtype=dtype) * dtype(5.0)):
      displacement, shift = space.periodic_general(box)
      energy_fn = energy.lennard_jones_pair(displacement)
      init_fn, _ = simulate.npt_csvr(
        energy_fn,
        shift,
        dt=dtype(1e-3),
        pressure=dtype(0.4),
        kT=temperature,
        tau_p=tau_p,
        tau_t=dtype(0.1),
      )
      state = init_fn(
        random.PRNGKey(0),
        position,
        box=box,
        mass=mass,
        momenta=momenta,
        box_velocity=box_velocity,
      )
      expected_box = np.eye(3, dtype=dtype) * box if np.ndim(box) == 0 else box
      expected_box_mass = dtype(3 * particle_count) * temperature * tau_p**2

      self.assertAllClose(state.momentum, momenta)
      self.assertAllClose(state.velocity, momenta / mass[:, None])
      self.assertAllClose(state.box, expected_box)
      self.assertAllClose(state.box_mass, expected_box_mass)
      self.assertAllClose(state.box_momentum, expected_box_mass * box_velocity)
      self.assertFalse(np.allclose(np.sum(state.momentum, axis=0), 0))

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': f'_dtype={dtype.__name__}',
        'dtype': dtype,
      }
      for dtype in DTYPE
    )
  )
  def test_npt_csvr_requires_three_dimensions(self, dtype):
    box = np.eye(2, dtype=dtype) * dtype(5.0)
    _, shift = space.periodic_general(box)
    energy_fn = lambda position, **kwargs: np.sum(position**2)
    init_fn, _ = simulate.npt_csvr(
      energy_fn,
      shift,
      dt=dtype(1e-3),
      pressure=dtype(0.4),
      kT=dtype(1.0),
      tau_p=dtype(1.0),
      tau_t=dtype(0.1),
    )

    with self.assertRaisesRegex(ValueError, 'only three dimensions'):
      init_fn(random.PRNGKey(0), np.zeros((4, 2), dtype=dtype), box=box)

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': f'dtype={dtype.__name__}',
        'dtype': dtype,
      }
      for dtype in DTYPE
    )
  )
  def test_npt_nose_hoover_lammps(self, dtype):
    key = random.PRNGKey(0)

    box, pos, vel = test_util.load_lammps_npt_test_case(dtype)

    displacement, shift = space.periodic_general(box)
    dist_fun = space.metric(displacement)
    neighbor_fn, energy_fn = energy.stillinger_weber_neighbor_list(
      displacement, box
    )

    units = {
      'mass': 1,
      'distance': 1,
      'time': 98.22694788,
      'energy': 1,
      'velocity': 0.01018051,
      'force': 1.0,
      'torque ': 1,
      'temperature': 8.617330337217213e-05,
      'pressure': 6.241509125883258e-07,
    }

    fs = 1e-3 * units['time']
    ps = units['time']

    dt = fs
    write_every = 100
    T_init = 300 * units['temperature']
    P_init = 0.0 * units['pressure']
    Mass = 28.0855 * units['mass']
    key = random.PRNGKey(121)
    key, split = random.split(key)

    nbrs = neighbor_fn.allocate(pos, box=box, extra_capacity=8)
    init_fn, apply_fn = simulate.npt_nose_hoover(
      energy_fn, shift, dt=dt, pressure=P_init, kT=T_init
    )
    state = init_fn(key, pos, box=box, neighbor=nbrs)

    def step_fn(i, state_nbrs_buffers):
      state, nbrs, buffers = state_nbrs_buffers
      state = apply_fn(state, neighbor=nbrs)
      nbrs = nbrs.update(state.position)
      buffers['kT'] = (
        buffers['kT']
        .at[i]
        .set(quantity.temperature(momentum=state.momentum, mass=state.mass))
      )
      KE = quantity.kinetic_energy(momentum=state.momentum, mass=Mass)
      buffers['P'] = (
        buffers['P']
        .at[i]
        .set(
          quantity.pressure(
            energy_fn,
            state.position,
            box=box,
            kinetic_energy=KE,
            neighbor=nbrs,
          )
        )
      )
      buffers['H'] = (
        buffers['H']
        .at[i]
        .set(
          simulate.npt_nose_hoover_invariant(
            energy_fn, state, pressure=P_init, kT=T_init, neighbor=nbrs
          )
        )
      )
      return state, nbrs, buffers

    buffers = {
      'kT': np.zeros((DYNAMICS_STEPS,)),
      'P': np.zeros((DYNAMICS_STEPS,)),
      'H': np.zeros((DYNAMICS_STEPS,)),
    }

    state, nbrs, buffers = lax.fori_loop(
      0, DYNAMICS_STEPS, step_fn, (state, nbrs, buffers)
    )

    kT_tol = 1e-2
    P_tol = 2e-3 if dtype == np.float64 else 2e-3

    self.assertAllClose(
      np.mean(buffers['kT'][-DYNAMICS_STEPS // 2 :]),
      T_init,
      atol=kT_tol,
      rtol=kT_tol,
    )
    self.assertAllClose(
      np.mean(buffers['P'][-DYNAMICS_STEPS // 2 :]),
      P_init,
      atol=P_tol,
      rtol=P_tol,
    )
    H_tol = 2e-3
    self.assertAllClose(
      buffers['H'],
      np.ones((DYNAMICS_STEPS,), dtype=dtype) * buffers['H'][0],
      rtol=H_tol,
      atol=H_tol,
    )

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': f'_dtype={dtype.__name__}',
        'dtype': dtype,
      }
      for dtype in DTYPE
    )
  )
  def test_init_functions_accept_explicit_momenta(self, dtype):
    key = random.PRNGKey(0)
    alt_key = random.PRNGKey(1)
    particle_count = 4
    spatial_dimension = 2

    R = np.arange(particle_count * spatial_dimension, dtype=dtype).reshape(
      (particle_count, spatial_dimension)
    )
    mass = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    momenta = np.arange(
      particle_count * spatial_dimension, dtype=dtype
    ).reshape((particle_count, spatial_dimension)) / dtype(10.0)

    _, free_shift = space.free()
    box = np.eye(spatial_dimension, dtype=dtype) * dtype(5.0)
    _, periodic_shift = space.periodic_general(box)

    free_energy = lambda positions, **kwargs: np.sum(positions**2)
    periodic_energy = lambda positions, box, **kwargs: np.sum(positions**2)

    thermostat_cases = [
      (
        'nve',
        simulate.nve(free_energy, free_shift, 1e-3)[0],
        {'kT': dtype(1.0)},
      ),
      (
        'nvt_nose_hoover',
        simulate.nvt_nose_hoover(free_energy, free_shift, 1e-3, dtype(1.0))[0],
        {},
      ),
      (
        'npt_nose_hoover',
        simulate.npt_nose_hoover(
          periodic_energy, periodic_shift, 1e-3, dtype(1.0), dtype(1.0)
        )[0],
        {'box': box},
      ),
      (
        'nvt_langevin',
        simulate.nvt_langevin(
          free_energy, free_shift, 1e-3, dtype(1.0), gamma=dtype(0.1)
        )[0],
        {},
      ),
      (
        'temp_rescale',
        simulate.temp_rescale(
          free_energy,
          free_shift,
          1e-3,
          dtype(1.0),
          window=dtype(0.2),
          fraction=dtype(0.5),
        )[0],
        {},
      ),
      (
        'temp_berendsen',
        simulate.temp_berendsen(
          free_energy, free_shift, 1e-3, dtype(1.0), tau=dtype(0.5)
        )[0],
        {},
      ),
      ('nvk', simulate.nvk(free_energy, free_shift, 1e-3, dtype(1.0))[0], {}),
      (
        'temp_csvr',
        simulate.temp_csvr(
          free_energy, free_shift, 1e-3, dtype(1.0), tau=dtype(0.5)
        )[0],
        {},
      ),
    ]

    for _, init_fn, extra_kwargs in thermostat_cases:
      state = init_fn(key, R, mass=mass, momenta=momenta, **extra_kwargs)
      alt_state = init_fn(
        alt_key, R, mass=mass, momenta=momenta, **extra_kwargs
      )
      self.assertAllClose(state.momentum, momenta)
      self.assertAllClose(state.velocity, momenta / mass[:, None])
      self.assertAllClose(alt_state.momentum, momenta)

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': f'_dtype={dtype.__name__}',
        'dtype': dtype,
      }
      for dtype in DTYPE
    )
  )
  def test_init_functions_randomize_momenta_when_omitted(self, dtype):
    R = np.zeros((4, 2), dtype=dtype)
    mass = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    _, shift = space.free()
    energy_fn = lambda positions, **kwargs: np.sum(positions**2)

    nve_init_fn, _ = simulate.nve(energy_fn, shift, 1e-3)
    nvt_langevin_init_fn, _ = simulate.nvt_langevin(
      energy_fn, shift, 1e-3, dtype(1.0), gamma=dtype(0.1)
    )

    nve_state = nve_init_fn(random.PRNGKey(0), R, kT=dtype(1.0), mass=mass)
    nve_alt_state = nve_init_fn(random.PRNGKey(1), R, kT=dtype(1.0), mass=mass)
    self.assertFalse(np.allclose(nve_state.momentum, nve_alt_state.momentum))

    langevin_state = nvt_langevin_init_fn(random.PRNGKey(0), R, mass=mass)
    langevin_alt_state = nvt_langevin_init_fn(random.PRNGKey(1), R, mass=mass)
    self.assertFalse(
      np.allclose(langevin_state.momentum, langevin_alt_state.momentum)
    )

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
        'spatial_dimension': dim,
        'dtype': dtype,
      }
      for dim in SPATIAL_DIMENSION
      for dtype in DTYPE
    )
  )
  def test_nvt_langevin(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, R_key, R0_key, T_key, masses_key = random.split(key, 5)

      R = random.normal(
        R_key, (LANGEVIN_PARTICLE_COUNT, spatial_dimension), dtype=dtype
      )
      R0 = random.normal(
        R0_key, (LANGEVIN_PARTICLE_COUNT, spatial_dimension), dtype=dtype
      )
      _, shift = space.free()

      E = functools.partial(
        lambda R, R0, **kwargs: np.sum((R - R0) ** 2), R0=R0
      )

      T = random.uniform(T_key, (), minval=0.3, maxval=1.4, dtype=dtype)
      mass = random.uniform(
        masses_key,
        (LANGEVIN_PARTICLE_COUNT,),
        minval=0.1,
        maxval=10.0,
        dtype=dtype,
      )
      init_fn, apply_fn = simulate.nvt_langevin(
        E, shift, f32(1e-2), T, gamma=f32(0.3)
      )
      apply_fn = jit(apply_fn)

      state = init_fn(key, R, mass=mass, T_initial=dtype(1.0))

      T_list = []
      for step in range(LANGEVIN_DYNAMICS_STEPS):
        state = apply_fn(state)
        if step > 4000 and step % 100 == 0:
          T_list += [kT_fn(state.momentum, state.mass)]

      # TODO(schsam): It would be good to check Gaussinity of R and V in the
      # noninteracting case.
      T_emp = np.mean(np.array(T_list))
      assert np.abs(T_emp - T) < 0.1
      assert state.position.dtype == dtype

  def test_langevin_harmonic(self):
    alpha = 1.0
    E = lambda x: jnp.sum(0.5 * alpha * x**2)
    displacement, shift = space.free()

    N = 200000
    steps = 1000
    kT = 0.25
    dt = 1e-4
    gamma = 3
    mass = 3.0
    tol = 1e-3

    X = jnp.ones((N, 1, 1))
    key = random.split(random.PRNGKey(0), N)

    init_fn, step_fn = simulate.nvt_langevin(E, shift, dt, kT, gamma, False)
    step_fn = jit(vmap(step_fn))

    state = vmap(init_fn, (0, 0, None))(key, X, mass)
    p0 = state.momentum

    for i in range(steps):
      state = step_fn(state)

    # Compare mean position and momentum autocorrelation with theoretical
    # prediction.

    d = jnp.sqrt(gamma**2 / 4 - alpha / mass)

    beta_1 = gamma / 2 + d
    beta_2 = gamma / 2 - d
    A = -beta_2 / (beta_1 - beta_2)
    B = beta_1 / (beta_1 - beta_2)
    exp1 = lambda t: jnp.exp(-beta_1 * t)
    exp2 = lambda t: jnp.exp(-beta_2 * t)
    Z = kT / (2 * d * mass)

    pos_fn = lambda t: A * exp1(t) + B * exp2(t)
    mom_fn = lambda t: Z * (-beta_2 * exp2(t) + beta_1 * exp1(t)) * mass**2

    t = steps * dt
    self.assertAllClose(jnp.mean(state.position), pos_fn(t), rtol=tol, atol=tol)
    self.assertAllClose(
      jnp.mean(state.momentum * p0), mom_fn(t), rtol=tol, atol=tol
    )

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
        'spatial_dimension': dim,
        'dtype': dtype,
      }
      for dim in SPATIAL_DIMENSION
      for dtype in DTYPE
    )
  )
  def test_brownian(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)
    key, T_split, mass_split = random.split(key, 3)

    _, shift = space.free()
    energy_fn = lambda R, **kwargs: f32(0)

    R = np.zeros((BROWNIAN_PARTICLE_COUNT, spatial_dimension), dtype=dtype)
    mass = random.uniform(mass_split, (), minval=0.1, maxval=10.0, dtype=dtype)
    T = random.uniform(T_split, (), minval=0.3, maxval=1.4, dtype=dtype)

    dt = f32(1e-2)
    gamma = f32(0.1)

    init_fn, apply_fn = simulate.brownian(energy_fn, shift, dt, T, gamma=gamma)
    apply_fn = jit(apply_fn)

    state = init_fn(key, R, mass)

    sim_t = f32(BROWNIAN_DYNAMICS_STEPS * dt)
    for _ in range(BROWNIAN_DYNAMICS_STEPS):
      state = apply_fn(state)

    msd = np.var(state.position)
    th_msd = dtype(2 * T / (mass * gamma) * sim_t)
    assert np.abs(msd - th_msd) / msd < 1e-2
    assert state.position.dtype == dtype

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': f'_dtype={dtype.__name__}',
        'dtype': dtype,
      }
      for dtype in DTYPE
    )
  )
  def test_swap_mc_jammed(self, dtype):
    key = random.PRNGKey(0)

    state = test_util.load_jammed_state('simulation_test_state.npy', dtype)
    space_fn = space.periodic(state.box[0, 0])
    displacement_fn, shift_fn = space_fn

    sigma = np.diag(state.sigma)[state.species]

    energy_fn = lambda dr, sigma: energy.soft_sphere(dr, sigma=sigma)
    neighbor_fn = partition.neighbor_list(
      displacement_fn, state.box[0, 0], np.max(sigma) + 0.1, dr_threshold=0.5
    )

    kT = 1e-2
    t_md = 0.1
    N_swap = 10
    init_fn, apply_fn = simulate.hybrid_swap_mc(
      space_fn, energy_fn, neighbor_fn, 1e-3, kT, t_md, N_swap
    )
    state = init_fn(key, state.real_position, sigma)

    Ts = np.zeros((DYNAMICS_STEPS,))

    def step_fn(i, state_and_temp):
      state, temp = state_and_temp
      state = apply_fn(state)
      temp = temp.at[i].set(kT_fn(state.md.momentum, 1.0))
      return state, temp

    state, Ts = lax.fori_loop(0, DYNAMICS_STEPS, step_fn, (state, Ts))

    tol = 5e-4
    self.assertAllClose(
      Ts[10:], kT * np.ones((DYNAMICS_STEPS - 10)), rtol=5e-1, atol=5e-3
    )
    self.assertAllClose(np.mean(Ts[10:]), kT, rtol=tol, atol=tol)
    self.assertTrue(not np.all(state.sigma == sigma))

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': f'_dim={dim}_dtype={dtype.__name__}',
        'spatial_dimension': dim,
        'dtype': dtype,
      }
      for dim in SPATIAL_DIMENSION
      for dtype in DTYPE
    )
  )
  def test_nvk_ensemble(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    box_size = quantity.box_size_at_number_density(
      PARTICLE_COUNT, f32(1.2), spatial_dimension
    )
    displacement_fn, shift_fn = space.periodic(box_size)

    bonds_i = np.arange(PARTICLE_COUNT)
    bonds_j = np.roll(bonds_i, 1)
    bonds = np.stack([bonds_i, bonds_j])

    E = energy.simple_spring_bond(displacement_fn, bonds)

    invariant = ke_fn

    for _ in range(STOCHASTIC_SAMPLES):
      key, pos_key, vel_key, T_key, masses_key = random.split(key, 5)

      R = box_size * random.uniform(
        pos_key, (PARTICLE_COUNT, spatial_dimension), dtype=dtype
      )
      T = random.uniform(T_key, (), minval=0.3, maxval=1.4, dtype=dtype)
      mass = 1 + random.uniform(masses_key, (PARTICLE_COUNT,), dtype=dtype)
      init_fn, apply_fn = simulate.nvk(E, shift_fn, 1e-3, T)
      apply_fn = jit(apply_fn)

      state = init_fn(vel_key, R, mass=mass)

      initial = invariant(state.momentum, state.mass)

      for _ in range(DYNAMICS_STEPS):
        state = apply_fn(state)

      tol = 5e-4 if dtype is f32 else 1e-6
      self.assertAllClose(
        invariant(state.momentum, state.mass), initial, rtol=tol
      )
      self.assertEqual(state.position.dtype, dtype)

  @parameterized.named_parameters(
    test_util.cases_from_list(
      {
        'testcase_name': '_dim={}_dtype={}'.format(dim, dtype.__name__),
        'spatial_dimension': dim,
        'dtype': dtype,
      }
      for dim in SPATIAL_DIMENSION
      for dtype in DTYPE
    )
  )
  def test_temp_csvr(self, spatial_dimension, dtype):
    key = random.PRNGKey(0)

    for _ in range(STOCHASTIC_SAMPLES):
      key, R_key, R0_key, T_key, masses_key = random.split(key, 5)

      R = random.normal(
        R_key, (LANGEVIN_PARTICLE_COUNT, spatial_dimension), dtype=dtype
      )
      R0 = random.normal(
        R0_key, (LANGEVIN_PARTICLE_COUNT, spatial_dimension), dtype=dtype
      )
      _, shift = space.free()

      E = functools.partial(
        lambda R, R0, **kwargs: np.sum((R - R0) ** 2), R0=R0
      )

      T = random.uniform(T_key, (), minval=0.3, maxval=1.4, dtype=dtype)
      mass = random.uniform(
        masses_key,
        (LANGEVIN_PARTICLE_COUNT,),
        minval=0.1,
        maxval=10.0,
        dtype=dtype,
      )
      init_fn, apply_fn = simulate.temp_csvr(
        E, shift, f32(1e-2), T, tau=100 * f32(1e-2)
      )
      apply_fn = jit(apply_fn)

      state = init_fn(key, R, mass=mass, T_initial=dtype(1.0))

      T_list = []
      for step in range(LANGEVIN_DYNAMICS_STEPS):
        state = apply_fn(state)
        if step > 4000 and step % 100 == 0:
          T_list += [kT_fn(state.momentum, state.mass)]

      T_emp = np.mean(np.array(T_list))
      assert np.abs(T_emp - T) < 0.1
      assert state.position.dtype == dtype


class ReplicaExchangeTest(test_util.JAXMDTestCase):
  """Check ensemble weights, state transformations, and complete swap sweeps."""

  @parameterized.product(
    integrator=['langevin', 'csvr', 'npt_csvr'], dtype=DTYPE
  )
  def test_weights_and_temperature_transformation(self, integrator, dtype):
    """Exchange kinetics reversibly without changing configurations or caches."""
    R = np.array([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]], dtype=dtype)
    E = lambda R, **kwargs: np.sum(R**2)
    if integrator == 'npt_csvr':
      _, shift = space.periodic_general(dtype(3.0))
      init, step = simulate.npt_csvr(E, shift, 0.001, 0.2, 1.0, 1.0, 0.1)
      state = init(random.PRNGKey(1), R, box=dtype(3.0), box_velocity=0.2)
      parameters = {'pressure': dtype(0.2)}
      transform = {'pressure': dtype(0.2), 'new_pressure': dtype(0.7)}
      volume = quantity.volume(3, state.box)
      expected = -(E(R) + 0.2 * volume)
    else:
      _, shift = space.free()
      if integrator == 'langevin':
        init, step = simulate.nvt_langevin(E, shift, 0.001, 1.0)
      else:
        init, step = simulate.temp_csvr(E, shift, 0.001, 1.0, 0.1)
      state = init(random.PRNGKey(1), R)
      parameters, transform = {}, {}
      expected = -E(R)
    weight = jit(
      lambda s: simulate.log_probability(s, E, dtype(1.0), **parameters)
    )
    self.assertAllClose(weight(state), expected)
    changed = jit(
      lambda s: simulate.rescale_temperature(
        s, dtype(1.0), dtype(4.0), **transform
      )
    )(state)
    self.assertAllClose(changed.momentum, 2 * state.momentum)
    self.assertAllClose(
      simulate.kinetic_energy(changed), 4 * simulate.kinetic_energy(state)
    )
    self.assertAllClose(changed.position, state.position)
    self.assertAllClose(changed.force, state.force)
    onp.testing.assert_array_equal(changed.rng, state.rng)
    self.assertEqual(changed.momentum.dtype, dtype)
    if integrator == 'npt_csvr':
      self.assertAllClose(changed.box_momentum, 2 * state.box_momentum)
      self.assertAllClose(changed.box_mass, state.box_mass)
      self.assertAllClose(changed.box, state.box)
      self.assertAllClose(changed.dUdV, state.dUdV)
      self.assertAllClose(changed.potential_energy, state.potential_energy)
      kinetic = simulate.kinetic_energy(state) + state.box_momentum**2 / (
        2 * state.box_mass
      )
      self.assertAllClose(
        changed.conserved_quantity - state.conserved_quantity,
        3 * kinetic + 0.5 * volume - 3 * np.log(volume),
      )
      reverse = {'pressure': dtype(0.7), 'new_pressure': dtype(0.2)}
    else:
      reverse = {}
    restored = simulate.rescale_temperature(
      changed, dtype(4.0), dtype(1.0), **reverse
    )
    for a, b in zip(
      jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(restored)
    ):
      if a.dtype == onp.dtype('uint32'):
        onp.testing.assert_array_equal(a, b)
      else:
        self.assertAllClose(a, b, atol=1e-5)
    advanced = jit(lambda s: step(s, kT=dtype(4.0), **parameters))(changed)
    self.assertTrue(onp.all(onp.isfinite(advanced.momentum)))

  def test_csvr_dynamic_temperature(self):
    """The CSVR step must use the supplied replica temperature."""
    E = lambda R, **kwargs: np.sum(R**2)
    _, shift = space.free()
    init, step = simulate.temp_csvr(E, shift, 0.01, 1.0, 0.1)
    state = init(random.PRNGKey(3), np.ones((2, 3)))
    self.assertFalse(
      onp.allclose(step(state, kT=1.0).momentum, step(state, kT=4.0).momentum)
    )

  @parameterized.parameters(2, 3, 4, 5, 8)
  def test_pair_rounds(self, count):
    """Every unordered pair appears once, with no replica repeated in a round."""
    rounds = onp.asarray(replica_exchange._pair_rounds(count))
    pairs = set()
    for row in rounds:
      self.assertEqual(len(onp.unique(row)), row.size)
      pairs.update(tuple(sorted(pair)) for pair in row)
    self.assertEqual(len(pairs), count * (count - 1) // 2)
    self.assertEqual(rounds.shape[0], count - 1 + count % 2)

  @parameterized.product(
    count=[2, 3, 4, 5], integrator=['langevin', 'csvr', 'npt_csvr']
  )
  def test_stride_and_complete_sweep(self, count, integrator):
    """Equal ensembles accept all pairs only on exact stride boundaries."""
    R = np.array([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]])
    E = lambda R, **kwargs: np.sum(R**2)
    temperatures = np.ones(count)
    pressures = None
    if integrator == 'npt_csvr':
      _, shift = space.periodic_general(3.0)
      init_md, step_md = simulate.npt_csvr(E, shift, 0.001, 0.2, 1.0, 1.0, 0.1)
      replicas = vmap(lambda key: init_md(key, R, 3.0))(
        random.split(random.PRNGKey(1), count)
      )
      pressures = np.full(count, 0.2)
    else:
      _, shift = space.free()
      if integrator == 'langevin':
        init_md, step_md = simulate.nvt_langevin(E, shift, 0.001, 1.0)
      else:
        init_md, step_md = simulate.temp_csvr(E, shift, 0.001, 1.0, 0.1)
      replicas = vmap(lambda key: init_md(key, R))(
        random.split(random.PRNGKey(1), count)
      )
    init, step = replica_exchange.replica_exchange(
      E, step_md, temperatures, pressures, stride=3
    )
    initial = init(random.PRNGKey(2), replicas)
    before = jit(lambda s: lax.fori_loop(0, 2, lambda _, s: step(s), s))(
      initial
    )
    onp.testing.assert_array_equal(before.ensemble_indices, onp.arange(count))
    onp.testing.assert_array_equal(
      before.swap_accepts, onp.zeros((count, count))
    )
    onp.testing.assert_array_equal(before.rng, initial.rng)
    expected_md = vmap(
      lambda md: step_md(
        md, kT=1.0, **({} if pressures is None else {'pressure': 0.2})
      )
    )(before.replicas)
    after = jit(step)(before)
    self.assertEqual(int(after.step), 3)
    onp.testing.assert_array_equal(
      after.swap_accepts, onp.ones((count, count)) - onp.eye(count)
    )
    onp.testing.assert_array_equal(
      onp.sort(after.ensemble_indices), onp.arange(count)
    )
    self.assertAllClose(
      after.replicas.position, expected_md.position, atol=1e-12
    )
    self.assertAllClose(after.replicas.force, expected_md.force, atol=1e-12)
    again = jit(lambda s: lax.fori_loop(0, 3, lambda _, s: step(s), s))(after)
    onp.testing.assert_array_equal(
      again.swap_accepts, 2 * (onp.ones((count, count)) - onp.eye(count))
    )

  def test_acceptance_sign(self):
    """A high-energy configuration cannot move into a much colder ensemble."""
    E = lambda R, **kwargs: np.sum(R**2)
    _, shift = space.free()
    init_md, _ = simulate.nvt_langevin(E, shift, 0.01, 1.0)
    positions = np.array([[[0.0, 0.0, 0.0]], [[100.0, 0.0, 0.0]]])
    replicas = vmap(init_md)(random.split(random.PRNGKey(1), 2), positions)
    init, step = replica_exchange.replica_exchange(
      E, lambda s, **kw: s, [1.0, 2.0], stride=1
    )
    result = jit(step)(init(random.PRNGKey(2), replicas))
    onp.testing.assert_array_equal(result.ensemble_indices, [0, 1])
    onp.testing.assert_array_equal(result.swap_accepts, onp.zeros((2, 2)))
    reversed_state = init(random.PRNGKey(2), replicas, ensemble_indices=[1, 0])
    result = jit(step)(reversed_state)
    onp.testing.assert_array_equal(result.ensemble_indices, [0, 1])
    onp.testing.assert_array_equal(result.swap_accepts, [[0, 1], [1, 0]])

  @parameterized.parameters('nvt', 'npt')
  def test_exchange_preserves_exact_ensemble(self, ensemble):
    """Exchange exact independent samples and recover analytic ensemble moments."""
    count, samples, particles = 3, 12000, 2
    temperatures = np.array([0.8, 1.0, 1.3])
    pressures = np.array([0.6, 0.8, 1.0]) if ensemble == 'npt' else None
    keys = random.split(random.PRNGKey(10), 6)
    momentum = (
      random.normal(keys[0], (samples, count, particles, 3))
      * np.sqrt(temperatures)[None, :, None, None]
    )
    if ensemble == 'nvt':
      # U = |q|^2 / 2 has six independent Gaussian configurational degrees.
      E = lambda R, **kwargs: np.sum(R**2) / 2
      position = (
        random.normal(keys[1], momentum.shape)
        * np.sqrt(temperatures)[None, :, None, None]
      )
      replicas = simulate.NVTLangevinState(
        position,
        momentum,
        -position,
        np.ones((samples, count)),
        random.split(keys[2], samples * count).reshape(samples, count, 2),
      )
    else:
      # An unconstrained ideal gas has V ~ Gamma(N+1, kT/P).
      assert pressures is not None
      E = lambda R, **kwargs: np.zeros((), dtype=R.dtype)
      volume = (
        random.gamma(keys[1], particles + 1.0, (samples, count))
        * temperatures
        / pressures
      )
      position = random.uniform(keys[2], momentum.shape)
      boxes = volume[:, :, None, None] ** (1 / 3) * np.eye(3)
      zeros = np.zeros((samples, count))
      replicas = simulate.NPTCSVRState(
        position,
        momentum,
        np.zeros_like(position),
        np.ones_like(zeros),
        boxes,
        zeros,
        random.normal(keys[3], zeros.shape) * np.sqrt(temperatures),
        np.ones_like(zeros),
        zeros,
        zeros,
        random.split(keys[4], samples * count).reshape(samples, count, 2),
        zeros,
      )
    init, step = replica_exchange.replica_exchange(
      E, lambda s, **kw: s, temperatures, pressures, stride=1
    )
    batched = vmap(init)(random.split(keys[5], samples), replicas)
    result = jit(vmap(step))(batched)
    order = np.argsort(result.ensemble_indices, axis=1)
    rows = np.arange(samples)[:, None]
    kinetic = np.sum(result.replicas.momentum**2, axis=(-2, -1)) / 2
    kinetic = kinetic[rows, order]
    self.assertAllClose(np.mean(kinetic, axis=0), 3 * temperatures, rtol=0.025)
    self.assertAllClose(np.var(kinetic, axis=0), 3 * temperatures**2, rtol=0.06)
    if ensemble == 'nvt':
      potential = np.sum(result.replicas.position**2, axis=(-2, -1)) / 2
      potential = potential[rows, order]
      self.assertAllClose(
        np.mean(potential, axis=0), 3 * temperatures, rtol=0.025
      )
      self.assertAllClose(
        np.var(potential, axis=0), 3 * temperatures**2, rtol=0.06
      )
    else:
      assert pressures is not None
      volume = np.linalg.det(result.replicas.reference_box)[rows, order]
      self.assertAllClose(
        np.mean(volume, axis=0), 3 * temperatures / pressures, rtol=0.025
      )
      self.assertAllClose(
        np.var(volume, axis=0), 3 * (temperatures / pressures) ** 2, rtol=0.06
      )
      piston_kinetic = (result.replicas.box_momentum**2 / 2)[rows, order]
      self.assertAllClose(
        np.mean(piston_kinetic, axis=0), temperatures / 2, rtol=0.035
      )
      self.assertAllClose(
        np.var(piston_kinetic, axis=0), temperatures**2 / 2, rtol=0.08
      )

  def test_index_file_and_sorting(self):
    """Samples at an exchange step use the mapping written after that exchange."""
    state = replica_exchange.ReplicaExchangeState(
      None,
      np.array([0, 1, 2]),
      np.array(0),
      random.PRNGKey(0),
      np.zeros((3, 3)),
    )
    file = io.StringIO()
    replica_exchange.write_swap_indices(file, state)
    replica_exchange.write_swap_indices(
      file, state.set(step=np.array(10), ensemble_indices=np.array([2, 0, 1]))
    )
    mappings = onp.loadtxt(io.StringIO(file.getvalue()), dtype=int, ndmin=2)
    values = onp.arange(4 * 3 * 2).reshape(4, 3, 2)
    sorted_values = replica_exchange.sort_by_ensemble(
      [0, 9, 10, 11], values, mappings[:, 0], mappings[:, 1:]
    )
    self.assertAllClose(sorted_values[:2], values[:2])
    self.assertAllClose(sorted_values[2:], values[2:, [1, 2, 0]])
    with self.assertRaises(ValueError):
      replica_exchange.sort_by_ensemble([0], values[:1], [1], [[0, 1, 2]])
    with self.assertRaises(ValueError):
      replica_exchange.sort_by_ensemble([0], values[:1], [0], [[0, 0, 2]])

  def test_invalid_parameters(self):
    """Reject malformed ladders and unsupported state types before running MD."""
    E = lambda R, **kwargs: np.sum(R**2)
    for temperatures in ([1.0], [0.0, 1.0], [1.0, onp.nan], [[1.0, 2.0]]):
      with self.assertRaises(ValueError):
        replica_exchange.replica_exchange(E, lambda s: s, temperatures)
    invalid_strides: list[Any] = [0, -1, 1.5, True]
    for stride in invalid_strides:
      with self.assertRaises(ValueError):
        replica_exchange.replica_exchange(
          E, lambda s: s, [1.0, 2.0], stride=stride
        )
    for pressures in ([1.0], [1.0, onp.inf]):
      with self.assertRaises(ValueError):
        replica_exchange.replica_exchange(
          E, lambda s: s, [1.0, 2.0], pressure=pressures
        )
    _, shift = space.free()
    init, _ = simulate.nve(E, shift)
    state = init(random.PRNGKey(1), np.ones((2, 3)), kT=1.0)
    with self.assertRaises(TypeError):
      simulate.log_probability(state, E, 1.0)
    with self.assertRaises(TypeError):
      simulate.rescale_temperature(state, 1.0, 2.0)
    init_md, _ = simulate.nvt_langevin(E, shift, 0.01, 1.0)
    replicas = vmap(lambda key: init_md(key, np.ones((2, 3))))(
      random.split(random.PRNGKey(1), 2)
    )
    init_exchange, step_exchange = replica_exchange.replica_exchange(
      E, lambda s, **kwargs: s, [1.0, 2.0]
    )
    with self.assertRaises(ValueError):
      init_exchange(random.PRNGKey(2), replicas, ensemble_indices=[0, 0])
    with self.assertRaises(ValueError):
      init_exchange(
        random.PRNGKey(2), jax.tree_util.tree_map(lambda x: x[0], replicas)
      )
    init_npt, _ = replica_exchange.replica_exchange(
      E, lambda s, **kwargs: s, [1.0, 2.0], [0.4, 0.8]
    )
    with self.assertRaises(ValueError):
      init_npt(random.PRNGKey(2), replicas)
    state = init_exchange(random.PRNGKey(2), replicas)
    with self.assertRaises(ValueError):
      step_exchange(state, kT=1.0)


if __name__ == '__main__':
  absltest.main()
