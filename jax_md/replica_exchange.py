# Copyright 2026 Google LLC
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

"""Temperature and pressure replica exchange with continuous MD trajectories."""

from typing import Any, Callable, TextIO

import jax
from jax import lax, random, vmap
import jax.numpy as jnp
import numpy as np

from jax_md import dataclasses, simulate


@dataclasses.dataclass
class ReplicaExchangeState:
  """Batched MD states and their current thermodynamic labels.

  ``replicas`` has a leading trajectory axis on every dynamic leaf, as produced
  by ``vmap(init_md_fn)``. ``ensemble_indices[r]`` identifies the ensemble
  currently assigned to trajectory ``r``. ``step`` counts completed MD steps.
  ``swap_accepts[i, j]`` counts accepted exchanges between ensemble labels.
  Each unordered pair is attempted exactly once every exchange stride.
  ``rng`` is independent of the random keys used by the MD integrators.
  """

  replicas: Any
  ensemble_indices: jax.Array
  step: jax.Array
  rng: jax.Array
  swap_accepts: jax.Array


def _pair_rounds(count):
  """Partition the complete graph into rounds with no overlapping pairs."""
  ring = list(range(count + count % 2))
  rounds = []
  for _ in range(len(ring) - 1):
    rounds.append(
      [
        (i, j)
        for i, j in zip(
          ring[: len(ring) // 2], reversed(ring[len(ring) // 2 :])
        )
        if i < count and j < count
      ]
    )
    ring = [ring[0], ring[-1], *ring[1:-1]]
  return jnp.asarray(rounds, dtype=jnp.int32)


def replica_exchange(
  energy_fn: Callable,
  apply_md_fn: Callable,
  kT,
  pressure=None,
  stride: int = 100,
) -> simulate.Simulator:
  """Wrap a canonical MD step with deterministic all-pairs replica exchange.

  Args:
    energy_fn: The shared scalar potential energy function. For NPT it must
      accept the current ``box``. Temperature-dependent Hamiltonians, different
      particle masses or constraints, and exchanges between NVT and NPT are
      not supported.
    apply_md_fn: A single-replica MD step accepting dynamic ``kT`` and, for NPT,
      ``pressure``. MD states must implement
      :func:`jax_md.simulate.log_probability` and
      :func:`jax_md.simulate.rescale_temperature`.
    kT: A fixed one-dimensional array of positive ensemble temperatures in
      energy units, one entry per replica.
    pressure: An array matching ``kT`` for NPT, or ``None`` for NVT.
    stride: Positive integer MD steps between exchange events.

  Returns:
    ``init_fn(key, replicas, ensemble_indices=None)`` and ``apply_fn(state,
    **kwargs)``. The caller initializes the batched MD states at the supplied
    labels. Each apply call advances one MD step, then exchanges labels if the
    completed step is divisible by ``stride``. Outputs at that step belong to
    the post-exchange labels. Arguments other than ensemble parameters are
    shared across replicas and forwarded to MD and energy evaluation.

  Every event attempts all unordered ensemble pairs in round-robin order.
  Within a round, pairs are independent and vectorized; rounds run sequentially
  with updated labels. Potential energies are evaluated once per trajectory.
  Kinetic variables are transformed once, after the final label assignment.
  Pair decisions cost O(R**2); physical configurations and neighbor lists stay
  with their trajectories. Each Metropolis move preserves the product target
  distribution, although a complete deterministic sweep need not be reversible.

  File I/O belongs outside JIT: call :func:`write_swap_indices` initially and
  after exchange events. Save the whole state for an exact restart.
  """
  temperatures = np.asarray(kT)
  if (
    temperatures.ndim != 1
    or len(temperatures) < 2
    or not np.all(np.isfinite(temperatures) & (temperatures > 0))
  ):
    raise ValueError(
      'kT must contain at least two finite positive temperatures.'
    )
  if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
    raise ValueError('stride must be a positive integer.')
  parameters = {'kT': jnp.asarray(kT)}
  if pressure is not None:
    if np.shape(pressure) != temperatures.shape or not np.all(
      np.isfinite(pressure)
    ):
      raise ValueError('pressure must be finite and have the same shape as kT.')
    parameters['pressure'] = jnp.asarray(pressure)
  count = len(temperatures)
  rounds = _pair_rounds(count)

  def init_fn(key, replicas, ensemble_indices=None):
    """Attach labels and an independent exchange key to batched MD states."""
    leaves = jax.tree_util.tree_leaves(replicas)
    if not leaves or any(
      np.ndim(x) == 0 or np.shape(x)[0] != count for x in leaves
    ):
      raise ValueError('Every MD state leaf must have a leading replica axis.')
    is_npt = hasattr(type(replicas), 'box')
    if is_npt != (pressure is not None):
      raise ValueError('NPT states require pressure; NVT states must omit it.')
    if ensemble_indices is None:
      ensemble_indices = np.arange(count)
    if not np.array_equal(np.sort(ensemble_indices), np.arange(count)):
      raise ValueError(
        'ensemble_indices must be a permutation of replica labels.'
      )
    return ReplicaExchangeState(
      replicas,
      jnp.asarray(ensemble_indices, dtype=jnp.int32),
      jnp.zeros((), dtype=jnp.int32),
      key,
      jnp.zeros((count, count), dtype=jnp.int32),
    )

  def exchange(state, kwargs):
    """Attempt one complete sweep, reusing energies while labels change."""

    def weights(md):
      box_kwargs = {} if pressure is None else {'box': md.box}
      potential = energy_fn(md.position, **box_kwargs, **kwargs)
      cached_energy = lambda position, **unused: potential
      return vmap(
        lambda targets: simulate.log_probability(md, cached_energy, **targets)
      )(parameters)

    log_weights = vmap(weights)(state.replicas)
    key, next_key = random.split(state.rng)
    uniforms = random.uniform(key, rounds.shape[:2], dtype=log_weights.dtype)

    def attempt(carry, inputs):
      indices, counts = carry
      pairs, uniform = inputs
      i, j = pairs[:, 0], pairs[:, 1]
      trajectories = (
        jnp.zeros_like(indices)
        .at[indices]
        .set(jnp.arange(count, dtype=indices.dtype))
      )
      a, b = trajectories[i], trajectories[j]
      log_ratio = (log_weights[a, j] - log_weights[a, i]) + (
        log_weights[b, i] - log_weights[b, j]
      )
      accepted = jnp.log(uniform) < jnp.minimum(0.0, log_ratio)
      indices = indices.at[a].set(jnp.where(accepted, j, i))
      indices = indices.at[b].set(jnp.where(accepted, i, j))
      counts = counts.at[i, j].add(accepted.astype(counts.dtype))
      counts = counts.at[j, i].add(accepted.astype(counts.dtype))
      return (indices, counts), None

    (indices, counts), _ = lax.scan(
      attempt, (state.ensemble_indices, state.swap_accepts), (rounds, uniforms)
    )
    old = jax.tree_util.tree_map(
      lambda x: x[state.ensemble_indices], parameters
    )
    new = jax.tree_util.tree_map(lambda x: x[indices], parameters)

    def transform(md, before, after):
      pressures = (
        {}
        if pressure is None
        else {'pressure': before['pressure'], 'new_pressure': after['pressure']}
      )
      return simulate.rescale_temperature(
        md, before['kT'], after['kT'], **pressures
      )

    return state.set(
      replicas=vmap(transform)(state.replicas, old, new),
      ensemble_indices=indices,
      swap_accepts=counts,
      rng=next_key,
    )

  def apply_fn(state, **kwargs):
    """Advance all replicas together and exchange at the prescribed stride."""
    if 'kT' in kwargs or 'pressure' in kwargs:
      raise ValueError(
        'Replica temperatures and pressures are fixed by the ladder.'
      )
    dtype = jax.tree_util.tree_leaves(state.replicas.position)[0].dtype
    targets = jax.tree_util.tree_map(
      lambda x: jnp.asarray(x[state.ensemble_indices], dtype=dtype), parameters
    )
    replicas = vmap(lambda md, target: apply_md_fn(md, **target, **kwargs))(
      state.replicas, targets
    )
    state = state.set(replicas=replicas, step=state.step + 1)
    return lax.cond(
      state.step % stride == 0,
      lambda s: exchange(s, kwargs),
      lambda s: s,
      state,
    )

  return init_fn, apply_fn


def write_swap_indices(file: TextIO, state: ReplicaExchangeState):
  """Write ``step label_for_trajectory_0 ...`` after an event, outside JIT.

  Write the initial state at step zero, then each exchange event, including
  rejected sweeps. Sample trajectory outputs after writing the corresponding
  mapping. NumPy/JAX conversion transfers only the indices and step to the host.
  """
  labels = ' '.join(str(int(i)) for i in np.asarray(state.ensemble_indices))
  file.write(f'{int(state.step)} {labels}\n')


def sort_by_ensemble(steps, values, swap_steps, swap_indices):
  """Sort trajectory-indexed samples into fixed-ensemble order on the host.

  ``values`` has shape ``[frames, replicas, ...]``. Mapping rows apply from
  ``swap_steps`` onward, including samples at the exact exchange step. An
  initial mapping covering the earliest sample is required. This works for
  positions, boxes, and scalar observables and preserves their data types.
  """
  steps, values = np.asarray(steps), np.asarray(values)
  swap_steps, swap_indices = np.asarray(swap_steps), np.asarray(swap_indices)
  if steps.ndim != 1 or values.ndim < 2 or values.shape[0] != len(steps):
    raise ValueError('values must have shape [len(steps), replicas, ...].')
  if (
    swap_steps.ndim != 1
    or len(swap_steps) == 0
    or swap_indices.shape != (len(swap_steps), values.shape[1])
    or np.any(np.diff(swap_steps) <= 0)
    or np.any(np.diff(steps) < 0)
    or not np.issubdtype(swap_indices.dtype, np.integer)
    or not np.all(np.sort(swap_indices, axis=1) == np.arange(values.shape[1]))
  ):
    raise ValueError(
      'Mappings must be permutations at strictly increasing steps.'
    )
  rows = np.searchsorted(swap_steps, steps, side='right') - 1
  if np.any(rows < 0):
    raise ValueError('An initial mapping must cover the earliest sample.')
  order = np.argsort(swap_indices[rows], axis=1)
  return values[np.arange(len(steps))[:, None], order]
