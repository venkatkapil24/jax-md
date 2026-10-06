"""Run four LJ NPT CSVR replicas and write post-exchange trajectory samples.

Run from the repository with JAX_ENABLE_X64=1. Sort the resulting archive with
examples/sort_replica_exchange.py to obtain one trajectory per ensemble.
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random, vmap

from jax_md import energy, quantity, replica_exchange, simulate, space

jax.config.update('jax_enable_x64', True)

NSTEPS_SIM = 1000 if os.environ.get('READTHEDOCS', False) else 50000
STRIDE = 100
kT = jnp.array([0.8, 0.8, 1.0, 1.0])
pressure = jnp.array([0.4, 0.8, 0.4, 0.8])
box = (64 / 0.7) ** (1 / 3)
grid = jnp.stack(jnp.meshgrid(*[jnp.arange(4)] * 3, indexing='ij'), axis=-1)
position = (grid.reshape(-1, 3) + 0.5) / 4
displacement, shift = space.periodic_general(box)
energy_fn = energy.lennard_jones_pair(displacement, r_onset=1.8, r_cutoff=2.0)
init_md, step_md = simulate.npt_csvr(
  energy_fn, shift, dt=0.002, kT=1.0, pressure=0.4, tau_p=1.0, tau_t=0.1
)
keys = random.split(random.PRNGKey(0), len(kT) + 1)
replicas = vmap(
  lambda key, t, p: init_md(key, position, box, kT=t, pressure=p)
)(keys[:-1], kT, pressure)
init, step = replica_exchange.replica_exchange(
  energy_fn, step_md, kT, pressure, stride=STRIDE
)
state = init(keys[-1], replicas)
advance = jax.jit(lambda s: lax.fori_loop(0, STRIDE, lambda _, s: step(s), s))
output = Path('replica_exchange_run')
output.mkdir(exist_ok=True)
samples = {
  name: []
  for name in (
    'steps',
    'position',
    'box',
    'potential_energy',
    'temperature',
    'volume',
    'pressure',
  )
}

with (output / 'swap_indices').open('w') as swaps:
  swaps.write('# step ensemble_label_for_each_continuous_trajectory\n')
  for frame in range(NSTEPS_SIM // STRIDE + 1):
    if frame:
      state = advance(state)
    replica_exchange.write_swap_indices(swaps, state)
    md = state.replicas
    volumes = vmap(lambda s: quantity.volume(3, s.box))(md)
    kinetic = vmap(simulate.kinetic_energy)(md)
    values = (
      state.step,
      md.position,
      vmap(lambda s: s.box)(md),
      md.potential_energy,
      2 * kinetic / position.size,
      volumes,
      (2 * kinetic - md.dUdV) / (3 * volumes),
    )
    for name, value in zip(samples, values):
      samples[name].append(np.asarray(value))

np.savez(output / 'trajectories.npz', **samples)
print(f'Wrote {NSTEPS_SIM} steps to {output.resolve()}')
print('Accepted exchanges between ensemble labels:')
print(np.asarray(state.swap_accepts))
