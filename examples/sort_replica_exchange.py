"""Regroup a replica-exchange NPZ trajectory archive by thermodynamic label.

Usage: python examples/sort_replica_exchange.py swap_indices trajectories.npz
       sorted_trajectories.npz

Input archives contain steps [frames] and sample arrays [frames, replicas, ...].
The swap file contains step followed by each trajectory's post-exchange label.
"""

import argparse

import numpy as np

from jax_md import replica_exchange


def main():
  """Read continuous trajectories, sort every sampled field, and save them."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('swap_indices')
  parser.add_argument('trajectories')
  parser.add_argument('output')
  args = parser.parse_args()
  swaps = np.loadtxt(args.swap_indices, dtype=np.int64, ndmin=2)
  with np.load(args.trajectories, allow_pickle=False) as raw:
    sorted_data = {'steps': raw['steps']}
    for name in raw.files:
      if name != 'steps':
        sorted_data[name] = replica_exchange.sort_by_ensemble(
          raw['steps'], raw[name], swaps[:, 0], swaps[:, 1:]
        )
  np.savez(args.output, **sorted_data)
  print(f'Wrote ensemble-sorted samples to {args.output}')


if __name__ == '__main__':
  main()
