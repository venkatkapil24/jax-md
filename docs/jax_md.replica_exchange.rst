Replica Exchange
================

.. default-role:: code
.. automodule:: jax_md.replica_exchange

.. autofunction:: replica_exchange
.. autofunction:: write_swap_indices
.. autofunction:: sort_by_ensemble
.. autoclass:: ReplicaExchangeState

Built-in adapters support NVT Langevin, NVT CSVR, and NPT CSVR. Nose-Hoover
integrators are outside the scope of this implementation. NPT Langevin can use
the same exchange driver once its state adapters are registered.

The runnable ``examples/replica_exchange_lj.py`` example writes continuous
trajectory data to ``replica_exchange_run/trajectories.npz`` and a complete
post-exchange mapping to ``replica_exchange_run/swap_indices``. Run
``examples/sort_replica_exchange.py`` with those two paths and an output archive
path to regroup the data by fixed thermodynamic ensemble. Archive arrays have
axes ``[frames, replicas, ...]`` and include a one-dimensional ``steps`` array.

.. code-block:: bash

   JAX_ENABLE_X64=1 python examples/replica_exchange_lj.py
   python examples/sort_replica_exchange.py \
       replica_exchange_run/swap_indices \
       replica_exchange_run/trajectories.npz \
       replica_exchange_run/ensembles.npz

Each trajectory retains its positions, box, forces, neighbor-list state, and MD
random key. Only labels and kinetic variables change at an accepted exchange.
For NPT, particle configurations use a common coordinate representation and
the same number of particles. Volume-dependent measure factors are common to
all labels and cancel in exchange ratios.

The acceptance matrix counts individual accepted pair moves. Later moves in
the same sweep can undo earlier label changes, so the mapping file records
only the final assignment. Acceptance counts alone do not measure mixing;
use the mapping history to examine ensemble visits and round trips.
