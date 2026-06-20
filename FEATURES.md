# JAX MD Features

This document summarizes the main components of the library.

## Spaces ([`space.py`](https://jax-md.readthedocs.io/en/main/jax_md.space.html))

In general we must have a way of computing the pairwise distance between atoms. We must also have efficient strategies for moving atoms in some space that may or may not be globally isomorphic to R^N. For example, periodic boundary conditions are commonplace in simulations and must be respected. Spaces are defined as a pair of functions, `(displacement_fn, shift_fn)`. Given two points `displacement_fn(R_1, R_2)` computes the displacement vector between the two points. If you would like to compute displacement vectors between all pairs of points in a given `(N, dim)` matrix the function `space.map_product` appropriately vectorizes `displacement_fn`. It is often useful to define a metric instead of a displacement function in which case you can use the helper function `space.metric` to convert a displacement function to a metric function. Given a point and a shift `shift_fn(R, dR)` displaces the point `R` by an amount `dR`.

The following spaces are currently supported:
- [`space.free()`](https://jax-md.readthedocs.io/en/main/jax_md.space.html?highlight=free#jax_md.space.free) specifies a space with free boundary conditions.
- [`space.periodic(box_size)`](https://jax-md.readthedocs.io/en/main/jax_md.space.html?highlight=periodic#jax_md.space.periodic) specifies a space with periodic boundary conditions of side length `box_size`.
- [`space.periodic_general(box)`](https://jax-md.readthedocs.io/en/main/jax_md.space.html?highlight=periodic_general#jax_md.space.periodic_general) specifies a space as a periodic parellelopiped formed by transforming the unit cube by an affine transformation `box`.

Example:

```python
from jax_md import space
box_size = 25.0
displacement_fn, shift_fn = space.periodic(box_size)
```

## Potential Energy ([`energy.py`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html))

In the simplest case, molecular dynamics calculations are often based on a pair potential that is defined by a user. This then is used to compute a total energy whose negative gradient gives forces. One of the very nice things about JAX is that we get forces for free! The second part of the code is devoted to computing energies.

We provide the following classical potentials:
- [`energy.soft_sphere`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html?highlight=soft_sphere#jax_md.energy.soft_sphere) a soft sphere whose energy increases as the overlap of the spheres to some power, `alpha`.
- [`energy.lennard_jones`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html?highlight=lennard_jones#jax_md.energy.lennard_jones) a standard 12-6 Lennard-Jones potential.
- [`energy.morse`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html?highlight=morse#jax_md.energy.morse) a morse potential.
- [`energy.tersoff`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html?highlight=tersoff#jax_md.energy.tersoff) the Tersoff potential for simulating semiconducting materials. Can load parameters from LAMMPS files.
- [`energy.eam`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html?highlight=eam#jax_md.energy.eam) embedded atom model potential with ability to load parameters from LAMMPS files.
- [`energy.stillinger_weber`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html?highlight=stillinger_weber#jax_md.energy.stillinger_weber) used to model Silicon-like systems.
- [`energy.bks`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html?highlight=bks#jax_md.energy.bks) Beest-Kramer-van Santen potential used to model silica.
- [`energy.gupta`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html?highlight=gupta#jax_md.energy.gupta) used to model gold nanoclusters.

We also provide the following neural network potentials:
- [`energy.behler_parrinello`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html?highlight=behler_parrinello#jax_md.energy.behler_parrinello) a widely used fixed-feature neural network architecture for molecular systems.
- [`energy.graph_network`](https://jax-md.readthedocs.io/en/main/jax_md.energy.html?highlight=graph_network#jax_md.energy.graph_network) a deep graph neural network designed for energy fitting.

For finite-ranged potentials it is often useful to consider only interactions within a certain neighborhood. We include the `_neighbor_list` modifier to the above potentials that uses a list of neighbors (see below) for optimization.

Example:

```python
import jax.numpy as np
from jax import random
from jax_md import energy, quantity
N = 1000
spatial_dimension = 2
key = random.PRNGKey(0)
R = random.uniform(key, (N, spatial_dimension), minval=0.0, maxval=1.0)
energy_fn = energy.lennard_jones_pair(displacement_fn)
print('E = {}'.format(energy_fn(R)))
force_fn = quantity.force(energy_fn)
print('Total Squared Force = {}'.format(np.sum(force_fn(R) ** 2)))
```

## Dynamics ([`simulate.py`](https://jax-md.readthedocs.io/en/main/jax_md.simulate.html), [`minimize.py`](https://jax-md.readthedocs.io/en/main/jax_md.minimize.html))

Given an energy function and a system, there are a number of dynamics are useful to simulate. The simulation code is based on the structure of the optimizers found in JAX. In particular, each simulation function returns an initialization function and an update function. The initialization function takes a set of positions and creates the necessary dynamical state variables. The update function does a single step of dynamics to the dynamical state variables and returns an updated state.

We include a several different kinds of dynamics. However, there is certainly room to add more for e.g. constant strain simulations.

It is often desirable to find an energy minimum of the system. We provide two methods to do this. We provide simple gradient descent minimization. This is mostly for pedagogical purposes, since it often performs poorly. We additionally include the FIRE algorithm which often sees significantly faster convergence. Moreover a common experiment to run in the context of molecular dynamics is to simulate a system with a fixed volume and temperature.

We provide the following dynamics:
- [`simulate.nve`](https://jax-md.readthedocs.io/en/main/jax_md.simulate.html?highlight=nve#jax_md.simulate.nve) Constant energy simulation; numerically integrates Newton's laws directly.
- [`simulate.nvt_nose_hoover`](https://jax-md.readthedocs.io/en/main/jax_md.simulate.html?highlight=nvt_nose_hoover#jax_md.simulate.nvt_nose_hoover) Uses Nose-Hoover chain to simulate a constant temperature system.
- [`simulate.npt_nose_hoover`](https://jax-md.readthedocs.io/en/main/jax_md.simulate.html?highlight=nvp_nose_hoover#jax_md.simulate.nvp_nose_hoover) Uses Nose-Hoover chain to simulate a system at constant pressure and temperature.
- [`simulate.nvt_langevin`](https://jax-md.readthedocs.io/en/main/jax_md.simulate.html?highlight=nvt_langevin#jax_md.simulate.nvt_langevin) Simulates a system by numerically integrating the Langevin stochastic differential equation.
- [`simulate.hybrid_swap_mc`](https://jax-md.readthedocs.io/en/main/jax_md.simulate.html?highlight=hybrid_swap_mc#jax_md.simulate.hybrid_swap_mc) Alternates NVT dynamics with Monte-Carlo swapping moves to generate low energy glasses.
- [`simulate.brownian`](https://jax-md.readthedocs.io/en/main/jax_md.simulate.html?highlight=brownian#jax_md.simulate.brownian) Simulates brownian motion.
- [`minimize.gradient_descent`](https://jax-md.readthedocs.io/en/main/jax_md.minimize.html?highlight=gradient_descent#jax_md.minimize.gradient_descent) Minimizes a system using gradient descent.
- [`minimize.fire_descent`](https://jax-md.readthedocs.io/en/main/jax_md.minimize.html?highlight=fire_descent#jax_md.minimize.fire_descent) Minimizes a system using the fast inertial relaxation engine.

Example:

```python
from jax_md import simulate
temperature = 1.0
dt = 1e-3
init, update = simulate.nvt_nose_hoover(energy_fn, shift_fn, dt, temperature)
state = init(key, R)
for _ in range(100):
  state = update(state)
R = state.position
```

## Spatial Partitioning ([`partition.py`](https://jax-md.readthedocs.io/en/main/jax_md.partition.html))

In many applications, it is useful to construct spatial partitions of particles / objects in a simulation.

We provide the following methods:
- [`partition.cell_list`](https://jax-md.readthedocs.io/en/main/jax_md.partition.html?highlight=cell_list#jax_md.partition.cell_list) Partitions objects (and metadata) into a grid of cells.
- [`partition.neighbor_list`](https://jax-md.readthedocs.io/en/main/jax_md.partition.html?highlight=neighbor_list#jax_md.partition.neighbor_list) Constructs a set of neighbors within some cutoff distance for each object in a simulation.

Cell List Example:
```python
from jax_md import partition

cell_size = 5.0
capacity = 10
cell_list_fn = partition.cell_list(box_size, cell_size, capacity)
cell_list_data = cell_list_fn.allocate(R)
```

Neighbor List Example:
```python
from jax_md import partition

neighbor_list_fn = partition.neighbor_list(displacement_fn, box_size, cell_size)
neighbors = neighbor_list_fn.allocate(R) # Create a new neighbor list.

# Do some simulating....

neighbors = neighbors.update(R)  # Update the neighbor list without resizing.
if neighbors.did_buffer_overflow:  # Couldn't fit all the neighbors into the list.
  neighbors = neighbor_list_fn.allocate(R)  # So create a new neighbor list.
```

There are three different formats of neighbor list supported: `Dense`, `Sparse`, and `OrderedSparse`. `Dense` neighbor lists store neighbors in an `(particle_count, neighbors_per_particle)` array, `Sparse` neighbor lists store neighbors in a `(2, total_neighbors)` array of pairs, `OrderedSparse` neighbor lists are like `Sparse` neighbor lists, but they only store pairs such that `i < j`.
