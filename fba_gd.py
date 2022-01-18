"""Flux Balance Analysis via gradient descent."""
import abc
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
from scipy.sparse import csr_matrix

from reaction_network import ReactionNetwork
from scheme import Molecule, Reaction

ArrayT = Union[np.ndarray, jnp.ndarray]


class ObjectiveComponent(abc.ABC):
    """Abstract base class for components of an objective function to be optimized."""

    @abc.abstractmethod
    def prepare_targets(self, target_values: dict) -> Optional[ArrayT]:
        """Converts a dict of target values into an array, suitable to be passed to residual()."""
        raise NotImplementedError()

    @abc.abstractmethod
    def residual(self, velocities: ArrayT, dm_dt: ArrayT, targets: ArrayT) -> ArrayT:
        """Returns a vector of singular residual values, all to be minimized to achieve the objective."""
        raise NotImplementedError()


class SteadyStateObjective(ObjectiveComponent):
    """Calculates the deviation of the system from steady state, for network intermediates."""

    def __init__(self, network: ReactionNetwork, intermediates: Iterable[Molecule], weight: float = 1.0):
        self.indices = np.array([network.reactant_index(m) for m in intermediates])
        self.weight = weight

    def prepare_targets(self, target_values: dict = None) -> Optional[ArrayT]:
        """SteadyStateObjective does not use solve-time target values; always returns None."""
        return None

    def residual(self, velocities: ArrayT, dm_dt: ArrayT, targets: Optional[ArrayT] = None) -> ArrayT:
        """Returns the subset of dm/dt affecting intermediates, which should all be zero."""
        return dm_dt[self.indices] * self.weight


class VelocityBoundsObjective(ObjectiveComponent):
    """Penalizes reaction velocities outside of specified bounds."""

    def __init__(self, network: ReactionNetwork, bounds: Mapping[Reaction, Tuple[float, float]], weight: float = 1.0):
        """Initializes the objective with defined upper and lower bounds."""
        self.network = network
        self.indices = np.array([network.reaction_index(r) for r in bounds])
        self.bounds = {reaction: (lb, ub) for reaction, (lb, ub) in bounds.items()}
        self.weight = weight

    def prepare_targets(self, target_values: dict = None) -> Optional[ArrayT]:
        """Prepares an array of upper and lower bounds.

        Args:
            target_values: {reaction: (lb, ub)} _overriding_ any bounds specified at initialization.

        Returns:
            2D numpy array with shape (2, #targets). Any reaction missing from target_values (including if
            target_values is None) defaults to the bounds specified on initialization.
        """
        if target_values is not None:
            # Copy initialized bounds and update with target values as specified.
            bounds = dict(self.bounds)
            bounds.update(target_values)
        else:
            # Safe to use initialized bounds without copying
            bounds = self.bounds

        return self.network.reaction_vector(bounds, (-np.inf, np.inf))[self.indices].T

    def residual(self, velocities: ArrayT, dm_dt: ArrayT, targets: ArrayT) -> ArrayT:
        """Returns a vector of numbers, zero within bounds, negative for below lb, or positive for above ub."""
        lb, ub = targets
        shortfall = jnp.minimum(0, velocities[self.indices] - lb)
        excess = jnp.maximum(0, velocities[self.indices] - ub)
        return shortfall + excess


class TargetDmdtObjective(ObjectiveComponent):
    """Calculates the deviation from target rates of change (dm/dt) for specified molecules."""

    def __init__(self, network: ReactionNetwork, target_molecules: Iterable[Molecule], weight: float = 1.0):
        self.network = network
        self.indices = np.array([network.reactant_index(m) for m in target_molecules])
        self.weight = weight

    def prepare_targets(self, target_values: Mapping[str, Any]) -> Optional[ArrayT]:
        """Converts a dict {molecule: dmdt} into a vector of target values."""
        return self.network.reactant_vector(target_values)[self.indices]

    def residual(self, velocities: ArrayT, dm_dt: ArrayT, targets: ArrayT) -> ArrayT:
        """Returns the excess or shortfall of the actual dm/dt vs the target, for all target molecules."""
        return (dm_dt[self.indices] - targets) * self.weight


class TargetVelocityObjective(ObjectiveComponent):
    """Calculates the deviation from target velocities for specified reactions."""

    def __init__(self, network: ReactionNetwork, target_reactions: Iterable[Reaction], weight: float = 1.0):
        self.network = network
        self.indices = np.array([network.reaction_index(r) for r in target_reactions])
        self.weight = weight

    def prepare_targets(self, target_values: dict) -> Optional[ArrayT]:
        """Converts a dict {reaction_id: velocity} into a vector of target values."""
        return self.network.reaction_vector(target_values)[self.indices]

    def residual(self, velocities: ArrayT, dm_dt: ArrayT, targets: ArrayT) -> ArrayT:
        """Returns the excess or shortfall of the actual velocity vs the target, for all target reactions."""
        return (velocities[self.indices] - targets) * self.weight


@dataclass
class FbaResult:
    """Reaction velocities and dm/dt for an FBA solution, with metrics."""
    seed: int
    velocities: Mapping[Reaction, float]
    dm_dt: Mapping[Molecule, float]
    ss_residual: np.ndarray


class GradientDescentFba:
    """Solves an FBA problem with kinetic and/or homeostatic objectives, by gradient descent."""

    def __init__(self,
                 reactions: Iterable[Reaction],
                 exchanges: Iterable[Molecule],
                 target_metabolites: Iterable[str]):
        """Initialize this FBA solver.

        Args:
            reactions: a list of reaction dicts following the knowledge base structure. Expected keys are "reaction id",
                "stoichiometry", "is reversible".
            exchanges: ids of molecules on the boundary, which may flow in or out of the system.
            target_metabolites: ids of molecules with production targets.
        """
        exchanges = set(exchanges)
        target_metabolites = set(target_metabolites)

        # Iterate once through the list of reactions
        network = ReactionNetwork()
        irreversible_reactions = []
        for reaction in reactions:
            network.add_reaction(reaction)
            if not reaction.reversible:
                irreversible_reactions.append(reaction.id)
        self.network = network

        # All FBA problems have a steady-state objective, for all intermediates.
        self._objectives = {}
        self.add_objective("steady-state",
                           SteadyStateObjective(network,
                                                (m for m in network.reactants()
                                                 if m not in exchanges and m not in target_metabolites)))
        # Apply any reversibility constraints with a bounds objective.
        if irreversible_reactions:
            self.add_objective("irreversibility",
                               VelocityBoundsObjective(network,
                                                       {reaction_id: (0, np.inf)
                                                        for reaction_id in irreversible_reactions}))

    def add_objective(self, objective_id: str, objective: ObjectiveComponent):
        self._objectives[objective_id] = objective

    def residuals(self, velocities: ArrayT, objective_targets: Mapping[str, ArrayT]) -> Mapping[str, ArrayT]:
        """Calculates the residual for each component of the overall objective function.

        Args:
            velocities: vector of velocities (rates) for all reactions in the network.
            objective_targets: dict of target value vectors for each objective component. The shape and values of these
                targets depend on the individual objectives. Missing are permitted, if the individual objective accepts
                None.

        Returns:
            A dict of residual vectors, supplied by each objective component.
        """
        dm_dt = self.network.s_matrix @ velocities

        residuals = {}
        for objective_id, objective in self._objectives.items():
            targets = objective_targets.get(objective_id, None)
            residuals[objective_id] = objective.residual(velocities, dm_dt, targets)
        return residuals

    def solve(self,
              objective_targets: Mapping[str, dict],
              initial_velocities: Optional[Mapping[Reaction, float]] = None,
              rng_seed: int = None,
              **kwargs) -> FbaResult:
        """Performs the optimization to solve the specified FBA problem.

        Args:
            objective_targets: {objective_id: {key: value}} for each objective component. The details of these targets
                depend on the individual objectives. Missing targets are permitted, if the individual objective accepts
                None.
            initial_velocities: (optional) {reaction_id: velocity} as a starting point for optimization. For repeated
                solutions with evolving objective targets, starting from the previous solution can improve performance.
                If None, a random starting point is used.
            rng_seed: (optional) seed for the random number generator, when randomizing the starting point. Provided
                as an arg to support reproducibility; if None then a suitable seed is chosen.
            kwargs: Any additional keyword arguments will be passed through to scipy.optimize.least_squares.

        Returns:
            FbaResult containing optimized reaction velocities, and resulting rate of change per metabolite (dm/dt).
        """
        # Set up x0 with or without random variation, and truncate to bounds.
        if initial_velocities is not None:
            x0 = jnp.asarray(self.network.reaction_vector(initial_velocities))
        else:
            # Random starting point.
            if rng_seed is None:
                rng_seed = int(time.time() * 1000)
            num_reactions = self.network.shape[1]
            x0 = jax.random.uniform(jax.random.PRNGKey(rng_seed), (num_reactions,))

        target_values = {}
        for objective_id, objective in self._objectives.items():
            targets = objective.prepare_targets(objective_targets.get(objective_id))
            if targets is not None:
                target_values[objective_id] = jnp.asarray(targets)

        # Overall residual is a flattened vector of the (weighted) residuals of individual objectives.
        def loss(v):
            return jnp.concatenate(list(self.residuals(v, target_values).values()))

        jac = jax.jit(jax.jacfwd(loss))

        # Perform the actual gradient descent, and extract the result.
        soln = scipy.optimize.least_squares(jax.jit(loss), x0, jac=lambda x: csr_matrix(jac(x)), **kwargs)

        # Perform the actual gradient descent, and extract the result.
        dm_dt = self.network.s_matrix @ soln.x
        ss_residual = self._objectives["steady-state"].residual(soln.x, dm_dt, None)
        return FbaResult(seed=rng_seed,
                         velocities=self.network.reaction_values(soln.x),
                         dm_dt=self.network.reactant_values(dm_dt),
                         ss_residual=ss_residual)
