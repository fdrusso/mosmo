"""Flux Balance Analysis via gradient descent."""
import abc
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
from scipy.sparse import csr_matrix

from model.reaction_network import ReactionNetwork
from model.core import Molecule, Reaction

ArrayT = Union[np.ndarray, jnp.ndarray]


# Residual aggregation functions - all satisfy Callable[[ArrayT], float].

def l_pt5(residual: ArrayT) -> float:
    """Aggregates a residual vector into a scalar, as sum(sqrt(abs(x))); values contribute sub-linearly to total."""
    return jnp.sum(jnp.sqrt(jnp.abs(residual)))


def l1(residual: ArrayT) -> float:
    """Aggregates a residual vector into a scalar, as sum(abs(x)); values contribute linearly to total."""
    return jnp.sum(jnp.abs(residual))


def l2(residual: ArrayT) -> float:
    """Aggregates a residual vector into a scalar, as sum(square(x)); values contribute super-linearly to total."""
    return jnp.sum(jnp.square(residual))


class Objective(abc.ABC):
    """Superclass for components of a flux optimization objective.

    Individual Objectives can focus on any number of reaction velocities, dM/dt values, or both. The general pattern
    is that given specific velocity and dmdt vectors, an Objective produces a (constant length) vector of residual
    values. These residuals are aggregated to a single weighted value, which contributes to the overall loss function
    of the flux optimization problem.

    Some but not all Objectives have adjustable parameters (e.g. target values or thresholds) that can be changed
    without altering the overall structure of the optimization problem.
    """

    def __init__(self, aggfun: Callable[[ArrayT], float] = l2, weight: float = 1.0):
        """General init for any Objective.

        Args:
            aggfun: the residual aggregation function, typically l_pt5, l1, or l2, but any Callable that fits the
                signature will do
            weight: the weight of this term in the overall objective function.
        """
        self.aggfun = aggfun
        self.weight = weight

    def update_params(self, params: Any):
        """Updates parameters of this objective, if any."""
        pass

    def params(self) -> Optional[ArrayT]:
        """Returns the current values of all adjustable params, suitable to pass to residual() or loss()."""
        return None

    @abc.abstractmethod
    def residual(self, velocities: ArrayT, dmdt: ArrayT, params: Optional[ArrayT]) -> ArrayT:
        """Returns a vector of residual values whose weighted aggregate value is to be minimized."""
        raise NotImplementedError()

    def loss(self, velocities: ArrayT, dmdt: ArrayT, params: Optional[ArrayT]) -> float:
        return self.weight * self.aggfun(self.residual(velocities, dmdt, params))


class SteadyStateObjective(Objective):
    """Penalizes any non-zero dM/dt values for specified intermediates in the reaction network."""

    def __init__(self, network: ReactionNetwork, intermediates: Iterable[Molecule],
                 aggfun: Callable[[ArrayT], float] = l2, weight: float = 1.0):
        super().__init__(aggfun, weight)
        self.network = network
        self.indices = np.array([network.reactant_index(m) for m in intermediates], dtype=np.int32)

    def residual(self, velocities: ArrayT, dmdt: ArrayT, params=None) -> jnp.ndarray:
        """Ignores velocities; returns dM/dt values for all configured intermediates."""
        return dmdt[self.indices]


class IrreversibilityObjective(Objective):
    """Penalizes negative velocity values for irreversible reactions in the network."""

    def __init__(self, network: ReactionNetwork, aggfun: Callable[[ArrayT], float] = l2, weight: float = 1.0):
        super().__init__(aggfun, weight)
        self.network = network
        self.indices = np.array([i for i, reaction in enumerate(network.reactions()) if not reaction.reversible],
                                dtype=np.int32)

    def residual(self, velocities: ArrayT, dmdt: ArrayT, params=None) -> jnp.ndarray:
        """Returns value of any negative velocity, or 0 for positive velocity, for all irreversible reactions."""
        return jnp.minimum(0, velocities[self.indices])


class ProductionObjective(Objective):
    """Penalizes deviation of dM/dt from a target value or range, for select reactants.

    The target value(s) for any reactant can be changed via update_params, although the set of reactants being
    targeted cannot. Targets are specified as either:
    - {target: value} for a specific target value
    - {target: (lb, ub)} for a range of equally acceptable values

    Residuals are the shortfall vs a lower bound or target, or excess vs an upper bound or target.
    """

    def __init__(self,
                 network: ReactionNetwork,
                 targets: Mapping[Molecule, Union[float, Tuple[Optional[float], Optional[float]]]],
                 aggfun: Callable[[ArrayT], float] = l2,
                 weight: float = 1.0):
        super().__init__(aggfun, weight)
        self.network = network
        self.indices = np.array([network.reactant_index(met) for met in targets], dtype=np.int32)
        self.bounds = np.full((self.indices.shape[0], 2), [-np.inf, np.inf]).T
        self.update_params(targets)

    def update_params(self, targets: Mapping[Molecule, Union[float, Tuple[Optional[float], Optional[float]]]]):
        """Updates some or all target dM/dt values."""
        for i, met_idx in enumerate(self.indices):
            met = self.network.reactant(met_idx)
            if met in targets:
                target = targets[met]
                if isinstance(target, float) or isinstance(target, int):
                    target = (target, target)
                self.bounds[0][i] = target[0] if target[0] is not None else -np.inf
                self.bounds[1][i] = target[1] if target[1] is not None else np.inf

    def params(self) -> Optional[ArrayT]:
        """Returns an array of shape (2, #targets) with lower and upper target bounds."""
        return self.bounds

    def residual(self, velocities: ArrayT, dmdt: ArrayT, bounds: ArrayT) -> jnp.ndarray:
        """Calculates shortfall (as a negative) or excess dM/dt for select reactants vs target values or bounds."""
        shortfall = jnp.minimum(0, dmdt[self.indices] - bounds[0])
        excess = jnp.maximum(0, dmdt[self.indices] - bounds[1])
        return shortfall + excess


class VelocityObjective(Objective):
    """Penalizes deviation of velocity from a target value or range, for select reactions.

    The target value(s) for any reaction can be changed via update_params, although the set of reactions being
    targeted cannot. Targets are specified as either:
    - {target: value} for a specific target value
    - {target: (lb, ub)} for a range of equally acceptable values

    Residuals are the shortfall vs a lower bound or target, or excess vs an upper bound or target.
    """

    def __init__(self,
                 network: ReactionNetwork,
                 targets: Mapping[Reaction, Union[float, Tuple[Optional[float], Optional[float]]]],
                 aggfun: Callable[[ArrayT], float] = l2,
                 weight: float = 1.0):
        super().__init__(aggfun, weight)
        self.network = network
        self.indices = np.array([network.reaction_index(rxn) for rxn in targets], dtype=np.int32)
        self.bounds = np.full((self.indices.shape[0], 2), [-np.inf, np.inf]).T
        self.update_params(targets)

    def update_params(self, targets: Mapping[Reaction, Union[float, Tuple[Optional[float], Optional[float]]]]):
        """Updates some or all target velocity values."""
        for i, rxn_idx in enumerate(self.indices):
            rxn = self.network.reaction(rxn_idx)
            if rxn in targets:
                target = targets[rxn]
                if isinstance(target, float) or isinstance(target, int):
                    target = (target, target)
                self.bounds[0][i] = target[0] if target[0] is not None else -np.inf
                self.bounds[1][i] = target[1] if target[1] is not None else np.inf

    def params(self) -> Optional[ArrayT]:
        """Returns an array of shape (2, #targets) with lower and upper target bounds."""
        return self.bounds

    def residual(self, velocities: ArrayT, dmdt: ArrayT, bounds: ArrayT) -> jnp.ndarray:
        """Calculates shortfall (as a negative) or excess velocity for select reactions vs target values or bounds."""
        shortfall = jnp.minimum(0, velocities[self.indices] - bounds[0])
        excess = jnp.maximum(0, velocities[self.indices] - bounds[1])
        return shortfall + excess


@dataclass
class FbaResult:
    """Reaction velocities and dm/dt for an FBA solution, with fitness metric."""
    v0: ArrayT
    velocities: ArrayT
    dmdt: ArrayT
    fit: float


class FbaGd:
    """Defines and solves a Flux Balance Analysis problem via gradient descent.

    The problem is specified via a set of Objective components, each constraining some aspect of the solution. All
    FBA problems include balancing reaction velocities such that network intermediates are at steady state, i.e.
    have a rate of change (dM/dt) of 0.  Any reactions defined as reversible=False should have non-negative velocity
    in the solution. Finally, by default we include a sparsity term to minimize unnecessary flux, e.g. in so-called
    'futile cycles'. Other objectives are provided by the caller, and may take any form that evaluates a potential
    solution in terms of velocities, dM/dt or both.

    For performance, this class uses a JAX jit-compiled function and jacobian, defining the problem structure and
    solution gradient, respectively. Neither changes over the lifetime of an FbaGd instance, although numerical
    parameters of any objective component may be adjusted without restriction between calls to solve(). As an example,
    if a problem is defined as:

        problem = FbaGd(network, intermediates, {'production': ProductionObjective(network, {a: (1.5, 2.3)} )})

    Then the following calls are valid:

        problem.update_params({'production': {a: (0, 1.7)} )
        problem.update_params({'production': {a: 2.0} )
        problem.update_params({'production': {a: None} )

    But not:

        problem.update_params({'production': {b: 5} )
    """

    def __init__(self,
                 network: ReactionNetwork,
                 intermediates: Iterable[Molecule],
                 objectives: Mapping[str, Objective],
                 w_fitness: float = 1e4,
                 w_sparsity: float = 1e-4):
        """Defines the FBA problem to be solved.

        Args:
            network: the reaction network
            intermediates: metabolites (reactants) that are internal to the network, and should be at steady-state
                in any solution
            objectives: named components of the overall objective function to be optimized
            w_fitness: the relative weight of solution fitness terms (steady-state and irreversibility)
            w_sparsity: the relative weight of the solution sparsity term
        """
        self.network = network

        # Fitness and sparsity are universal objectives for FBA
        self.objectives: Dict[str, Objective] = {
            'steady-state': SteadyStateObjective(network, intermediates, weight=w_fitness),
            'irreversibility': IrreversibilityObjective(network, weight=w_fitness),
            'sparsity': VelocityObjective(network,
                                          {r: 0 for r in network.reactions()},
                                          aggfun=l_pt5,
                                          weight=w_sparsity),
        }

        # Additional objectives are defined by the caller
        self.objectives.update(objectives)

        # The loss function takes objective params as explicit arguments so jax.jit will not fold them into constants
        def loss(v, *params):
            dmdt = self.network.s_matrix @ v
            return sum(objective.loss(v, dmdt, p) for objective, p in zip(self.objectives.values(), params))

        # Cache the jitted loss and jacobian functions
        self._loss_jit = jax.jit(loss)
        self._loss_jac = jax.jit(jax.jacfwd(loss))

    def update_params(self, updates):
        for name, params in updates.items():
            self.objectives[name].update_params(params)

    def solve(self, v0: Optional[ArrayT] = None, seed: Optional[jax.random.PRNGKey] = None, **kw_args) -> FbaResult:
        """Solves the FBA problem as currently specified.

        Args:
            v0: a vector of velocities used as a starting point for optimization
            seed: random seed used to generate v0 if none is provided. Ignored if v0 is provided. If neither v0 nor
                seed is provided, a suitable random seed is chosen.
            kw_args: additional keyword args passed through to the underlying scipy.optimize.minimize()

        Returns:
            FbaResult specifying the solution.
        """
        if v0 is None:
            if seed is None:
                seed = jax.random.PRNGKey(int(time.time() * 1000))
            v0 = jax.random.normal(seed, self.network.shape[1:])

        params = tuple(objective.params() for objective in self.objectives.values())
        soln = scipy.optimize.minimize(fun=self._loss_jit, args=params, x0=v0, jac=self._loss_jac, **kw_args)

        dmdt = self.network.s_matrix @ soln.x
        return FbaResult(v0=v0,
                         velocities=soln.x,
                         dmdt=dmdt,
                         fit=sum(float(self.objectives[name].loss(soln.x, dmdt, None))
                                 for name in ['steady-state', 'irreversibility']))