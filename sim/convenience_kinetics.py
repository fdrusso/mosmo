"""Generalized biochemical reaction and network kinetics.

Implemented from Liebermeister, W., Klipp, E. Bringing metabolic networks to life: convenience rate law and
thermodynamic constraints. Theor Biol Med Model 3, 41 (2006). https://doi.org/10.1186/1742-4682-3-41

This paper defines a general-purpose rate law based on reasonable, if not necessarily universal, assumptions. The
general formula (using notation consistent with the authors') is
$$
v(a,b) = E
\frac{
    k_{+}^{cat} \prod\limits_i \tilde{a}_i + k_{-}^{cat} \prod\limits_j \tilde{b}_j
}{
    \prod\limits_i (1 + \tilde{a}_i) + \prod\limits_j (1 + \tilde{b}_j) - 1
}
$$

This implementation uses pure JAX vector calculations.
"""
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np

from model.core import Molecule, Reaction
from model.reaction_network import ReactionNetwork

ArrayT = Union[np.ndarray, jnp.ndarray]


@dataclass
class ReactionKinetics:
    """Dataclass to hold kinetic parameters associated with an enzymatic Reaction."""
    reaction: Reaction  # The reaction
    kcat_f: float  # kcat of the forward reaction
    kcat_r: float  # kcat of the reverse reaction
    km: Mapping[Molecule, float]  # Km of each substrate and product with respect to the reaction.


class ConvenienceKinetics:
    """JAX-friendly implementation of Convenience Kinetics.

    Calculations rely heavily on arrays with 3 dimensions: reaction, side (i.e. substrate or product), and reactant.
    For consistency and parallelization we want all dimensions to be constant (not ragged). In general this means
    padding the internal data structures with ones, so that multiplying across each row is unaffected.
    """

    def __init__(self,
                 network: ReactionNetwork,
                 kinetics: Optional[Mapping[Reaction, ReactionKinetics]] = None):
        """Constructs a ConvenienceKinetics object.

        Args:
            network: The network of reactions and reactants being modeled.
            kinetics: Kinetic parameters for each Reaction in the network. If not supplied here, kinetic data must be
                supplied when doing the actual calculations, via reaction_rates or dstate_dt. Even if supplied here,
                these parameters may be overridden at calculation time.
        """
        self.network = network

        # Build up a list of substrate and product indices for each reaction.
        width = 0
        ragged_indices = []
        for reaction in network.reactions():
            reaction_indices = [[], []]  # [substrates, products]
            for reactant, count in reaction.stoichiometry.items():
                idx = network.reactant_index(reactant)
                if count < 0:
                    reaction_indices[0].extend([idx] * -count)
                else:
                    reaction_indices[1].extend([idx] * count)

            width = max(width, len(reaction_indices[0]), len(reaction_indices[1]))
            ragged_indices.append(reaction_indices)

        # Build a regularized array of indices, padded with -1, and a corresponding mask padded with 0.
        indices = -np.ones((network.shape[1], 2, width), dtype=int)
        mask = np.zeros((network.shape[1], 2, width), dtype=int)
        for i, reaction_indices in enumerate(ragged_indices):
            indices[i, 0, :len(reaction_indices[0])] = reaction_indices[0]
            indices[i, 1, :len(reaction_indices[1])] = reaction_indices[1]
            mask[i, 0, :len(reaction_indices[0])] = 1
            mask[i, 1, :len(reaction_indices[1])] = 1

        self.width_ = width
        self.indices_ = indices
        self.mask_ = mask

        # Save kinetic parameters for each reaction, if given
        if kinetics is not None:
            self.kcats_, self.kms_ = self.param_arrays(kinetics)
        else:
            self.kcats_ = None
            self.kms_ = None

    def param_arrays(self, kinetics: Mapping[Reaction, ReactionKinetics]) -> Tuple[np.ndarray, np.ndarray]:
        """Processes ReactionKinetics structure into parameter arrays used for rate calculations."""
        kcats = np.zeros((self.network.shape[1], 2))
        kms = np.ones((self.network.shape[1], 2, self.width_))

        # indices_ first dimension is collinear with network.reactions
        for i, reaction_indices in enumerate(self.indices_):
            # Require that each reaction's kinetics are included, i.e. allow this to throw an error if not.
            reaction_kinetics = kinetics[self.network.reaction(i)]
            kcats[i, 0] = reaction_kinetics.kcat_f
            kcats[i, 1] = reaction_kinetics.kcat_r

            # Use reaction_indices as the source of truth for what km value belongs where.
            for j, side in enumerate(reaction_indices):
                for k, reactant_idx in enumerate(side):
                    if reactant_idx >= 0:
                        kms[i, j, k] = reaction_kinetics.km.get(self.network.reactant(reactant_idx), 1)
        return kcats, kms

    def reaction_rates(self,
                       state: ArrayT,
                       enzyme_conc: ArrayT,
                       kcats: Optional[ArrayT] = None,
                       kms: Optional[ArrayT] = None) -> ArrayT:
        """Calculates current reaction rates using the convenience kinetics formula.

        Args:
            state: Array of concentration values collinear with network.reactants().
            enzyme_conc: Array of enzyme concentrations collinear with network.reactions().
            kcats: Array with shape (#reactions, 2), as returned from param_arrays(). Overrides any kcat values
                configured on construction.
            kms: Array with shape (#reactions, 2, max(#reactants)), as returned from param_arrays(). Overrides any km
                values configured on construction.

        Returns:
            1d array of reaction rates, collinear with network.reactions().
        """
        # Use kinetic parameters as supplied, or fall back to configured intrinsic values
        if kcats is None:
            kcats = self.kcats_
        if kms is None:
            kms = self.kms_

        # $\tilde{a} = a_i / {km}^a_i for all i; \tilde{b} = b_j / {km}^b_j for all j$, padded with ones as necessary.
        # Appending [1] to the state vector means any index of -1 translates to unity, i.e. a no-op for multiplication.
        state_norm = jnp.append(state, 1)[self.indices_] / kms

        # $k_{+}^{cat} \prod_i{\tilde{a}_i} + k_{-}^{cat} \prod_j{\tilde{b}_j}$.
        numerator = jnp.sum(kcats * jnp.prod(state_norm, axis=-1), axis=-1)

        # $\prod_i{(1 + \tilde{a}_i)} + \prod_j{(1 + \tilde{b}_j)} - 1$
        # state_norm + mask means (1 + \tilde{a}_i) for all real values, and 1 (i.e. a no-op for multiplication).
        # for all padded values.
        denominator = jnp.sum(jnp.prod(state_norm + self.mask_, axis=-1), axis=-1) - 1

        return enzyme_conc * numerator / denominator

    def dstate_dt(self,
                  state: ArrayT,
                  enzyme_conc: ArrayT,
                  kcats: Optional[ArrayT] = None,
                  kms: Optional[ArrayT] = None) -> ArrayT:
        """Calculates current rate of change per reactant using the convenience kinetics formula.

        Args:
            state: Array of concentration values collinear with network.reactants().
            enzyme_conc: Array of enzyme concentrations collinear with network.reactions().
            kcats: Array with shape (#reactions, 2), as returned from param_arrays(). Overrides any kcat values
                configured on construction.
            kms: Array with shape (#reactions, 2, max(#reactants)), as returned from param_arrays(). Overrides any km
                values configured on construction.

        Returns:
            1d array of rates of change, collinear with network.reactants().
        """
        return self.network.s_matrix @ self.reaction_rates(state, enzyme_conc, kcats, kms)
