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
from typing import Dict, Iterable, List, Mapping, Optional, Union

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
    km: Mapping[Molecule, float]  # Km of each substrate and product with respect to the reaction
    ka: Mapping[Molecule, float]  # Binding constants of all activators
    ki: Mapping[Molecule, float]  # Binding constants of all inhibitors


class Ligands:
    """Calculates occupancy of designated ligands across a set of reactions.

    A central concept used by Convenience Kinetics is the occupancy, i.e. degree of binding, of various sets of ligands
    (substrates, products, activators, inhibitors) to the enzyme catalyzing each reaction. The paper refers to this as
    'normalized concentration', denoted $\tilde{a}$, and defined as $a_i / K^M_{a_i}$ for all i in concentration vector
    a. Equations expressed with this notation are relatively compact and interpretable.

    This implementation is optimized for uniform vectorized calculations across multiple reactions in a network. The
    number of molecules in a given set can vary for each reaction, leading to a ragged array as the natural
    representation. Here we use constant-width arrays for the molecules and their corresponding binding constants,
    padding as appropriate so as not to affect the final result.
    """
    def __init__(self,
                 network: ReactionNetwork,
                 ligand_lists: Iterable[Iterable[Molecule]],
                 constants: Iterable[Mapping[Molecule, float]]):
        """Initialize the Ligands set for multiple reactions.

        Args:
            network: the reaction network being modeled. Manages mapping between Reaction and Molecule objects and
                corresponding indices in packed arrays.
            ligand_lists: One list of molecules per reaction in the network. If multiple instances of a molecule
                participate in the same reaction in the same role, that molecule is repeated as appropriate. The list
                for any reaction may be empty.
            constants: Defined binding constants for all molecules in a given ligand_list
        """
        self.network = network

        ragged_indices = []
        width = 0
        for ligand_list in ligand_lists:
            indices = [network.reactant_index(ligand) for ligand in ligand_list]
            ragged_indices.append(indices)
            width = max(width, len(indices))

        # -1 as a default index lets us get a default value by appending it to the state vector.
        padded_indices = np.full((len(ragged_indices), width), -1, dtype=int)
        # But keep track of which values are real and which are padded.
        mask = np.zeros((len(ragged_indices), width), dtype=int)
        for i, indices in enumerate(ragged_indices):
            padded_indices[i, :len(indices)] = indices
            mask[i, :len(indices)] = 1

        self.width = width
        self.indices = padded_indices
        self.mask = mask

        # Pack the binding constants into an identically indexed array.
        self.constants = self.pack(constants, default=1.0)

    def pack(self, values: Iterable[Mapping[Molecule, float]], default: float = 0.0) -> np.ndarray:
        """Packs binding constants (or any other values) into an array that aligns with this Ligands set.

        Args:
            values: Values associated with each molecule for each reaction.
            default: The default value used to pad the resulting array.

        Returns:
            An array with shape (#reactions, width) that aligns with self.indices and self.mask.
        """
        packed_values = np.full(self.indices.shape, default)
        for i, (row_indices, row_values) in enumerate(zip(self.indices, values)):
            for j, ligand_index in enumerate(row_indices):
                if self.mask[i, j]:
                    packed_values[i, j] = row_values.get(self.network.reactant(ligand_index), default)
        return packed_values

    def unpack(self, packed_values: ArrayT) -> List[Dict[Molecule, float]]:
        """Unpacks an array of values into a list of dicts indexed by molecule, one per reaction.

        Args:
            packed_values: An array with shape (#reactions, width) that aligns with self.indices and self.mask

        Returns:
            Values associated with each molecule for each reaction. If a given molecule is repeated in the Ligands
            set for a given reaction, only the last value for that molecule is kept.
        """
        values = []
        for row_indices, row_mask, row in zip(self.indices, self.mask, packed_values):
            values.append({self.network.reactant(ligand_index): value
                           for ligand_index, mask_value, value in zip(row_indices, row_mask, row)
                           if mask_value})
        return values

    def occupancy(self, state: ArrayT, constants: Optional[ArrayT] = None, default: float = 1.0) -> jnp.ndarray:
        """Calculates occupancy across this Ligand set, given a state vector.

        Args:
            state: a vector of state (i.e. concentration) values collinear with self.network.reactants().
            constants: may override the intrinsic binding constants defined for this Ligands set on initialization.
            default: The default value used for all padded elements of the array.

        Returns:
            An array of shape (#reactions, width), with values y / k for each state value y and constant k, padded
            with the default as appropriate.
        """
        if constants is None:
            constants = self.constants
        # Appending [default] to the state vector means any index of -1 dereferences to the default value.
        return jnp.append(state, default)[self.indices] / constants


class ConvenienceKinetics:
    """JAX-friendly implementation of Convenience Kinetics.

    Calculations rely heavily on arrays with 3 dimensions: reaction, side (i.e. substrate or product), and reactant.
    For consistency and parallelization we want all dimensions to be constant (not ragged). In general this means
    padding the internal data structures with ones, so that multiplying across each row is unaffected.
    """

    def __init__(self,
                 network: ReactionNetwork,
                 kinetics: Mapping[Reaction, ReactionKinetics]):
        """Constructs a ConvenienceKinetics object.

        Args:
            network: The network of reactions and reactants being modeled.
            kinetics: Kinetic parameters for each Reaction in the network.
        """
        self.network = network
        self.kcats = np.array(
            [[kinetics[reaction].kcat_f, kinetics[reaction].kcat_b] for reaction in network.reactions()])

        # Substrates and products represented by one Ligands set each, with Km's.
        substrates = []
        products = []
        constants = []
        for reaction in network.reactions():
            substrates.append([])
            products.append([])
            for reactant, count in reaction.stoichiometry.items():
                if count < 0:
                    substrates[-1].extend([reactant] * -count)
                else:
                    products[-1].extend([reactant] * count)

            constants.append(kinetics[reaction].km)

        self.substrates = Ligands(network, substrates, constants)
        self.products = Ligands(network, products, constants)

        # Activators and inhibitors represented by one Ligands set each, with Ka's or Ki's respectively.
        activators = []
        kas = []
        inhibitors = []
        kis = []
        for reaction in network.reactions():
            reaction_kinetics = kinetics[reaction]
            activators.append(reaction_kinetics.ka.keys())
            kas.append(reaction_kinetics.ka)
            inhibitors.append(reaction_kinetics.ki.keys())
            kis.append(reaction_kinetics.ki)

        self.activators = Ligands(network, activators, kas)
        self.inhibitors = Ligands(network, inhibitors, kis)

    def reaction_rates(self, state: ArrayT, enzyme_conc: ArrayT) -> ArrayT:
        """Calculates current reaction rates using the convenience kinetics formula.

        Args:
            state: Array of concentration values collinear with network.reactants().
            enzyme_conc: Array of enzyme concentrations collinear with network.reactions().

        Returns:
            1d array of reaction rates, collinear with network.reactions().
        """
        kcats = self.kcats

        # $\tilde{a} = a_i / {km}^a_i for all i; \tilde{b} = b_j / {km}^b_j for all j$, padded with ones as necessary.
        occupancy_s = self.substrates.occupancy(state, default=1)
        occupancy_p = self.products.occupancy(state, default=1)

        # $k_{+}^{cat} \prod_i{\tilde{a}_i} + k_{-}^{cat} \prod_j{\tilde{b}_j}$.
        numerator = kcats[:, 0] * jnp.prod(occupancy_s, axis=-1) + kcats[:, 1] * jnp.prod(occupancy_p, axis=-1)

        # $\prod_i{(1 + \tilde{a}_i)} + \prod_j{(1 + \tilde{b}_j)} - 1$
        denominator = jnp.prod(occupancy_s * self.substrates.mask + 1, axis=-1) + jnp.prod(
            occupancy_p * self.products.mask + 1, axis=-1) - 1

        # Activation: $\prod_i{\frac{a_i}{a_i + K^A_i} = \prod_i{\frac{a_i / K^A_i}{a_i / K^A_i + 1}$
        occupancy_a = self.activators.occupancy(state, default=1)
        activation = jnp.prod(occupancy_a / (occupancy_a * self.activators.mask + 1), axis=-1)
        # Inhibition: $\prod_i{\frac{K^I_i}{a_i + K^I_i} = \prod_i{\frac{1}{a_i / K^I_i + 1}$
        occupancy_i = self.inhibitors.occupancy(state, default=0)
        inhibition = jnp.prod(1 / (occupancy_i + 1), axis=-1)

        return enzyme_conc * activation * inhibition * numerator / denominator

    def dstate_dt(self, state: ArrayT, enzyme_conc: ArrayT) -> ArrayT:
        """Calculates current rate of change per reactant using the convenience kinetics formula.

        Args:
            state: Array of concentration values collinear with network.reactants().
            enzyme_conc: Array of enzyme concentrations collinear with network.reactions().

        Returns:
            1d array of rates of change, collinear with network.reactants().
        """
        return self.network.s_matrix @ self.reaction_rates(state, enzyme_conc)
