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

This implementation uses pure JAX vector calculations, relying heavily on arrays with one row per reaction. For
consistency and parallelization we want all rows to be constant-width (not ragged). In general this means padding
internal data structures with ones, so that multiplying across each row is unaffected.
"""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Union

import jax.numpy as jnp
import numpy as np

from model.core import Molecule, Reaction
from model.reaction_network import ReactionNetwork

ArrayT = Union[np.ndarray, jnp.ndarray]
R = 8.314463e-3  # or more precisely, exactly 8.31446261815324e-3 kilojoule per kelvin per mole
RT = R * 298.15  # T = 298.15 K = 25 C


class Ligands:
    """Facilitates calculating binding of designated ligands across a set of reactions.

    A central concept used by Convenience Kinetics is the concentrations of various sets of ligands (substrates,
    products, activators, inhibitors) relative to their binding constants to the enzyme catalyzing each reaction. The
    paper refers to this as 'normalized concentration', denoted $\tilde{a}$, and defined as $a_i / K^M_{a_i}$ for all i
    in concentration vector a. Equations expressed with this notation are relatively compact and interpretable.

    This implementation is optimized for uniform vectorized calculations across multiple reactions in a network. The
    number of molecules in a given set can vary for each reaction, leading to a ragged array as the natural
    representation. Here we use constant-width arrays for the molecules and their corresponding binding constants,
    padding as appropriate so as not to affect the final result.
    """

    def __init__(self,
                 network: ReactionNetwork,
                 ligand_lists: Iterable[Iterable[Molecule]]):
        """Initialize the Ligands set for multiple reactions.

        Args:
            network: the reaction network being modeled. Manages mapping between Reaction and Molecule objects and
                corresponding indices in packed arrays.
            ligand_lists: One list of molecules per reaction in the network. If multiple instances of a molecule
                participate in the same reaction in the same role, that molecule is repeated as appropriate. The list
                for any reaction may be empty.
        """
        self.network = network

        ragged_indices = []
        width = 0
        for ligand_list in ligand_lists:
            indices = [network.reactant_index(ligand) for ligand in ligand_list]
            ragged_indices.append(indices)
            width = max(width, len(indices))

        # -1 as a default index lets us supply a padded value at calculation time.
        padded_indices = np.full((len(ragged_indices), width), -1, dtype=int)
        # But keep track of which values are real and which are padded.
        mask = np.zeros((len(ragged_indices), width), dtype=int)
        for i, indices in enumerate(ragged_indices):
            padded_indices[i, :len(indices)] = indices
            mask[i, :len(indices)] = 1

        self.width = width
        self.indices = padded_indices
        self.mask = mask

    def pack(self,
             values: Iterable[Mapping[Molecule, float]],
             default: float = 0.0,
             padding: float = 0.0) -> np.ndarray:
        """Packs binding constants (or any other values) into an array that aligns with this Ligands set.

        Args:
            values: Values associated with each molecule for each reaction.
            default: The default value for molecules missing from `values`.
            padding: The value used to pad the resulting array to constant width.

        Returns:
            An array with shape (#reactions, width) that aligns with indices and mask.
        """
        packed_values = np.full(self.indices.shape, padding, dtype=float)
        for i, (row_indices, row_values) in enumerate(zip(self.indices, values)):
            for j, ligand_index in enumerate(row_indices):
                if self.mask[i, j]:
                    packed_values[i, j] = row_values.get(self.network.reactant(ligand_index), default)
        return packed_values

    def unpack(self, packed_values: ArrayT) -> List[Dict[Molecule, float]]:
        """Unpacks an array of values into a list of dicts indexed by molecule, one per reaction.

        Args:
            packed_values: An array with shape (#reactions, width) that aligns with indices and mask

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

    def tilde(self, state: ArrayT, constants: ArrayT, padding: float = 1.0) -> jnp.ndarray:
        """Concentration relative to Km across this Ligand set, given a state vector.

        Km is the ligand concentration where enzyme binding is half-saturated. This method expresses absolute ligand
        concentrations relative to each Km, denoted in the Convenience Kinetics paper with a tilde over the
        concentration vector. The actual degree of binding for ligand a is $\tilde{a} / (1 + \tilde{a}$.

        Args:
            state: a vector of state (i.e. concentration) values collinear with self.network.reactants().
            constants: array with shape (#reactions, width) that aligns with indices and mask.
            padding: The value used for all padded elements of the array.

        Returns:
            An array of shape (#reactions, width), with values y / k for each state value y and constant k, padded
            with the default as appropriate.
        """
        # Appending [default] to the state vector means any index of -1 dereferences to the default value.
        return jnp.append(state, padding)[self.indices] / constants


@dataclass
class ReactionKinetics:
    """Dataclass to hold kinetic parameters associated with an enzymatic Reaction."""
    reaction: Reaction  # The reaction
    kcat_f: float  # kcat of the forward reaction
    kcat_b: float  # kcat of the reverse (back) reaction
    km: Mapping[Molecule, float]  # Km of each substrate and product with respect to the reaction
    ka: Mapping[Molecule, float]  # Binding constants of all activators
    ki: Mapping[Molecule, float]  # Binding constants of all inhibitors


@dataclass
class PackedNetworkKinetics:
    """Holds calculation-ready arrays of kinetic constants."""
    kcats_f: ArrayT
    kcats_b: ArrayT
    kms_s: ArrayT
    kms_p: ArrayT
    kas: ArrayT
    kis: ArrayT


class ConvenienceKinetics:
    """JAX-friendly implementation of Convenience Kinetics."""

    def __init__(self,
                 network: ReactionNetwork,
                 kinetics: Mapping[Reaction, ReactionKinetics]):
        """Constructs a ConvenienceKinetics object.

        Args:
            network: The network of reactions and reactants being modeled.
            kinetics: Kinetic parameters for each Reaction in the network.
        """
        self.network = network

        # Substrates, products, activators, and inhibitors are represented by one Ligands set each.
        substrates = []
        products = []
        activators = []
        inhibitors = []
        for reaction in network.reactions():
            substrates.append([])
            products.append([])
            for reactant, count in reaction.stoichiometry.items():
                if count < 0:
                    substrates[-1].extend([reactant] * -count)
                else:
                    products[-1].extend([reactant] * count)

            reaction_kinetics = kinetics[reaction]
            activators.append(reaction_kinetics.ka.keys())
            inhibitors.append(reaction_kinetics.ki.keys())

        self.substrates = Ligands(network, substrates)
        self.products = Ligands(network, products)
        self.activators = Ligands(network, activators)
        self.inhibitors = Ligands(network, inhibitors)

        # Cache calculation-ready arrays of kinetic constants.
        self.kinetics = self.pack(kinetics)

    def pack(self, kinetics: Mapping[Reaction, ReactionKinetics]) -> PackedNetworkKinetics:
        """Generates calculation-ready arrays of kinetic constants from ReactionKinetics per Reaction."""
        kcats_f = []
        kcats_b = []
        kms = []
        kas = []
        kis = []
        for reaction in self.network.reactions():
            reaction_kinetics = kinetics[reaction]
            kcats_f.append(reaction_kinetics.kcat_f)
            kcats_b.append(reaction_kinetics.kcat_b)
            kms.append(reaction_kinetics.km)
            kas.append(reaction_kinetics.ka)
            kis.append(reaction_kinetics.ki)

        return PackedNetworkKinetics(
            kcats_f=np.array(kcats_f, dtype=float),
            kcats_b=np.array(kcats_b, dtype=float),
            kms_s=self.substrates.pack(kms, default=0.1, padding=1),
            kms_p=self.products.pack(kms, default=0.1, padding=1),
            kas=self.activators.pack(kas, default=1e-7, padding=1),
            kis=self.inhibitors.pack(kis, default=1e5, padding=1))

    def reaction_rates(self,
                       state: ArrayT,
                       enzyme_conc: ArrayT,
                       kinetics: Optional[PackedNetworkKinetics] = None) -> ArrayT:
        """Calculates current reaction rates using the convenience kinetics formula.

        Args:
            state: Array of concentration values collinear with network.reactants().
            enzyme_conc: Array of enzyme concentrations collinear with network.reactions().
            kinetics: May override intrinsic kinetics defined for this network.

        Returns:
            1d array of reaction rates, collinear with network.reactions().
        """
        if kinetics is None:
            kinetics = self.kinetics

        # $\tilde{a} = a_i / {km}^a_i for all i; \tilde{b} = b_j / {km}^b_j for all j$, padded with ones as necessary.
        tilde_s = self.substrates.tilde(state, kinetics.kms_s, padding=1)
        tilde_p = self.products.tilde(state, kinetics.kms_p, padding=1)

        # $k_{+}^{cat} \prod_i{\tilde{a}_i} + k_{-}^{cat} \prod_j{\tilde{b}_j}$.
        numerator = kinetics.kcats_f * jnp.prod(tilde_s, axis=-1) - kinetics.kcats_b * jnp.prod(tilde_p, axis=-1)

        # $\prod_i{(1 + \tilde{a}_i)} + \prod_j{(1 + \tilde{b}_j)} - 1$
        # tilde_y + mask = (1 + tilde_y) for real values, and 1 for padded values.
        denominator = jnp.prod(tilde_s + self.substrates.mask, axis=-1) + jnp.prod(
            tilde_p + self.products.mask, axis=-1) - 1

        # Activation: $\prod_i{\frac{a_i}{a_i + K^A_i} = \prod_i{\frac{a_i / K^A_i}{a_i / K^A_i + 1}$
        tilde_a = self.activators.tilde(state, kinetics.kas, padding=1)
        activation = jnp.prod(tilde_a / (tilde_a + self.activators.mask), axis=-1)
        # Inhibition: $\prod_i{\frac{K^I_i}{a_i + K^I_i} = \prod_i{\frac{1}{a_i / K^I_i + 1}$
        # Since tilde_i is padded with zero already, we can use (tilde_i + 1) directly.
        tilde_i = self.inhibitors.tilde(state, kinetics.kis, padding=0)
        inhibition = jnp.prod(1 / (tilde_i + 1), axis=-1)

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


def generate_kcats(kms_s: ArrayT, kms_p: ArrayT, kvs: ArrayT, dgrs: ArrayT) -> jnp.ndarray:
    """Generates thermodynamically consistent forward and back kcats, given ΔG, Km's and a velocity constant.

    Base on the Haldane relationship, -ΔG/RT = ln(K) = ln(kcat+) - ln(kcat-) + sum(n ln(Km)).

    Args:
        kms_s: array of substrate Km values (mM), with shape (#rxns, max(#substrates)), padded with ones.
        kms_p: array of product Km values (mM), with shape (#rxns, max(#products)), padded with ones.
        kvs: array of velocity constants, with shape (#rxns,)
        dgrs: reaction ΔGs (kilojoule / mole, mM standard), array of shape (#rxns,).

    Returns:
        An array of shape (#rxns, 2), with forward and back kcats per reaction.
    """
    # $ln(k_{cat}^{+}) - ln(k_{cat}^{-}) = -\frac{\Delta{G}_r}{RT} - \sum_i{(n_i ln({K_M}_i))}$
    diffs = -dgrs / RT + jnp.sum(jnp.log(kms_s), axis=-1) - jnp.sum(jnp.log(kms_p), axis=-1)
    # e^(kvs +/- diffs/2)
    return jnp.exp(kvs + diffs * jnp.array([[+0.5], [-0.5]])).T
