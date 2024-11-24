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
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Set, Union

import jax.numpy as jnp
import numpy as np

from mosmo.model import Molecule, Reaction, Pathway

ArrayT = Union[np.ndarray, jnp.ndarray]
ParamT = Union[float, ArrayT]
R = 8.314463e-3  # or more precisely, exactly 8.31446261815324e-3 kilojoule per kelvin per mole


class Ligands:
    """Maps a collection of molecules per reaction within a Pathway onto a constant-width array.

    Conceptually, represents molecules in the same role (e.g. substrates or inhibitors) across all reactions in a given
    network. Functionally, maps positions in an array layout to specific (reaction, molecule) pairs, to facilitate
    uniform vectorized calculations. Since the number of molecules in a given role may vary from reaction to reaction,
    we pad the array to constant width with a value that does not affect calculations (i.e. 0 for addition or 1 for
    multiplication).
    """

    def __init__(self,
                 network: Pathway,
                 reaction_ligands: Mapping[Reaction, Iterable[Molecule]]):
        """Initialize the Ligands set for multiple reactions.

        Args:
            network: the reaction network being modeled.
            reaction_ligands: One list of molecules per reaction in the network. If multiple instances of a molecule
                participate in the same reaction in the same role, that molecule is repeated as appropriate. The list
                for any reaction may be empty, or the reaction itself may be absent.
        """
        self.network = network

        ragged_indices = []
        width = 0
        for reaction in network.reactions:
            indices = [network.molecules.index_of(ligand) for ligand in reaction_ligands.get(reaction, [])]
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

    def pack_values(self,
                    values: Mapping[Reaction, Mapping[Molecule, ParamT]],
                    default: ParamT = 0.0,
                    padding: ParamT = 0.0) -> jnp.ndarray:
        """Packs values per (reaction, molecule) pair into an array that aligns with this Ligands set.

        Intended for values that vary for a given molecule across reactions, such as Km. The values may be scalars or
        arrays, depending on the structure of the intended calculations. All such values must have the same shape.

        Args:
            values: Values associated with each molecule for each reaction.
            default: The default for molecules that are validly part of the Ligands set, but missing from `values`.
            padding: The value used to pad the resulting array to constant width.

        Returns:
            An array with shape (#reactions, width) that aligns with indices and mask.
        """
        packed_values = []
        for i, reaction in enumerate(self.network.reactions):
            row_values = values.get(reaction, {})
            row = []
            for j, ligand_index in enumerate(self.indices[i]):
                if self.mask[i, j]:
                    row.append(row_values.get(self.network.molecules[ligand_index], default))
                else:
                    row.append(padding)
            packed_values.append(row)

        # Put the (reaction, molecule) axes at the end to support broadcasting. For scalar values this is a no-op.
        return jnp.moveaxis(jnp.array(packed_values), (0, 1), (-2, -1))

    def unpack_values(self, packed_values: ArrayT) -> Dict[Reaction, Dict[Molecule, ParamT]]:
        """Unpacks an array of values into a structure indexed by reaction and  molecule.

        Args:
            packed_values: An array with shape (#reactions, width) that aligns with indices and mask

        Returns:
            Values associated with each molecule for each reaction. If a given molecule is repeated in the Ligands
            set for a given reaction, only one value for that molecule is kept.
        """
        # Put the (reaction, molecule) axes at the front for indexing. For an array of scalar values this is a no-op.
        _packed_values = jnp.moveaxis(packed_values, (-2, -1), (0, 1))
        values = {}
        for i, reaction in enumerate(self.network.reactions):
            row_values = {}
            for j, ligand_index in enumerate(self.indices[i]):
                if self.mask[i, j]:
                    row_values[self.network.molecules[ligand_index]] = packed_values[i, j]
            values[reaction] = row_values
        return values

    def map_state(self, state: ArrayT, padding: float = 1.0) -> jnp.ndarray:
        """Maps a state vector into an array that aligns or broadcasts with others generated by this Ligands set.

        Intended for values that are consistent for a molecule across all reaction, such as the current concentration.
        This method works equally for a 1d state vector, or for an arbitrary array of state vectors, in order to support
        efficient vectorized or ensemble calculations.

        Args:
            state: A vector of state (i.e. concentration) values collinear with network molecules, or a vector of such
                state vectors.
            padding: The value used for all padded elements of the array, generally 1 for multiplication operations,
                or 0 for addition.

        Returns:
            An array with shape (..., #reactions, width), depending on the shape of state. The last two axes align
            with this Ligands set's indices and mask.
        """
        # These operations work equally well for arrays of any dimension. Appending `padding` before indexing lets
        # the index -1 map to the padded value.
        _state = jnp.append(state, jnp.full(state.shape[:-1] + (1,), padding), axis=-1)
        return _state[..., self.indices]


@dataclass
class ReactionKinetics:
    """Dataclass to hold kinetic parameters associated with an enzymatic Reaction."""
    kcat_f: ParamT  # kcat(s) of the forward reaction
    kcat_b: ParamT  # kcat(s) of the reverse (back) reaction
    km: Mapping[Molecule, ParamT]  # Km(s) of each substrate and product with respect to the reaction
    ka: Mapping[Molecule, ParamT]  # Binding constants of all activators
    ki: Mapping[Molecule, ParamT]  # Binding constants of all inhibitors

    @staticmethod
    def thermo_consistent(
            reaction: Reaction,
            delta_g: ParamT,
            km: Optional[Mapping[Molecule, ParamT]] = None,
            kv: Optional[ParamT] = None,
            kcat_f: Optional[ParamT] = None,
            kcat_b: Optional[ParamT] = None,
            ka: Optional[Mapping[Molecule, ParamT]] = None,
            ki: Optional[Mapping[Molecule, ParamT]] = None,
            default_km: float = 0.1,
            temperature: float = 298.15,
            ignore: Optional[Set[Molecule]] = None,
    ):
        """Generates thermodynamically consistent kinetics, given ΔG, Kms, and velocity constant.

        Calculates forward and back kcat values based on the Haldane relationship:
            -ΔG/RT = ln(K) = ln(kcat+) - ln(kcat-) + sum(n ln(Km))

        The calculation may override supplied values for kcat_f and/or kcat_b to maintain thermodynamic consistency.

        Any of the kinetic parameters may be provided as scalars, or as arrays of values for ensemble models. The only
        requirement is that all such arrays must be compatible according to numpy broadcast rules.

        Args:
            reaction: the reaction
            delta_g: standard ΔG of the reaction
            km: Km of each molecule with respect to the reaction, where known
            kv: velocity constant, a measure of catalytic efficiency. If not supplied, `kv` will be calculated from
                supplied values for `kcat_f` and/or `kcat_b`, or fall back to a reasonable default.
            kcat_f: kcat of the forward reaction
            kcat_b: kcat of the back reaction
            ka: activators and corresponding Ka, if any
            ki: inhibitors and correspond Ki, if any
            default_km: Km assumed for any molecules missing from `km`
            temperature: the temperature used for thermodynamic calculations. Defaults to standard room temperature
            ignore: Molecules to ignore for the purpose of calculating kinetics. Typically this includes e.g. water and
                protons, not because they are irrelevant, but because their effect on kinetics cannot be differentiated
                under buffered aqueous reaction conditions.

        Returns:
            A complete and thermodynamically consistent ReactionKinetics object.
        """
        # All operations work for scalars, or any arrays that broadcast together.
        RT = R * temperature
        km = km or {}
        ignore = ignore or set()
        sum_ln_km = sum(
            count * np.log(km.get(molecule, default_km))
            for molecule, count in reaction.stoichiometry.items()
            if molecule not in ignore
        )
        diff = -delta_g / RT + sum_ln_km

        # Decision tree to calculate velocity constant(s) while respecting any passed-in kcats.
        if kv is None:
            if kcat_f is not None and kcat_b is not None:
                kv = (np.log(kcat_f) + np.log(kcat_b)) / 2
            elif kcat_f is not None:
                kv = np.log(kcat_f) - diff / 2
            elif kcat_b is not None:
                kv = np.log(kcat_b) + diff / 2
            else:
                kv = 0.

        return ReactionKinetics(
            kcat_f=np.exp(kv + diff / 2),
            kcat_b=np.exp(kv - diff / 2),
            km={
                molecule: km.get(molecule, default_km)
                for molecule in reaction.stoichiometry
                if molecule not in ignore
            },
            ka=dict(ka or {}),
            ki=dict(ki or {}),
        )


# Define an empty ReactionKinetics object for missing data.
ReactionKinetics.NONE = ReactionKinetics(kcat_f=0., kcat_b=0., km={}, ka={}, ki={})


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
                 network: Pathway,
                 reaction_kinetics: Mapping[Reaction, ReactionKinetics],
                 ignore: Optional[Iterable[Molecule]] = None):
        """Constructs a ConvenienceKinetics object.

        Args:
            network: The network of reactions and molecules being modeled.
            reaction_kinetics: Kinetic parameters for each Reaction in the network.
            ignore: Molecules to ignore for the purpose of calculating kinetics. Typically this includes e.g. water and
                protons, not because they are irrelevant, but because their effect on kinetics cannot be differentiated
                under buffered aqueous reaction conditions.
        """
        self.network = network
        self.ignore = set()
        if ignore is not None:
            self.ignore.update(ignore)

        # Substrates, products, activators, and inhibitors are represented by one Ligands set each.
        substrates = defaultdict(list)
        products = defaultdict(list)
        activators = defaultdict(list)
        inhibitors = defaultdict(list)
        for reaction, kinetics in reaction_kinetics.items():
            for molecule, count in reaction.stoichiometry.items():
                if molecule not in self.ignore:
                    if count < 0:
                        substrates[reaction].extend([molecule] * -count)
                    else:
                        products[reaction].extend([molecule] * count)

            activators[reaction].extend(kinetics.ka.keys())
            inhibitors[reaction].extend(kinetics.ki.keys())

        self.substrates = Ligands(network, substrates)
        self.products = Ligands(network, products)
        self.activators = Ligands(network, activators)
        self.inhibitors = Ligands(network, inhibitors)

        # Cache calculation-ready arrays of kinetic parameters.
        self.kparms = self.pack_kinetics(reaction_kinetics)

    def pack_kinetics(self, reaction_kinetics: Mapping[Reaction, ReactionKinetics]) -> PackedNetworkKinetics:
        """Generates calculation-ready arrays of kinetic constants from ReactionKinetics per Reaction."""
        kcats_f = []
        kcats_b = []
        kms = {}
        kas = {}
        kis = {}
        for reaction in self.network.reactions:
            kinetics = reaction_kinetics.get(reaction, ReactionKinetics.NONE)
            kcats_f.append(kinetics.kcat_f)
            kcats_b.append(kinetics.kcat_b)
            kms[reaction] = kinetics.km
            kas[reaction] = kinetics.ka
            kis[reaction] = kinetics.ki

        return PackedNetworkKinetics(
            kcats_f=np.array(kcats_f, dtype=float),
            kcats_b=np.array(kcats_b, dtype=float),
            kms_s=self.substrates.pack_values(kms, default=0.1, padding=1),
            kms_p=self.products.pack_values(kms, default=0.1, padding=1),
            kas=self.activators.pack_values(kas, default=1e-7, padding=1),
            kis=self.inhibitors.pack_values(kis, default=1e5, padding=1))

    def unpack_kinetics(self, kparms: Optional[PackedNetworkKinetics] = None) -> Mapping[Reaction, ReactionKinetics]:
        """Converts PackedKinetics back to more easily accessible ReactionKinetics."""
        kparms = kparms or self.kparms
        kms_s = self.substrates.unpack_values(kparms.kms_s)
        kms_p = self.products.unpack_values(kparms.kms_p)
        kas = self.activators.unpack_values(kparms.kas)
        kis = self.inhibitors.unpack_values(kparms.kis)

        reaction_kinetics = {}
        for i, reaction in enumerate(self.network.reactions):
            # km=kms_s[i] | kms_p[i], but not supported in current version of python
            km = {}
            km.update(kms_s[reaction])
            km.update(kms_p[reaction])
            reaction_kinetics[reaction] = ReactionKinetics(
                kcat_f=kparms.kcats_f[i],
                kcat_b=kparms.kcats_b[i],
                km=km,
                ka=kas[reaction],
                ki=kis[reaction],
            )
        return reaction_kinetics

    def adjust_kinetics(self, dgrs: ArrayT, kvs: ArrayT, temperature=298.15):
        """Generates thermodynamically consistent kcats, given ΔG plus a velocity constant per reaction.

        Based on the Haldane relationship, -ΔG/RT = ln(K) = ln(kcat+) - ln(kcat-) + sum(n ln(Km)).

        Args:
            dgrs: reaction ΔGs (kilojoule / mole, mM standard), array of shape (#rxns,).
            kvs: array of velocity constants, with shape (#rxns,).
            temperature: the temperature used in thermodynamic calculations. Defaults to 295.15 (25°C).
        """
        RT = R * temperature

        # $ln(k_{cat}^{+}) - ln(k_{cat}^{-}) = -\frac{\Delta{G}_r}{RT} - \sum_i{(n_i ln({K_M}_i))}$
        ln_km_s = jnp.log(self.kparms.kms_s) * self.substrates.mask
        ln_km_p = jnp.log(self.kparms.kms_p) * self.products.mask
        diffs = -dgrs / RT + jnp.sum(ln_km_s, axis=-1) - jnp.sum(ln_km_p, axis=-1)

        # e^(kvs +/- diffs/2)
        kcats = jnp.exp(kvs + diffs * jnp.array([[+0.5], [-0.5]]))
        self.kparms.kcats_f = kcats[0]
        self.kparms.kcats_b = kcats[1]

    def reaction_rates(self,
                       state: ArrayT,
                       enzyme_conc: ArrayT,
                       kparms: Optional[PackedNetworkKinetics] = None) -> ArrayT:
        """Calculates current reaction rates using the convenience kinetics formula.

        Args:
            state: A vector of state (i.e. concentration) values collinear with network molecules, or a vector of such
                state vectors.
            enzyme_conc: Array of enzyme concentrations collinear with network.reactions().
            kparms: May override intrinsic kinetics defined for this network.

        Returns:
            Array with rates for all reactions in the network. Its shape is the result of broadcast operations among
            the inputs `state` and `enzyme_conc`, plus any structure in `kparms`. For 1d inputs, the result has shape
            (#reactions, ).
        """
        kparms = kparms or self.kparms

        # $\tilde{a} = a_i / {km}^a_i for all i; \tilde{b} = b_j / {km}^b_j for all j$, padded with ones as necessary.
        tilde_s = self.substrates.map_state(state, padding=1.) / kparms.kms_s
        tilde_p = self.products.map_state(state, padding=1.) / kparms.kms_p

        # $k_{+}^{cat} \prod_i{\tilde{a}_i} + k_{-}^{cat} \prod_j{\tilde{b}_j}$.
        numerator = kparms.kcats_f * jnp.prod(tilde_s, axis=-1) - kparms.kcats_b * jnp.prod(tilde_p, axis=-1)

        # $\prod_i{(1 + \tilde{a}_i)} + \prod_j{(1 + \tilde{b}_j)} - 1$
        # tilde_y + mask = (1 + tilde_y) for real values, and 1 for padded values.
        denominator = jnp.prod(tilde_s + self.substrates.mask, axis=-1) + jnp.prod(
            tilde_p + self.products.mask, axis=-1) - 1

        # Activation: $\prod_i{\frac{a_i}{a_i + K^A_i} = \prod_i{\frac{a_i / K^A_i}{a_i / K^A_i + 1}$
        tilde_a = self.activators.map_state(state, padding=1.) / kparms.kas
        activation = jnp.prod(tilde_a / (tilde_a + self.activators.mask), axis=-1)
        # Inhibition: $\prod_i{\frac{K^I_i}{a_i + K^I_i} = \prod_i{\frac{1}{a_i / K^I_i + 1}$
        # Since tilde_i is padded with zero already, we can use (tilde_i + 1) directly.
        tilde_i = self.inhibitors.map_state(state, padding=0.) / kparms.kis
        inhibition = jnp.prod(1 / (tilde_i + 1), axis=-1)

        return enzyme_conc * activation * inhibition * numerator / denominator

    def dstate_dt(self, state: ArrayT, enzyme_conc: ArrayT, kparms: Optional[PackedNetworkKinetics] = None) -> ArrayT:
        """Calculates current rate of change per molecule using the convenience kinetics formula.

        Args:
            state: A vector of state (i.e. concentration) values collinear with network molecules, or a vector of such
                state vectors.
            enzyme_conc: Array of enzyme concentrations collinear with network.reactions().
            kparms: May override intrinsic kinetics defined for this network.

        Returns:
            Array with rates of change for all state quantities. Its shape is the result of broadcast operations among
            the inputs `state` and `enzyme_conc`, plus any structure in `kparms`. For 1d inputs, the result has shape
            (#molecules, ).
        """
        return self.reaction_rates(state, enzyme_conc, kparms) @ self.network.s_matrix.T
