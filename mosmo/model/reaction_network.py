"""General representation of a network of stoichiometric reactions.

An idealized representation of a network of reactions, in which every reaction is all-or-nothing,
with strictly defined stoichiometry. The evolution of this network over time is determined by a
stoichiometry matrix, with reactants as the rows and reactions as the columns.

Some characteristics of typical reaction networks:
- In almost all cases we can assume the stoichiometry matrix to be sparse.
- For real biochemical systems, the total number of nonzero values is closer to linear with respect
  to the number of rows or columns than to the product of the two.
- Real reactions never involve more than a few reactants, so columns will likely never have more than
  a few nonzero values.
- The converse is not true: reactants such as ATP or water may each participate in many reactions, so
  the corresponding rows may have many nonzero values.
"""
from typing import Any, Iterable, Iterator, Mapping, Optional, Tuple

import numpy as np

from mosmo.model.core import Molecule, Reaction


class ReactionNetwork:
    """General representation of a network of stoichiometric reactions.

    This class serves two main functions:
    - Constructs a representation of the network as a (sparse) matrix of stoichiometry coefficients
      for each reactant (row) in each reaction (column).
    - Provides a mapping between the strictly numerically indexed rows and columns of this S matrix
      and the semantic Molecules and Reactions they correspond to.
    """

    def __init__(self, reactions: Optional[Iterable[Reaction]] = None):
        """Initialize this reaction network.

        Args:
            reactions: a list of Reactions included in this network.
        """

        # Defer construction of the stoichiometry matrix until it is needed.
        self._s_matrix = None

        # Build maps between string id and numerical array index, for reactions and reactants.
        self._reactions = []
        self._reaction_index = {}
        self._reactants = []
        self._reactant_index = {}
        if reactions is not None:
            for reaction in reactions:
                self.add_reaction(reaction)

    def add_reaction(self, reaction: Reaction):
        """Adds a reaction to the network.

        Args:
            reaction: the reaction to add to the network.
        """
        self._reaction_index[reaction] = len(self._reactions)  # Next index
        self._reactions.append(reaction)
        for reactant in reaction.stoichiometry:
            if reactant not in self._reactant_index:
                self._reactant_index[reactant] = len(self._reactants)
                self._reactants.append(reactant)
        # Force reconstruction of the stoichiometry matrix.
        self._s_matrix = None

    @property
    def s_matrix(self) -> np.ndarray:
        """The 2D stoichiometry matrix describing this reaction network."""
        if self._s_matrix is None:
            s_matrix = np.zeros(self.shape)
            for reaction in self._reactions:
                for reactant, coeff in reaction.stoichiometry.items():
                    # (reactant, reaction) is guaranteed unique
                    s_matrix[self._reactant_index[reactant], self._reaction_index[reaction]] = coeff
            self._s_matrix = s_matrix
        return self._s_matrix

    @property
    def shape(self) -> Tuple[int, int]:
        """The 2D shape of this network, (#molecules, #reactions)."""
        return len(self._reactants), len(self._reactions)

    def reactions(self) -> Iterator[Reaction]:
        """Iterates through reactions in their indexed order."""
        for reaction in self._reactions:
            yield reaction

    def reaction(self, i: int) -> Reaction:
        """The reaction at index i."""
        return self._reactions[i]

    def reaction_index(self, reaction: Reaction) -> Optional[int]:
        """The index of the reaction, or None if it is not part of the network."""
        return self._reaction_index.get(reaction, None)

    def reaction_vector(self, data: Mapping[Reaction, Any], default: Any = 0) -> np.ndarray:
        """Converts a dict of {reaction: value} to a 1D vector for numpy ops."""
        return np.array([data.get(reaction, default) for reaction in self._reactions], dtype=float)

    def reaction_values(self, values: Iterable[Any]) -> Mapping[Reaction, Any]:
        """Converts an array of values to a {reaction: value} dict."""
        return {reaction: value for reaction, value in zip(self._reactions, values)}

    def reactants(self) -> Iterator[Molecule]:
        """Iterates through reactants in their indexed order."""
        for reactant in self._reactants:
            yield reactant

    def reactant(self, i: int) -> Molecule:
        """The reactant at index i."""
        return self._reactants[i]

    def reactant_index(self, reactant: Molecule) -> Optional[int]:
        """The index of the reactant, or None if it is not part of the network."""
        return self._reactant_index.get(reactant, None)

    def reactant_vector(self, data: Mapping[str, Any], default: Any = 0) -> np.ndarray:
        """Converts a dict of {reactant: value} to a 1D vector for numpy ops."""
        return np.array([data.get(reactant, default) for reactant in self._reactants], dtype=float)

    def reactant_values(self, values: Iterable[Any]) -> Mapping[Molecule, Any]:
        """Converts an array of values to a {reactant: value} dict."""
        return {reactant: value for reactant, value in zip(self._reactants, values)}
