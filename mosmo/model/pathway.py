"""General representation of a network of stoichiometric reactions.

An idealized representation of a pathway as a network of reactions, in which every reaction is all-or-nothing,
with strictly defined stoichiometry. The evolution of this network over time is determined by a
stoichiometry matrix, with molecules as the rows and reactions as the columns.

Some characteristics of typical reaction networks:
- In almost all cases we can assume the stoichiometry matrix to be sparse.
- For real biochemical systems, the total number of nonzero values is closer to linear with respect
  to the number of rows or columns than to the product of the two.
- Real reactions never involve more than a few molecules, so columns will likely never have more than
  a few nonzero values.
- The converse is not true: molecules such as ATP or water may each participate in many reactions, so
  the corresponding rows may have many nonzero values.
"""
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np

from .base import KbEntry
from .core import Molecule, Reaction

KE = TypeVar("KE", bound=KbEntry)


class Index(Sequence[KE]):
    """Provides a mapping between a numerically indexed vector or array and the KbEntry items they correspond to.

    An Index behaves as a list with set semantics, i.e. any item appears at most once and therefore has a unique
    numerical position. It is useful for moving back and forth between packed values in numerically-indexed vectors
    (e.g. numpy arrays) and the semantic objects represented at each position. Random access by position is supported
    directly by subscripting. Random access by item is supported by index_of().
    """

    def __init__(self, items: Optional[Iterable[KE]] = None):
        self._items = []
        self._index = {}
        if items is not None:
            self.update(items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[KE]:
        """Iterates through the items of the index in the same order as an indexed vector or array."""
        return iter(self._items)

    def __contains__(self, item) -> bool:
        """Constant-time containment test."""
        return item in self._index

    def __getitem__(self, index: Union[int, slice]) -> KE:
        """Returns the item at the numerical index (or slice)."""
        return self._items[index]

    def add(self, item: KE):
        """Adds an item to the index using set semantics; i.e. this is a no-op for existing items."""
        if item not in self._index:
            self._index[item] = len(self._items)  # Next index
            self._items.append(item)

    def update(self, items: Iterable[KE]):
        """Adds a collection of items to the index using set semantics."""
        for item in items:
            self.add(item)

    def index_of(self, item: KE) -> Optional[int]:
        """Returns the numerical position of the item, or None if not present."""
        return self._index.get(item, None)

    def pack(self, data: Mapping[KE, Any], default: Any = 0) -> np.ndarray:
        """Converts a dict of {item: value} to a 1D vector for numpy ops."""
        return np.array([data.get(item, default) for item in self._items], dtype=float)

    def unpack(self, values: Iterable[Any]) -> Mapping[KE, Any]:
        """Converts an array of values to an {item: value} dict."""
        return {item: value for item, value in zip(self._items, values)}

    def labels(self) -> Sequence[str]:
        """Presents the items in the index as a list of (string) labels."""
        return [item.label for item in self._items]


class Pathway(KbEntry):
    """General representation of a network of stoichiometric reactions.

    Broadens the basic biological meaning of 'pathway' to include any reaction network large or small, with or
    without a graphical representation.

    This class serves the following functions:
    - Constructs a representation of the network as a (sparse) matrix of stoichiometry coefficients
      for each molecule (row) in each reaction (column).
    - Provides a mapping between the strictly numerically indexed rows and columns of this S matrix
      and the semantic Molecules and Reactions they correspond to.
    - Connects the mathematical behavior of the reaction network to a graphical representation of the pathway, as
      understood by biochemists or molecular biologists.
    """

    def __init__(self, reactions: Optional[Iterable[Reaction]] = None, diagram=None, **kwargs):
        """Initialize this pathway, based on a set of reactions.

        Args:
            reactions: a collection of Reactions included in this network. Any repeated reactions will be
                included once.
            diagram: json-compatible structure defining a pathway diagram consistent with the contents of
                this pathway.
        """
        super().__init__(**kwargs)
        self.diagram = diagram

        # Defer construction of the stoichiometry matrix until it is needed.
        self._s_matrix = None

        # Prepare indices for reactions and molecules.
        self.reactions: Index[Reaction] = Index()
        self.molecules: Index[Molecule] = Index()
        if reactions is not None:
            for reaction in reactions:
                self.add_reaction(reaction)

    def add_reaction(self, reaction: Reaction):
        """Adds a reaction to the pathway.

        Args:
            reaction: the reaction to add.
        """
        self.reactions.add(reaction)
        self.molecules.update(reaction.stoichiometry.keys())

        # Force reconstruction of the stoichiometry matrix.
        self._s_matrix = None

    @property
    def s_matrix(self) -> np.ndarray:
        """The 2D stoichiometry matrix describing this reaction network mathematically."""
        if self._s_matrix is None:
            s_matrix = np.zeros(self.shape)
            for reaction in self.reactions:
                for molecule, coeff in reaction.stoichiometry.items():
                    # (molecule, reaction) is guaranteed unique
                    s_matrix[self.molecules.index_of(molecule), self.reactions.index_of(reaction)] = coeff
            self._s_matrix = s_matrix
        return self._s_matrix

    @property
    def shape(self) -> Tuple[int, int]:
        """The 2D shape of this network, (#molecules, #reactions)."""
        return len(self.molecules), len(self.reactions)

    def _data_items(self):
        return super()._data_items() | {
            'size': f'{len(self.reactions)} reactions over {len(self.molecules)} molecules.',
            'reactions': list(self.reactions),
        }

    def __eq__(self, other):
        return self.same_as(other)

    def __hash__(self):
        return hash((type(self), self.id))

    def __add__(self, other):
        """Combines this network with another Pathway, or a single Reaction."""
        if isinstance(other, Pathway):
            return Pathway(
                id=f"{self.id}+{other.id}",
                name=f"{self.name} + {other.name}",
                reactions=list(self.reactions) + list(other.reactions)
            )
        elif isinstance(other, Reaction):
            return Pathway(
                id=f"{self.id}+{other.id}",
                name=f"{self.name} + {other.label}",
                reactions=list(self.reactions) + [other]
            )
        else:
            raise ValueError(f"Pathway cannot be combined with type [{type(other)}]")

    @property
    def steps(self):
        """Supports legacy usage of Pathway.steps."""
        return list(self.reactions)

    @property
    def metabolites(self):
        """Supports legacy usage of Pathway.metabolites."""
        return list(self.molecules)

    @property
    def enzymes(self):
        """Supports legacy usage of Pathway.enzymes."""
        return list(rxn.catalyst for rxn in self.reactions if rxn.catalyst is not None)

    @property
    def reactants(self):
        """Supports legacy usage of ReactionNetwork.reactants."""
        return self.molecules


# Support legacy usage of the ReactionNetwork class
ReactionNetwork = Pathway
