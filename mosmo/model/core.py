"""Core classes defining objects and concepts used to construct models of molecular systems."""
import collections
from dataclasses import dataclass
from typing import List, Mapping, Optional, Tuple

from .base import KbEntry


@dataclass
class Variation:
    """Describes the nature of a dimension of variation, as a choice among a list of named forms."""
    name: str
    form_names: List[str]


@dataclass
class Specialization:
    """Specialization of the parent, or generalization of the child, identified by a set of form names.

    Variation and Specialization extend the is_a relationship used in many ontologies, by adding
    addressability. That is, we don't just declare that <child> is_a <parent>, but that <child> is
    _the_ [foo, bar] form of the parent. As a practical example we may have an entry in our KB for
    glucose. We also know that glucose has D and L stereoisomers, and that because of ring-chain
    tautamerism, a given molecule may be in the open-chain, α, or β configurations. So, glucose is
    the parent concept, and β-D-glucose is the [D, β] form of glucose.
    """
    parent_id: str
    form: Tuple[str]
    child_id: str


@dataclass
class Molecule(KbEntry):
    """A molecule or molecule-like entity that may participate in a molecular system."""
    formula: Optional[str] = None
    """Chemical formula of this molecule."""

    mass: Optional[float] = None
    """Mass of one molecule, in daltons (or of a mole, in grams)."""

    charge: Optional[int] = None
    """Electric charge of the molecule."""

    inchi: Optional[str] = None
    """InChI string describing the structure (https://en.wikipedia.org/wiki/International_Chemical_Identifier)."""

    variations: Optional[List[Variation]] = None
    """Defines the ways in which molecules of this type may vary.
    
    Many molecules can vary in protonation state, conformation, modification at specific sites, etc.
    Each Variation defines one such dimension of variation.
    """

    canonical_form: Optional[Specialization] = None
    """Defines this molecule as a specific form (i.e. this is the child) of some canonical reference form."""

    default_form: Optional[Specialization] = None
    """For a general molecule, defines a more specific assumed form (i.e. this is the parent).

    As a specific example, we most often refer simply to ATP. But ATP technically has multiple protonation
    states, with slightly different mass and different charge. For simplicity we continue to refer simply 
    to ATP, but define that its default form is ATP [4-].
    """

    def _data_items(self):
        items = super()._data_items() | {
            'formula': self.formula,
            'mass': self.mass,
            'charge': self.charge,
            'inchi': self.inchi,
        }
        if self.canonical_form:
            items['canonical_form'] = self.canonical_form.parent_id
        if self.default_form:
            items['default_form'] = self.default_form.child_id
        return items

    def __eq__(self, other):
        return self.same_as(other)

    def __hash__(self):
        return hash((type(self), self.id))

    def __repr__(self):
        return f"[{self.id}] {self.name or ''}"


@dataclass
class Reaction(KbEntry):
    """A process transforming one set of molecules into another set of molecules in defined proportions."""
    stoichiometry: Mapping[Molecule, float] = None
    """The molecules transformed by this reaction. Substrates have negative stoichiometry, products positive."""

    catalyst: Optional[Molecule] = None
    """A single molecule (though possibly a complex) catalyzing this reaction. Neither consumed nor produced."""

    reversible: bool = True
    """Whether or not this reaction should be treated as reversible"""

    @property
    def equation(self):
        """Human-readable compact summary of the reaction."""
        def molecule_term(molecule: Molecule, count: float) -> str:
            if count == 1:
                return molecule.label
            else:
                return f'{count} {molecule.label}'

        lhs = [molecule_term(molecule, -count) for molecule, count in self.stoichiometry.items() if count < 0]
        rhs = [molecule_term(molecule, count) for molecule, count in self.stoichiometry.items() if count > 0]
        arrow = ' <=> ' if self.reversible else ' => '

        return ' + '.join(lhs) + arrow + ' + '.join(rhs)

    def _data_items(self):
        return super()._data_items() | {
            'equation': self.equation,
            'reversible': self.reversible,
            'catalyst': self.catalyst,
        }

    def __eq__(self, other):
        return self.same_as(other)

    def __hash__(self):
        return hash((type(self), self.id))

    def __repr__(self):
        return f"[{self.id}] {self.equation}"

    def __add__(self, other):
        """Combines this reaction with another."""
        # Trick to support sum(): Adding any kind of 0 is supported
        if not other:
            return self
        if not isinstance(other, Reaction):
            raise ValueError(f"Reaction cannot be combined with type [{type(other)}]")

        stoichiometry = collections.Counter()
        stoichiometry.update(self.stoichiometry)
        stoichiometry.update(other.stoichiometry)
        return Reaction(
            id = self.id + "+" + other.id,
            db = None,
            stoichiometry = {molecule: count for molecule, count in stoichiometry.items() if count != 0},
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (other * -1)

    def __mul__(self, other):
        """Multiplies the effect of this reaction proportionally across all reactants."""
        if not isinstance(other, (int, float)):
            raise ValueError(f"Reaction cannot be multiplied by type [{type(other)}]")

        return Reaction(
            id = str(other) + "*" + self.id,
            db = None,
            stoichiometry = {molecule: other * count for molecule, count in self.stoichiometry.items()},
        )

    __rmul__ = __mul__
