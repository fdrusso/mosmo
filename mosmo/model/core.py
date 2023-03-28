"""Core classes defining objects and concepts used to construct models of molecular systems."""
from dataclasses import dataclass
from typing import List, Mapping, Optional, Set, Tuple

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

    def __eq__(self, other):
        return type(self) == type(other) and self.id == other.id

    def __hash__(self):
        return hash((type(self), self.id))

    def __repr__(self):
        return f"[{self.id}] {self.name}"


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
    def formula(self):
        """Human-readable compact summary of the reaction."""
        def reactant_term(reactant: Molecule, count: float) -> str:
            if count == 1:
                return reactant.label
            else:
                return f'{count} {reactant.label}'

        lhs = [reactant_term(reactant, -count) for reactant, count in self.stoichiometry.items() if count < 0]
        rhs = [reactant_term(reactant, count) for reactant, count in self.stoichiometry.items() if count > 0]
        arrow = ' <=> ' if self.reversible else ' => '

        return ' + '.join(lhs) + arrow + ' + '.join(rhs)

    def __eq__(self, other):
        return type(self) == type(other) and self.id == other.id

    def __hash__(self):
        return hash((type(self), self.id))

    def __repr__(self):
        return f"[{self.id}] {self.formula}"


@dataclass
class Pathway(KbEntry):
    """A process encompassing multiple Reactions and their Molecules."""
    steps: Set[Reaction] = None

    metabolites: Set[Molecule] = None

    enzymes: Set[Molecule] = None

    def __eq__(self, other):
        return type(self) == type(other) and self.id == other.id

    def __hash__(self):
        return hash((type(self), self.id))
