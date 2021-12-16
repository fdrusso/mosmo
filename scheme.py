"""Core classes defining objects and concepts used to construct models of molecular systems."""

from dataclasses import dataclass
from typing import List, Mapping, Optional


@dataclass(eq=True, frozen=True)
class DbCrossRef:
    """A cross-reference to (essentially) the same object in an external database."""
    db: str
    """Short token identifying the referenced database."""

    id: str
    """The id of the corresponding object in the referenced database."""

    def __repr__(self):
        return f"{self.db}:{self.id}"

    @staticmethod
    def from_str(xref: str) -> "DbCrossRef":
        """Parses a typically formatted DB:ID string into a DbCrossRef."""
        parts = xref.split(":", 2)
        if len(parts) == 2:
            return DbCrossRef(*parts)
        else:
            return DbCrossRef("?", xref)


@dataclass
class Variation:
    """Describes the nature of a dimension of variation, as a choice among a list of named forms."""
    name: str
    form_names: List[str]


@dataclass
class Specialization:
    """Specialization of the parent, or generalization of the child, identified by a set of form names.

    Variation and Specialization extend the is_a relationship used in many ontologies, by adding addressability.
    That is, we don't just declare that <child> is_a <parent>, but that <child> is _the_ [foo, bar] version of the
    parent. As a practical example we may have an entity in our KB for glucose. But we know glucose can come in
    D and L stereoisomers, and also that because of ring-chain tautamerism, a given molecule may be in the open-chain,
    α, or β configurations. So, glucose is the parent concept, and β-D-glucose is the [β, D] version of glucose.
    """
    parent_id: str
    form: List[str]
    child_id: str


@dataclass
class KbObject:
    """Attributes common to first-class objects in the knowledge base."""
    _id: str
    """Immutable unique identifier."""

    name: str
    """Preferred name of the object. Brief but descriptive, suitable for lists."""

    shorthand: Optional[str] = None
    """Acronym, abbreviation, or other terse label, suitable for diagrams."""

    aka: Optional[List[str]] = None
    """Alternative names of the object."""

    crossref: Optional[List[DbCrossRef]] = None
    """Cross-references to (essentially) the same object in other databases."""

    def __eq__(self, other):
        return type(self) == type(other) and self._id == other._id

    def __hash__(self):
        return hash((type(self), self._id))

    @property
    def id(self):
        return self._id


@dataclass
class Molecule(KbObject):
    """A molecule or molecule-like entity that may participate in a molecular system."""
    formula: Optional[str] = None
    """Chemical formula of this molecule."""

    mass: Optional[float] = None
    """Mass, in daltons of one molecule."""

    charge: Optional[int] = None
    """Electric charge of the molecule."""

    inchi: Optional[str] = None
    """InChI string describing the structure (https://en.wikipedia.org/wiki/International_Chemical_Identifier)."""

    variations: Optional[List[Variation]] = None
    """Defines the ways in which molecules of this type may vary.
    
    Many molecules can vary in protonation state, conformation, modification at specific sites, etc. Each Variation
    defines one such dimension of variation.
    """

    canonical_form: Optional[Specialization] = None
    """Defines this molecule as a specific form (i.e. this is the child) of some canonical reference form."""

    default_form: Optional[Specialization] = None
    """For a general molecule, defines a more specific form (i.e. this is the parent) it is assumed to take.

    As a specific example, we most often refer simply to ATP. But ATP technically has multiple protonation states, with
    slightly different mass and different charge. For simplicity we continue to refer simply to ATP, but define that
    its default form is ATP [4-].
    """

    def __eq__(self, other):
        return type(self) == type(other) and self._id == other._id

    def __hash__(self):
        return hash((type(self), self._id))

    def __repr__(self):
        facts = [f"Molecule [{self.id}] {self.name}"]
        if self.formula:
            facts.append(f"formula: {self.formula}")
        if self.mass:
            facts.append(f"mass: {self.mass} Da")
        if self.charge is not None:
            facts.append(f"charge: {self.charge:+d}")
        return "\n  ".join(facts)


@dataclass
class Reaction(KbObject):
    """A process transforming one set of molecules into another set of molecules in defined proportions."""
    stoichiometry: Mapping[Molecule, float] = None
    """The molecules transformed by this reaction. Substrates have negative stoichiometry, products positive."""

    catalyst: Optional[Molecule] = None
    """A single molecule (though possibly a complex) catalyzing this reaction. Neither consumed nor produced."""

    reversible: bool = True
    """Whether or not this reaction should be treated as reversible"""

    def __eq__(self, other):
        return type(self) == type(other) and self._id == other._id

    def __hash__(self):
        return hash((type(self), self._id))
