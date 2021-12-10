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

    def __eq__(self, other):
        return type(self) == type(other) and self._id == other._id

    def __hash__(self):
        return hash((type(self), self._id))


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
