"""Base classes with universal attributes for Knowledge Base entries."""
from dataclasses import dataclass
from typing import List, Optional, Set


@dataclass(eq=True, order=True, frozen=True)
class DbXref:
    """A cross-reference to (essentially) the same entry, item, or concept in an external database."""
    db: Optional[str]
    """Short token identifying the referenced database. `None` indicates an unknown or ambiguous xref."""

    id: str
    """The id of the corresponding entry in the referenced database."""

    def __repr__(self):
        return f"{self.db}:{self.id}"

    @staticmethod
    def from_str(xref: str) -> "DbXref":
        """Parses a typically formatted DB:ID string into a DbXref."""
        parts = xref.split(":", 2)
        if len(parts) == 2:
            return DbXref(*parts)
        else:
            return DbXref(None, xref)


@dataclass
class KbEntry:
    """Attributes common to first-class entities, items, or concepts in the knowledge base."""
    _id: str
    """Immutable unique identifier."""

    name: str = ''
    """Preferred name of the entry. Brief but descriptive, suitable for lists."""

    shorthand: Optional[str] = None
    """Acronym, abbreviation, or other terse label, suitable for diagrams."""

    description: Optional[str] = None
    """Full description, suitable for a view focused on one entry at a time."""

    aka: Optional[List[str]] = None
    """Alternative names of the entry."""

    xrefs: Optional[Set[DbXref]] = None
    """Cross-references to (essentially) the same entry in other databases."""

    def __eq__(self, other):
        return type(self) == type(other) and self._id == other._id

    def __hash__(self):
        return hash((type(self), self._id))

    @property
    def id(self):
        return self._id

    @property
    def label(self):
        """Compact designator of the entry, e.g. for plot labels. Shorthand is preferred, otherwise ID."""
        return self.shorthand or self._id
