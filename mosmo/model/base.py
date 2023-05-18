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
    id: str
    """Persistent unique identifier."""

    db: Optional[str] = None
    """The database/dataset in which this is an entry."""

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

    def ref(self) -> DbXref:
        """A DbXref that refers to this KbEntry."""
        return DbXref(db=self.db, id=self.id)

    def same_as(self, other) -> bool:
        """Reusable by subclasses to simplify implementation of __eq__."""
        return (
            type(self) == type(other) and
            self.id == other.id and
            (self.db is None and other.db is None or self.db == other.db)
        )

    def __eq__(self, other):
        return self.same_as(other)

    def __hash__(self):
        return hash((type(self), self.id, self.db))

    @property
    def label(self):
        """Compact designator of the entry, e.g. for plot labels. Shorthand is preferred, otherwise ID."""
        return self.shorthand or self.id
