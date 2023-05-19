"""Base classes with universal attributes for Knowledge Base entries."""
from dataclasses import dataclass
from typing import List, Mapping, Optional, Set, Type, Union


@dataclass(eq=True, frozen=True)
class Datasource:
    """Part of the ecosystem hosting biological data."""
    id: str
    name: Optional[str] = None
    home: Optional[str] = None
    urlpat: Optional[Mapping[Type, str]] = None


class _Registry:
    """Provides quick access to defined datasources."""

    def __init__(self):
        self.datasources = {}

    def has(self, id: str) -> bool:
        """Safe test for datasource existence."""
        return id in self.datasources

    def define(self, datasource: Datasource) -> Datasource:
        """Adds a datasource definition to the registry."""
        if self.has(datasource.id):
            raise ValueError(f"Datasource {datasource.id} is already defined")
        self.datasources[datasource.id] = datasource
        self.__dict__[datasource.id] = datasource
        return datasource

    def get(self, id: str, create: bool = True) -> Optional[Datasource]:
        """Retrieves a datasource definition, created on demand if necessary."""
        if id not in self.datasources and create:
            self.define(Datasource(id=id))
        return self.datasources.get(id, None)


DS = _Registry()


class DbXref:
    """A reference to an entry in an external database."""
    def __init__(self, db: Optional[Union[str, Datasource]], id: str, clazz: Optional[Type] = None):
        if type(db) == Datasource:
            self.db = db
        elif type(db) == str:
            self.db = DS.get(db)
        else:
            self.db = None
        self.id = id
        self._clazz = clazz

    def __repr__(self):
        if self.db:
            return f"{self.db.id}:{self.id}"
        else:
            return self.id

    def __eq__(self, other):
        return (type(self) == type(other)
                and (self.db is None and other.db is None or self.db == other.db)
                and self.id == other.id)

    def __hash__(self):
        return hash((self.db, self.id))

    def url(self) -> Optional[str]:
        if self.db is not None and self.db.urlpat:
            if self._clazz and self._clazz in self.db.urlpat:
                return self.db.urlpat[self._clazz].format(id=self.id)
            elif len(self.db.urlpat) == 1:
                # Just assume the lookup by id will work
                return next(iter(self.db.urlpat.values())).format(id=self.id)
        return None

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
        return DbXref(db=self.db, id=self.id, clazz=type(self))

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
