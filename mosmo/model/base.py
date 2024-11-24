"""Base classes with universal attributes for Knowledge Base entries."""
import textwrap
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Set, Type


@dataclass(frozen=True, eq=True, order=True)
class Datasource:
    """Part of the ecosystem hosting biological data."""
    id: str
    name: Optional[str] = field(default=None, compare=False)
    home: Optional[str] = field(default=None, compare=False)
    urlpat: Optional[Mapping[Type, str]] = field(default=None, compare=False)

    def __repr__(self):
        return f'[{self.id}] {self.name}'


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


@dataclass(frozen=True, eq=True, repr=False)
class DbXref:
    """A reference to an entry in a datasource."""
    id: str
    db: Optional[Datasource] = None

    def __repr__(self):
        if self.db:
            return f"{self.db.id}:{self.id}"
        else:
            return self.id

    def url(self, clazz: Optional[Type] = None) -> Optional[str]:
        """Constructs a URL linking to the datasource's record of this entry.

        Some but not all datasources contain entries of multiple types, with corresponding differences in URL pattern.
        In this case the type of object must be provided to construct an unambiguous URL.
        """
        if self.db is not None and self.db.urlpat:
            if len(self.db.urlpat) == 1:
                urlpat = next(iter(self.db.urlpat.values()))
                return urlpat.format(id=self.id)
            elif clazz in self.db.urlpat:
                urlpat = self.db.urlpat[clazz]
                return urlpat.format(id=self.id)
        return None

    @staticmethod
    def from_str(xref: str) -> "DbXref":
        """Parses a typically formatted DB:ID string into a DbXref."""
        parts = xref.split(":", 2)
        if len(parts) == 2:
            return DbXref(parts[1], DS.get(parts[0]))
        else:
            return DbXref(xref)


@dataclass
class KbEntry:
    """Attributes common to first-class entities, items, or concepts in the knowledge base.

    Attributes:
        id: Persistent unique identifier.
        db: The database/datasource in which this is an entry.
        name: Preferred name of the entry. Brief but descriptive, suitable for lists.
        shorthand: Acronym, abbreviation, or other terse label, suitable for graphs or diagrams.
        description: Full description, suitable for a view focused on one entry at a time.
        aka: Alternative names of the entry.
        xrefs: Cross-references to (essentially) the same entry in other databases.

    All attributes are optional on init, though id and db in particular are important for most functionality.
    """
    id: Optional[str] = None
    db: Optional[Datasource] = None
    name: Optional[str] = None
    shorthand: Optional[str] = None
    description: Optional[str] = None
    aka: Optional[List[str]] = None
    xrefs: Optional[Set[DbXref]] = None

    def ref(self) -> DbXref:
        """A DbXref that refers to this KbEntry."""
        return DbXref(id=self.id, db=self.db)

    def url(self):
        if self.db:
            return self.ref().url(type(self))
        else:
            return None

    def xref_urls(self):
        return [xref.url(type(self)) for xref in self.xrefs] if self.xrefs else []

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

    def __repr__(self):
        return f"[{self.id}]{' (' + self.shorthand + ')' if self.shorthand else ''} {self.name or ''}"

    @property
    def label(self):
        """Compact designator of the entry, e.g. for plot labels. Shorthand is preferred, otherwise ID."""
        return self.shorthand or self.id

    def data(self, max_width=60, sep=': ', indent='    '):
        """Displays the contents of this entry. Intended for interactive contexts only.

        This function delegates to _data_items, which can be overridden by subclasses to add more information.
        """

        lines = [str(self.ref())]
        items = self._data_items()
        label_width = max(len(label) for label in items.keys())
        item_width = max_width - label_width - len(sep)
        for label, value in items.items():
            if isinstance(value, list):
                lines.append(label + sep)
                lines.extend(f'{indent}{v}' for v in value)
            else:
                value = str(value)
                if len(value) <= item_width:
                    lines.append(label + sep + value)
                else:
                    lines.append(label + sep)
                    lines.extend(textwrap.wrap(value, width=max_width, initial_indent=indent, subsequent_indent=indent))
        print('\n'.join(lines))

    def _data_items(self):
        """Key-value pairs for data(). Subclasses may override to supply additional items."""
        return {
            'name': self.name,
            'shorthand': self.shorthand,
            'aka': self.aka,
            'description': self.description,
            'xrefs': sorted(str(xref) for xref in self.xrefs) if self.xrefs else None
        }
