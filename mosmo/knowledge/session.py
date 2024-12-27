"""Interactive Knowledge Base session optionally backed by persistent MongoDB storage.

A Session is organized around Datasets, each representing a collection of KbEntry of uniform type. Typically,
a Dataset also has some domain-relevant meaning, such as all entries from a specific external datasource. When
backed by storage, a Dataset corresponds to a pymongo Collection. Translation to and from the underlying document
format is managed by a Codec.
"""
import collections
from contextlib import contextmanager
import copy
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Type, Union

from pymongo import MongoClient

from mosmo.knowledge import codecs
from mosmo.model import Datasource, DbXref, KbEntry


@dataclass(eq=True, order=True, frozen=True)
class Dataset:
    """A defined collection of entries in the Knowledge Base.

    Entries in a dataset are all of the same type, and associated with a single Datasource. This corresponds to a
    single Collection in the underlying mongo db used for persistence.
    """
    name: str
    datasource: Datasource
    content_type: Type[KbEntry]
    client_db: str
    collection: str
    codec: codecs.Codec
    canonical: bool = False

    def __repr__(self):
        return f'{self.name}: ({self.datasource.id}/{self.content_type.__name__}) [{self.client_db}.{self.collection}]'


def _as_xref(q: Union[DbXref, KbEntry, str]) -> DbXref:
    """Attempts to coerce the query to a DbXref."""
    if isinstance(q, DbXref):
        return q
    elif isinstance(q, str):
        return DbXref.from_str(q)
    elif isinstance(q, KbEntry):
        return q.ref()
    else:
        raise TypeError(f"{q} cannot be converted to DbXref.")


class Session:
    def __init__(self, client: Optional[MongoClient] = None, schema: Iterable[Dataset] = None):
        """Initializes a KB session.

        Args:
            client: connection to a local or remote MongoDB server. If None, the session performs as an in-memory cache
                and no data is persisted.
            schema: Dataset definitions for the contents of the KB.
        """
        self.client = client
        self.schema: Dict[str, Dataset] = {}
        self.by_source: Dict[Datasource, Dict[Type, Dataset]] = collections.defaultdict(dict)
        self.canon: Dict[Type, Dataset] = {}
        self._cache: Dict[Dataset, Dict[Any, KbEntry]] = {}
        self.writable: Dict[Dataset, bool] = {}

        if schema:
            for dataset in schema:
                self.define_dataset(dataset)

    def define_dataset(self, dataset: Dataset):
        """Adds a dataset to the schema of this session. Datasets start out locked for writing."""
        if dataset.name in self.schema or dataset.name in self.__dict__:
            raise ValueError(f'Name collision: {dataset.name} is an existing attribute of Session')

        self.schema[dataset.name] = dataset
        self.by_source[dataset.datasource][dataset.content_type] = dataset
        self.writable[dataset] = False

        if dataset.canonical:
            if dataset.content_type in self.canon:
                raise ValueError(f"Attempt to redefine canonical dataset for {dataset.content_type}.")
            self.canon[dataset.content_type] = dataset

        # Make dataset definitions easily accessible as attributes of the Session.
        self.__dict__[dataset.name] = dataset

        # The cache is not just to save round-trips to the datastore, but to maximize reuse of decoded instances.
        self._cache[dataset] = {}

    def find_dataset(self, db: Datasource, clazz: Optional[Type] = None):
        """Finds the physical dataset associated with a logical datasource (and type), if any."""
        sources = self.by_source.get(db, {})
        if len(sources) == 1:
            return next(iter(sources.values()))
        else:
            return  sources.get(clazz)

    @contextmanager
    def unlock(self, *datasets):
        """Unlocks select datasets, or all datasets, to allow writing (putting) entries.

        The dataset lock/unlock mechanism is intended not for bulletproof security, but to avoid unintended changes
        to the knowledge base.

        Usage:
            with session.unlock(ds):
                session.put(ds, entry)
                session.remove(entry)
        """
        if not datasets:
            datasets = self.schema.values()
        try:
            for dataset in datasets:
                self.writable[dataset] = True
            yield datasets
        finally:
            for dataset in datasets:
                self.writable[dataset] = False

    def clear_cache(self, *datasets):
        """Clears cached entries for select datasets, or all datasets."""
        if not datasets:
            datasets = self.schema.values()
        for dataset in datasets:
            self._cache[dataset].clear()

    def _cache_value(self, dataset: Dataset, doc) -> KbEntry:
        """Decodes a document from storage into the in-memory cache for the specified dataset."""
        if doc['_id'] not in self._cache[dataset]:
            entry = dataset.codec.decode(doc)
            if entry.db is None:
                entry.db = dataset.datasource
            self._cache[dataset][doc['_id']] = entry
        return self._cache[dataset][doc['_id']]

    def get(self, dataset: Dataset, id: str) -> Optional[KbEntry]:
        """Retrieves the specified entry from the KB by ID, if it exists."""
        if dataset is None:
            return None

        if id not in self._cache[dataset] and self.client is not None:
            doc = self.client[dataset.client_db][dataset.collection].find_one(id)
            if doc:
                self._cache_value(dataset, doc)
        return self._cache[dataset].get(id)

    def deref(self, q: Union[DbXref, KbEntry, str], clazz: Optional[Type] = None) -> Optional[KbEntry]:
        """Retrieves the entry referred to by a DbXref or its string representation."""
        xref = _as_xref(q)
        dataset = self.find_dataset(xref.db, clazz)
        if dataset:
            return self.get(dataset, xref.id)
        else:
            return None

    def put(self, dataset: Dataset, entry: KbEntry, bypass_cache: bool = False):
        """Persists an entry to the KB, in the given dataset.

        The entry's db attribute reflects the dataset where it is persisted. If it is not currently associated with a
        dataset, its db attribute will be updated. If it is already part of a different dataset, a copy will be
        persisted instead.

        Note that changing an entry's db attribute changes how it tests equality, and how its hash is calculated.
        Use with caution.

        Args:
             dataset: the dataset where the entry will be persisted.
             entry: the entry to be persisted.
             bypass_cache: if True, the entry is persisted straight to the underlying databases, bypassing the session
                cache. May save session memory if a large number of entries are persisted.

        Raises:
            ValueError on an attempt to write to a locked dataset.
        """
        if not self.writable[dataset]:
            raise ValueError(f'Dataset [{dataset.name}] is locked.')

        if entry.db is None:
            entry.db = dataset.datasource
        elif entry.db != dataset.datasource:
            entry = copy.deepcopy(entry)
            entry.db = dataset.datasource

        if not bypass_cache:
            self._cache[dataset][entry.id] = entry
        else:
            # Even when bypassing the cache, make sure the cache itself is not now stale.
            self._cache[dataset].pop(entry.id, None)

        if self.client is not None:
            doc = dataset.codec.encode(entry)
            self.client[dataset.client_db][dataset.collection].replace_one({'_id': entry.id}, doc, upsert=True)

    def remove(self, entry: KbEntry):
        """Removes an entry from underlying storage.

        The entry's `db` attribute is used to identify where the entry is to be removed, both from the local cache
        and underlying storage. The entry instance itself is unchanged.

        Args:
             entry: the entry to be removed.

        Raises:
            ValueError on an attempt to write to a locked dataset.
        """
        dataset = self.find_dataset(entry.db)
        if dataset:
            if not self.writable[dataset]:
                raise ValueError(f'Dataset [{dataset.name}] is locked.')

            if self.client is not None:
                self.client[dataset.client_db][dataset.collection].delete_one({'_id': entry.id})
            self._cache[dataset].pop(entry.id)

    def find(self, dataset: Dataset, name: str, include_aka=True) -> List[KbEntry]:
        """Finds any number of KB entries matching the given name, optionally as an AKA."""
        found = set()
        docs = []
        for doc in self.client[dataset.client_db][dataset.collection].find(
                {'name': name}).collation({'locale': 'en', 'strength': 1}):
            if doc['_id'] not in found:
                docs.append(doc)
                found.add(doc['_id'])
        if include_aka:
            for doc in self.client[dataset.client_db][dataset.collection].find(
                    {'aka': name}).collation({'locale': 'en', 'strength': 1}):
                if doc['_id'] not in found:
                    docs.append(doc)
                    found.add(doc['_id'])
        return [self._cache_value(dataset, doc) for doc in docs]

    def find_one(self, dataset: Dataset, name: str, include_aka=True, strict: bool = False) -> Optional[KbEntry]:
        """Returns the first KB entry matching the given name, if any."""
        found = self.find(dataset, name, include_aka)
        if strict:
            if not found:
                raise ValueError(f'No hits to {name} found in {dataset.name}')
            elif len(found) > 1:
                warnings.warn(f'Multiple hits to {name} found in {dataset.name}')
        return found[0] if found else None

    def xref(self, dataset: Dataset, q: Union[DbXref, KbEntry, str]) -> List[KbEntry]:
        """Finds any number of entries in the dataset cross-referenced to the given query."""
        xref = _as_xref(q)
        query = {'xrefs.id': xref.id}
        if xref.db:
            query['xrefs.db'] = xref.db.id

        results = []
        for doc in self.client[dataset.client_db][dataset.collection].find(query).collation(
                {'locale': 'en', 'strength': 1}):
            results.append(self._cache_value(dataset, doc))
        return results

    def xref_one(self, dataset: Dataset, q: Union[DbXref, KbEntry, str], strict: bool = False) -> Optional[KbEntry]:
        """Returns the first entry in the dataset cross-referenced to the given query, if any."""
        xrefs = self.xref(dataset, q)
        if strict:
            if not xrefs:
                raise ValueError(f'No xref to {q} found in {dataset.name}')
            elif len(xrefs) > 1:
                warnings.warn(f'Multiple xrefs to {q} found in {dataset.name}')
        return xrefs[0] if xrefs else None

    def __call__(self, q) -> Optional[KbEntry]:
        """Convenience interface to the KB.

        Attempts to return a unique entry as intended by the caller. For unambiguous queries this works well, e.g.:
        - Fully specified DB:ID xref (as a DbXref or a string)
        - Unique ID of an entry in a canonical dataset
        - ID of an entry in any dataset, provided it is unique across all datasets

        This call will always return the first entry it finds, so if the conditions above do not hold, the result may
        not be as expected. CALLER BEWARE.

        This interface is provided to reduce friction in accessing KB entries interactively. Since multiple calls to the
        underlying DB may be needed to resolve any query, this is unlikely to be the most efficient way to access large
        numbers of entries. Consider using more specific methods as appropriate.

        Args:
            q: a DbXref or string sufficient to identify a unique entry in the knowledge base.

        Returns:
            A single KbEntry identified by q, or None.
        """
        # Fully specified xref (string or object)
        entry = self.deref(q)
        if entry:
            return entry

        # ID in a canonical dataset
        for dataset in self.canon.values():
            entry = self.get(dataset, q)
            if entry:
                return entry

        # ID in a non-canonical dataset
        for dataset in self.schema.values():
            if self.canon.get(dataset.content_type) == dataset:
                continue  # already tried
            entry = self.get(dataset, q)
            if entry:
                return entry


class XrefCodec(codecs.Codec):
    """Session-aware Codec encoding a KbEntry as a DbXref.

    This allows persistent references across Datasets, but does _not_ enforce referential integrity.
    """

    def __init__(self, session: Session, clazz: Type):
        self.session = session
        self.delegate = codecs.CODECS[DbXref]
        self.clazz = clazz

    def encode(self, entry):
        return self.delegate.encode(entry.ref())

    def decode(self, doc):
        xref = self.delegate.decode(doc)
        if self.session:
            obj = self.session.deref(xref, self.clazz)
            if obj:
                return obj
        return self.clazz(id=xref.id, db=xref.db)
