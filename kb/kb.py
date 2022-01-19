"""Knowledge Base for Molecular Systems Modeling."""
import abc
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Type

import pymongo

from scheme import DbXref, KbEntry, Molecule, Reaction, Pathway, Specialization, Variation


# Codecs define the encoding/decoding schema for KB objects in the storage layer. PyMongo does have a mechanism to
# define custom types, but it is not flexible enough to do this cleanly in our case. The main cost is we need to define
# the schema relationships with explicit types, rather than having the system infer them. That is a reasonable
# constraint.

class Codec(abc.ABC):
    """Base class for all Codecs.

    The semantics of the base class are difficult to capture using python type hints. The intent is that an instance of
    a given subclass translates between a python object of defined type, and a corresponding pymongo document.
    """
    @abc.abstractmethod
    def encode(self, obj):
        """Converts a python object into a pymongo document or fragment."""
        raise NotImplementedError()

    @abc.abstractmethod
    def decode(self, doc):
        """Converts a pymongo into a python object."""
        raise NotImplementedError()


class AsIsCodec(Codec):
    """No-op codec passes everything through encode and decode as-is."""
    def encode(self, obj):
        return obj

    def decode(self, doc):
        return doc


AS_IS = AsIsCodec()


class ListCodec(Codec):
    """Encodes/decodes a python iterable type to a json-compatible list."""
    def __init__(self, item_codec: Codec = None, list_type: Callable[[Iterable], Iterable] = list):
        self.list_type = list_type
        self.item_codec = item_codec or AS_IS

    def encode(self, items):
        return list(self.item_codec.encode(item) for item in items)

    def decode(self, doc):
        return self.list_type(self.item_codec.decode(item) for item in doc)


class MappingCodec(Codec):
    """Encodes/decodes a python mapping type to a json list of tuples."""
    def __init__(self,
                 key_codec: Codec = None,
                 value_codec: Codec = None,
                 mapping_type: Callable[[Mapping], Mapping] = dict):
        self.mapping_type = mapping_type
        self.key_codec = key_codec or AS_IS
        self.value_code = value_codec or AS_IS

    def encode(self, mapping):
        return [[self.key_codec.encode(k), self.value_code.encode(v)] for k, v in mapping.items()]

    def decode(self, doc):
        return self.mapping_type({self.key_codec.decode(k): self.value_code.decode(v) for k, v in doc})


class ObjectCodec(Codec):
    """Encodes/decodes a python instance to a json-compatible dict."""
    def __init__(self, clazz, codec_map: Mapping[str, Codec] = None):
        self.clazz = clazz
        self.codec_map = codec_map or {}

    def encode(self, obj):
        doc = {}
        for k, v in obj.__dict__.items():
            if v is not None:
                doc[k] = self.codec_map.get(k, AS_IS).encode(v)
        return doc

    def decode(self, doc):
        args = {}
        for k, v in doc.items():
            args[k] = self.codec_map.get(k, AS_IS).decode(v)
        return self.clazz(**args)


class ObjectIdCodec(Codec):
    """(Lossy) codec that encodes only the ID of an object, and decodes it as an otherwise empty object."""
    def __init__(self, clazz):
        self.clazz = clazz

    def encode(self, obj):
        return {'_id': obj.id}

    def decode(self, doc):
        return self.clazz(doc['_id'])


MOLSTUB = ObjectIdCodec(Molecule)
RXNSTUB = ObjectIdCodec(Reaction)

CODECS = {
    DbXref: ObjectCodec(DbXref),
    Variation: ObjectCodec(Variation, {'form_names': ListCodec()}),
    Specialization: ObjectCodec(Specialization, {'form': ListCodec(list_type=tuple)}),
}

CODECS[KbEntry] = ObjectCodec(KbEntry, {
    'xrefs': ListCodec(item_codec=CODECS[DbXref], list_type=set),
})

CODECS[Molecule] = ObjectCodec(Molecule, {
    'xrefs': ListCodec(item_codec=CODECS[DbXref], list_type=set),
    'variations': ListCodec(item_codec=CODECS[Variation]),
    'canonical_form': CODECS[Specialization],
    'default_form': CODECS[Specialization],
})

CODECS[Reaction] = ObjectCodec(Reaction, {
    'xrefs': ListCodec(item_codec=CODECS[DbXref], list_type=set),
    'stoichiometry': MappingCodec(key_codec=MOLSTUB),
    'catalyst': MOLSTUB,
})

CODECS[Pathway] = ObjectCodec(Pathway, {
    'xrefs': ListCodec(item_codec=CODECS[DbXref], list_type=set),
    'metabolites': ListCodec(item_codec=MOLSTUB),
    'steps': ListCodec(item_codec=RXNSTUB),
    'enzymes': ListCodec(item_codec=MOLSTUB),
})


@dataclass(eq=True, order=True, frozen=True)
class Dataset:
    db: str
    collection: str
    content_type: Type[KbEntry]
    codec: Codec = None

    def __repr__(self):
        return f'{self.db}.{self.collection} [{self.content_type.__name__}]'


class LookupCodec(Codec):
    """Encodes a KbEntry by its ID only; decodes by looking it up in a given dataset."""

    def __init__(self, source, dataset):
        self._source = source
        self._dataset = dataset

    def encode(self, obj):
        return obj.id

    def decode(self, id):
        # TODO: Remove this when switchover is complete
        try:
            id = id['_id']
        except (AttributeError, TypeError):
            # The expected case -- id was an id not a doc
            pass
        return self._source.get(self._dataset, id)


class Session:
    def __init__(self, uri: str = 'mongodb://127.0.0.1:27017', schema: Mapping[str, Dataset] = None):
        self._uri = uri
        self._client = None
        self.schema = {}
        self._cache = {}
        if schema:
            for name, dataset in schema.items():
                self.define_dataset(name, dataset)

    def define_dataset(self, name, dataset):
        """Adds a dataset to the schema of this session."""
        if name not in self.schema and name not in self.__dict__:
            self.schema[name] = dataset

            # The cache is not just to save round-trips to the datastore, but to maximize reuse of decoded instances.
            self._cache[dataset] = {}

            # Make the dataset definitions easily accessible as attributes.
            self.__dict__[name] = dataset
        else:
            raise ValueError(f'Name collision: {name} is an existing attribute of Session')

    @property
    def client(self) -> pymongo.MongoClient:
        """The session's connection to the underlying datastore."""
        if self._client is None:
            self._client = pymongo.MongoClient(self._uri)
        return self._client

    def _cache_value(self, dataset, doc):
        if doc['_id'] not in self._cache[dataset]:
            codec = dataset.codec or CODECS[dataset.content_type]
            self._cache[dataset][doc['_id']] = codec.decode(doc)
        return self._cache[dataset][doc['_id']]

    def get(self, dataset, id):
        """Retrieves the specified entry from the KB by ID, if it exists."""
        if id not in self._cache[dataset]:
            doc = self.client[dataset.db][dataset.collection].find_one(id)
            if doc:
                self._cache_value(dataset, doc)
        return self._cache[dataset].get(id)

    def put(self, dataset, obj, bypass_cache=False):
        """Persists the object to the KB, in the given dataset."""
        if not bypass_cache:
            self._cache[dataset][obj.id] = obj

        codec = dataset.codec or CODECS[dataset.content_type]
        doc = codec.encode(obj)
        self.client[dataset.db][dataset.collection].replace_one({'_id': obj.id}, doc, upsert=True)

    def find(self, dataset, name, include_aka=True):
        """Finds any number of KB entries matching the given name, optionally as an AKA."""
        found = set()
        docs = []
        for doc in self.client[dataset.db][dataset.collection].find(
                {'name': name}).collation({'locale': 'en', 'strength': 1}):
            if doc['_id'] not in found:
                docs.append(doc)
                found.add(doc['_id'])
        if include_aka:
            for doc in self.client[dataset.db][dataset.collection].find(
                    {'aka': name}).collation({'locale': 'en', 'strength': 1}):
                if doc['_id'] not in found:
                    docs.append(doc)
                    found.add(doc['_id'])
        return [self._cache_value(dataset, doc) for doc in docs]

    def xref(self, dataset, xref_id, xref_db=None):
        """Finds any number of KB entries cross-referenced to the given ID."""
        query = {'xrefs.id': xref_id}
        if xref_db:
            query['xrefs.db'] = xref_db

        results = []
        for doc in self.client[dataset.db][dataset.collection].find(query).collation({'locale': 'en', 'strength': 1}):
            results.append(self._cache_value(dataset, doc))
        return results


def configure_kb(uri=None):
    """Returns a Session object configured to access all reference and KB datasets."""
    session = Session(uri=uri)

    # Reference datasets (local copies of external sources)
    session.define_dataset('EC', Dataset('ref', 'EC', KbEntry))
    session.define_dataset('GO', Dataset('ref', 'GO', KbEntry))
    session.define_dataset('CHEBI', Dataset('ref', 'CHEBI', Molecule))
    session.define_dataset('RHEA', Dataset('ref', 'RHEA', Reaction, ObjectCodec(Reaction, {
        'xrefs': ListCodec(item_codec=CODECS[DbXref], list_type=set),
        'stoichiometry': MappingCodec(key_codec=LookupCodec(session, session.CHEBI)),
        'catalyst': MOLSTUB,
    })))

    # The KB proper - compiled, reconciled, integrated
    session.define_dataset('compounds', Dataset('kb', 'compounds', Molecule))
    session.define_dataset('reactions', Dataset('kb', 'reactions', Reaction, ObjectCodec(Reaction, {
        'xrefs': ListCodec(item_codec=CODECS[DbXref], list_type=set),
        'stoichiometry': MappingCodec(key_codec=LookupCodec(session, session.compounds)),
        'catalyst': MOLSTUB,
    })))
    session.define_dataset('pathways', Dataset('kb', 'pathways', Pathway, ObjectCodec(Pathway, {
        'xrefs': ListCodec(item_codec=CODECS[DbXref], list_type=set),
        'metabolites': ListCodec(item_codec=LookupCodec(session, session.compounds)),
        'steps': ListCodec(item_codec=LookupCodec(session, session.reactions)),
        'enzymes': ListCodec(item_codec=MOLSTUB),
    })))
    return session
