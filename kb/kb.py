"""Knowledge Base for Molecular Systems Modeling."""
from typing import Callable, Iterable, List, Optional

import pymongo

from scheme import DbXref, KbEntry, Molecule, Reaction, Specialization, Variation


class Connection:
    """Manages the connection to the MongoDB storage layer."""

    def __init__(self, uri: str = 'mongodb://127.0.0.1:27017', db=None):
        self._uri = uri
        self._client = None

    def connect(self, uri: Optional[str] = None):
        if uri is not None:
            self._uri = uri
        self._client = pymongo.MongoClient(self._uri)

    @property
    def client(self) -> pymongo.MongoClient:
        if self._client is None:
            self.connect()
        return self._client


KB = Connection().client.kb
REFDB = Connection().client.ref


# Defines the encoding/decoding schema for KB objects in the storage layer. PyMongo does have a mechanism to define
# custom types, but it is not flexible enough to do this cleanly in our case. The main cost is we need to define the
# schema relationships with explicit types, rather than having the system infer them. That is a reasonable constraint.

class AsIsCodec:
    """No-op codec passes everything through encode and decode as-is."""
    def encode(self, obj):
        return obj

    def decode(self, data):
        return data


class ObjectCodec:
    """Encodes/decodes a python instance to a json-compatible dict."""

    def __init__(self, clazz, codec_map=None, selective=False):
        self.clazz = clazz
        self.codec_map = codec_map or {}
        self.selective = selective

    def encode(self, obj):
        data = {}
        for k, v in obj.__dict__.items():
            if v is not None and (not self.selective or k in self.codec_map):
                data[k] = self.codec_map.get(k, AS_IS).encode(v)
        return data

    def decode(self, data):
        args = {}
        for k, v in data.items():
            if not self.selective or k in self.codec_map:
                args[k] = self.codec_map.get(k, AS_IS).decode(v)
        return self.clazz(**args)


class ListCodec:
    """Encodes/decodes a python iterable type to a json-compatible list."""

    def __init__(self, item_codec=None, list_type:Callable[[Iterable], Iterable] = list):
        self.list_type = list_type
        self.item_codec = item_codec or AS_IS

    def encode(self, items):
        return list(self.item_codec.encode(item) for item in items)

    def decode(self, data):
        return self.list_type(self.item_codec.decode(datum) for datum in data)


class MappingCodec:
    """Encodes/decodes a python mapping type to a json list of tuples."""

    def __init__(self, key_codec=None, value_codec=None, mapping_type=dict):
        self.mapping_type = mapping_type
        self.key_codec = key_codec or AS_IS
        self.value_code = value_codec or AS_IS

    def encode(self, mapping):
        return [[self.key_codec.encode(k), self.value_code.encode(v)] for k, v in mapping.items()]

    def decode(self, data):
        return self.mapping_type({self.key_codec.decode(k): self.value_code.decode(v) for k, v in data})


AS_IS = AsIsCodec()

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
    'stoichiometry': MappingCodec(key_codec=ObjectCodec(Molecule, {'_id': AS_IS, 'name': AS_IS}, selective=True)),
    'catalyst': ObjectCodec(Molecule, {'_id': AS_IS, 'name': AS_IS}, selective=True),
})


def _get(id, source, codec):
    doc = source.find_one(id)
    if doc:
        return codec.decode(doc)
    else:
        return None


def _find(name, source, codec, include_aka=True):
    found = set()
    docs = []
    for doc in source.find({'name': name}).collation({'locale': 'en', 'strength': 1}):
        if doc['_id'] not in found:
            docs.append(doc)
            found.add(doc['_id'])
    if include_aka:
        for doc in source.find({'aka': name}).collation({'locale': 'en', 'strength': 1}):
            if doc['_id'] not in found:
                docs.append(doc)
                found.add(doc['_id'])
    return [codec.decode(doc) for doc in docs]


def _xref(xref_id, xref_db, source, codec):
    query = {'xrefs.id': xref_id}
    if xref_db:
        query['xrefs.db'] = xref_db

    results = []
    for doc in source.find(query).collation({'locale': 'en', 'strength': 1}):
        results.append(codec.decode(doc))
    return results


def get_molecule(compound_id, source=KB.compounds) -> Optional[Molecule]:
    """Retrieve a single molecule by ID."""
    return _get(compound_id, source, CODECS[Molecule])


def find_molecules(name, source=KB.compounds, include_aka=True) -> List[Molecule]:
    """Retrieve molecules by name, or AKA. Matches case-insensitively on full name."""
    return _find(name, source, CODECS[Molecule], include_aka)


def xref_molecules(xref_id, xref_db=None, source=KB.compounds) -> List[Molecule]:
    """Retrieve molecules by xref. Matches case-insensitively on ID, and optionally, db."""
    return _xref(xref_id, xref_db, source, CODECS[Molecule])


def get_reaction(compound_id, source=KB.compounds) -> Optional[Reaction]:
    """Retrieve a single reaction by ID."""
    return _get(compound_id, source, CODECS[Reaction])


def find_reactions(name, source=KB.compounds, include_aka=True) -> List[Reaction]:
    """Retrieve reactions by name, or AKA. Matches case-insensitively on full name."""
    return _find(name, source, CODECS[Reaction], include_aka)


def xref_reactions(xref_id, xref_db=None, source=KB.compounds) -> List[Reaction]:
    """Retrieve reactions by xref. Matches case-insensitively on ID, and optionally, db."""
    return _xref(xref_id, xref_db, source, CODECS[Reaction])
