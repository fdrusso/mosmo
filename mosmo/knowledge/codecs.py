"""Encode and decode JSON (BSON) documents as KB classes.

Codecs define the encoding/decoding schema for KB objects in the storage layer (MongoDB). PyMongo does have a mechanism
to define custom types, but it is not flexible enough to work cleanly in our case. The main cost of the approach
implemented here is that we need to define schema relationships explicitly, with defined types, rather than relying on
the system to infer them. This seems a manageable constraint.
"""
import abc
from typing import Callable, Iterable, Mapping

from mosmo.model import DS, DbXref, KbEntry, Molecule, Reaction, Pathway, Specialization, Variation


class Codec(abc.ABC):
    """Base class for all Codecs."""

    # Implementation Note: Mongo stores data as "documents" that use JSON semantics, i.e. each document is a dict whose
    # values may be scalars, lists, or dicts. To express the semantics of a given codec using python type hints and
    # generics devolves into a specification for JSON itself, which is beyond the scope of what we're trying to do here,
    # and ultimately makes Codec usage _less_ readable. Instead, we rely on subclasses to define and enforce typing of
    # encoded types and resulting documents or fragments.

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
        self.value_codec = value_codec or AS_IS

    def encode(self, mapping):
        return [(self.key_codec.encode(k), self.value_codec.encode(v)) for k, v in mapping.items()]

    def decode(self, doc):
        return self.mapping_type({self.key_codec.decode(k): self.value_codec.decode(v) for k, v in doc})


class ObjectCodec(Codec):
    """Encodes/decodes a python instance to a json-compatible dict."""

    def __init__(self, clazz, codec_map: Mapping[str, Codec] = None, rename: Mapping[str, str] = None):
        self.clazz = clazz
        self.codec_map = codec_map or {}
        self.encoded_name = {}
        self.decoded_name = {}
        if rename:
            for decoded, encoded in rename.items():
                self.encoded_name[decoded] = encoded
                self.decoded_name[encoded] = decoded

    def encode(self, obj):
        doc = {}
        for k, v in obj.__dict__.items():
            codec = self.codec_map.get(k, AS_IS)
            if codec is not None and v is not None:
                k = self.encoded_name.get(k, k)
                doc[k] = codec.encode(v)
        return doc

    def decode(self, doc):
        args = {}
        for k, v in doc.items():
            codec = self.codec_map.get(k, AS_IS)
            k = self.decoded_name.get(k, k)
            args[k] = codec.decode(v)
        return self.clazz(**args)


class ObjectIdCodec(Codec):
    """(Lossy) codec that encodes only the ID of an object, and decodes it as an otherwise empty object.

    This is useful for cases where the document includes only a reference to the object, but the caller either does
    not know, or does not care about the details of that object.
    """

    def __init__(self, clazz):
        self.clazz = clazz

    def encode(self, obj):
        return obj.id

    def decode(self, id):
        return self.clazz(id)


DB_ID = ObjectIdCodec(lambda id: DS.get(id))
MOL_ID = ObjectIdCodec(Molecule)
RXN_ID = ObjectIdCodec(Reaction)

CODECS = {
    DbXref: ObjectCodec(DbXref, {'db': DB_ID}),
    Variation: ObjectCodec(Variation, {'form_names': ListCodec()}),
    Specialization: ObjectCodec(Specialization, {'form': ListCodec(list_type=tuple)}),
}

CODECS[KbEntry] = ObjectCodec(
    KbEntry,
    codec_map={
        'db': DB_ID,
        'xrefs': ListCodec(item_codec=CODECS[DbXref], list_type=set)
    },
    rename={"id": "_id"}
)

CODECS[Molecule] = ObjectCodec(
    Molecule,
    codec_map={
        'db': DB_ID,
        'xrefs': ListCodec(item_codec=CODECS[DbXref], list_type=set),
        'variations': ListCodec(item_codec=CODECS[Variation]),
        'canonical_form': CODECS[Specialization],
        'default_form': CODECS[Specialization],
    },
    rename={"id": "_id"}
)

CODECS[Reaction] = ObjectCodec(
    Reaction,
    codec_map={
        'db': DB_ID,
        'xrefs': ListCodec(item_codec=CODECS[DbXref], list_type=set),
        'stoichiometry': MappingCodec(key_codec=MOL_ID),
        'catalyst': MOL_ID,
    },
    rename={"id": "_id"}
)

CODECS[Pathway] = ObjectCodec(
    Pathway,
    codec_map={
        'db': DB_ID,
        'xrefs': ListCodec(item_codec=CODECS[DbXref], list_type=set),
        'metabolites': ListCodec(item_codec=MOL_ID),
        'steps': ListCodec(item_codec=RXN_ID),
        'enzymes': ListCodec(item_codec=MOL_ID),
    },
    rename={"id": "_id"}
)
