"""Knowledge Base for Molecular Systems Modeling.

This constitutes a schema definition for a mosmo.knowledge.session.Session. Conventionally this would be managed
with a schema.xml or schema.json file, with a parser that does all the actual configuration. But it is easier, more
powerful, and just as maintainable to express the schema directly via python code.
"""
from pymongo import MongoClient

from mosmo.knowledge.codecs import AS_IS, CODECS, ListCodec, LookupCodec, MappingCodec, ObjectCodec, ChainLookupCodec
from mosmo.model import DS, KbEntry, Molecule, Reaction, Pathway, DbXref
from mosmo.knowledge.molfeatures import FEATURE_TYPES
from .session import Dataset, Session, XrefCodec


def configure_kb(uri: str = 'mongodb://127.0.0.1:27017'):
    """Returns a Session object configured to access all reference and canonical KB datasets."""
    session = Session(MongoClient(uri))

    # Define codecs for model.core types.
    codex = dict(CODECS)

    feature_codec = LookupCodec(FEATURE_TYPES, 'label')
    variant_codec = ChainLookupCodec(feature_codec, 'feature', 'label')
    formname_codec = ListCodec(item_codec=variant_codec, list_type=tuple)

    codex[Molecule] = ObjectCodec(
        Molecule,
        parent=codex[KbEntry],
        codec_map={
            'formula': AS_IS,
            'mass': AS_IS,
            'charge': AS_IS,
            'structure': AS_IS,
            'canonical_form': AS_IS,
            'form_name': formname_codec,
            'features': MappingCodec(value_codec=feature_codec),
            'child_forms': MappingCodec(key_codec=formname_codec),
            'default_form': AS_IS,
        })

    codex[Reaction] = ObjectCodec(
        Reaction,
        parent=codex[KbEntry],
        codec_map={
            'stoichiometry': MappingCodec(key_codec=XrefCodec(session, Molecule)),
            'catalyst': XrefCodec(session, Molecule),
            'reversible': AS_IS,
        })

    codex[Pathway] = ObjectCodec(
        Pathway,
        parent=codex[KbEntry],
        codec_map={
            'reactions': ListCodec(item_codec=XrefCodec(session, Reaction)),
            'diagram': AS_IS,
        })

    # Reference datasets (local copies of external sources)
    session.define_dataset(Dataset('EC', DS.EC, KbEntry, 'ref', 'EC', codex[KbEntry]))
    session.define_dataset(Dataset('GO', DS.GO, KbEntry, 'ref', 'GO', codex[KbEntry]))
    session.define_dataset(Dataset('CHEBI', DS.CHEBI, Molecule, 'ref', 'CHEBI', codex[Molecule]))
    session.define_dataset(Dataset('RHEA', DS.RHEA, Reaction, 'ref', 'RHEA', codex[Reaction]))

    # The KB proper - compiled, reconciled, integrated, canonical
    session.define_dataset(
        Dataset('compounds', DS.CANON, Molecule, 'kb', 'compounds', codex[Molecule], canonical=True))
    session.define_dataset(
        Dataset('reactions', DS.CANON, Reaction, 'kb', 'reactions', codex[Reaction], canonical=True))
    session.define_dataset(
        Dataset('pathways', DS.CANON, Pathway, 'kb', 'pathways', codex[Pathway], canonical=True))
    return session
