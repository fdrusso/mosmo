"""Knowledge Base for Molecular Systems Modeling.

This constitutes a schema definition for a mosmo.knowledge.session.Session. Conventionally this would be managed
with a schema.xml or schema.json file, with a parser that does all the actual configuration. But it is easier, more
powerful, and just as maintainable to express the schema directly via python code.
"""
from pymongo import MongoClient

from mosmo.knowledge.codecs import AS_IS, CODECS, ListCodec, MappingCodec, ObjectCodec
from mosmo.model import DS, KbEntry, Molecule, Reaction, Pathway, Specialization, Variation
from .session import Dataset, Session, XrefCodec

import mosmo.knowledge.datasources  # KEEP: Defines standard datasources referred to below.


def configure_kb(uri: str = 'mongodb://127.0.0.1:27017'):
    """Returns a Session object configured to access all reference and canonical KB datasets."""
    session = Session(MongoClient(uri))

    # Define codecs for model.core types.
    codecs = dict(CODECS)
    codecs[Variation] = ObjectCodec(Variation, codec_map={'name': AS_IS, 'form_names': AS_IS})

    codecs[Specialization] = ObjectCodec(
        Specialization,
        codec_map={
            'parent_id': AS_IS,
            'form': ListCodec(list_type=tuple),
            'child_id': AS_IS,
        })

    codecs[Molecule] = ObjectCodec(
        Molecule,
        parent=codecs[KbEntry],
        codec_map={
            'formula': AS_IS,
            'mass': AS_IS,
            'charge': AS_IS,
            'inchi': AS_IS,
            'variations': ListCodec(item_codec=CODECS[Variation]),
            'canonical_form': CODECS[Specialization],
            'default_form': CODECS[Specialization],
        })

    codecs[Reaction] = ObjectCodec(
        Reaction,
        parent=codecs[KbEntry],
        codec_map={
            'stoichiometry': MappingCodec(key_codec=XrefCodec(session, Molecule)),
            'catalyst': XrefCodec(session, Molecule),
            'reversible': AS_IS,
        })

    codecs[Pathway] = ObjectCodec(
        Pathway,
        parent=codecs[KbEntry],
        codec_map={
            'reactions': ListCodec(item_codec=XrefCodec(session, Reaction)),
            'diagram': AS_IS,
        })

    # Reference datasets (local copies of external sources)
    session.define_dataset(Dataset('EC', DS.EC, KbEntry, 'ref', 'EC', codecs[KbEntry]))
    session.define_dataset(Dataset('GO', DS.GO, KbEntry, 'ref', 'GO', codecs[KbEntry]))
    session.define_dataset(Dataset('CHEBI', DS.CHEBI, Molecule, 'ref', 'CHEBI', codecs[Molecule]))
    session.define_dataset(Dataset('RHEA', DS.RHEA, Reaction, 'ref', 'RHEA', codecs[Reaction]))

    # The KB proper - compiled, reconciled, integrated, canonical
    session.define_dataset(
        Dataset('compounds', DS.CANON, Molecule, 'kb', 'compounds', codecs[Molecule], canonical=True))
    session.define_dataset(
        Dataset('reactions', DS.CANON, Reaction, 'kb', 'reactions', codecs[Reaction], canonical=True))
    session.define_dataset(
        Dataset('pathways', DS.CANON, Pathway, 'kb', 'pathways', codecs[Pathway], canonical=True))
    return session
