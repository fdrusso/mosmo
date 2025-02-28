# Knowledge Representation in MoSMo

The mosmo framework is designed with interactive modeling in mind, particularly in the context of a Jupyter
notebook. An interactive session may include an exchange such as follows:
```python
from mosmo.knowledge import kb
KB = kb.configure_kb()

pgi = KB.xref_one(KB.RHEA, 'EC:5.3.1.9')
print(f'[{pgi.ref}] {pgi.name}')
for mol, count in pgi.stoichiometry.items():
    print(f'  {count:+d} [{mol.ref}] {mol.name}')
```
 
Which produces the output:
```raw
[RHEA:11816] glucose-6-phosphate isomerase
  -1 [CHEBI:58225] alpha-D-glucose 6-phosphate(2-)
  +1 [CHEBI:57634] beta-D-fructofuranose 6-phosphate(2-)
```

Briefly, this retrieves a reaction from the RHEA dataset, cross-referenced to EC number 5.3.1.9
(phosphoglucoisomerase, or PGI), then prints out some information about the reaction and the molecules that
participate in it. We will discuss in more detail what is underneath such an exchange, both in terms of the
KnowledgeBase code, and the conceptual model that supports it.

## Knowledge Model

The world of bioinformatics relies on a decentralized collection of information and knowledge across the web,
provided and maintained by academic, commercial, and private sources worldwide. The mosmo framework works to
make this knowledge available for interactive modeling, while maintaining connections to the broader
bioinformatics ecosystem. This begins with python classes that represent the units of knowledge available from
the various sources, both in general and specific to the molecular systems modeling domain. These are defined in the
`mosmo.model` package.

### Datasources and Entries
The `Datasource` class describes a source maintaining a collection of knowledge on the web. Most users won't
need to interact with a `Datasource` object directly, except as part of an identifier such as 'CHEBI:58225',
where CHEBI is the datasource. The system pre-defines many datasource such as CHEBI, KEGG, GO, etc. There is
also a built-in registry object named `DS`, which provides easy access to these pre-defined datasources. For
instance, typing:
```python
print(DS.KEGG)
```
produces:
```raw
[KEGG] Kyoto Encyclopedia of Genes and Genomes
```

A `Datasource` object also has a `home` attribute for its homepage, and a series of patterns to generate URLs
for specific entries (more below).

The `KbEntry` class represents one entry in such a datasource, alternately called an entry, record, or object.
Two key attributes are `db`, referring to the providing datasource, and `id`, which is unique within the datasource.
It also has additional attributes common to most kinds of entries, though again the terminology may differ across
sources.
- name: The preferred name of the entry, short but descriptive. Expected to fit on a single line in most contexts.
- aka: Additional names referring to the same entry. Although most datasources try to keep things unambiguous,
  it is not uncommon for akas to be shared across multiple entries, so take care when searching based on aka.
- shorthand: A very short abbreviation, acronym or other label suitable for identifying the entry in charts or diagrams.
- description: A longer description of the entry, with no restriction on length.
- xrefs: Cross-references to entries in other datasources (see below).

On a technical note, `KbEntry` is hashable, which in python terms means it may belong to sets, and may be used as
a key in a dictionary. In practice this means mosmo-based code tends to work directly with fully fleshed-out objects
such as reactions or molecules, as in the example above. This is opposed to much bioinformatics code that passes
around only id strings, forcing an additional lookup before using any actual information.

### Cross-references and URLs
The concept of a cross-reference (or xref) is universal across the bioinformatics knowledge ecosystem, arising from
overlap in the various datasources. For instance, RHEA:11816, KEGG:R00771, and METACYC:PGLUCISOM-RXN all describe
(essentially) the same reaction, as do EC:5.3.1.9 and GO:0004347. Note however that "essentially" is important here,
given differences in editorial standards across these sources. In general, each datasource does attempt to maintain
useful cross-references to others, but we cannot always expect consistency in what each source considers to be
"the same" entry in another source.

Cross-references are represented in the `mosmo.model` package by the `DbXref` class. Like `KbEntry`, `DbXref` has the
attributes `db` and `id`, but it is otherwise empty, serving only as a reference. A `DbXref` can however produce a URL
to bring up the entry in the original datasource. So for example with PGI:
```python
for xref in pgi.xrefs:
    print(xref, xref.url(Reaction))
```

produces:
```raw
EC:5.3.1.9 https://enzyme.expasy.org/EC/5.3.1.9
GO:0004347 http://amigo.geneontology.org/amigo/term/GO:0004347
KEGG:R00771 https://www.genome.jp/entry/R00771
METACYC:PGLUCISOM-RXN https://metacyc.org/META/NEW-IMAGE?object=PGLUCISOM-RXN
```

While `Datasource`, `KbEntry`, and `DbXref` are all inspired by the world of bioinformatics, they are actually
completely domain-agnostic. These classes could equally be used to represent catalogs of automobile parts, music,
airline schedules, or pretty much anything. Next we will discuss classes that are specific to the domain of
molecular systems.

### Molecules
### Reactions
### Pathways

## Knowledge Base
### Dataset vs Datasource
### Reference and Canon
### Codecs
### The KB session