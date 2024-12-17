"""Standard Datasource definitions for the domain of Molecular Systems Modelling."""
from mosmo.model import Datasource, DS, KbEntry, Molecule, Reaction, Pathway


DS.define(Datasource(
    id="CANON",
    name="Curated Knowledge Base",
    home="https://github.com/fdrusso/mosmo"
))

DS.define(Datasource(
    id="BIGG",
    name="BiGG Models",
    home="http://bigg.ucsd.edu/",
    urlpat={
        Molecule: "http://bigg.ucsd.edu/search?query={id}",
        Reaction: "http://bigg.ucsd.edu/search?query={id}",
    }
))

DS.define(Datasource(
    id="CAS",
    name="Chemical Abstracts Service",
    home="https://www.cas.org/cas-data/cas-registry"
))

DS.define(Datasource(
    id="CHEBI",
    name="Chemical Entities of Biological Interest (ChEBI)",
    home="https://www.ebi.ac.uk/chebi/",
    urlpat={
        Molecule: "http://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:{id}"
    }
))

DS.define(Datasource(
    id="EC",
    name="Enzyme Commission",
    home="https://enzyme.expasy.org/",
    urlpat={
        KbEntry: "https://enzyme.expasy.org/EC/{id}"
    }
))

DS.define(Datasource(
    id="ECOCYC",
    name="EcoCyc: Encyclopedia of E. coli Genes and Metabolism",
    home="https://ecocyc.org/",
    urlpat={
        Molecule: "https://ecocyc.org/compound?id={id}",
        Reaction: "https://ecocyc.org/ECOLI/NEW-IMAGE?object={id}",
        Pathway: "https://ecocyc.org/ECOLI/NEW-IMAGE?object={id}",
    }
))

DS.define(Datasource(
    id="GO",
    name="Gene Ontology",
    home="http://geneontology.org/",
    urlpat={
        KbEntry: "http://amigo.geneontology.org/amigo/term/GO:{id}"
    }
))

DS.define(Datasource(
    id="KEGG",
    name="Kyoto Encyclopedia of Genes and Genomes",
    home="https://www.genome.jp/kegg/",
    urlpat={
        Molecule: "https://www.genome.jp/entry/{id}",
        Reaction: "https://www.genome.jp/entry/{id}",
        Pathway: "https://www.genome.jp/pathway/{id}"
    }
))

DS.define(Datasource(
    id="LINCS",
    name="The Library of Integrated Network-Based Cellular Signatures (LINCS)",
    home="https://lincsportal.ccs.miami.edu/SmallMolecules/",
    urlpat={
        Molecule: "https://lincsportal.ccs.miami.edu/SmallMolecules/view/{id}"
    }
))

DS.define(Datasource(
    id="MACIE",
    name="Mechanism Annotation and Classification in Enzymes",
    home="https://www.ebi.ac.uk/thornton-srv/m-csa/",
    urlpat={
        Reaction: "https://www.ebi.ac.uk/thornton-srv/m-csa/entry/{id}"
    }
))

DS.define(Datasource(
    id="METACYC",
    name="MetaCyc: Metabolic Pathways From all Domains of Life",
    home="https://metacyc.org/",
    urlpat={
        Molecule: "https://metacyc.org/compound?id={id}",
        Reaction: "https://metacyc.org/META/NEW-IMAGE?object={id}",
        Pathway: "https://metacyc.org/META/NEW-IMAGE?object={id}",
    }
))

DS.define(Datasource(
    id="METANETX",
    name="MetaNetX",
    home="https://www.metanetx.org/",
    urlpat={
        Molecule: "https://www.metanetx.org/chem_info/{id}",
        Reaction: "https://www.metanetx.org/equa_info/{id}"
    }
))

DS.define(Datasource(
    id="REACT",
    name="Reactome",
    home="https://reactome.org/",
    urlpat={
        Reaction: "https://www.reactome.org/content/detail/{id}"
    }
))

DS.define(Datasource(
    id="RHEA",
    name="Rhea, the reaction knowledgebase",
    home="https://www.rhea-db.org/",
    urlpat={
        Reaction: "https://www.rhea-db.org/rhea/{id}"
    }
))

DS.define(Datasource(
    id="WIKI",
    name="Wikipedia",
    home="https://en.wikipedia.org/",
    urlpat={
        KbEntry: "https://en.wikipedia.org/wiki/{id}"
    }
))
