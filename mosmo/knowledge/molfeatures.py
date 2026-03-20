"""Standard VariableFeatures available for use on Molecules."""
from mosmo.model import VariableFeature, FEATURE_TYPES

FEATURE_TYPES.register(VariableFeature('DL', ['D', 'L']))

FEATURE_TYPES.register(VariableFeature(
    'PhosphoSugars',
    ['NONE',
     ('1P', 'P1'), ('2P', 'P2'), ('3P', 'P3'), ('4P', 'P4'), ('5P', 'P5'), ('6P', 'P6'), ('7P', 'P7'),
     'bis15', 'bis16', 'bis17']))

FEATURE_TYPES.register(VariableFeature(
    'Tautomerism',
    ['open', ('r5α', 'R5A'), ('r5β', 'R5B'), ('r6α', 'R6A'), ('r6β', 'R6B')]))

FEATURE_TYPES.register(VariableFeature(
    'Protonation',
    [('4-', 'M4'), ('3-', 'M3'), ('2-', 'M2'), ('1-', 'M1'),
     'neut',
     ('1+', 'P1'), ('2+', 'P2'), ('3+', 'P3'), ('4+', 'P4')]))

FEATURE_TYPES.register(VariableFeature('NucleotideForm', ['base', 'ribo', 'deoxy']))

FEATURE_TYPES.register(VariableFeature('PhosphoNucleotide', ['mono', 'di', 'tri', 'cyc35', 'cyc23']))

FEATURE_TYPES.register(VariableFeature('Chelation', ['mg']))
