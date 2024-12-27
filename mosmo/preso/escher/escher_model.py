from typing import Iterable

from mosmo.model.core import Molecule, Reaction


def escher_model(model_name: str, reactions: Iterable[Reaction]):
    def escher_reaction(reaction: Reaction) -> dict:
        return {
          'id': reaction.id,
          'name': reaction.label,
          'metabolites': {reactant.id: count for reactant, count in reaction.stoichiometry.items()},
          'lower_bound': -1000.0 if reaction.reversible else 0.0,
          'upper_bound': 1000.0,
          'gene_reaction_rule': reaction.catalyst.id if reaction.catalyst else '',
        }

    def escher_metabolite(reactant: Molecule) -> dict:
        return {
          'id': reactant.id,
          'name': reactant.label,
          'compartment': 'any',
          'charge': reactant.charge,
          'formula': reactant.formula,
        }

    def escher_gene(catalyst: Molecule) -> dict:
        return {
          'id': catalyst.id,
          'name': catalyst.label,
        }

    reactions_json = []
    metabolites_json = {}
    genes_json = {}
    for reaction in reactions:
        reactions_json.append(escher_reaction(reaction))
        for reactant in reaction.stoichiometry:
            if reactant not in metabolites_json:
                metabolites_json[reactant] = escher_metabolite(reactant)
        if reaction.catalyst and reaction.catalyst not in genes_json:
            genes_json[reaction.catalyst] = escher_gene(reaction.catalyst)

    return {
        'id': model_name,
        'version': '1',
        'metabolites': list(metabolites_json.values()),
        'reactions': reactions_json,
        'genes': list(genes_json.values()),
        'compartments': {'any': 'anywhere'},
    }
