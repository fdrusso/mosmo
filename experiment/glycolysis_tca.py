from bson import ObjectId
from vivarium.core.composer import Composer
from vivarium.core.engine import Engine, pf
from vivarium.core.types import Processes

from kb import kb
from process.fba_process import FbaProcess
from process.util import Clamp, Drain

KB = kb.configure_kb()


class MultiProcessModel(Composer):
    def __init__(self, config: dict):
        super().__init__(config)

    def generate_processes(self, config: dict) -> Processes:
        processes = {}
        boundaries = {}
        if 'clamp' in config:
            processes['clamp'] = Clamp(**config['clamp'])
            boundaries.update({met: None for met in config['clamp']['targets']})
        if 'drain' in config:
            processes['drain'] = Drain(**config['drain'])
            boundaries.update({met: None for met in config['drain']['rates']})
        for fba_config in config.get('fba_process', []):
            processes[fba_config['name']] = FbaProcess({**fba_config, 'boundaries': boundaries})

        return processes

    def generate_topology(self, config: dict):
        topology = {
            'clamp': {
                'metabolites': ('metabolites',),
            },
            'drain': {
                'metabolites': ('metabolites',),
            }
        }
        for fba_config in config.get('fba_process', []):
            topology[fba_config['name']] = {
                'metabolites': ('metabolites',),
                'fluxes': ('fluxes',),
            }
        return topology


def main():
    glycolysis = KB.get(KB.pathways, ObjectId("61e21657e4819e9d1a81f65f"))
    ppp = KB.get(KB.pathways, ObjectId("61e21657e4819e9d1a81f660"))
    tca = KB.get(KB.pathways, ObjectId("61e21657e4819e9d1a81f661"))
    glx_shunt = KB.get(KB.pathways, ObjectId("61e21657e4819e9d1a81f65e"))

    acCoA = KB.get(KB.compounds, 'accoa')
    akg = KB.get(KB.compounds, 'akg')
    oaa = KB.get(KB.compounds, 'oaa')

    # Concentrations in mM, from http://book.bionumbers.org/what-are-the-concentrations-of-free-metabolites-in-cells
    concs = {KB.get(KB.compounds, met_id): conc for met_id, conc in {
        'akg': 0.44,
        'accoa': 0.61,
        'adp': 0.55,
        'amp': 0.28,
        'atp': 9.6,
        'co2': 0.01,
        'coa': 1.4,
        'h+': 1e-7,
        'h2o': 55500,
        'nad.ox': 2.6,
        'nad.red': 0.083,
        'nadp.ox': 0.0021,
        'nadp.red': 0.12,
        'oaa': 0.1,  # no data
        'pep': 0.18,
        'pi': 10.,  # no data
        'pyr': 0.1,  # no data
        'q.ox': 0.01,  # no data
        'q.red': 0.01,  # no data
        'Glc.D.ext': 10.0,  # environment
    }.items()}

    config = {
        'fba_process': [
            {
                'name': 'Glycolysis+',
                'reactions': [
                    *glycolysis.steps,
                    *ppp.steps,
                    KB.get(KB.reactions, 'pts.glc'),
                ],
                'drivers': {
                    acCoA: concs[acCoA],
                },
            },
            {
                'name': 'TCA',
                'reactions': [
                    *tca.steps,
                    *glx_shunt.steps,
                    KB.get(KB.reactions, 'mae.nad'),
                    KB.get(KB.reactions, 'mae.nadp'),
                    KB.get(KB.reactions, 'ppc'),
                    KB.get(KB.reactions, 'ppck'),
                ],
                'drivers': {
                    acCoA: concs[acCoA],
                    akg: concs[akg],
                    oaa: concs[oaa],
                },
                'gain': 0.3,
            },
        ],
        'clamp': {
            'targets': {
                met: conc for met, conc in concs.items() if met != acCoA
            }
        },
        'drain': {
            'rates': {
                acCoA: concs[acCoA] * 0.01,
                akg: concs[akg] * 0.03,
                oaa: concs[oaa] * 0.02,
            }
        },
    }
    composer = MultiProcessModel(config)
    composite = composer.generate()

    # Build and run the engine
    sim = Engine(composite=composite,
                 initial_state={
                     'metabolites': {
                         met.id: conc for met, conc in concs.items()
                     }
                 })
    total_time = 10
    sim.update(total_time)

    # get the data
    data = sim.emitter.get_data()
    print(pf(data))


if __name__ == '__main__':
    main()
