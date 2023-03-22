from vivarium.core.composer import Composer
from vivarium.core.engine import Engine, pf
from vivarium.core.types import Processes

from mosmo.knowledge import kb
from viv.process.fba_process import FbaProcess
from viv.process.util import Clamp, Drain

KB = kb.configure_kb()


class MultiProcessModel(Composer):
    def __init__(self, config: dict):
        super().__init__(config)

    def generate_processes(self, config: dict) -> Processes:
        processes = {}

        # Clamped and/or drained quantities are assumed to be boundaries for other processes.
        boundaries = {}
        if 'clamp' in config:
            processes['clamp'] = Clamp(**config['clamp'])
            boundaries.update({met: None for met in config['clamp']['targets']})
        if 'drain' in config:
            processes['drain'] = Drain(**config['drain'])
            boundaries.update({met: None for met in config['drain']['rates']})

        for process_config in config.get('fba_process', []):
            # Combine global boundaries with process-specific boundaries, if any
            process_boundaries = {*boundaries}
            if 'boundaries' in process_config:
                process_boundaries.update(process_config['boundaries'])
            # Combined boundaries will override entry in process_confif if it is present.
            processes[process_config['name']] = FbaProcess({**process_config, 'boundaries': process_boundaries})

        return processes

    def generate_topology(self, config: dict):
        topology = {}
        if 'clamp' in config:
            topology['clamp'] = {
                'metabolites': ('metabolites',),
            }
        if 'drain' in config:
            topology['drain'] = {
                'metabolites': ('metabolites',),
            }
        for fba_config in config.get('fba_process', []):
            topology[fba_config['name']] = {
                'metabolites': ('metabolites',),
                'fluxes': ('fluxes',),
                'pid_data': ('pid_data',),
            }
        return topology


# Concentrations in mM, from http://book.bionumbers.org/what-are-the-concentrations-of-free-metabolites-in-cells
POOLS = {KB.get(KB.compounds, met_id): conc for met_id, conc in [
    ('akg', 0.44),
    ('accoa', 0.61),
    ('adp', 0.55),
    ('amp', 0.28),
    ('atp', 9.6),
    ('co2', 0.01),
    ('coa', 1.4),
    ('h+', 1e-7),
    ('h2o', 55500),
    ('nad.ox', 2.6),
    ('nad.red', 0.083),
    ('nadp.ox', 0.0021),
    ('nadp.red', 0.12),
    ('oaa', 0.1),  # no data
    ('pep', 0.18),
    ('pi', 10.),  # no data
    ('pyr', 0.1),  # no data
    ('q.ox', 0.01),  # no data
    ('q.red', 0.01),  # no data
    ('Glc.D.ext', 10.0),  # environment
]}


def build_config():
    glycolysis = KB.find(KB.pathways, 'glycolysis')[0]
    ppp = KB.find(KB.pathways, 'pentose phosphate')[0]
    tca = KB.find(KB.pathways, 'tca')[0]
    glx_shunt = KB.find(KB.pathways, 'glyoxylate shunt')[0]

    pep = KB.get(KB.compounds, 'pep')
    pyr = KB.get(KB.compounds, 'pyr')
    acCoA = KB.get(KB.compounds, 'accoa')
    akg = KB.get(KB.compounds, 'akg')
    oaa = KB.get(KB.compounds, 'oaa')

    return {
        'fba_process': [
            {
                'name': 'Glycolysis+',
                'reactions': [
                    *glycolysis.steps,
                    *ppp.steps,
                    KB.get(KB.reactions, 'pts.glc'),
                ],
                'futile_cycles': [
                    [KB.get(KB.reactions, 'pfk'), KB.get(KB.reactions, 'fbp')],
                    [KB.get(KB.reactions, 'pyk'), KB.get(KB.reactions, 'pps')],
                ],
                'drivers': {
                    acCoA: POOLS[acCoA],
                    pep: POOLS[pep],
                    pyr: POOLS[pyr],
                },
                'pid_kp': 0.5,
                'pid_ki': 0.1,
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
                    akg: POOLS[akg],
                    oaa: POOLS[oaa],
                },
                'boundaries': [
                    pyr,
                    pep,
                    acCoA,
                ],
                'pid_kp': 0.5,
                'pid_ki': 0.1,
            },
        ],
        'clamp': {
            'targets': {
                met: conc for met, conc in POOLS.items() if met not in (pyr, pep, acCoA, oaa, akg)
            }
        },
        'drain': {
            'rates': {
                acCoA: POOLS[acCoA] * 0.01,
                akg: POOLS[akg] * 0.03,
                oaa: POOLS[oaa] * 0.02,
            }
        },
    }


def main():
    config = build_config()
    composer = MultiProcessModel(config)
    composite = composer.generate()

    # Build and run the engine
    sim = Engine(composite=composite,
                 initial_state={
                     'metabolites': {
                         met.id: conc for met, conc in POOLS.items()
                     }
                 })
    total_time = 10
    sim.update(total_time)

    # get the data
    data = sim.emitter.get_data()
    print(pf(data))


if __name__ == '__main__':
    main()
