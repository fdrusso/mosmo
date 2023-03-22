from vivarium.core.composer import Composer
from vivarium.core.engine import Engine, pf
from vivarium.core.types import Processes

from mosmo.knowledge import kb
from viv.process.fba_process import FbaProcess
from viv.process.util import Clamp, Drain

KB = kb.configure_kb()


class SimpleModel(Composer):
    def __init__(self, config: dict):
        super().__init__(config)

    def generate_processes(self, config: dict) -> Processes:
        processes = {}
        boundaries = set()
        if 'clamp' in config:
            boundaries.update(met for met in config['clamp']['targets'])
            processes['clamp'] = Clamp(**config['clamp'])
        if 'drain' in config:
            boundaries.update(met for met in config['drain']['rates'])
            processes['drain'] = Drain(**config['drain'])
        processes['fba_process'] = FbaProcess({**config['fba_process'], 'boundaries': boundaries})
        return processes

    def generate_topology(self, config: dict):
        return {
            'fba_process': {
                'metabolites': ('metabolites',),
                'fluxes': ('fluxes',),
                'pid_data': ('pid_data',),
            },
            'clamp': {
                'metabolites': ('metabolites',),
            },
            'drain': {
                'metabolites': ('metabolites',),
            }
        }


POOLS = {KB.get(KB.compounds, met_id): conc for met_id, conc in [
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
    ('pi', 10.),  # no data
    ('Glc.D.ext', 10.0),  # environment
]}


def build_config():
    glycolysis = KB.find(KB.pathways, 'glycolysis')[0]
    reactions = glycolysis.steps + [KB.get(KB.reactions, 'pts.glc')]
    acCoA = KB.get(KB.compounds, 'accoa')

    return {
        'fba_process': {
            'reactions': reactions,
            'futile_cycles': [
                [KB.get(KB.reactions, 'pfk'), KB.get(KB.reactions, 'fbp')],
                [KB.get(KB.reactions, 'pyk'), KB.get(KB.reactions, 'pps')],
            ],
            'drivers': {
                acCoA: POOLS[acCoA],
            },
            'pid_ki': .1,
            # 'pid_kd': 0,
        },
        'clamp': {
            'targets': {
                met: conc for met, conc in POOLS.items() if met != acCoA
            }
        },
        'drain': {
            'rates': {
                acCoA: POOLS[acCoA] * 0.05,
            }
        },
    }


def main():
    composer = SimpleModel(build_config())
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
