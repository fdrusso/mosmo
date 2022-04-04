from typing import Iterable, Mapping, Union

from vivarium.core.process import Process
from vivarium.core.types import State, Update

from model.core import Molecule, Reaction
from model.reaction_network import ReactionNetwork
from sim.fba_gd import FbaGd, ProductionObjective


class FbaProcess(Process):
    """A Vivarium Process simulating a ReactionNetwork via gradient-descent FBA."""

    defaults = {
        'reactions': [],
        'drivers': {},
        'boundaries': [],
        'gain': 0.5,
    }

    def __init__(self, config):
        config = {**FbaProcess.defaults, **config}
        super().__init__(config)

        self.network = ReactionNetwork(config['reactions'])
        self.drivers = config['drivers']
        self.boundaries = set(config['boundaries'])
        self.gain = config['gain']

        # Set up the FBA problem. Everything not declared as a driver or boundary is an intermediate.
        self.intermediates = [met for met in self.network.reactants()
                              if met not in self.drivers and met not in self.boundaries]
        self.fba = FbaGd(self.network, self.intermediates, {
            'drivers': ProductionObjective(self.network, {met: 0.0 for met, target in self.drivers.items()})
        })

    def ports_schema(self):
        return {
            'metabolites': {
                met.id: {'_default': 0.0, '_emit': True} for met in (self.boundaries | self.drivers.keys())
            },
            'fluxes': {
                rxn.id: {'_updater': 'set', '_emit': True} for rxn in self.network.reactions()
            },
        }

    def next_update(self, time_step: Union[float, int], states: State) -> Update:
        # Update the FBA problem with target rates based on current boundary metabolite pools.
        targets = {}
        for met, target in self.drivers.items():
            # Current and target values are concentrations, in mM.
            current = states['metabolites'][met.id]
            # Target rates depend on displacement, invariant to time_step, controlled by a gain parameter.
            targets[met] = (target - current) * self.gain
        self.fba.update_params({'drivers': targets})

        # Solve the problem and return updates
        soln = self.fba.solve()
        rates = self.network.reactant_values(soln.dmdt)
        velocities = self.network.reaction_values(soln.velocities)
        return {
            'metabolites': {
                met.id: rate * time_step for met, rate in rates.items()
                if met not in self.drivers and met not in self.boundaries
            },
            'fluxes': {rxn.id: velocity for rxn, velocity in velocities.items()},
        }
