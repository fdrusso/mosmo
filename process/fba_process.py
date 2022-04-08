from typing import Union

from vivarium.core.process import Process
from vivarium.core.types import State, Update

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
        # Super __init__ sets self.parameters from defaults + config
        super().__init__(config)

        self.network = ReactionNetwork(self.parameters['reactions'])
        self.drivers = self.parameters['drivers']
        self.boundaries = set(self.parameters['boundaries']) | self.drivers.keys()  # Drivers are also boundaries.
        self.gain = self.parameters['gain']

        # Set up the FBA problem. Everything not declared as a driver or boundary is an intermediate.
        self.intermediates = [met for met in self.network.reactants() if met not in self.boundaries]
        self.fba = FbaGd(self.network, self.intermediates, {
            'drivers': ProductionObjective(self.network, {met: 0.0 for met, target in self.drivers.items()})
        })

    def ports_schema(self):
        return {
            'metabolites': {
                met.id: {'_default': 0.0, '_emit': True} for met in self.boundaries
            },
            'fluxes': {
                rxn.id: {'_default': 0.0, '_updater': 'set', '_emit': True} for rxn in self.network.reactions()
            },
        }

    def production_targets(self, states):
        """Calculates target production rates for all driver metabolites, based on current state."""
        # TODO(fdrusso): All of this is preliminary logic. Expect it to evolve. A lot.
        targets = {}
        for met, target in self.drivers.items():
            # Current and target values are concentrations, in mM.
            current = states['metabolites'][met.id]
            # Target rates depend on displacement, invariant to time_step, controlled by a gain parameter.
            targets[met] = (target - current) * self.gain
        return targets

    def ports_update(self, dmdt, velocities, time_step):
        return {
            'metabolites': {
                met.id: rate * time_step for met, rate in dmdt.items()
            },
            'fluxes': {rxn.id: velocity for rxn, velocity in velocities.items()},
        }

    def next_update(self, time_step: Union[float, int], states: State) -> Update:
        # Update the FBA problem with target rates based on current boundary metabolite pools.
        self.fba.update_params({'drivers': self.production_targets(states)})

        # Solve the problem and return updates
        soln = self.fba.solve()

        # Report rates of change for boundary metabolites, and flux for all reactions.
        dmdt = {met: soln.dmdt[self.network.reactant_index(met)] for met in self.boundaries}
        velocities = self.network.reaction_values(soln.velocities)
        return self.ports_update(dmdt, velocities, time_step)
