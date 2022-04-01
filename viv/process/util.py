from typing import Mapping, Optional, Union

from vivarium.core.process import Process, Step
from vivarium.core.types import State, Update

from kb import kb

KB = kb.configure_kb()


class Clamp(Step):
    """A Vivarium Process that clamps one or more metabolite pools at set values."""

    def __init__(self, targets: Mapping[str, float]):
        super().__init__()
        self.targets = {KB.get(KB.compounds, met_id): target for met_id, target in targets.items()}

    def ports_schema(self):
        return {
            'metabolites': {
                met.id: {'_default': 0.0, '_emit': True} for met in self.targets
            },
        }

    def next_update(self, time_step: Union[float, int], states: State) -> Update:
        """Runs instantaneously to replenish deficits or drain excess."""
        adjustment = {}
        for met, target in self.targets.items():
            adjustment[met.id] = (target - states['metabolites'][met.id])

        return {'metabolites': adjustment}


class Drain(Process):
    """A Vivarium Process that depletes one or more metabolite pools at set rates, subject to availability."""

    def __init__(self, rates: Mapping[str, float], back_off: float = 0.5):
        super().__init__()
        self.rates = {KB.get(KB.compounds, met_id): rate for met_id, rate in rates.items()}
        self.back_off = back_off

    def ports_schema(self):
        return {
            'metabolites': {
                met.id: {'_default': 0.0, '_emit': True} for met in self.rates
            },
        }

    def next_update(self, time_step: Union[float, int], states: State) -> Update:
        efflux = {}
        for met, rate in self.rates.items():
            requested = rate * time_step
            available = states['metabolites'][met.id] * self.back_off
            efflux[met.id] = -min(requested, available)

        return {'metabolites': efflux}
