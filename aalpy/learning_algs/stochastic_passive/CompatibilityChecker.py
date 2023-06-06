from abc import ABC, abstractmethod
from collections import defaultdict
from math import sqrt, log

from aalpy.learning_algs.stochastic_passive.FPTA import AlergiaPtaNode


class CompatibilityChecker(ABC):

    @abstractmethod
    def are_states_different(self, a: AlergiaPtaNode, b: AlergiaPtaNode, **kwargs) -> bool:
        pass


class HoeffdingCompatibility(CompatibilityChecker):
    def __init__(self, eps):
        self.eps = eps

    def hoeffding_bound(self, a : dict, b : dict):
        n1 = sum(a.values())
        n2 = sum(b.values())

        if n1 * n2 == 0:
            return False

        for o in set(a.keys()).union(b.keys()):
            a_freq = a[o] if o in a else 0
            b_freq = b[o] if o in b else 0

            if abs(a_freq / n1 - b_freq / n2) > ((sqrt(1 / n1) + sqrt(1 / n2)) * sqrt(0.5 * log(2 / self.eps))):
                return True
        return False

    def get_two_stage_dict(self, input_dict : dict):
        ret = defaultdict(dict)
        for (in_sym, out_sym), value in input_dict.items():
            ret[in_sym][out_sym] = value
        return ret

    def are_states_different(self, a: AlergiaPtaNode, b: AlergiaPtaNode, **kwargs):
        # no data availably for any node
        if len(a.input_frequency) * len(b.input_frequency) == 0:
            return False

        # assuming tuples are used for IOAlergia and not as Alergia outputs
        if not isinstance(list(a.input_frequency.keys())[0], tuple):
            return self.hoeffding_bound(a.input_frequency, b.input_frequency)

        # IOAlergia: check hoeffding bound conditioned on inputs
        a_dict, b_dict = (self.get_two_stage_dict(x.input_frequency) for x in [a,b])

        for key in filter(lambda x : x in a_dict.keys(), b_dict.keys()):
            if self.hoeffding_bound(a_dict[key], b_dict[key]) == True:
                return True
        return False

