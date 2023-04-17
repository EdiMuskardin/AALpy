import random
from collections import defaultdict

from aalpy.base import Automaton, AutomatonState


class MdpState(AutomatonState):
    def __init__(self, state_id, output=None):
        super().__init__(state_id)
        self.output = output
        # each child is a tuple (Node(output), probability)
        self.transitions = defaultdict(list)


class Mdp(Automaton):
    """Markov Decision Process."""
    def __init__(self, initial_state: MdpState, states: list):
        super().__init__(initial_state, states)

    def reset_to_initial(self):
        self.current_state = self.initial_state

    def step(self, letter):
        """Next step is determined based on transition probabilities of the current state.

        Args:

            letter: input

        Returns:

            output of the current state
        """
        if letter is None:
            return self.current_state.output

        probability_distributions = [i[1] for i in self.current_state.transitions[letter]]
        states = [i[0] for i in self.current_state.transitions[letter]]

        new_state = random.choices(states, probability_distributions, k=1)[0]

        self.current_state = new_state
        return self.current_state.output

    def step_to(self, inp, out):
        """Performs a step on the automaton based on the input `inp` and output `out`.

        Args:

            inp: input
            out: output

        Returns:

            output of the reached state, None otherwise
        """
        for new_state in self.current_state.transitions[inp]:
            if new_state[0].output == out:
                self.current_state = new_state[0]
                return out
        return None

    def to_interval_mdp(self, observation_table, confidence=0.9, method='normal'):
        from aalpy.automata import interval_mdp_from_learning_data
        return interval_mdp_from_learning_data(self, observation_table, confidence, method)

# def to_interval_mdp(self, confidence=0.9, sample_size=100, method='normal'):
    #     from statsmodels.stats.proportion import proportion_confint
    #     from aalpy.automata.IntervalMdp import IntervalMdpState, IntervalMdp
    #
    #     # copy mdp
    #     state_dict = dict()
    #     for state in self.states:
    #         state_dict[state.state_id] = IntervalMdpState(state.state_id, output=state.output)
    #
    #     for state in self.states:
    #         for i, node_output_list in state.transitions.items():
    #             for node, probability in node_output_list:
    #                 if probability != 1.:
    #                     lower, upper = proportion_confint(int(sample_size * probability), sample_size, confidence, method)
    #                     state_dict[state.state_id].transitions[i].append((state_dict[node.state_id], (lower, upper)))
    #                 else:
    #                     state_dict[state.state_id].transitions[i].append((state_dict[node.state_id], (0, 1)))
    #
    #     return IntervalMdp(state_dict[self.initial_state.state_id], list(state_dict.values()))
