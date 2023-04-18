from collections import defaultdict

from aalpy.base import Automaton, AutomatonState


class IntervalMdpState(AutomatonState):
    def __init__(self, state_id, output=None):
        super().__init__(state_id)
        self.output = output
        # each child is a tuple (Node(output), (lower_bound, upper_bound))
        self.transitions = defaultdict(list)


class IntervalMdp(Automaton):

    def __init__(self, initial_state: IntervalMdpState, states: list):
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
        pass

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


def interval_mdp_from_learning_data(learned_model, observation_table, confidence=0.9, method='normal'):
    from statsmodels.stats.proportion import proportion_confint

    # copy mdp
    state_dict = dict()
    for state in learned_model.states:
        state_dict[state.state_id] = IntervalMdpState(state.state_id, output=state.output)

    for state in learned_model.states:
        for i, node_output_list in state.transitions.items():
            for node, probability in node_output_list:
                if probability != 1.:
                    freq_dict = observation_table.T[state.prefix][(i,)]
                    total_sum = sum(freq_dict.values())

                    lower, upper = proportion_confint(observation_table.T[state.prefix][(i,)][node.output],
                                                      total_sum, 1-confidence, method)
                    state_dict[state.state_id].transitions[i].append((state_dict[node.state_id], (lower, upper)))
                else:
                    state_dict[state.state_id].transitions[i].append((state_dict[node.state_id], (1, 1)))


    return IntervalMdp(state_dict[learned_model.initial_state.state_id], list(state_dict.values()))



