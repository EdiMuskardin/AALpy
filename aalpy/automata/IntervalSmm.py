from collections import defaultdict

from aalpy.automata import IntervalMdpState, IntervalMdp
from aalpy.base import AutomatonState, Automaton


class IntervalSmmState(AutomatonState):
    """ """

    def __init__(self, state_id):
        super().__init__(state_id)
        # each child is a tuple (newNode, output, (lower_bound, upper_bound))
        self.transitions = defaultdict(list)


class IntervalSmm(Automaton):

    def __init__(self, initial_state: IntervalSmmState, states: list):
        super().__init__(initial_state, states)

    def reset_to_initial(self):
        self.current_state = self.initial_state

    def step(self, letter):
        pass

    def step_to(self, inp, out):
        """Performs a step on the automaton based on the input `inp` and output `out`.

        Args:

            inp: input
            out: output

        Returns:

            output of the reached state, None otherwise

        """
        for (new_state, output, prob) in self.current_state.transitions[inp]:
            if output == out:
                self.current_state = new_state
                return out
        return

    def to_interval_mdp(self):
        inputs = self.get_input_alphabet()
        mdp_states = []
        smm_state_to_mdp_state = dict()
        init_state = IntervalMdpState("0", "___start___")
        mdp_states.append(init_state)
        for s in self.states:
            incoming_edges = defaultdict(list)
            incoming_outputs = set()
            for pre_s in self.states:
                for i in inputs:
                    incoming_edges[i] += filter(lambda t: t[0] == s, pre_s.transitions[i])
                    incoming_outputs.update(map(lambda t: t[1], incoming_edges[i]))
            state_id = 0
            for o in incoming_outputs:
                new_state_id = s.state_id + str(state_id)
                state_id += 1
                new_state = IntervalMdpState(new_state_id, o)
                mdp_states.append(new_state)
                smm_state_to_mdp_state[(s.state_id, o)] = new_state

        for s in self.states:
            mdp_states_for_s = {mdp_state for (s_id, o), mdp_state in smm_state_to_mdp_state.items() if
                                s_id == s.state_id}
            for i in inputs:
                for outgoing_t in s.transitions[i]:
                    target_smm_state = outgoing_t[0]
                    output = outgoing_t[1]
                    prob = outgoing_t[2]
                    target_mdp_state = smm_state_to_mdp_state[(target_smm_state.state_id, output)]
                    for mdp_state in mdp_states_for_s:
                        mdp_state.transitions[i].append((target_mdp_state, prob))
                    if s == self.initial_state:
                        init_state.transitions[i].append((target_mdp_state, prob))
        return IntervalMdp(init_state, mdp_states)


def interval_smm_from_learning_data(learned_model, observation_table, confidence=0.9, method='normal'):
    from statsmodels.stats.proportion import proportion_confint

    # copy mdp
    state_dict = dict()
    for state in learned_model.states:
        state_dict[state.state_id] = IntervalSmmState(state.state_id)

    for state in learned_model.states:
        for i, node_output_list in state.transitions.items():
            for node, output, probability in node_output_list:
                if probability != 1.:
                    freq_dict = observation_table.T[state.prefix][(i,)]
                    total_sum = sum(freq_dict.values())

                    lower, upper = proportion_confint(observation_table.T[state.prefix][(i,)][output],
                                                      total_sum, 1-confidence, method)
                    state_dict[state.state_id].transitions[i].append((state_dict[node.state_id], output, (lower, upper)))
                else:
                    state_dict[state.state_id].transitions[i].append((state_dict[node.state_id], output, (1, 1)))

    initial_state_id = learned_model.initial_state.state_id
    return IntervalSmm(state_dict[initial_state_id], list(state_dict.values()))
