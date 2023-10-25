from collections import defaultdict
from typing import Union

from aalpy.base import Automaton, AutomatonState


class SevpaAlphabet:
    def __init__(self, internal_alphabet: list, call_alphabet: list, return_alphabet: list,
                 exclusive_call_return_pairs: dict = None):
        self.internal_alphabet = internal_alphabet
        self.call_alphabet = call_alphabet
        self.return_alphabet = return_alphabet
        self.exclusive_call_return_pairs = exclusive_call_return_pairs

    def get_merged_alphabet(self) -> list:
        alphabet = list()
        alphabet.extend(self.internal_alphabet)
        alphabet.extend(self.call_alphabet)
        alphabet.extend(self.return_alphabet)
        return alphabet

    def __str__(self):
        return f'Internal: {self.internal_alphabet} Call: {self.call_alphabet} Return: {self.return_alphabet}'


class SevpaState(AutomatonState):
    """
    Single state of a 1-SEVPA.
    """

    def __init__(self, state_id, is_accepting=False):
        super().__init__(state_id)
        self.transitions = defaultdict(list[SevpaTransition])
        self.is_accepting = is_accepting


class SevpaTransition:
    def __init__(self, start: SevpaState, target: SevpaState, symbol, action, stack_guard=None):
        self.start = start
        self.target = target
        self.symbol = symbol
        self.action = action
        self.stack_guard = stack_guard

    def __str__(self):
        return f"{self.symbol}: {self.start.state_id} --> {self.target.state_id} | {self.action}: {self.stack_guard}"


class Sevpa(Automaton):
    empty = "_"

    def __init__(self, initial_state: SevpaState, states: list[SevpaState], input_alphabet: SevpaAlphabet):
        super().__init__(initial_state, states)
        self.initial_state = initial_state
        self.states = states
        self.input_alphabet = input_alphabet
        self.current_state = None
        self.stack = []
        self.error_state_reached = False

        # alphabet sets for faster inclusion checks (as in SevpaAlphabet we have lists, for reproducibility)
        self.internal_set = set(self.input_alphabet.internal_alphabet)
        self.call_set = set(self.input_alphabet.call_alphabet)
        self.return_set = set(self.input_alphabet.return_alphabet)

    def reset_to_initial(self):
        super().reset_to_initial()
        self.current_state = self.initial_state
        self.stack = [self.empty]
        self.error_state_reached = False
        return self.current_state.is_accepting and self.stack[-1] == self.empty

    def step(self, letter):
        if self.error_state_reached:
            return False

        if letter is None:
            return self.current_state.is_accepting and self.stack[-1] == self.empty

        if letter in self.call_set:
            self.stack.append((self.current_state.state_id, letter))
            self.current_state = self.initial_state
            return self.current_state.is_accepting and self.stack[-1] == self.empty

        # get possible transitions
        transitions = self.current_state.transitions[letter]
        taken_transition = None
        for t in transitions:
            if t.symbol in self.return_set:
                if t.stack_guard == self.stack[-1]:
                    taken_transition = t
                    break
            elif t.symbol in self.internal_set:
                taken_transition = t
                break
            else:
                assert False

        # No transition is possible
        if not taken_transition:
            self.error_state_reached = True
            return False

        self.current_state = taken_transition.target

        if taken_transition.action == 'pop':
            # empty stack elem should always be on the stack
            if len(self.stack) <= 1:
                self.error_state_reached = True
                return False
            self.stack.pop()

        return self.current_state.is_accepting and self.stack[-1] == self.empty

    def get_state_by_id(self, state_id) -> Union[SevpaState, None]:
        for state in self.states:
            if state.state_id == state_id:
                return state
        return None

    def execute_sequence(self, origin_state, seq):
        if origin_state.prefix != self.initial_state.prefix:
            assert False, 'execute_sequance for Sevpa only is only supported from the initial state.'
        self.reset_to_initial()
        self.current_state = origin_state
        return [self.step(s) for s in seq]

    def to_state_setup(self):
        state_setup_dict = {}

        # ensure prefixes are computed
        # self.compute_prefixes()
        # TODO
        sorted_states = sorted(self.states, key=lambda x: len(x.state_id))
        for s in sorted_states:
            state_setup_dict[s.state_id] = (
                s.is_accepting, {k: (v.target.state_id, v.action) for k, v in s.transitions.items()})

        return state_setup_dict

    @staticmethod
    def from_state_setup(state_setup: dict, init_state_id, input_alphabet: SevpaAlphabet):

        # build states with state_id and output
        states = {key: SevpaState(key, val[0]) for key, val in state_setup.items()}
        states[Sevpa.error_state.state_id] = Sevpa.error_state  # PdaState(Pda.error_state,False)

        # add transitions to states
        for state_id, state in states.items():
            if state_id == Sevpa.error_state.state_id:
                continue
            for _input, trans_spec in state_setup[state_id][1].items():
                for (target_state_id, action, stack_guard) in trans_spec:
                    if action == 'pop':
                        assert stack_guard[0] in states
                        assert stack_guard[1] in input_alphabet.call_alphabet
                        stack_guard = (stack_guard[0], stack_guard[1])
                        trans = SevpaTransition(start=state, target=states[target_state_id], symbol=_input,
                                                action=action, stack_guard=stack_guard)
                    elif action is None:
                        trans = SevpaTransition(start=state, target=states[target_state_id], symbol=_input,
                                                action=None, stack_guard=None)
                    else:
                        assert False, 'Action must either be "pop" or None, note that there are no push actions ' \
                                      'definitions in SEVPA'

                    state.transitions[_input].append(trans)

        init_state = states[init_state_id]
        return Sevpa(init_state, [state for state in states.values()], input_alphabet)

    def transform_access_sequance(self, state=None, stack_content=None) -> list[str]:

        word = []
        calling_state = self.initial_state if not state else state
        stack = self.stack if not stack_content else stack_content

        for index, stack_elem in enumerate(stack):
            # skip the first element because it's the start of the stack '_
            if index == 0:
                continue
            from_state_id = stack_elem[0]  # the corresponding state where the stack element got pushed from
            call_letter = stack_elem[1]  # the call letter that was pushed on the stack
            from_state = self.get_state_by_id(from_state_id)
            if from_state.prefix != ():
                word.extend(from_state.prefix)
            word.append(call_letter)
        word.extend(calling_state.prefix)
        return word

    @staticmethod
    def create_daisy_hypothesis(initial_state, alphabet):

        for i in alphabet.internal_alphabet:
            trans = SevpaTransition(start=initial_state, target=initial_state, symbol=i, action=None)
            initial_state.transitions[i].append(trans)

        for c in alphabet.call_alphabet:
            for r in alphabet.return_alphabet:
                trans = SevpaTransition(start=initial_state, target=initial_state, symbol=r, action='pop',
                                        stack_guard=(initial_state.state_id, c))
                initial_state.transitions[r].append(trans)

        return Sevpa(initial_state, [initial_state], alphabet)