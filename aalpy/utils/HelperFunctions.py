import random
import string
from collections import defaultdict


def extend_set(list_to_extend: list, new_elements: list) -> list:
    """
    Helper function to extend a list while maintaining set property.
    They are stored as lists, so with this function set property is maintained.
    :return

    Returns:

        list of elements that were added to the set

    """
    set_repr = set(list_to_extend)
    added_elements = [s for s in new_elements if s not in set_repr]
    list_to_extend.extend(added_elements)
    return added_elements


def all_prefixes(li):
    """
    Returns all prefixes of a list.

    Args:
      li: list from which to compute all prefixes

    Returns:
      list of all prefixes

    """
    return [tuple(li[:i + 1]) for i in range(len(li))]


def all_suffixes(li):
    """
    Returns all suffixes of a list.

    Args:
      li: list from which to compute all suffixes

    Returns:
      list of all suffixes

    """
    return [tuple(li[len(li) - i - 1:]) for i in range(len(li))]


def profile_function(function: callable, sort_key='cumtime'):
    """

    Args:
      function: callable: 
      sort_key:  (Default value = 'cumtime')

    Returns:
        prints the profiling results
    """
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    function()
    pr.disable()
    pr.print_stats(sort=sort_key)


def random_string_generator(size=10, chars=string.ascii_lowercase + string.digits):
    """

    Args:

      size:  (Default value = 10)
      chars:  (Default value = string.ascii_lowercase + string.digits)

    Returns:

        a random string of length size
    """
    import random
    return ''.join(random.choice(chars) for _ in range(size))


def print_learning_info(info: dict):
    """
    Print learning statistics.
    """
    print('-----------------------------------')
    print('Learning Finished.')
    print('Learning Rounds:  {}'.format(info['learning_rounds']))
    print('Number of states: {}'.format(info['automaton_size']))
    print('Time (in seconds)')
    print('  Total                : {}'.format(info['total_time']))
    print('  Learning algorithm   : {}'.format(info['learning_time']))
    print('  Conformance checking : {}'.format(info['eq_oracle_time']))
    print('Learning Algorithm')
    print(' # Membership Queries  : {}'.format(info['queries_learning']))
    if 'cache_saved' in info.keys():
        print(' # MQ Saved by Caching : {}'.format(info['cache_saved']))
    print(' # Steps               : {}'.format(info['steps_learning']))
    print('Equivalence Query')
    print(' # Membership Queries  : {}'.format(info['queries_eq_oracle']))
    print(' # Steps               : {}'.format(info['steps_eq_oracle']))
    print('-----------------------------------')


def print_observation_table(ot, table_type):
    """
    Prints the whole observation table.

    Args:

        ot: observation table
        table_type: 'det', 'non-det', or 'stoc'

    """
    if table_type == 'det':
        s_set, extended_s, e_set, table = ot.S, ot.s_dot_a(), ot.E, ot.T
    elif table_type == 'non-det':
        s_set, extended_s, e_set = ot.S, ot.get_extended_S(), ot.E
        table = ot.sul.cache.get_table(s_set + extended_s, e_set)
    elif table_type == 'abstracted-non-det':
        s_set, extended_s, e_set, table = ot.S, ot.S_dot_A, ot.E, ot.T
    else:
        s_set, extended_s, e_set, table = ot.S, ot.get_extended_s(), ot.E, ot.T

    headers = [str(e) for e in e_set]
    s_rows = []
    extended_rows = []
    headers.insert(0, 'Prefixes / E set')
    for s in s_set:
        row = [str(s)]
        if table_type == 'det':
            row.extend(str(e) for e in table[s])
        else:
            row.extend(str(table[s][e]) for e in e_set)
        s_rows.append(row)
    for s in extended_s:
        row = [str(s)]
        if table_type == 'det':
            row.extend(str(e) for e in table[s])
        else:
            row.extend(str(table[s][e]) for e in e_set)
        extended_rows.append(row)

    table = [headers] + s_rows
    columns = defaultdict(int)
    for i in table + extended_rows:
        for index, el in enumerate(i):
            columns[index] = max(columns[index], len(el))

    row_len = 0
    for row in table:
        row = "|".join(element.ljust(columns[ind] + 1) for ind, element in enumerate(row))
        print("-" * len(row))
        row_len = len(row)
        print(row)
    print('=' * row_len)
    for row in extended_rows:
        row = "|".join(element.ljust(columns[ind] + 1) for ind, element in enumerate(row))
        print("-" * len(row))
        print(row)
    print('-' * row_len)


def is_suffix_of(suffix, trace) -> bool:
    """

    Args:
      suffix: target suffix
      trace: trace in question

    Returns:

        True if suffix is the suffix of trace.
    """
    if len(trace) < len(suffix):
        return False
    else:
        return trace[-len(suffix):] == suffix


def get_cex_prefixes(cex, automaton_type):
    """
    Returns all prefixes of the stochastic automaton.

    Args:
        cex: counterexample
        automaton_type: `mdp` or `smm`

    Returns:

        all prefixes of the counterexample based on the `automaton_type`
    """
    if automaton_type == 'mdp':
        return [tuple(cex[:i + 1]) for i in range(0, len(cex), 2)]
    return [tuple(cex[:i]) for i in range(0, len(cex) + 1, 2)]


def get_available_oracles_and_err_msg():
    from aalpy.oracles import RandomWalkEqOracle
    from aalpy.oracles import RandomWordEqOracle
    available_oracles = {RandomWalkEqOracle, RandomWordEqOracle}

    available_oracles_msg = 'Warning! Only Random Walk and Random Word oracles are supported for non-deterministic and ' \
                            'stochastic learning. If you have implemented the custom oracle, set the custom_oracle ' \
                            'flag to True. '

    return available_oracles, available_oracles_msg


def make_input_complete(automaton, missing_transition_go_to='self_loop'):
    """
    Makes the automaton input complete/enabled. If a input is not defined in a state, it will lead to the self loop.
    In case of Mealy Machines, Stochastic Mealy machines and ONFSM 'epsilon' is used as output.
    Self loop simply loops the transition back to state which was not input complete,
    whereas 'sink_state' leads all transitions to a newly added sink state. If transitions have output
    (Mealy machines and their derivatives), 'epsilon' is used as an output value. If a state has an output value,
    it is either False (in case of DFA) or 'sink_state' in case of Moore machines and its derivatives.

    Args:

        automaton: automaton that is potentially not input complete
        missing_transition_go_to: either 'self_loop' or 'sink_state'.

    Returns:

        an input complete automaton
    """
    from aalpy.base import DeterministicAutomaton
    from aalpy.automata import Dfa, MooreState, MealyMachine, Mdp, StochasticMealyMachine, Onfsm, \
        DfaState, MealyState, MooreMachine, OnfsmState, MdpState, StochasticMealyState

    assert missing_transition_go_to in {'self_loop', 'sink_state'}

    input_al = automaton.get_input_alphabet()

    if automaton.is_input_complete():
        return automaton

    sink_state = None
    sink_state_type_dict = {Dfa: DfaState(state_id='sink', is_accepting=False),
                            MooreMachine: MooreState(state_id='sink', output='sink_state'),
                            MealyMachine: MealyState(state_id='sink'),
                            Onfsm: OnfsmState(state_id='sink'),
                            Mdp: MdpState(state_id='sink', output='sink_state'),
                            StochasticMealyMachine: StochasticMealyState(state_id='sink')}

    if missing_transition_go_to == 'sink_state':
        sink_state = sink_state_type_dict[automaton.__class__]
        automaton.states.append(sink_state)

    for state in automaton.states:
        for i in input_al:
            if i not in state.transitions.keys():
                if missing_transition_go_to == 'self_loop':
                    if isinstance(automaton, DeterministicAutomaton):
                        state.transitions[i] = state
                        if isinstance(automaton, MealyMachine):
                            state.output_fun[i] = 'epsilon'
                    if isinstance(automaton, Onfsm):
                        state.transitions[i].append(('epsilon', state))
                    if isinstance(automaton, Mdp):
                        state.transitions[i].append((state, 1.))
                    if isinstance(automaton, StochasticMealyMachine):
                        state.transitions[i].append((state, 'epsilon', 1.))
                else:
                    if isinstance(automaton, Dfa):
                        state.transitions[i] = sink_state
                    if isinstance(automaton, MooreMachine):
                        state.transitions[i] = sink_state
                    if isinstance(automaton, MealyMachine):
                        state.transitions[i] = sink_state
                        state.output_fun[i] = 'epsilon'
                    if isinstance(automaton, Onfsm):
                        state.transitions[i].append(('epsilon', sink_state))
                    if isinstance(automaton, Mdp):
                        state.transitions[i].append((sink_state, 1.))
                    if isinstance(automaton, StochasticMealyMachine):
                        state.transitions[i].append((sink_state, 'epsilon', 1.))

    return automaton


def convert_i_o_traces_for_RPNI(sequences):
    """
    Converts a list of input-output sequences to RPNI format.
    Eg. [[(1,'a'), (2,'b'), (3,'c')], [(6,'7'), (4,'e'), (3,'c')]] to
    [((1,), 'a'), ((1, 2), 'b'), ((1, 2, 3), 'c'), ((6,), '7'), ((6, 4), 'e'), ((6, 4, 3), 'c')]
    """
    rpni_sequences = set()

    for s in sequences:
        for i in range(len(s)):
            inputs = tuple([io[0] for io in s[:i + 1]])
            rpni_sequences.add((inputs, s[i][1]))

    return list(rpni_sequences)


def visualize_classification_tree(root_node):
    from pydot import Dot, Node, Edge

    graph = Dot('classification_tree', graph_type='digraph')
    root_node_dot = Node(id(root_node), shape='box',
                         label=f'Distinguishing String:\n{root_node.distinguishing_string}')
    graph.add_node(root_node_dot)

    queue = [(root_node, root_node_dot)]

    while queue:
        origin_node, origin_node_dot = queue.pop(0)

        for key, child in origin_node.children.items():
            if child.is_leaf():
                destination_dot = Node(id(child), label=f'Access String:\n{child.access_string}')
            else:
                destination_dot = Node(id(child), shape='box',
                                       label=f'Distinguishing String:\n{child.distinguishing_string}')
                queue.append((child, destination_dot))
            graph.add_node(destination_dot)
            graph.add_edge(Edge(origin_node_dot, destination_dot, label=key))

    # print(graph.to_string())
    graph.write(path='classification_tree.pdf', format='pdf')


def is_balanced(input_seq, vpa_alphabet):
    counter = 0
    for i in input_seq:
        if i in vpa_alphabet.call_alphabet:
            counter += 1
        if i in vpa_alphabet.return_alphabet:
            counter -= 1
        if counter < 0:
            return False
    return counter == 0


def generate_input_output_data_from_automata(model, num_sequances=4000, min_seq_len=1, max_seq_len=16):

    alphabet = model.get_input_alphabet()
    input_output_sequances = []
    while len(input_output_sequances) < num_sequances:
        sequance = []
        for _ in range(random.randint(min_seq_len, max_seq_len)):
            sequance.append(random.choice(alphabet))

        model.reset_to_initial()
        outputs = model.execute_sequence(model.initial_state, sequance)

        input_output_sequances.append(list(zip(sequance, outputs)))

    return input_output_sequances


def generate_input_output_data_from_vpa(vpa, num_sequances=1000, max_seq_len=16, max_attempts=None):
    alphabet = vpa.input_alphabet.get_merged_alphabet()
    data_set = set()

    num_nominal_tries = num_sequances // 2
    num_generation_attempts = 0

    max_attempts = max_attempts if max_attempts is not None else num_sequances * 50

    while len(data_set) < num_sequances:
        num_generation_attempts += 1
        # it can happen that it is not possible to generate num_sequances balanced words of lenght <= max_seq_len
        if num_generation_attempts == max_attempts:
            break

        sequance = ()

        vpa.reset_to_initial()

        for _ in range(max_seq_len):
            if num_generation_attempts <= num_nominal_tries:
                possible_transitions = list(vpa.current_state.transitions.keys())
                if not possible_transitions:
                    break
                chosen_input = random.choice(possible_transitions)
            else:
                chosen_input = random.choice(alphabet)
            sequance += (chosen_input,)

            output = vpa.step(chosen_input)
            #if vpa.is_balanced(sequance):
            data_set.add((sequance, output))

    data_set = list(data_set)
    return data_set
