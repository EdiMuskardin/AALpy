import time

from aalpy.base import Oracle, SUL
from aalpy.utils.HelperFunctions import extend_set, print_learning_info, print_observation_table, all_prefixes
from .ClassificationTree import ClassificationTree, CTInternalNode, CTLeafNode
from .CounterExampleProcessing import longest_prefix_cex_processing, rs_cex_processing
from .DiscriminationTree import DiscriminationTree, DTStateNode, DTDiscriminatorNode
from .KV_helpers import state_name_gen, prettify_hypothesis
from .ObservationTable import ObservationTable
from .TTTHypothesis import TTTHypothesis
from .TTT_helper_functions import link, close_transitions, rs_split_cex
from ...SULs import DfaSUL
from ...base.SUL import CacheSUL
from aalpy.automata import Dfa, DfaState

counterexample_processing_strategy = [None, 'rs']
print_options = [0, 1, 2, 3]

# TODO implement print_level


def run_KV(alphabet: list, sul: SUL, eq_oracle: Oracle, automaton_type='dfa', cex_processing=None,
           max_learning_rounds=None, return_data=False, print_level=2, pretty_state_names=True,
           reuse_counterexamples=False,
           ):
    """
    Executes TTT algorithm.

    Args:

        alphabet: input alphabet

        sul: system under learning

        eq_oracle: equivalence oracle

        automaton_type: type of automaton to be learned. Currently only 'dfa' supported.

        max_learning_rounds: number of learning rounds after which learning will terminate (Default value = None)

        return_data: if True, a map containing all information(runtime/#queries/#steps) will be returned
            (Default value = False)

        print_level: 0 - None, 1 - just results, 2 - current round and hypothesis size, 3 - educational/debug
            (Default value = 2)

        pretty_state_names: if False, the resulting dfa's state names will be the ones generated during learning.
                            if True, generic 's0'-sX' state names will be used
            (Default value = True)

        reuse_counterexamples: Slight improvement over the original KV algorithm. If True, a counterexample will be
                               reused until the hypothesis accepts it.
            (Default value = False)

        use_rs_cex_processing: Improvement over the original KV algorithm, use Rivest & Schapire to split counterexamples

    Returns:

        automaton of type automaton_type (dict containing all information about learning if 'return_data' is True)

    """

    assert print_level in print_options
    assert cex_processing in counterexample_processing_strategy
    assert automaton_type == 'dfa'
    assert isinstance(sul, DfaSUL)

    start_time = time.time()
    eq_query_time = 0
    learning_rounds = 0

    # Do a membership query on the empty string to determine whether
    # the start state of the SUL is accepting or rejecting
    empty_string_mq = sul.query((None,))[-1]

    # Construct a hypothesis automaton that consists simply of this
    # single (accepting or rejecting) state with self-loops for
    # all transitions.
    initial_state = DfaState(state_id=(None,),
                             is_accepting=empty_string_mq)
    for a in alphabet:
        initial_state.transitions[a] = initial_state

    hypothesis = Dfa(initial_state=initial_state,
                     states=[initial_state])

    # Perform an equivalence query on this automaton
    eq_query_start = time.time()
    cex = eq_oracle.find_cex(hypothesis)
    eq_query_time += time.time() - eq_query_start
    if cex is None:
        return hypothesis
    else:
        cex = tuple(cex)
        if reuse_counterexamples:
            supposed_result = not hypothesis.get_result(cex)
    print(f"processing {cex=}")

    # initialise the classification tree to have a root
    # labeled with the empty word as the distinguishing string
    # and two leafs labeled with access strings cex and empty word
    ctree = ClassificationTree(alphabet=alphabet,
                               sul=sul,
                               cex=cex,
                               empty_is_true=empty_string_mq)

    cex_list = []  # not needed, just here to check if cex get reused
    while True:
        learning_rounds += 1
        if max_learning_rounds and learning_rounds - 1 == max_learning_rounds:
            break

        hypothesis = ctree.gen_hypothesis()

        if print_level > 1:
            print(f'Hypothesis {learning_rounds}: {len(hypothesis.states)} states.')

        if print_level == 3:
            # TODO: print classification tree
            pass

        if reuse_counterexamples and hypothesis.get_result(cex) != supposed_result:
            # our hypothesis still doesn't get the supposed result for the former counterexample -> reuse it
            pass
        else:
            # Perform an equivalence query on this automaton
            eq_query_start = time.time()
            cex = eq_oracle.find_cex(hypothesis)
            eq_query_time += time.time() - eq_query_start

            if cex is None:
                break
            else:
                cex = tuple(cex)
                if cex in cex_list:
                    if reuse_counterexamples:
                        assert False
                if reuse_counterexamples:
                    supposed_result = not hypothesis.get_result(cex)
                cex_list.append(cex)

            if print_level == 3:
                print('Counterexample', cex)

        if cex_processing == 'rs':
            ctree.update_rs(cex, hypothesis)
        else:
            ctree.update(cex, hypothesis)

    total_time = round(time.time() - start_time, 2)
    eq_query_time = round(eq_query_time, 2)
    learning_time = round(total_time - eq_query_time, 2)

    info = {
        'learning_rounds': learning_rounds,
        'automaton_size': len(hypothesis.states),
        'queries_learning': sul.num_queries,
        'steps_learning': sul.num_steps,
        'queries_eq_oracle': eq_oracle.num_queries,
        'steps_eq_oracle': eq_oracle.num_steps,
        'learning_time': learning_time,
        'eq_oracle_time': eq_query_time,
        'total_time': total_time,
        'classification_tree': ctree
    }

    prettify_hypothesis(hypothesis, alphabet, keep_access_strings=not pretty_state_names)

    if print_level > 0:
        print_learning_info(info)

    if return_data:
        return hypothesis, info

    return hypothesis
