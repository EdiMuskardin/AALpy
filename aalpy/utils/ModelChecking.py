import os
from queue import Queue
import re
from collections import defaultdict
from typing import Tuple

import itertools as it

import aalpy.paths
from aalpy.SULs import MealySUL, DfaSUL, MooreSUL
from aalpy.automata import Mdp, StochasticMealyMachine, MealyMachine, Dfa, MooreMachine, MooreState, MealyState, \
    DfaState
from aalpy.base import DeterministicAutomaton, SUL, AutomatonState
from random import choices

prism_prob_output_regex = re.compile("Result: (\d+\.\d+)")


def get_properties_file(exp_name):
    property_files = {
        'first_grid': aalpy.paths.path_to_properties + 'first_eval.props',
        'second_grid': aalpy.paths.path_to_properties + 'second_eval.props',
        'shared_coin': aalpy.paths.path_to_properties + 'shared_coin_eval.props',
        'slot_machine': aalpy.paths.path_to_properties + 'slot_machine_eval.props',
        'mqtt': aalpy.paths.path_to_properties + 'emqtt_two_client.props',
        'tcp': aalpy.paths.path_to_properties + 'tcp_eval.props'
    }
    return property_files[exp_name]


def get_correct_prop_values(exp_name):
    correct_model_properties = {
        'first_grid': {'prob1': 0.96217534, 'prob2': 0.6499274956800001, 'prob3': 0.6911765746880001},
        'second_grid': {'prob1': 0.93480795088125, 'prob2': 0.6711947700000002, 'prob3': 0.9742903305241055,
                        'prob4': 0.14244219329051103},
        'shared_coin': {'prob1': 0.10694382182657244, 'prob2': 0.5555528623795738, 'prob3': 0.3333324384052837,
                        'prob4': 0.42857002816478273, 'prob5': 0.001708984375, 'prob6': 0.266845703125,
                        'prob7': 0.244384765625, 'prob8': 0.263427734375},
        'slot_machine': {'prob1': 0.36380049887344645, 'prob2': 0.6445910164135946, 'prob3': 1.0, 'prob4': 0.159,
                         'prob5': 0.28567, 'prob6': 0.2500000000000001, 'prob7': 0.025445087448668406},
        'mqtt': {'prob1': 0.9612, 'prob2': 0.34390000000000004, 'prob3': 0.6513215599000001, 'prob4': 0.814697981114816,
                 'prob5': 0.7290000000000001},
        'tcp': {'prob1': 0.19, 'prob2': 0.5695327900000001, 'prob3': 0.7712320754503901, 'prob4': 0.8784233454094308}
    }
    return list(correct_model_properties[exp_name].values())


def _target_string(target, orig_id_to_int_id, is_interval_mdp):
    target_state = target[0]
    target_prob = target[1]
    target_id = orig_id_to_int_id[target_state.state_id]
    if not is_interval_mdp:
        return f"{target_prob} : (loc'={target_id})"
    else:
        return f"[{target_prob[0]},{target_prob[1]}]  : (loc'={target_id})"


def _sanitize_for_prism(symbol):
    if symbol in ["mdp", "init", "module", "endmodule", "label"]:
        return "___" + symbol + "___"
    else:
        return symbol


def mdp_2_prism_format(mdp: Mdp, name: str, output_path=None, is_interval_mdp=False):
    """
    Translates MDP to Prims modelling language.

    Args:

        mdp: markov decision process

        name: name of the mdp/experiment

        output_path: output file (Default value = None)

    """
    module_string = "mdp"
    module_string += os.linesep
    module_string += f"module {name}"
    module_string += os.linesep

    nr_states = len(mdp.states)
    orig_id_to_int_id = dict()
    for i, s in enumerate(mdp.states):
        orig_id_to_int_id[s.state_id] = i
    module_string += "loc : [0..{}] init {};".format(nr_states, orig_id_to_int_id[mdp.initial_state.state_id])
    module_string += os.linesep

    # print transitions
    for source in mdp.states:
        source_id = orig_id_to_int_id[source.state_id]
        for inp in source.transitions.keys():
            if source.transitions[inp]:
                target_strings = \
                    map(lambda target: _target_string(target, orig_id_to_int_id, is_interval_mdp), source.transitions[inp])
                target_joined = " + ".join(target_strings)
                module_string += f"[{_sanitize_for_prism(inp)}] loc={source_id} -> {os.linesep} {target_joined};"
                module_string += os.linesep
    module_string += "endmodule"
    module_string += os.linesep
    # labelling function
    output_to_state_id = defaultdict(list)
    for s in mdp.states:
        joined_output = s.output
        outputs = joined_output.split("__")
        for o in outputs:
            if o:
                output_to_state_id[o].append(orig_id_to_int_id[s.state_id])

    for output, states in output_to_state_id.items():
        state_propositions = map(lambda s_id: "loc={}".format(s_id), states)
        state_disjunction = "|".join(state_propositions)
        output_string = _sanitize_for_prism(output)
        module_string += f"label \"{output_string}\" = {state_disjunction};"
        module_string += os.linesep

    if output_path:
        with open(output_path, "w") as text_file:
            text_file.write(module_string)
    return module_string


def evaluate_all_properties(prism_file_name, properties_file_name):
    import subprocess
    import io
    from os import path

    prism_file = aalpy.paths.path_to_prism.split('/')[-1]
    path_to_prism_file = aalpy.paths.path_to_prism[:-len(prism_file)]

    file_abs_path = path.abspath(prism_file_name)
    properties_als_path = path.abspath(properties_file_name)
    results = {}
    proc = subprocess.Popen(
        [aalpy.paths.path_to_prism, file_abs_path, properties_als_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=path_to_prism_file)
    for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
        if not line:
            break
        else:
            match = prism_prob_output_regex.match(line)
            if match:
                results[f'prop{len(results) + 1}'] = float(match.group(1))
    proc.kill()
    return results


def model_check_properties(model: Mdp, properties: str):
    """

    Args:
        model: Markov Decision Process that serves as a basis for model checking.
        properties: Properties file. It should point to a file under the path_to_properties folder.

    Returns:

        results of model checking
    """
    from os import remove
    from aalpy.utils import mdp_2_prism_format
    mdp_2_prism_format(mdp=model, name='mc_exp', output_path=f'mc_exp.prism')

    prism_model_path = f'mc_exp.prism'

    data = evaluate_all_properties(prism_model_path, properties)

    remove(prism_model_path)

    return data


def model_check_experiment(path_to_properties, correct_prop_values, mdp, precision=4):
    """
    For our stochastic experiments you can use this function.
    For example, check learn_stochastic_system_and_do_model_checking in Examples.py

    Args:
        path_to_properties: path to the properties file
        correct_prop_values: correct values of all properties. In list, where property at index i corresponds to the
            i-th element of the list.
        mdp: MDP
        precision: precision to which round up results

    Returns:

        results of model checking and absolute differance to the correct results
    """
    model_checking_results = model_check_properties(mdp, path_to_properties)

    diff_2_correct = dict()
    for ind, val in enumerate(model_checking_results.values()):
        diff_2_correct[f'prop{ind + 1}'] = round(abs(correct_prop_values[ind] - val), precision)

    results = {key: round(val, precision) for key, val in model_checking_results.items()}
    return results, diff_2_correct


def stop_based_on_confidence(hypothesis, property_based_stopping, print_level=2):
    """

    Args:

        hypothesis: Markov decision process
        property_based_stopping: a tuple (path to properties file, list of correct property values, max allowed error)
        print_level: 2 or 3 if output of model checking is to be printed during learning

    Returns:

        True if absolute error for all properties is smaller then property_based_stopping[2]
    """
    from aalpy.automata.StochasticMealyMachine import smm_to_mdp_conversion

    path_2_prop = property_based_stopping[0]
    correct_values = property_based_stopping[1]
    error_bound = property_based_stopping[2]

    model = hypothesis
    if isinstance(hypothesis, StochasticMealyMachine):
        model = smm_to_mdp_conversion(hypothesis)

    res, diff = model_check_experiment(path_2_prop, correct_values, model)

    if print_level >= 2:
        print('Error for each property:', [round(d * 100, 2) for d in diff.values()])
    if not diff:
        return False

    for d in diff.values():
        if d > error_bound:
            return False

    return True

def bisimilar(a1: DeterministicAutomaton, a2: DeterministicAutomaton):
    """
    Checks whether the provided automata are bisimilar
    """

    if a1.__class__ != a2.__class__:
        raise ValueError("tried to check bisimilarity of distinct automaton types")
    supported_automaton_types = (Dfa, MooreMachine, MealyMachine)
    if not isinstance(a1, supported_automaton_types):
        raise NotImplementedError(f"bisimilarity is not implemented for {a1.__class__.__name__}. Supported: {', '.join(t.__name__ for t in supported_automaton_types)}")

    to_check : Queue[Tuple[AutomatonState, AutomatonState]] = Queue()
    to_check.put((a1.initial_state, a2.initial_state))
    requirements = dict()
    requirements[(a1.initial_state, a2.initial_state)] = []

    while not to_check.empty():
        s1, s2 = to_check.get()
        if (isinstance(s1, DfaState)) and s1.is_accepting != s2.is_accepting or \
           (isinstance(s1, MooreState) and s1.output != s2.output) or \
           (isinstance(s1, MealyState) and s1.output_fun != s2.output_fun):
            return requirements[(s1, s2)]

        t1, t2 = s1.transitions, s2.transitions
        # not using sets -> deterministic but inefficient
        # could use set to detect and slow only if detection?
        for t in it.chain(t1.keys(), t2.keys()):
            if (t in t1.keys()) != (t in t2.keys()):
                return requirements[(s1, s2)] + [t]

        for t in t1.keys():
            c1, c2 = t1[t], t2[t]
            if (c1, c2) not in requirements:
                requirements[(c1, c2)] = requirements[(s1, s2)] + [t]
                to_check.put((c1, c2))

def compare_automata(aut_1: DeterministicAutomaton, aut_2: DeterministicAutomaton, num_cex=10):
    """
    Finds cases of non-conformance between first and second automaton. This is done by performing RandomW equivalence
    check. It is possible that number of found counterexamples is smaller than num_cex, as no counterexample will be a
    suffix of a previously found counterexample.

    Args:

        aut_1: first automaton

        aut_2: second automaton

        num_cex: max. number of searches for counterexamples

    Returns:

        A list of input sequences that revel different behaviour on both automata. Counterexamples are sorted by length.
    """
    #
    from aalpy.oracles import RandomWMethodEqOracle

    type_map = {MooreMachine: MooreSUL, Dfa: DfaSUL, MealyMachine: MealySUL}
    assert set(aut_1.get_input_alphabet()) == set(aut_2.get_input_alphabet())

    input_al = aut_1.get_input_alphabet()
    # larger automaton is used as hypothesis, as then test-cases will contain prefixes leading to states
    # not in smaller automaton
    base_automaton, test_automaton = (aut_1, aut_2) if aut_1.size < aut_2.size else (aut_2, aut_1)
    base_sul = type_map[type(base_automaton)](base_automaton)

    # compute prefixes for all states of the test automaton (needed for advanced eq. oracle)
    for state in test_automaton.states:
        if not state.prefix:
            state.prefix = test_automaton.get_shortest_path(test_automaton.initial_state, state)

    # setup  the eq oracle
    eq_oracle = RandomWMethodEqOracle(input_al, base_sul, walks_per_state=min(100, len(input_al) * 10), walk_len=10)

    found_cex = []
    # to avoid near "infinite" loops due to while loop and set requirement
    # that is, if you can only find 1 cex and all other cexs are suffixes of that cex, first while condition will never
    # be reached
    failsafe_counter = 0
    failsafe_stopping = num_cex * 100
    while len(found_cex) < num_cex or failsafe_counter == failsafe_stopping:
        cex = eq_oracle.find_cex(test_automaton)
        # if no counterexample can be found terminate the loop
        if cex is None:
            break
        if cex not in found_cex:
            found_cex.append(cex)
        failsafe_counter += 1

    found_cex.sort(key=len)

    return found_cex


class TestCaseWrapperSUL(SUL):
    def __init__(self, sul):
        super().__init__()
        self.sul = sul
        self.test_cases = []
        self.test_case_inputs = None
        self.test_case_outputs = None

    def pre(self):
        self.test_case_inputs = []
        self.test_case_outputs = []
        return self.sul.pre()

    def post(self):
        if self.test_case_inputs and self.test_case_outputs:
            self.test_cases.append((tuple(self.test_case_inputs), tuple(self.test_case_outputs)))
        return self.sul.post()

    def step(self, letter):
        output = self.sul.step(letter)
        self.test_case_inputs.append(letter)
        self.test_case_outputs.append(output)
        return output


def generate_test_cases(automaton: DeterministicAutomaton, oracle):
    """
    Uses parametrized eq. oracle to construct test cases on the automaton.
    If automaton are big (200+ states), increase recursion depth if necessary (eg. sys.setrecursionlimit(10000)).

    Args:

        automaton: deterministic automaton that serves as a basis for test case generation
        oracle: oracle that will construct test-cases and record inputs and outputs

    Returns:

        List of test cases, where each testcase is a tuple containing two elements, and input and an output sequance.
    """
    from copy import deepcopy

    type_map = {MooreMachine: MooreSUL, Dfa: DfaSUL, MealyMachine: MealySUL}

    automaton_copy = deepcopy(automaton)
    base_sul = type_map[type(automaton_copy)](automaton_copy)

    wrapped_sul = TestCaseWrapperSUL(base_sul)
    oracle.sul = wrapped_sul
    # no counterexamples can be found
    cex = oracle.find_cex(automaton)
    assert cex is None
    return wrapped_sul.test_cases


def statistical_model_checking(model, goals, max_num_steps, num_tests=105967):
    """


    Args:
        model: model on which model checking is performed
        goals: set of goal outputs
        max_num_steps: bounded length of tests
        num_tests: num of tests that will be performed

    Returns:

        num of tests containing element of goals set / num_tests
    """
    def compute_output_sequence(model, seq):
        model.reset_to_initial()
        observed_outputs = {model.step(i) for i in seq}
        return observed_outputs

    goal_reached = 0
    inputs = model.get_input_alphabet()
    for _ in range(num_tests):
        test_sequence = choices(inputs, k=max_num_steps)
        outputs = compute_output_sequence(model, test_sequence)
        if goals & outputs:
            goal_reached += 1

    return goal_reached / num_tests