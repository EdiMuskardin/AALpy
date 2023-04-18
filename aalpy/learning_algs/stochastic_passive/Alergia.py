import time
from collections import defaultdict
from bisect import insort

from aalpy.automata import MarkovChain, MdpState, Mdp, McState, StochasticMealyState, \
    StochasticMealyMachine, IntervalMdpState, IntervalMdp
from aalpy.learning_algs.stochastic_passive.CompatibilityChecker import HoeffdingCompatibility
from aalpy.learning_algs.stochastic_passive.FPTA import create_fpta

state_automaton_map = {'mc': (McState, MarkovChain), 'mdp': (MdpState, Mdp),
                       'smm': (StochasticMealyState, StochasticMealyMachine),
                       'interval_mdp': (IntervalMdpState, IntervalMdp)}


class Alergia:
    def __init__(self, data, automaton_type, eps=0.005, compatibility_checker=None, optimize_for='accuracy',
                 interval_confidence=None, interval_method=None, print_info=False):

        assert eps == 'auto' or 0 < eps <= 2
        assert optimize_for in {'memory', 'accuracy'}

        self.automaton_type = automaton_type
        self.print_info = print_info
        self.optimize_for = optimize_for
        self.interval_conf = interval_confidence
        self.interval_method = interval_method

        if eps == 'auto':
            eps = 10 / sum(len(d) - 1 for d in data)  # len - 1 to ignore initial output

        self.diff_checker = HoeffdingCompatibility(eps) if not compatibility_checker else compatibility_checker

        pta_start = time.time()

        self.immutableTreeRoot, self.mutableTreeRoot = create_fpta(data, automaton_type, optimize_for)

        pta_time = round(time.time() - pta_start, 2)
        if self.print_info:
            print(f'PTA Construction Time:  {pta_time}')

    def compatibility_test(self, a, b):
        if self.automaton_type != 'smm' and a.output != b.output:
            return False

        if not a.children.values() or not b.children.values():
            return True

        if self.diff_checker.are_states_different(a, b):
            return False

        for el in set(a.children.keys()).intersection(b.children.keys()):
            if not self.compatibility_test(a.children[el], b.children[el]):
                return False

        return True

    def merge(self, q_r, q_b):
        t_q_b = self.get_blue_node(q_b)
        b_prefix = q_b.getPrefix()
        to_update = self.mutableTreeRoot
        for p in b_prefix[:-1]:
            to_update = to_update.children[p]

        to_update.children[b_prefix[-1]] = q_r

        self.fold(q_r, q_b, t_q_b)

    def fold(self, red, blue, blue_tree_node):
        for i, c in blue.children.items():
            if i in red.children.keys():
                red.input_frequency[i] += blue_tree_node.input_frequency[i]
                self.fold(red.children[i], blue.children[i], self.get_blue_node(blue.children[i]))
            else:
                red.children[i] = blue.children[i]
                red.input_frequency[i] = blue_tree_node.input_frequency[i]

    def run(self):
        start_time = time.time()

        red = [self.mutableTreeRoot]  # representative nodes and will be included in the final output model
        blue = self.mutableTreeRoot.successors()  # intermediate successors scheduled for testing

        while blue:
            lex_min_blue = min(list(blue), key=lambda x: len(x.getPrefix()))
            merged = False

            for q_r in red:
                if self.compatibility_test(self.get_blue_node(q_r), self.get_blue_node(lex_min_blue)):
                    self.merge(q_r, lex_min_blue)
                    merged = True
                    break

            if not merged:
                insort(red, lex_min_blue)

            blue.clear()

            for r in red:
                for s in r.successors():
                    if s not in red:
                        blue.append(s)

        assert sorted(red, key=lambda x: len(x.getPrefix())) == red

        self.normalize(red)

        for i, r in enumerate(red):
            r.state_id = f'q{i}'

        if self.print_info:
            print(f'Alergia Learning Time: {round(time.time() - start_time, 2)}')
            print(f'Alergia Learned {len(red)} state automaton.')

        return self.to_automaton(red)

    def normalize(self, red):
        red_sorted = sorted(list(red), key=lambda x: len(x.getPrefix()))
        for r in red_sorted:
            r.children_prob = dict()  # Initializing in here saves many unnecessary initializations
            if self.automaton_type == 'mc':
                total_output = sum(r.input_frequency.values())
                for i in r.input_frequency.keys():
                    r.children_prob[i] = r.input_frequency[i] / total_output
            else:
                outputs_per_input = defaultdict(int)
                for io, freq in r.input_frequency.items():
                    outputs_per_input[io[0]] += freq
                for io in r.input_frequency.keys():
                    if self.automaton_type != 'interval_mdp':
                        r.children_prob[io] = r.input_frequency[io] / outputs_per_input[io[0]]
                    else:
                        from statsmodels.stats.proportion import proportion_confint
                        if r.input_frequency[io] == outputs_per_input[io[0]]:
                            lower, upper = 1, 1
                        else:
                            lower, upper = proportion_confint(r.input_frequency[io],
                                                              outputs_per_input[io[0]],
                                                              1 - self.interval_conf,
                                                              self.interval_method)
                        r.children_prob[io] = (lower, upper)

    def get_blue_node(self, red_node):
        if self.optimize_for == 'memory':
            return red_node
        blue = self.immutableTreeRoot
        for p in red_node.getPrefix():
            blue = blue.children[p]
        return blue

    def to_automaton(self, red):
        s_c = state_automaton_map[self.automaton_type][0]
        a_c = state_automaton_map[self.automaton_type][1]

        states = []
        initial_state = None
        red_mdp_map = dict()
        for s in red:
            if self.automaton_type != 'smm':
                automaton_state = s_c(s.state_id, output=s.output)
            else:
                automaton_state = s_c(s.state_id)

            automaton_state.prefix = s.getPrefix()
            states.append(automaton_state)
            red_mdp_map[tuple(s.getPrefix())] = automaton_state
            red_mdp_map[automaton_state.state_id] = s
            if not s.getPrefix():
                initial_state = automaton_state

        for s in states:
            red_eq = red_mdp_map[s.state_id]
            for io, c in red_eq.children.items():
                destination = red_mdp_map[tuple(c.getPrefix())]
                i = io if self.automaton_type == 'mc' else io[0]
                if self.automaton_type == 'mdp' or 'interval_mdp':
                    s.transitions[i].append((destination, red_eq.children_prob[io]))
                elif self.automaton_type == 'mc':
                    s.transitions.append((destination, red_eq.children_prob[i]))
                elif self.automaton_type == 'smm':
                    s.transitions[i].append((destination, io[1], red_eq.children_prob[io]))
                else:
                    s.transitions[i] = destination

        return a_c(initial_state, states)


def run_Alergia(data, automaton_type, eps=0.005, compatibility_checker=None, optimize_for='accuracy',
                print_info=False, **kwargs):
    """
    Run Alergia or IOAlergia on provided data.

    Args:

        data: data either in a form [[I,I,I],[I,I,I],...] if learning Markov Chains or [[O,(I,O),(I,O)...],
        [O,(I,O_,...],..,] if learning MDPs (I represents input, O output).
        Note that in whole data first symbol of each entry should be the same (Initial output of the MDP/MC).

        eps: epsilon value if you are using default HoeffdingCompatibility. If it is set to 'auto' it will be computed
        as 10/(all steps in the data)

        automaton_type: either 'mdp' if you wish to learn an MDP, 'mc' if you want to learn Markov Chain, or 'smm' if
        you want to learn stochastic Mealy machine

        optimize_for: either 'memory' or 'accuracy'. memory will use 50% less memory, but will be more inaccurate.

        compatibility_checker: impl. of class CompatibilityChecker, HoeffdingCompatibility with eps value by default

        kwargs: interval_confidence and interval_method

        print_info:

    Returns:

        mdp, smm, or markov chain
    """
    assert automaton_type in {'mdp', 'mc', 'smm', 'interval_mdp'}
    method, interval_confidence = None, None
    if automaton_type == 'interval_mdp':
        method = kwargs.get('interval_method', 'normal')
        interval_confidence = kwargs.get('interval_confidence', 0.9)

    alergia = Alergia(data, eps=eps, automaton_type=automaton_type, optimize_for=optimize_for,
                      compatibility_checker=compatibility_checker, interval_confidence=interval_confidence,
                      interval_method=method,
                      print_info=print_info)
    model = alergia.run()
    del alergia.mutableTreeRoot, alergia.immutableTreeRoot, alergia
    return model


def run_JAlergia(path_to_data_file, automaton_type, path_to_jAlergia_jar, eps=0.005,
                 heap_memory='-Xmx2048M', optimize_for='accuracy'):
    """
    Run Alergia or IOAlergia on provided data.

    Args:

        path_to_data_file: either a data in a list of lists or a path to file containing data. 
        Form [[I,I,I],[I,I,I],...] if learning Markov Chains or
        [[O,I,O,I,O...], [O,(I,O_,...],..,] if learning MDPs (I represents input, O output).
        Note that in whole data first symbol of each entry should be the same (Initial output of the MDP/MC).

        eps: epsilon value
        
        heap_memory: java heap memory flag, increase if heap is full
        
        optimize_for: either 'memory' or 'accuracy'. memory will use 50% less memory, but will be more inaccurate.

        automaton_type: either 'mdp' if you wish to learn an MDP, 'mc' if you want to learn Markov Chain,
         or 'smm' if you
                        want to learn stochastic Mealy machine


    Returns:

        learnedModel
    """
    # TODO rename path_to_data_file to data in next versions of AALpy after 20. may
    assert automaton_type in {'mdp', 'smm', 'mc'}
    assert optimize_for in {'memory', 'accuracy'}

    import os
    import subprocess
    from aalpy.utils.FileHandler import load_automaton_from_file

    save_file = "jAlergiaModel.dot"
    delete_tmp_file = False
    if os.path.exists(save_file):
        os.remove(save_file)

    if os.path.exists(path_to_jAlergia_jar):
        path_to_jAlergia_jar = os.path.abspath(path_to_jAlergia_jar)
    else:
        print(f'JAlergia jar not found at {path_to_jAlergia_jar}.')
        return

    if isinstance(path_to_data_file, str):
        if os.path.exists(path_to_data_file):
            abs_path = os.path.abspath(path_to_data_file)
        else:
            print('Input file not found.')
            return
    else:
        if not isinstance(path_to_data_file, (list, tuple)):
            print('Data should be either a list of sequences or a path to the data file.')
        with open('jAlergiaInputs.txt', 'w') as f:
            for seq in path_to_data_file:
                f.write(','.join([str(s) for s in seq])+'\n')
        delete_tmp_file = True
        abs_path = os.path.abspath('jAlergiaInputs.txt')

    optimize_for = optimize_for[:3]

    subprocess.call(['java', heap_memory, '-jar', path_to_jAlergia_jar, '-input', abs_path, '-eps', str(eps), '-type',
                     automaton_type, '-optim', optimize_for])

    if not os.path.exists(save_file):
        print("JAlergia error occurred.")
        return

    model = load_automaton_from_file(save_file, automaton_type=automaton_type)
    os.remove(save_file)
    if delete_tmp_file:
        os.remove('jAlergiaInputs.txt')

    return model
