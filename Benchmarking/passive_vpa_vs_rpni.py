from aalpy import run_RPNI, run_PAPNI, load_automaton_from_file, AutomatonSUL
from aalpy.utils import convert_i_o_traces_for_RPNI, generate_input_output_data_from_vpa
from aalpy.utils.BenchmarkVpaModels import get_all_VPAs


def calculate_f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def calculate_precision_recall_f1(true_positives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = calculate_f1_score(precision, recall)

    return precision, recall, f1


def compare_rpni_and_papni(original_model, rpni_model, papni_model, training_dataset, num_sequances, max_seq_len):
    # generate test sequances
    test_data = generate_input_output_data_from_vpa(original_model, num_sequances, max_seq_len)

    # exclude ones that were used for model training, as that ones will be correctly classified
    test_data = [x for x in test_data if x not in training_dataset]

    num_positive_words = len([x for x in test_data if x[1]])

    print('Comparing RPNI and PAPNI on this dataset:')
    print(f'# positive: {num_positive_words}')
    print(f'# negative: {len(test_data) - num_positive_words}')

    def evaluate_model(learned_model, test_data):
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for input_seq, correct_output in test_data:
            learned_model.reset_to_initial()
            learned_output = learned_model.execute_sequence(learned_model.initial_state, input_seq)[-1]

            if learned_output and correct_output:
                true_positives += 1
            elif learned_output and not correct_output:
                false_positives += 1
            elif not learned_output and correct_output:
                false_negatives += 1

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = calculate_f1_score(precision, recall)

        return precision, recall, f1

    rpni_error = evaluate_model(rpni_model, test_data)
    papni_error = evaluate_model(papni_model, test_data)

    print(f'RPNI size {rpni_model.size} vs {papni_model.size} PAPNI size')
    print(f'RPNI   precision, recall, f1: {rpni_error}')
    print(f'PAPNI  precision, recall, f1: {papni_error}')


def get_sequances_from_active_sevpa(model):
    from aalpy import SUL, run_KV, RandomWordEqOracle, SevpaAlphabet

    class CustomSUL(SUL):
        def __init__(self, automatonSUL):
            super(CustomSUL, self).__init__()
            self.sul = automatonSUL
            self.sequances = []

        def pre(self):
            self.tc = []
            self.sul.pre()

        def post(self):
            self.sequances.append(self.tc)
            self.sul.post()

        def step(self, letter):
            output = self.sul.step(letter)
            if letter is not None:
                self.tc.append((letter, output))
            return output

    vpa_alphabet = model.get_input_alphabet()
    alphabet = SevpaAlphabet(vpa_alphabet.internal_alphabet, vpa_alphabet.call_alphabet, vpa_alphabet.return_alphabet)
    sul = AutomatonSUL(model)
    sul = CustomSUL(sul)
    eq_oracle = RandomWordEqOracle(alphabet.get_merged_alphabet(), sul, num_walks=50000, min_walk_len=6,
                                   max_walk_len=18, reset_after_cex=False)
    # eq_oracle = BreadthFirstExplorationEqOracle(vpa_alphabet.get_merged_alphabet(), sul, 7)
    _ = run_KV(alphabet, sul, eq_oracle, automaton_type='vpa')

    sequances = convert_i_o_traces_for_RPNI(sul.sequances)

    return convert_i_o_traces_for_RPNI(sul.sequances)


def generate_positive_words_bfs(vpa, max_depth=8):
    from collections import deque

    data_set = []

    queue = deque()
    queue.append(((), 0))

    vpa_alphabet = vpa.get_input_alphabet()
    merged_alphabet = vpa_alphabet.get_merged_alphabet()

    while queue:
        current_sequance, stack_len = queue.popleft()

        if len(current_sequance) == max_depth:
            break

        for inp in merged_alphabet:
            new_stack_len = stack_len
            if inp in vpa_alphabet.call_alphabet:
                new_stack_len += 1
            elif inp in vpa_alphabet.return_alphabet:
                new_stack_len -= 1

            if new_stack_len == -1:
                continue

            new_seq = current_sequance + (inp,)
            if new_stack_len == 0:
                print(new_seq)
                data_set.append(new_seq)

            queue.append((new_seq, new_stack_len))

    return data_set


test_models = get_all_VPAs()

# test_models = [test_models[-1]]

random_data_generation = True

for inx, ground_truth in enumerate(test_models):
    print(f'-----------------------------------------------------------------')
    print(f'Experiment: {inx}')

    if random_data_generation:
        data = generate_input_output_data_from_vpa(ground_truth, num_sequances=1000, max_seq_len=8)
    else:
        data = get_sequances_from_active_sevpa(ground_truth)

    vpa_alphabet = ground_truth.get_input_alphabet()

    num_positive_words = len([x for x in data if x[1]])

    print(f'# positive: {num_positive_words}')
    print(f'# negative: {len(data) - num_positive_words}')

    rpni_model = run_RPNI(data, 'dfa', print_info=True, input_completeness='sink_state')

    papni_model = run_PAPNI(data, vpa_alphabet, print_info=True)

    compare_rpni_and_papni(ground_truth, rpni_model, papni_model, data, num_sequances=25000, max_seq_len=50)
