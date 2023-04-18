from random import randint, choice

from aalpy.SULs import MdpSUL
from aalpy.automata import interval_smm_from_learning_data, interval_mdp_from_learning_data, IntervalSmm
from aalpy.learning_algs import run_stochastic_Lstar, run_Alergia
from aalpy.oracles import RandomWordEqOracle
from aalpy.utils import get_faulty_coffee_machine_MDP, get_small_gridworld, load_automaton_from_file, get_small_pomdp
from aalpy.utils import mdp_2_prism_format

model_under_learning = load_automaton_from_file('DotModels/MDPs/first_grid.dot', 'mdp')

#model_under_learning = get_faulty_coffee_machine_MDP()
#model_under_learning = get_small_pomdp()

sul = MdpSUL(model_under_learning)
alphabet = model_under_learning.get_input_alphabet()
eq_oracle = RandomWordEqOracle(alphabet, sul, num_walks=100)


learned_model = run_stochastic_Lstar(alphabet, sul, eq_oracle, automaton_type='interval_mdp',
                                     interval_confidence=0.9, interval_method='normal',)
learned_model.visualize()
exit()

if isinstance(learned_model, IntervalSmm):
    learned_model = learned_model.to_interval_mdp()

prism_string = mdp_2_prism_format(learned_model, 'interval_mdp', is_interval_mdp=True, output_path='interval_mdp.prism')
print(prism_string)

exit()

sul = MdpSUL(model_under_learning)

data = []
for _ in range(100000):
    str_len = randint(5, 12)
    seq = [sul.pre()]
    for _ in range(str_len):
        i = choice(alphabet)
        o = sul.step(i)
        seq.append((i, o))
    sul.post()
    data.append(seq)

# run alergia with the data and automaton_type set to 'mdp' to True to learn a MDP
model = run_Alergia(data, automaton_type='interval_mdp', eps=0.005,
                    interval_confidence=0.8, interval_method='normal', print_info=True)
print(model)


# # if learning_type == 'mdp':
# #     learned_interval_mdp = learned_model.to_interval_mdp(data['observation_table'], confidence=0.8)
# # else:
# #     learned_interval_mdp = learned_model.to_interval_smm(data['observation_table'], confidence=0.8).to_interval_mdp()