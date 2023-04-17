from random import randint, choice

from aalpy.SULs import MdpSUL
from aalpy.automata import interval_smm_from_learning_data, interval_mdp_from_learning_data
from aalpy.learning_algs import run_stochastic_Lstar, run_Alergia
from aalpy.oracles import RandomWordEqOracle
from aalpy.utils import get_faulty_coffee_machine_MDP, get_small_gridworld, load_automaton_from_file
from aalpy.utils import mdp_2_prism_format
# coffee_m = load_automaton_from_file('DotModels/MDPs/first_grid.dot', 'mdp')
coffee_m = get_faulty_coffee_machine_MDP()


sul = MdpSUL(coffee_m)
alphabet = coffee_m.get_input_alphabet()
eq_oracle = RandomWordEqOracle(alphabet, sul, num_walks=100)

learning_type = 'mdp'

learned_model, data = run_stochastic_Lstar(alphabet, sul, eq_oracle, automaton_type=learning_type, return_data=True, )
if learning_type == 'mdp':
    learned_interval_mdp = learned_model.to_interval_mdp(data['observation_table'], confidence=0.8)
else:
    learned_interval_mdp = learned_model.to_interval_smm(data['observation_table'], confidence=0.8).to_interval_mdp()

prism_string = mdp_2_prism_format(learned_interval_mdp, 'interval_mdp', is_interval_mdp=True)
print(prism_string)


sul = MdpSUL(coffee_m)

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