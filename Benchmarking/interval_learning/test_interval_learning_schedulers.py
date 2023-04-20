import sys
sys.path.append('../..')

from random import randint, choice

from Benchmarking.interval_learning.PrismInterface import PrismInterface
from aalpy.SULs import MdpSUL
from aalpy.automata import interval_smm_from_learning_data, interval_mdp_from_learning_data, IntervalSmm
from aalpy.learning_algs import run_stochastic_Lstar, run_Alergia
from aalpy.oracles import RandomWordEqOracle
from aalpy.utils import get_faulty_coffee_machine_MDP, get_small_gridworld, load_automaton_from_file, get_small_pomdp
from aalpy.utils import mdp_2_prism_format

import aalpy.paths

aalpy.paths.path_to_prism = '/mnt/c/Users/muskardine/Desktop/interval_prism/prism-imc2/prism/bin/prism'
# aalpy.paths.path_to_prism = "C:/Program Files/prism-4.7/bin/prism.bat"


def evaluate_scheduler(scheduler, model, goal, num_steps, num_tests=2000,):
    num_reached = 0
    for _ in range(num_tests):
        model.reset_to_initial()
        scheduler.reset()
        step_counter = 0
        for _ in range(num_steps):
            action = scheduler.get_input()
            output = model.step(action)
            scheduler.step_to(action, output)
            if goal in output:
                num_reached += 1
                break

    return num_reached / num_tests


model_under_learning = load_automaton_from_file('../../DotModels/MDPs/second_grid.dot', 'mdp')

# model_under_learning = get_faulty_coffee_machine_MDP()
# model_under_learning = get_small_pomdp()

sul = MdpSUL(model_under_learning)
alphabet = model_under_learning.get_input_alphabet()
eq_oracle = RandomWordEqOracle(alphabet, sul, num_walks=100)

learning_type = 'smm'
learned_model, data = run_stochastic_Lstar(alphabet, sul, eq_oracle, automaton_type=learning_type,
                                           max_rounds=40,
                                           return_data=True)

if learning_type == 'mdp':
    learned_interval_mdp = learned_model.to_interval_mdp(data['observation_table'], confidence=0.9)
else:
    learned_interval_mdp = learned_model.to_interval_smm(data['observation_table'], confidence=0.9).to_interval_mdp()
    learned_model = learned_model.to_mdp()

for step_nums in [11, 12, 13, 15, 17]:
    #######################################

    goal_state = 'goal'
    num_steps = step_nums
    prism_interface_mdp = PrismInterface(goal_state, learned_model, num_steps=num_steps, operation='Pmax',
                                         add_step_counter=True, stepping_bound=20)
    normal_scheduler = prism_interface_mdp.scheduler

    print('--------------------------------')
    print(f'Max step number: {step_nums}')
    print('Testing schedulers on 2000 episodes:')
    normal_scheduler_res = evaluate_scheduler(normal_scheduler, model_under_learning, goal_state, num_steps=num_steps)
    print('Normal scheduler  (Pmax)    :', normal_scheduler_res)

    #######################################

    prism_interface_mdp = PrismInterface(goal_state, learned_interval_mdp, num_steps=num_steps, operation='Pmaxmin',
                                         add_step_counter=True, stepping_bound=20)
    interval_scheduler = prism_interface_mdp.scheduler

    interval_scheduler_res = evaluate_scheduler(interval_scheduler, model_under_learning, goal_state, num_steps=num_steps)
    print('Interval scheduler (Pmaxmin):', interval_scheduler_res)

    #######################################

    prism_interface_mdp = PrismInterface(goal_state, learned_interval_mdp, num_steps=num_steps, operation='Pmaxmax',
                                         add_step_counter=True, stepping_bound=20)
    interval_scheduler = prism_interface_mdp.scheduler

    interval_scheduler_res = evaluate_scheduler(interval_scheduler, model_under_learning, goal_state, num_steps=num_steps)
    print('Interval scheduler (Pmaxmax):', interval_scheduler_res)


