import os
import pathlib
import pandas as pd
import numpy as np
from aalpy.oracles import RandomWMethodEqOracle
from aalpy.oracles import PerfectKnowledgeEqOracle
from aalpy.oracles import StatePrefixEqOracle
from aalpy.oracles.SortedStateCoverageEqOracle import SortedStateCoverageEqOracle
from aalpy.oracles.InterleavedStateCoverageEqOracle import InterleavedStateCoverageEqOracle
from aalpy.oracles.StochasticStateCoverageEqOracle import StochasticStateCoverageEqOracle
from aalpy.SULs.AutomataSUL import AutomatonSUL
from aalpy.learning_algs.deterministic.LStar import run_Lstar
from aalpy.utils.FileHandler import load_automaton_from_file
import multiprocessing as mp

# print up to 1 decimal point
np.set_printoptions(precision=1)
# do not print in scientific notation
np.set_printoptions(suppress=True)
# print up to 3 decimal point
pd.options.display.float_format = '{:.3f}'.format

WALKS_PER_ROUND = {
        "TCP": 100000,
        "TLS": 10000,
        "MQTT": 10000
        }
WALKS_PER_STATE = {
        "TCP": 1000,
        "TLS": 1000,
        "MQTT": 1000
        }
WALK_LEN = {
        "TCP": 200,
        "TLS": 100,
        "MQTT": 100
        }

class Random(StatePrefixEqOracle):
    def __init__(self, alphabet, sul,
                 walks_per_round,
                 walks_per_state,
                 walk_len,
                 depth_first=False):
        super().__init__(alphabet, sul, walks_per_round, walks_per_state, walk_len, depth_first)

class StochasticLinear(StochasticStateCoverageEqOracle):
    def __init__(self, alphabet, sul,
                 walks_per_round,
                 walk_len,
                 prob_function='linear'):
        super().__init__(alphabet, sul, walks_per_round, walk_len, prob_function)

class StochasticSquare(StochasticStateCoverageEqOracle):
    def __init__(self, alphabet, sul,
                 walks_per_round,
                 walk_len,
                 prob_function='square'):
        super().__init__(alphabet, sul, walks_per_round, walk_len, prob_function)

class StochasticExponential(StochasticStateCoverageEqOracle):
    def __init__(self, alphabet, sul,
                 walks_per_round,
                 walk_len,
                 prob_function='exponential'):
        super().__init__(alphabet, sul, walks_per_round, walk_len, prob_function)

def process_oracle(alphabet, sul, oracle, correct_size, i):
    """
    Process the oracle and return the number of queries to the equivalence and membership oracles
    and whether the learned model has the correct size.

    Args:
        alphabet: input alphabet
        sul: system under learning
        oracle: equivalence oracle
        correct_size: correct size of the model
        i: index of the oracle
    """
    _, info = run_Lstar(alphabet, sul, oracle, 'mealy', return_data=True, print_level=0)
    print(f"{i} number of hypotheses: {len(info['intermediate_hypotheses'])}")
    return (i, info['queries_eq_oracle'],
               info['queries_learning'],
               1 if info['automaton_size'] != correct_size else 0)

def do_learning_experiments(model, alphabet, correct_size, prot):
    """
    Perform the learning experiments for the given model and alphabet.

    Args:
        model: model to learn
        alphabet: input alphabet
        correct_size: correct size of the model
    """
    # create a copy of the SUL for each oracle
    suls = [AutomatonSUL(model) for _ in range(NUM_ORACLES)]
    # initialize the oracles
    eq_oracles = [Random(alphabet, suls[0], WALKS_PER_ROUND[prot], WALKS_PER_STATE[prot], WALK_LEN[prot]),
                  StochasticLinear(alphabet, suls[1], WALKS_PER_ROUND[prot], WALK_LEN[prot]),
                  StochasticSquare(alphabet, suls[2], WALKS_PER_ROUND[prot], WALK_LEN[prot]),
                  StochasticExponential(alphabet, suls[3], WALKS_PER_ROUND[prot], WALK_LEN[prot])]
    # create the arguments for eache oracle's task
    tasks = [(alphabet, sul, oracle, correct_size, i)
             for i, (sul, oracle) in enumerate(zip(suls, eq_oracles))]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(process_oracle, tasks)

    return results


def main():
    ROOT = os.getcwd() + "/DotModels"
    # PROTOCOLS = ["ASML", "TLS", "MQTT", "EMV", "TCP"]
    # ORDER: TCP -> TLS -> MQTT
    PROTOCOLS = ["TLS", "MQTT"]
    DIRS = [pathlib.Path(ROOT + '/' + prot) for prot in PROTOCOLS]
    FILES = [file for dir in DIRS for file in dir.iterdir()]
    FILES_PER_PROT = {prot: len([file for file in DIRS[i].iterdir()]) for i, prot in enumerate(PROTOCOLS)}
    MODELS = [load_automaton_from_file(f, 'mealy') for f in FILES]
    
    EQ_QUERIES = np.zeros((len(MODELS), TIMES, NUM_ORACLES))
    MB_QUERIES = np.zeros((len(MODELS), TIMES, NUM_ORACLES))
    FAILURES   = np.zeros((len(MODELS), TIMES, NUM_ORACLES))
    # iterate over the models
    for index, (model, file) in enumerate(zip(MODELS, FILES)):
        # these variables can be shared among the processes
        prot = file.parent.stem
        correct_size = model.size
        alphabet = list(model.get_input_alphabet())
        # repeat the experiments to gather statistics
        for trial in range(TIMES):

            results = do_learning_experiments(model, alphabet, correct_size, prot)

            for i, eq_queries, mb_queries, failure in results:
                EQ_QUERIES[index, trial, i] = eq_queries
                MB_QUERIES[index, trial, i] = mb_queries
                FAILURES[index, trial, i] = failure
    
    prev = 0
    for prot in PROTOCOLS:
        items = FILES_PER_PROT[prot]
        np.save(f'eq_queries_{prot}.npy', EQ_QUERIES[prev:prev+items, :, :])
        np.save(f'mb_queries_{prot}.npy', MB_QUERIES[prev:prev+items, :, :])
        np.save(f'failures_{prot}.npy', FAILURES[prev:prev+items, :, :])
        prev += items

    for array, name in zip([EQ_QUERIES, MB_QUERIES, FAILURES],
                            ['eq_queries', 'mb_queries', 'failures']):
        averages = np.mean(array, axis=1)
        std_devs = np.std(array, axis=1)
        np.save(f'{name}.npy', array)
        np.save(f'{name}_averages.npy', averages)
        np.save(f'{name}_std_devs.npy', std_devs)

if __name__ == '__main__':
    TIMES = 1
    NUM_ORACLES = 4
    main()

