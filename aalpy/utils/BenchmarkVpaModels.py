import random

from aalpy.automata.Vpa import Vpa, VpaAlphabet


def vpa_L1():
    # we always ensure that n >= 1

    call_set = ['a']
    return_set = ['b']
    internal_set = []

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

    state_setup = {
        "q0": (False, {"a": [("q1", 'push', "a")], }),
        "q1": (False, {"a": [("q1", 'push', "a")], "b": [("q2", 'pop', "a")]}),
        "q2": (True, {"b": [("q2", 'pop', "a")]}),
    }
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_L2():
    call_set = ['a', 'b']
    return_set = ['c', 'd']
    internal_set = []

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

    state_setup = {
        "q0": (False, {"a": [("q1", 'push', "a")], "b": [("q1", 'push', "b")], }),
        "q1": (False, {"a": [("q1", 'push', "a")], "b": [("q1", 'push', "b")],
                       "c": [("q2", 'pop', "a"), ("q2", 'pop', "b")],
                       "d": [("q2", 'pop', "a"), ("q2", 'pop', "b")]}),
        "q2": (True, {"c": [("q2", 'pop', "a"), ("q2", 'pop', "b")],
                      "d": [("q2", 'pop', "a"), ("q2", 'pop', "b")]}),
    }
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_for_L3():
    call_set = ['a', 'c', 'b', 'd']
    return_set = ['e', 'g', 'f', 'h']
    internal_set = []

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

    state_setup = {
        "q0": (False, {"a": [("q0a", 'push', "a")],
                       "c": [("q0c", 'push', "c")],
                       }),
        "q0a": (False, {"b": [("q1", 'push', "b")]}),
        "q0c": (False, {"d": [("q1", 'push', "d")]}),
        "q1": (False, {"a": [("q1a", 'push', "a")],
                       "c": [("q1c", 'push', "c")],
                       "e": [("q2e", 'pop', "b"), ("q2e", 'pop', "d")],
                       "g": [("q2g", 'pop', "b"), ("q2g", 'pop', "d")],  # stack should actually be redundant
                       }),
        "q1a": (False, {"b": [("q1", 'push', "b")]}),
        "q1c": (False, {"d": [("q1", 'push', "d")]}),
        "q2e": (False, {"f": [("q2", 'pop', "a"), ("q2", 'pop', "c")]}),
        "q2g": (False, {"h": [("q2", 'pop', "a"), ("q2", 'pop', "c")]}),
        "q2": (True, {"e": [("q2e", 'pop', "b"), ("q2e", 'pop', "d")],
                      "g": [("q2g", 'pop', "b"), ("q2g", 'pop', "d")]})
    }
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_L3():
    call_set = ['a', 'b']
    return_set = ['c', 'd']
    internal_set = []

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

    state_setup = {
        "q0": (False, {"a": [("q01", 'push', "a")], }),
        "q01": (False, {"b": [("q1", 'push', "b")], }),

        "q1": (False, {"a": [("q11", 'push', "a")],
                       "c": [("q21", 'pop', "b")]}),
        "q11": (False, {"b": [("q1", 'push', "b")], }),
        "q21": (False, {"d": [("q2", 'pop', "a")]}),
        "q2": (True, {"c": [("q21", 'pop', "b")]}),
    }
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_L4():
    # Dyck order 2

    call_set = ['(', '[']
    return_set = [')', ']']
    internal_set = []

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

    state_setup = {
        "q0": (False, {"(": [("q1", 'push', '(')],
                       "[": [("q1", 'push', '[')],  # exclude empty seq
                       }),
        "q1": (True, {"(": [("q1", 'push', '(')],
                      "[": [("q1", 'push', '[')],
                      ")": [("q1", 'pop', "(")],
                      "]": [("q1", 'pop', "[")]
                      }),
    }
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_L6():
    # Dyck order 3

    call_set = ['(', '[', '{']
    return_set = [')', ']', '}']
    internal_set = []

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

    state_setup = {
        "q0": (False, {"(": [("q1", 'push', '(')],
                       "[": [("q1", 'push', '[')],
                       "{": [("q1", 'push', '{')],
                       }),
        "q1": (True, {"(": [("q1", 'push', '(')],
                      "[": [("q1", 'push', '[')],
                      "{": [("q1", 'push', '{')],
                      ")": [("q1", 'pop', "(")],
                      "]": [("q1", 'pop', "[")],
                      "}": [("q1", 'pop', "{")],
                      }),
    }
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_L8():
    # Dyck order 4

    call_set = ['(', '[', '{', '<']
    return_set = [')', ']', '}', '>']
    internal_set = []

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

    state_setup = {
        "q0": (False, {"(": [("q1", 'push', '(')],
                       "[": [("q1", 'push', '[')],
                       "{": [("q1", 'push', '{')],
                       "<": [("q1", 'push', '<')],
                       }),
        "q1": (True, {"(": [("q1", 'push', '(')],
                      "[": [("q1", 'push', '[')],
                      "{": [("q1", 'push', '{')],
                      "<": [("q1", 'push', '<')],
                      ")": [("q1", 'pop', "(")],
                      "]": [("q1", 'pop', "[")],
                      "}": [("q1", 'pop', "{")],
                      ">": [("q1", 'pop', "<")],
                      }),
    }
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_L9():
    # Dyck order 2 (single-nested)

    call_set = ['(', '[']
    return_set = [')', ']']
    internal_set = []

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

    state_setup = {
        "q0": (False, {"(": [("q1", 'push', "(")],
                       "[": [("q1", 'push', "[")],  # exclude empty seq
                       }),
        "q1": (False, {"(": [("q1", 'push', "(")],
                       "[": [("q1", 'push', "[")],
                       ")": [("q2", 'pop', "(")],
                       "]": [("q2", 'pop', "[")]}),
        "q2": (True, {
            ")": [("q2", 'pop', "(")],
            "]": [("q2", 'pop', "[")]
        }),
    }
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_L10():
    # Dyck order 1

    call_set = ['(']
    return_set = [')']
    internal_set = ['a', 'b', 'c']

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

    state_setup = {
        "q0": (False, {"(": [("q1", 'push', "(")],
                       "a": [("q1", None, None)],
                       "b": [("q1", None, None)],
                       "c": [("q1", None, None)],  # exclude empty seq
                       }),
        "q1": (True, {"(": [("q1", 'push', "(")],
                      ")": [("q1", 'pop', "(")],
                      "a": [("q1", None, None)],
                      "b": [("q1", None, None)],
                      "c": [("q1", None, None)]
                      }),
    }
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_L11():
    # Dyck order 2

    call_set = ['(', '[']
    return_set = [')', ']']
    internal_set = ['a', 'b', 'c']

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

    state_setup = {
        "q0": (False, {"(": [("q1", 'push', "(")],
                       "[": [("q1", 'push', "[")],
                       "a": [("q1", None, None)],
                       "b": [("q1", None, None)],
                       "c": [("q1", None, None)],  # exclude empty seq
                       }),
        "q1": (True, {"(": [("q1", 'push', "(")],
                      "[": [("q1", 'push', "[")],
                      ")": [("q1", 'pop', "(")],
                      "]": [("q1", 'pop', "[")],
                      "a": [("q1", None, None)],
                      "b": [("q1", None, None)],
                      "c": [("q1", None, None)]
                      }),
    }
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_L12():
    # Dyck order 1

    call_set = ['(']
    return_set = [')']
    internal_set = ['a', 'b', 'c', 'd']

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

    state_setup = {
        "q0": (False, {"(": [("q1", 'push', "(")],
                       "a": [("qa", None, None)],
                       "d": [("q1", None, None)],  # exclude empty seq
                       }),
        "q1": (True, {"(": [("q1", 'push', "(")],
                      ")": [("q1", 'pop', "(")],
                      "a": [("qa", None, None)],
                      "d": [("q1", None, None)],
                      }),
        "qa": (False, {"b": [("qb", None, None)],
                       }),
        "qb": (False, {"c": [("q1", None, None)],
                       })
    }
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_for_L16():
    # just a testing language

    call_set = ['a']
    return_set = ['b']
    internal_set = []

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

    state_setup = {
        "q0": (False, {"a": [("q1", 'push', "$")]}),
        "q1": (False, {"a": [("q1", 'push', "x")],
                       "b": [("q1", 'pop', "x"), ("q2", 'pop', "$")],
                       }),
        "q2": (True, {})
    }
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def vpa_for_odd_parentheses():
    # VPA for accepting only odd number of fully balanced parentheses
    # It accepts patterns like () and ((()))), but rejects odd pairs or multiple groups

    call_set = ['(']
    return_set = [')']
    internal_set = []

    input_alphabet = VpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

    state_setup = {
        "q0": (False, {
            "(": [("q1", 'push', "(")]
        }),
        "q1": (False, {
            "(": [("q0", 'push', "(")],
            ")": [("q2", 'pop', "(")]
        }),
        "q2": (True, {
            ")": [("q2", 'pop', "(")]
        }),
    }

    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def gen_arithmetic_data(num_sequances=3000, min_seq_len=2, max_seq_len=8):
    import ast
    from aalpy.base import SUL
    from aalpy.utils import convert_i_o_traces_for_RPNI

    class ArithmeticSUL(SUL):
        def __init__(self):
            super().__init__()
            self.string_under_test = ''

        def pre(self):
            self.string_under_test = ''

        def post(self):
            pass

        def step(self, letter):
            if letter:
                self.string_under_test += ' ' + letter if len(self.string_under_test) > 0 else letter

            try:
                # Parse the expression using ast
                parsed_expr = ast.parse(self.string_under_test, mode='eval')
                # Check if the parsed expression is a valid arithmetic expression
                is_valid = all(isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Name, ast.Load))
                               or isinstance(node, ast.operator) or isinstance(node, ast.expr_context)
                               or (isinstance(node, ast.BinOp) and isinstance(node.op, ast.operator))
                               for node in ast.walk(parsed_expr))
                return is_valid

            except SyntaxError:
                return False

    sul = ArithmeticSUL()

    alphabet = VpaAlphabet(internal_alphabet=['1', '+', ], call_alphabet=['(', ], return_alphabet=[')', ])
    merged_alphabet = alphabet.get_merged_alphabet()
    data = []
    while len(data) < num_sequances:
        seq = []
        for _ in range(random.randint(min_seq_len, max_seq_len)):
            seq.append(random.choice(merged_alphabet))

            outputs = sul.query(tuple(seq))
            data.append(list(zip(seq, outputs)))

    rpni_format = convert_i_o_traces_for_RPNI(data)

    return rpni_format, alphabet


def vpa_json():
    # Define call, return, and internal symbols for JSON
    call_set = ['{', '[']
    return_set = ['}', ']']
    internal_set = [':', ',', 'key', 'val']  # Keys, values, and separators as internal alphabet symbols

    # Define the input alphabet
    input_alphabet = VpaAlphabet(
        internal_alphabet=internal_set,
        call_alphabet=call_set,
        return_alphabet=return_set
    )

    # Define states and transitions
    state_setup = {
        "q0": (False, {
            "{": [("q1", 'push', '{')],
            "[": [("q2", 'push', '[')],
            "}": [("q0", 'pop', '{')],
            "]": [("q0", 'pop', '[')],
            "key": [("q3", None, None)],  # Expect a key in an object

        }),

        # Array parsing state
        "q2": (False, {
            "val": [("q5", None, None)],  # Allow multiple values in an array
            "[": [("q2", 'push', '[')],  # Nested arrays
            "]": [("q0", 'pop', '[')],  # End of array
        }),

        # After parsing a key
        "q3": (False, {
            ":": [("q4", None, None)],  # Key-value separator
        }),

        # After parsing a key-value separator
        "q4": (False, {
            "val": [("q5", None, None)],  # Expecting a value
            "{": [("q0", 'push', '{')],  # Nested object
            "[": [("q2", 'push', '[')],  # Nested array
        }),

        # After parsing a value in an object
        "q5": (False, {
            ",": [("q4", None, None)],  # Another key-value pair
            "}": [("q0", 'pop', '{')],  # End of object
            "]": [("q0", 'pop', '[')],  # End of object
        }),
    }

    # Construct and return the VPA
    vpa = Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=input_alphabet)
    return vpa


def get_all_VPAs():
    from aalpy import load_automaton_from_file
    arithmetics_vpa = load_automaton_from_file('../DotModels/arithmetics.dot', 'vpa')
    return [vpa_L1(), vpa_L2(), vpa_L3(), vpa_L4(),
            vpa_L6(), vpa_L8(), vpa_L9(), vpa_L10(),
            vpa_L11(), vpa_L12(), vpa_for_odd_parentheses(), vpa_for_odd_parentheses(), arithmetics_vpa]

#
# if __name__ == '__main__':
#     test_seq = [
#         ('{', '}'),
#         ('{', '[', ']', '}'),
#         ('[', ']', '{', '}'),
#         ('{', 'key', ':', 'val', '}'),
#         ('{', 'key', ':', '{', 'key', ':', 'val', '}', '}'),
#         ('{', 'key', ':', '[', ']', '}'),
#         ('{', 'key', ':', '[', 'val', ']', '}'),
#         ('{', 'key', ':', '[', 'val', ',', 'val', ',', 'val', ']', '}'),
#         ('{', 'key', ':', '[', 'val', ']', 'key', ':', 'val', '}'),
#
#     ]
#
#     vj = vpa_json()
#     vj.visualize()
#     for ts in test_seq:
#         print('-----------------------------------')
#         vj.reset_to_initial()
#         o = vj.execute_sequence(vj.initial_state, ts)
#         if not o[-1]:
#             print(ts)
#     exit()
