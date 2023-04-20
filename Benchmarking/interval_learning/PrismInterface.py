import os
from collections import defaultdict
from pathlib import Path

import aalpy.paths
from aalpy.utils import mdp_2_prism_format

cluster_center_cache = dict()


class Scheduler:
    def __init__(self, initial_state, transition_dict, label_dict, scheduler_dict):
        self.scheduler_dict = scheduler_dict
        self.initial_state = initial_state
        self.transition_dict = transition_dict
        self.label_dict = label_dict
        self.current_state = None

    def get_input(self):
        if self.current_state is None:
            print("Return none because current state is none")
            return None
        else:
            # print("Current state is not none")
            if self.current_state not in self.scheduler_dict:
                return None
            return self.scheduler_dict[self.current_state]

    def reset(self):
        self.current_state = self.initial_state

    def poss_step_to(self, input):
        output_labels = []
        trans_from_current = self.transition_dict[self.current_state]
        found_state = False
        for (prob, action, target_state) in trans_from_current:
            if action == input:
                output_labels.extend(self.label_dict[target_state])
        return output_labels

    def step_to(self, input, output):
        reached_state = None
        trans_from_current = self.transition_dict[self.current_state]
        found_state = False
        for (prob, action, target_state) in trans_from_current:
            if action == input and output in self.label_dict[target_state]:
                reached_state = self.current_state = target_state
                found_state = True
                break
        if not found_state:
            reached_state = None

        return reached_state

    def get_available_actions(self):
        trans_from_current = self.transition_dict[self.current_state]
        return list(set([action for prob, action, target_state in trans_from_current]))


class PrismInterface:
    def __init__(self, destination, model, num_steps=None, operation='Pmax', add_step_counter=True, stepping_bound=20):
        assert operation in {'Pmax', 'Pmaxmin', 'Pmaxmax', 'Pminmax', 'Pminmin'}
        self.tmp_dir = Path("tmp_prism")
        self.destination = destination
        self.model = model
        self.num_steps = num_steps
        self.operation = operation
        if type(destination) != list:
            destination = [destination]
        destination = "_or_".join(destination)
        self.tmp_mdp_file = (self.tmp_dir / f"po_rl_{destination}.prism")
        # self.tmp_prop_file = f"{self.tmp_dir_name}/po_rl.props"
        self.current_state = None
        self.tmp_dir.mkdir(exist_ok=True)
        self.prism_property = self.create_mc_query()
        mdp_2_prism_format(self.model, "porl", output_path=self.tmp_mdp_file, is_interval_mdp=operation != 'Pmax',
                           add_step_counter=add_step_counter, stepping_bound=stepping_bound)

        self.dot_file =  (self.tmp_dir.absolute() / f"sched_{destination}.dot")
        self.adv_file_name = (self.tmp_dir.absolute() / f"sched_{destination}.adv")
        self.concrete_model_name = str(self.tmp_dir.absolute() / f"concrete_model_{destination}")
        self.property_val = 0
        self.call_prism()
        self.parser = PrismSchedulerParser(self.adv_file_name if self.operation == 'Pmax' else self.dot_file,
                                           self.concrete_model_name + ".lab",
                                           self.concrete_model_name + ".tra")
        self.scheduler = Scheduler(self.parser.initial_state, self.parser.transition_dict,
                                   self.parser.label_dict, self.parser.scheduler_dict)
        os.remove(self.dot_file)
        os.remove(self.tmp_mdp_file)
        if os.path.exists(self.adv_file_name):
            os.remove(self.adv_file_name)
        os.remove(self.concrete_model_name + ".lab")
        os.remove(self.concrete_model_name + ".tra")

    def create_mc_query(self):
        if type(self.destination) != list:
            destination = [self.destination]
        else:
            destination = self.destination
        destination = "|".join(map(lambda d: f"\"{d}\"", destination))

        opt_string = self.operation
        prop = f"{opt_string}=?[F {destination}]" if not self.num_steps else \
            f'{opt_string}=?[F({destination} & steps < {self.num_steps})]'
        return prop

    def call_prism(self):
        import subprocess
        from os import path

        self.property_val = 0

        destination_in_model = False
        for s in self.model.states:
            if self.destination in s.output.split("__"):
                destination_in_model = True
                break

        if not destination_in_model:
            print('SCHEDULER NOT COMPUTED')
            return self.property_val

        prism_file = aalpy.paths.path_to_prism.split('/')[-1]
        path_to_prism_file = aalpy.paths.path_to_prism[:-len(prism_file)]
        file_abs_path = path.abspath(self.tmp_mdp_file)
        proc = subprocess.Popen(
            [aalpy.paths.path_to_prism, file_abs_path, "-pf", self.prism_property, "-noprob1", "-exportadv",
             self.adv_file_name, "-exportstrat", self.dot_file, "-exportmodel", f"{self.concrete_model_name}.all"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=path_to_prism_file)
        out = proc.communicate()[0]
        out = out.decode('utf-8').splitlines()
        for line in out:
            # print(line)
            if not line:
                continue
            if 'Syntax error' in line:
                print(line)
            else:
                if "Result:" in line:
                    end_index = len(line) if "(" not in line else line.index("(") - 1
                    try:
                        self.property_val = float(line[len("Result: "): end_index])
                        # if result_val < 1.0:
                        #    print(f"We cannot reach with absolute certainty, probability is {result_val}")
                    except:
                        print("Result parsing error")
        proc.kill()
        return self.property_val


class PrismSchedulerParser:
    def __init__(self, scheduler_file, label_file, transition_file):
        with open(scheduler_file, "r") as f:
            self.scheduler_file_content = f.readlines()
        with open(label_file, "r") as f:
            self.label_file_content = f.readlines()
        with open(transition_file, "r") as f:
            self.transition_file_content = f.readlines()
        self.label_dict = self.create_labels()
        self.transition_dict = self.create_transitions()
        self.scheduler_dict = self.parse_scheduler(is_dot_file='.dot' in str(scheduler_file))
        self.initial_state = next(filter(lambda e: "init" in e[1], self.label_dict.items()))[0]
        self.actions = set()
        for l in self.transition_dict.values():
            for _, action, _ in l:
                self.actions.add(action)
        self.actions = list(self.actions)

    def create_labels(self):
        label_dict = dict()
        header_line = self.label_file_content[0]
        label_lines = self.label_file_content[1:]
        header_dict = dict()
        split_header = header_line.split(" ")
        for s in split_header:
            label_id = s.strip().split("=")[0]
            label_name = s.strip().split("=")[1].replace('"', '')
            header_dict[label_id] = label_name
        for l in label_lines:
            state_id = int(l.split(":")[0])
            label_ids = l.split(":")[1].split(" ")
            label_names = set(
                map(lambda l_id: header_dict[l_id.strip()], filter(lambda l_id: l_id.strip(), label_ids)))
            label_dict[state_id] = label_names
        return label_dict

    def create_transitions(self):
        header_line = self.transition_file_content[0]
        transition_lines = self.transition_file_content[1:]
        transitions = defaultdict(list)
        for t in transition_lines:
            split_line = t.split(" ")
            source_state = int(split_line[0])
            target_state = int(split_line[2])
            prob = float(split_line[3]) if '[' not in split_line[3] else split_line[3]
            action = split_line[4].strip()
            transitions[source_state].append((prob, action, target_state))
        return transitions

    def parse_scheduler(self, is_dot_file):
        scheduler = dict()

        if not is_dot_file:
            header_line = self.scheduler_file_content[0]
            transition_lines = self.scheduler_file_content[1:]
            for t in transition_lines:
                split_line = t.split(" ")
                source_state = int(split_line[0])
                action = split_line[3].strip()
                if source_state in scheduler:
                    assert action == scheduler[source_state]
                else:
                    scheduler[source_state] = action
        else:
            for t in self.scheduler_file_content:
                split_line = t.split(":")
                scheduler[int(split_line[0])] = split_line[1].strip()

        return scheduler
