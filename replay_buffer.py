
import random
from gnn_module import PIEGNN, load_cnf_labels, construct_labels, process_cnf_batch
from inst_config import USE_M2K, LEVELS
import os
import pickle
import copy
import math
import glob


m2k = USE_M2K


def scan_noisy_data_folder(expname):

    """"Looks in the noisy data folder


    Returns a pair of lists (cnf, labelfile)
    """

    noisy_data_folder = f"data/noisy_data/{expname}/"

    cnf_file_folders = [f"level{k}" for k in range(0, LEVELS * 2)]
    # label_file_folders = [""] + [f"level{k}" for k in range(1, LEVELS * 2)]

    cnf_data_folders = [noisy_data_folder + k for k in cnf_file_folders]

    cnf_list = []
    label_list = []

    print("noisy data folder")
    print(noisy_data_folder)
    for cnf_folder in cnf_data_folders:
        print(cnf_folder)
        files_in_folder = glob.glob(cnf_folder + "/*")

        for file in files_in_folder:

            label_file_loc = file.replace("noisy_data", "noisy_data_labels")
            if os.path.exists(label_file_loc):
                cnf_list.append(file)
                label_list.append(label_file_loc)


    return cnf_list, label_list


class ReplayBuffer:

    def __init__(self):

        self.samples = {}

    def add_samples(self, new_dict):
        """
        Add samples and keep the shorter proofs, prefer non-random_shooting data if possible; for consistency
        sake.
        :param new_dict:
        :return:
        """
        for new_sample in new_dict:

            len_new_sample = len(new_dict[new_sample])
            if new_sample in self.samples:

                current_proof_length = len(self.samples[new_sample])
                if len_new_sample < current_proof_length:
                    self.samples[new_sample] = new_dict[new_sample]

                elif len_new_sample == current_proof_length:

                    # check if the current proof was from the original random shooting run
                    if "base_data/cnf2_var_renamed_140621" in self.samples[new_sample][0][0]:
                        print("Buffer from original, new one is not, replacing it.")
                        self.samples[new_sample] = new_dict[new_sample]
                    else:
                        pass
                        print("Not from original; new proof is same length as old one")
                else:
                    print("Old proof was shorter, keeping that one.")

            elif new_sample not in self.samples:
                print(f"Haven't seen a proof of this before, adding it to the buffer: {new_sample}")
                self.samples[new_sample] = new_dict[new_sample]

    def produce_sample(self, n):

        proved_theorems = list(self.samples.keys())

        if n > len(proved_theorems):
            n = len(proved_theorems)


        sample = random.sample(proved_theorems, k=n)
        print(sample)
        return [self.samples[k] for k in sample]

    def produce_sample_biased(self, n):
        """
        Sample deeper things more
        """
        proved_theorems = list(self.samples.keys())

        max_len = max([len(self.samples[k]) for k in proved_theorems])

        main_list = [[] for k in range(max_len)]

        for pt in proved_theorems:

            pt_proof = self.samples[pt]
            proof_length = len(pt_proof)

            main_list[proof_length].append(pt_proof)

        samples_per_length = math.floor( n / max_len)

        sampled_theorems = []
        for focus_length in range(max_len):
            sample = random.choices(main_list[focus_length], k=samples_per_length)
            sampled_theorems += sample

        return [self.samples[k] for k in sample]

    def save_buffer(self, name):

        path = f"data/replay_buffers_pickled/{name}.pkl"

        with open(path, 'wb') as pickled_buffer:
            pickle.dump(self, pickled_buffer)

    def load_buffer(self, name):

        path = f"data/replay_buffers_pickled/{name}.pkl"
        with open(path, 'rb') as old_buffer_file:
            old_buffer = pickle.load(old_buffer_file)

        self.samples = copy.deepcopy(old_buffer.samples)

    def merge_buffer(self, name):
        # merge another buffer into this one.

        path = f"data/replay_buffers_pickled/{name}.pkl"
        with open(path, 'rb') as buffer_to_merge:
            merge_buffer = pickle.load(buffer_to_merge)
        print(f"Before merge, we knew {len(self.samples)} different proofs")
        self.add_samples(merge_buffer)
        print(f"After merge, we knew {len(self.samples)} different proofs")




    # def write_current_content(self, filelocation):
    #
    #     with open(filelocation, "w") as buffer_file:



def scan_folder_for_proofs(case):
    if case == "original_data":
        proof_multilevel_dict = {}
        with open(f"lists/00all_probs_train_without_devel", "r") as train_split:

            train_theorems = train_split.readlines()
            train_theorems = [k.strip() for k in train_theorems]

        if m2k:
            with open("lists/M2k_list", "r") as m2k_file:

                restricted_set = m2k_file.readlines()
                restricted_set = [k.strip() for k in restricted_set]

                filtered_predict_theorems = []
                for theorem in train_theorems:

                    base_theorem, proof_id = theorem.split("__")
                    if base_theorem in restricted_set:
                        filtered_predict_theorems.append(theorem)

                train_theorems = filtered_predict_theorems

        train_theorems = train_theorems
        base_data_folder_cnf = f"data/base_data/"
        base_data_folder_lab = f"data/base_data/"

        cnf_file_folders = ["cnf2_var_renamed_140621", "random_shooting/gnn_data_fixed_25_5/second_step/inputs_var_renamed", ]
        label_file_folders = ["random_shooting/gnn_data_fixed_25_5/first_step/labels", "random_shooting/gnn_data_fixed_25_5/second_step/labels", ]

        train_data_cnf_files = []
        train_data_label_files = []
        train_data = []
        for theorem in train_theorems:
            for input_data_file_folder, label_data_file_folder in zip(cnf_file_folders, label_file_folders):
                cnf_file = base_data_folder_cnf + input_data_file_folder + "/" + theorem

                label_file = base_data_folder_lab + label_data_file_folder + "/" + theorem

                if os.path.isfile(cnf_file) and os.path.isfile(label_file):
                    print(cnf_file)
                    print(label_file)
                    if theorem not in proof_multilevel_dict:
                        proof_multilevel_dict[theorem] = [(cnf_file,label_file)]
                    elif theorem in proof_multilevel_dict:
                        proof_multilevel_dict[theorem] += [(cnf_file, label_file)]

    else:

        proof_multilevel_dict = {}
        with open(f"lists/00all_probs_train_without_devel", "r") as train_split:

            train_theorems = train_split.readlines()
            train_theorems = [k.strip() for k in train_theorems]

        if m2k:
            with open("lists/M2k_list", "r") as m2k_file:

                restricted_set = m2k_file.readlines()
                restricted_set = [k.strip() for k in restricted_set]

                filtered_predict_theorems = []
                for theorem in train_theorems:

                    base_theorem, proof_id = theorem.split("__")
                    if base_theorem in restricted_set:
                        filtered_predict_theorems.append(theorem)

                train_theorems = filtered_predict_theorems

        train_theorems = train_theorems
        base_data_folder_cnf = f"data/constructed_labels/{case}/cnfs/"
        base_data_folder_lab = f"data/constructed_labels/{case}/labels/"

        cnf_file_folders = [f"level{k}" for k in range(LEVELS*2)]
        label_file_folders = [f"level{k}" for k in range(LEVELS*2)]
        # cnf_file_folders = ["level0", "level1", "level2", "level3", "level4", "level5"]
        # label_file_folders = ["level0", "level1", "level2", "level3", "level4", "level5"]

        train_data_cnf_files = []
        train_data_label_files = []
        train_data = []
        for theorem in train_theorems:
            for input_data_file_folder, label_data_file_folder in zip(cnf_file_folders, label_file_folders):
                cnf_file = base_data_folder_cnf + input_data_file_folder + "/" + theorem
                # print(cnf_file)

                label_file = base_data_folder_lab + label_data_file_folder + "/" + theorem
                # print(label_file)
                if os.path.isfile(cnf_file) and os.path.isfile(label_file):
                    print(cnf_file)
                    print(label_file)
                    if theorem not in proof_multilevel_dict:
                        proof_multilevel_dict[theorem] = [(cnf_file, label_file)]
                    elif theorem in proof_multilevel_dict:
                        proof_multilevel_dict[theorem] += [(cnf_file, label_file)]
    print(f"Scanned folders, found {len(proof_multilevel_dict)} proofs.")

    return proof_multilevel_dict



