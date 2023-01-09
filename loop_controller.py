# This file should run the gnn in multilevel mode

import os

import argparse
import shutil
import random
import copy
#
from inst_config import LEVELS, PREMSEL, USE_M2K, EXPLORATION_SAMPLE_SIZE, PREMSEL_TRAIN_EPOCHS

cl = True

def split(a, n):
    n = min(n, len(a))
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


if cl:
    parser = argparse.ArgumentParser(description="Grid Search")

    parser.add_argument("--expname", type=str, help="Defines the output folder paths")
    parser.add_argument("--set", type=str, help="Only active if toplevel = 1. Can be train, val, test.")

    # parser.add_argument("--num_samples", type=int, help="How many samples to take from the model per clause")
    parser.add_argument("--num_samples", nargs="+", type=int, help="How many samples to take from the model per clause")
    parser.add_argument("--temperature", type=str, help="Softmax sampling temperature.")
    parser.add_argument("--model", type=str, help="Model file location")
    parser.add_argument("--epoch", type=int, help="Which checkpoint to take")
    parser.add_argument("--keep_clauses", type=int, help="Whether to propagate the original clauses to the next level")
    parser.add_argument("--beam", type=int, help="Whether we use the beam search")
    parser.add_argument("--rnn_its", type=int,
                        help="How many iterations to run the rnn (bounds the numbers of variables that can be assigned a symbol per clause).")
    parser.add_argument("--gpu", type=int, help="Which gpu to run on.")
    parser.add_argument("--inner_index", type=int, help="Which worker on the gpu this is")
    parser.add_argument("--inner_para_factor", type=int, help="Need to know how many gpus are running on 1 dataset")
    parser.add_argument("--seed", type=int, help="Global seed for all workers to keep sample consistent")
    parser.add_argument("--data_parallel_factor", type=int, help="Need to know how many GPUs there are if running parallel on the same dataset (probably either 1 (no parallelism) or 8 (running 1 experiment on all gpus")


    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    exp = args.expname
    dataset = args.set
    ns = args.num_samples
    temp = args.temperature
    model = args.model
    epoch = int(args.epoch)
    use_orig_clauses = int(args.keep_clauses)
    beam = int(args.beam)
    rnn_its = int(args.rnn_its)
    data_parallel_factor = args.data_parallel_factor
    inner_para_factor = args.inner_para_factor

    gpu_index = int(args.gpu)
    inner_index = int(args.inner_index)


else:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"3"
    exp = "dgxmove_check"
    dataset = "train"
    ns = [2, 2, 2]
    temp = 2
    model = "random_shooting_conditional_inst_no_shuffle"
    epoch = 18
    use_orig_clauses = 0
    beam = 0
    rnn_its = 6

# levelstrings = ['', '_level1', '_level2', '_level3', '_level4', '_level5', '_level6', '_level7', '_level8', '_level9']
levelstrings = [''] + [f'_level{k}' for k in range(1, LEVELS*2)]
reporting_strings = []


# I need to know which iteration we are in to know whether there exists a premise selector model
experiment_iteration = int(exp.split("_")[-1])

if experiment_iteration == 0:
    premise_selector = False
else:
    premise_selector = PREMSEL
# Top level (0)
no_levels = LEVELS
assert no_levels == len(ns)
# import random
seed = args.seed

def map_clause_types(string):

    string = string.replace(",plain,", ",axiom_redundant,")
    string = string.replace(",negated_conjecture,", ",axiom_useful,")

    return string

def map_clause_types_back(string):
    string = string.replace(",axiom_redundant,", ",plain,")
    string = string.replace(",axiom_useful,", ",negated_conjecture,")

    return string

def map_file(filename, writefolder):

    if os.path.getsize(filename):
        with open(filename, "r") as f1:

            lines = f1.readlines()
            lines = [map_clause_types(k) for k in lines]

            just_fn = filename.split("/")[-1]

        with open(writefolder + just_fn, "w") as wf:
            for line in lines:
                wf.write(line)


def map_file_back(filename, writefolder):

    if os.path.getsize(filename):
        with open(filename, "r") as f1:

            lines = f1.readlines()
            lines = [map_clause_types_back(k) for k in lines]

            just_fn = filename.split("/")[-1]

        with open(writefolder + just_fn, "w") as wf:
            for line in lines:
                wf.write(line)

for level in range(no_levels):

    if level == 0:
        print(f"EXECUTING LEVEL {level}")
        os.system(f"python -u predicting_loop.py --expname {exp} --gpu_index {args.gpu} --inner_index {args.inner_index} --seed {seed} --data_parallel_factor {data_parallel_factor}  --inner_para_factor {inner_para_factor} --beam {beam} --keep_clauses {use_orig_clauses} --model {model} --rnn_its {rnn_its} --epoch {epoch} --input a --toplevel 1 --set {dataset} --num_samples {ns[level]} --temperature {temp}")

    # Level 1

    # Premise selector should be here.
    # the input argument can perhaps be pruned but it could be nice to expose it (not infer it).
    elif level > 0:


        # If premise selector active:
        # First copy folder and put all "plain", "negated_conjecture" to "axiom_useful", "axiom_redundant"
        # maybe map them one-to-one, so I can go back.

        # then print them to:
        # expname ++ "ps"
        if premise_selector:
            sample_per_exploration_step = EXPLORATION_SAMPLE_SIZE
            print("Using premise selector!")
            if use_orig_clauses == 0:
                datafolder = f"data/inst_files/{exp + levelstrings[level-1]}/"
            elif use_orig_clauses == 1:
                print("Keeping original clauses")
                datafolder = f"data/inst_plus_orig_files/{exp + levelstrings[level-1]}/"

            # TODO copy these folders somewhere so the premsel data can be constructed.

            # probably have to turn premise selector off the first round.
            # make it only 2 steps?

            # TODO this is bad; need to only select the files that fall in this process' range

            # all_files = glob.glob(datafolder + "*")

            ########################### SELECTING THE RIGHT CHUNK OF DATA (PREMSEL MAKES IT NECESSARY HERE.)#################

            tr_list_file = f"lists/00all_probs_train_without_devel"

            with open(tr_list_file,
                      "r") as train_split:

                train_theorems = train_split.readlines()
                train_theorems = [k.strip() for k in train_theorems]


            val_list_file = f"lists/00all_probs_devel"


            with open(val_list_file, "r") as val_split:

                val_theorems = val_split.readlines()
                val_theorems = [k.strip() for k in val_theorems]


            test_list_file = f"lists/00all_probs_test"


            with open(test_list_file, "r") as test_split:

                test_theorems = test_split.readlines()
                test_theorems = [k.strip() for k in test_theorems]

            if dataset == "train":
                predict_theorems = train_theorems

            elif dataset == "val":
                predict_theorems = val_theorems

            elif dataset == "test":
                predict_theorems = test_theorems

            if USE_M2K:
                with open("lists/M2k_list", "r") as m2k_file:

                    restricted_set = m2k_file.readlines()
                    restricted_set = [k.strip() for k in restricted_set]

                    filtered_predict_theorems = []
                    for theorem in predict_theorems:

                        base_theorem, proof_id = theorem.split("__")
                        if base_theorem in restricted_set:
                            filtered_predict_theorems.append(theorem)

                    predict_theorems = filtered_predict_theorems

            # predict_theorems_sample = random.sample(predict_theorems, k=200)
            # random.seed(42)
            # predict_theorems = random.sample(predict_theorems, k=2000)

            random.seed(args.seed)  # seed I used for the grid search;
            if len(predict_theorems) < sample_per_exploration_step:
                sample_per_exploration_step = len(predict_theorems)
            predict_theorems_sample = random.sample(predict_theorems, k=sample_per_exploration_step)

            split_list = list(split(predict_theorems_sample, data_parallel_factor * inner_para_factor))

            print(f"The start of it looks like this:")
            print(split_list[gpu_index][:10])
            # predict_theorems = predict_theorems[:100]

            # Order is fixed because reading from file.

            # Pick the segment that is assigned to this gpu

            predict_theorems_split = split_list[gpu_index * inner_para_factor + inner_index]

            # predict_theorems = [k for k in predict_theorems_sample if k in predict_theorems_split]

            predict_theorems = predict_theorems_split
            ############################## CHUNK SELECTED

            all_files = [datafolder + k for k in predict_theorems]

            # glob the files printed at previous level eground plus orig files
            transition_in_folder = f"data/premsel_transition_in/{exp + levelstrings[level-1]}/"
            transition_out_folder = f"data/premsel_transition_out/{exp + levelstrings[level-1]}/"
            original_without_premsel_folder = f"data/no_premsel_copies/{exp + levelstrings[level-1]}/"

            if not os.path.exists(transition_in_folder):
                os.makedirs(transition_in_folder)

            if not os.path.exists(transition_out_folder):
                os.makedirs(transition_out_folder)

            if not os.path.exists(original_without_premsel_folder):
                os.makedirs(original_without_premsel_folder)

            # TODO figure out why copy2 is not always successful.
            for filename in all_files:
                just_fl = filename.split("/")[-1]
                shutil.copy2(filename, original_without_premsel_folder + just_fl)

            with open(f"data/premsel_comm_logs/{exp}_{level}", "a") as f:
                f.write("folders made\n")
            print("Folders made!")

            # map clausetypes
            input_list = []
            for filename in all_files:
                map_file(filename, transition_in_folder)
                input_list.append(transition_in_folder + filename.split("/")[-1])
            #
            with open(f"data/premsel_transition_in_lists/{exp + levelstrings[level-1]}_{gpu_index}_{inner_index}", "w") as prl_file:
                for premsel_input_filename in input_list:
                    prl_file.write(premsel_input_filename)
                    prl_file.write("\n")

            with open(f"data/premsel_comm_logs/{exp}_{level}", "a") as f:
                f.write("Transition in list made! \n")
            print("Transition in list made!")
            list_file_loc = f"data/premsel_transition_in_lists/{exp + levelstrings[level-1]}_{gpu_index}_{inner_index}"
            # write mapped files to new folder (premsel_transition_files_in)

            with open(f"data/premsel_comm_logs/{exp}_{level}", "a") as f:
                f.write(f"LFL : {list_file_loc} \n")
            print("Starting premise selector")

            exp_rem = copy.deepcopy(exp).replace("_test", "")
            exp_rem = exp_rem.replace("_val", "")
            with open(f"data/premsel_comm_logs/{exp}_{level}", "a") as f:
                f.write(f"exp_rem {exp_rem} \n")
            previous_exp_iter = '_'.join(exp_rem.split('_')[:-1]) + '_' + str(int(exp_rem.split("_")[-1]) - 1)
            with open(f"data/premsel_comm_logs/{exp}_{level}", "a") as f:
                f.write(f"prevexpiter {previous_exp_iter} \n")


            with open(f"data/premsel_comm_logs/{exp}_{level}", "a") as f:
                # f.write(f"python may22_c2i_premise_selector_cl.py --mode test_premise --model {previous_exp_iter} --expname {exp + levelstrings[level-1]} --input_filelist {list_file_loc} --input_folder 0 --epoch 19 --first_iter 0 ")
                f.write(f"python -u may22_c2i_premise_selector_cl.py --mode test_premise --gpu {args.gpu} --model {previous_exp_iter} --expname {exp + levelstrings[level-1]} --input_filelist {list_file_loc} --input_folder 0 --epoch {PREMSEL_TRAIN_EPOCHS -1} --first_iter 0 &> /{prefix}/piepejel/projects/iprover_instantiation/premsel_comm_logs/scriptoutput_{exp}_{level}")
                f.write("\n")


            os.system(f"python -u may22_c2i_premise_selector_cl.py --mode test_premise --gpu {args.gpu} --model {previous_exp_iter} --expname {exp + levelstrings[level-1]} --input_filelist {list_file_loc} --input_folder 0 --epoch {PREMSEL_TRAIN_EPOCHS -1} --first_iter 0 ")


            # shrunk_files = glob.glob(transition_out_folder + "*")
            shrunk_files = [transition_out_folder + k for k in predict_theorems]

            if use_orig_clauses == 0:
                folder = f"data/inst_files/{exp + levelstrings[level-1]}/"
            elif use_orig_clauses == 1:
                print("Keeping original clauses")
                folder = f"data/inst_plus_orig_files/{exp + levelstrings[level-1]}/"

            # map clausetypes back
            for filename in shrunk_files:
                map_file_back(filename, folder)



            print(f"EXECUTING LEVEL {level}")
            os.system(
                f"python -u predicting_loop.py --expname  {exp + levelstrings[level]} --gpu_index {args.gpu} --inner_index {args.inner_index} --seed {seed} --data_parallel_factor {data_parallel_factor} --inner_para_factor {inner_para_factor} --beam {beam} --keep_clauses {use_orig_clauses} --rnn_its {rnn_its} --model {model} --epoch {epoch} --input {exp + levelstrings[level - 1]} --toplevel 0 --set {dataset} --num_samples {ns[level]} --temperature {temp}")

        else:

            print(f"EXECUTING LEVEL {level}")
            os.system(f"python -u predicting_loop.py --expname  {exp + levelstrings[level]} --gpu_index {args.gpu} --inner_index {args.inner_index} --seed {seed} --data_parallel_factor {data_parallel_factor} --inner_para_factor {inner_para_factor} --beam {beam} --keep_clauses {use_orig_clauses} --rnn_its {rnn_its} --model {model} --epoch {epoch} --input {exp + levelstrings[level - 1]} --toplevel 0 --set {dataset} --num_samples {ns[level]} --temperature {temp}")

# last step wasn't copied.

if premise_selector:
    if use_orig_clauses == 0:
        datafolder = f"data/inst_files/{exp + levelstrings[no_levels-1]}/"
    elif use_orig_clauses == 1:
        print("Keeping original clauses")
        datafolder = f"data/inst_plus_orig_files/{exp + levelstrings[no_levels-1]}/"


    all_files = [datafolder + k for k in predict_theorems]
    original_without_premsel_folder = f"data/no_premsel_copies/{exp + levelstrings[no_levels-1]}/"

    if not os.path.exists(original_without_premsel_folder):
        os.makedirs(original_without_premsel_folder)

    for filename in all_files:
        just_fl = filename.split("/")[-1]
        shutil.copy2(filename, original_without_premsel_folder + just_fl)


