losses = []

# # # # # #
import random
import copy
import os
from gnn_module import load_cnf_only, PIEGNN, process_cnf_batch, construct_labels, extract_arities
from inst_config import USE_M2K, EXPLORATION_SAMPLE_SIZE, SIZE_FILTER, PREMSEL
import numpy as np
import string
import fcoplib as cop
import torch
import torch.nn as nn
import re
import argparse
import glob
from pathlib import Path
import time
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Drive the loop")

parser.add_argument("--expname", type=str, help="Defines the output folder paths")
parser.add_argument("--input", type=str, help="Where to find the input files (can be output folder of previous iteration)")
parser.add_argument("--toplevel", type=int, help="Whether to look for the original cnf files (this is the first iteration) or to just glob the input folder (output of previous iteratino)")
parser.add_argument("--set", type=str, help="Only active if toplevel = 1. Can be train, val, test.")

parser.add_argument("--num_samples", type=int, help="How many samples to take from the model per clause")

parser.add_argument("--temperature", type=str, help="Softmax sampling temperature.")

parser.add_argument("--model", type=str, help="Model file location")
parser.add_argument("--epoch", type= str, help="Which checkpoint to take")
parser.add_argument("--keep_clauses", type=int, help="Whether to propagate the original clauses to the next level")
parser.add_argument("--beam", type=int, help="Whether we use the beam search")
parser.add_argument("--rnn_its", type=int, help="How many iterations to run the rnn (bounds the numbers of variables that can be assigned a symbol per clause).")
parser.add_argument("--data_parallel_factor", type=int, help="Need to know how many gpus are running on 1 dataset")
parser.add_argument("--inner_para_factor", type=int, help="Need to know how many gpus are running on 1 dataset")

parser.add_argument("--gpu_index", type=int, help="Need to know which gpu we are")
parser.add_argument("--inner_index", type=int, help="Which worker on the gpu this is")
parser.add_argument("--seed", type=int, help="Random seed")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


utils_folder = f"utils/"

#

# equality_filter = True
# size_filter = 8000
# 8000 is a good size_filter value.
# 80000 used for paper
size_filter = SIZE_FILTER
no_layers = 10
dimensionality = 64
hidden_dim = 128
m2k = USE_M2K


sample_per_exploration_step = EXPLORATION_SAMPLE_SIZE
args = parser.parse_args()

expname = args.expname
input_folder = args.input
top_level = args.toplevel
predict_on = args.set
keep_orig_clauses = int(args.keep_clauses)

temp = float(args.temperature)
num_sam = int(args.num_samples)
sampling = not bool(int(args.beam))
rnn_its = int(args.rnn_its)
gpu_index = int(args.gpu_index)
inner_index =int(args.inner_index)
inner_para_factor = int(args.inner_para_factor)
datparafac = int(args.data_parallel_factor)

if top_level == 0:
    experiment_iteration = int(expname.split("_")[-2])

    if experiment_iteration == 0:
        premise_selector = False
    else:
        premise_selector = PREMSEL

elif top_level == 1:
    experiment_iteration = int(expname.split("_")[-1])

    if experiment_iteration == 0:
        premise_selector = False
    else:
        premise_selector = PREMSEL

def split(a, n):
    n = min(n, len(a))
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

if sampling:
    overshoot = 1
else:
    overshoot = num_sam # controls beam size?

if top_level == 1:


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

    if predict_on == "train":
        predict_theorems = train_theorems

    elif predict_on == "val":
        predict_theorems = val_theorems

    elif predict_on == "test":
        predict_theorems = test_theorems

    if m2k:
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

    print("Number of theorems to sample from: ")
    print(len(predict_theorems))
    print("Num samples")
    print(sample_per_exploration_step)


    predict_theorems_sample = random.sample(predict_theorems, k=sample_per_exploration_step)
    print("Num theorems we will use")
    print(len(predict_theorems_sample))
    assert 2 > 3
    split_list = list(split(predict_theorems_sample, datparafac*inner_para_factor))

    print(f"The start of it looks like this:")
    print(split_list[gpu_index][:10])
    # predict_theorems = predict_theorems[:100]


    # Order is fixed because reading from file.



    # Pick the segment that is assigned to this gpu

    predict_theorems_split = split_list[gpu_index*inner_para_factor + inner_index]

    # predict_theorems = [k for k in predict_theorems_sample if k in predict_theorems_split]

    predict_theorems = predict_theorems_split

    print(f"We actually used these: ")
    print(predict_theorems[:10])

    base_data_folder_cnf = f"./"

    cnf_file_folders = ["sample_data"]

    predict_data_cnf_files = []
    predict_data = []
    for theorem in predict_theorems:
        for input_data_file_folder in cnf_file_folders:
            cnf_file = base_data_folder_cnf + input_data_file_folder + "/" + theorem

            if os.path.isfile(cnf_file):

                try:
                    x, (l1, l2, l3), sig_size = load_cnf_only(cnf_file, return_sig_size=True)
                    print(cnf_file)
                    print(len(x.ini_nodes))
                    if (not (sig_size < overshoot) and len(x.ini_nodes) < size_filter):

                        try:
                            extract_arities(cnf_file, utils_folder)
                            predict_data.append((x, l1, l2, l3))
                            predict_data_cnf_files.append(cnf_file)
                        except ValueError as e:
                            print("There's a symbol with arity > 25")
                            print(e)
                    else:
                        print("Too many nodes")
                except ValueError as e:
                    print("Error")
                    print(e)

    assert len(predict_data) == len(predict_data_cnf_files)


elif top_level == 0:


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

    if predict_on == "train":
        predict_theorems = train_theorems

    elif predict_on == "val":
        predict_theorems = val_theorems

    elif predict_on == "test":
        predict_theorems = test_theorems

    # Taking small subset for testing
    # random.seed(42)
    # predict_theorems = random.sample(predict_theorems, k=2000)

    if m2k:
        with open("lists/M2k_list", "r") as m2k_file:

            restricted_set = m2k_file.readlines()
            restricted_set = [k.strip() for k in restricted_set]

            filtered_predict_theorems = []
            for theorem in predict_theorems:

                base_theorem, proof_id = theorem.split("__")
                if base_theorem in restricted_set:
                    filtered_predict_theorems.append(theorem)

            predict_theorems = filtered_predict_theorems


    # Order is fixed because reading from file.
    random.seed(args.seed)  # seed I used for the grid search;
    if len(predict_theorems) < sample_per_exploration_step:
        sample_per_exploration_step = len(predict_theorems)
    predict_theorems_sample = random.sample(predict_theorems, k=sample_per_exploration_step)
    print(f"Our assigned dataset has {len(predict_theorems)} theorems ")
    split_list = list(split(predict_theorems_sample, datparafac*inner_para_factor))
    print(len(split_list))
    print(len(split_list[0]))


    # Pick the segment that is assigned to this gpu (first version of this, we assume that 1 process = 1 gpu, but this might change later)

    predict_theorems = split_list[gpu_index*inner_para_factor + inner_index]
    print(f"Our assigned chunk has {len(predict_theorems)} theorems")
    print(f"The start of it looks like this:")
    print(predict_theorems[:10])
    if keep_orig_clauses == 0:
        datafolder = f"data/inst_files/{input_folder}/"
    elif keep_orig_clauses == 1:
        print("Keeping original clauses")
        datafolder = f"data/inst_plus_orig_files/{input_folder}/"

    predict_data_cnf_files = []
    predict_data = []
    files = glob.glob(datafolder + "*")

    print(f"We found {len(files)} files from the previous level")
    # this list will contain the files in the input folder, but only the ones assigned to this gpu process.

    filtered_files = []
    for fi in files:

        p_fi = Path(fi)
        print(p_fi.name)
        print(len(predict_theorems))
        if p_fi.name in predict_theorems:

            filtered_files.append(fi)

    print("We actually use these: ")
    print(filtered_files[:10])
    print(f"Of those, {len(filtered_files)} are from our chunk.")

    for file in filtered_files:
        if os.path.isfile(file):

            try:
                x, (l1, l2, l3), sig_size = load_cnf_only(file, return_sig_size=True)
                print(len(x.ini_nodes))
                print(file)
                if (not (sig_size < overshoot) and len(x.ini_nodes) < size_filter):

                    try:
                        extract_arities(file, utils_folder)
                        predict_data.append((x, l1, l2, l3))

                        predict_data_cnf_files.append(file)
                    except ValueError as e:
                        print("There's a symbol with arity > 25")
                        print(e)

                else:
                    print("Too many nodes or too few symbols.")
            except ValueError as e:
                print("Error")
                print(e)



print("Taking the levels into account, we have this state of training examples")
print(f"{len(predict_data)} training examples")


node_nums_tr = [len(k[0].ini_nodes) for k in predict_data]


train_data_zip = [(k, l) for (k, l) in zip(predict_data, predict_data_cnf_files) if len(k[0].ini_nodes) < size_filter]

train_data, train_data_cnf_files = zip(*train_data_zip)
print(f"{len(train_data)} predict examples after size filter")

no_iterations = 1
equality = True
do_nothing = False
# sampling = True
random_bsl = False
single_output = False
num_epochs = 400
batch_size = 1
spread = 1

# model_name = "random_shooting_conditional_inst_no_shuffle"
model_name = args.model
saved_epoch = int(args.epoch)

# server = "nfs"

stats = []
eval_info = []
epoch_losses = []
epoch_accuracies = []
val_epoch_losses = []
val_epoch_accuracies = []
epoch_finegrained_accuracies = []
val_epoch_finegrained_accuracies = []
epoch_finegrained_denoms = []
val_epoch_finegrained_denoms = []


model = PIEGNN(
    (dimensionality, dimensionality, dimensionality),
    (dimensionality, dimensionality, dimensionality),
    device=device,
    hidden_dim=dimensionality,
    layers=no_layers,
    residual=True,
    normalization="layer",
).to(device)

printed_file_list = []

for its in [rnn_its]:
    for beam_width in [num_sam]:
        for opt in [1]:
            for return_sequences in [num_sam]:
                if return_sequences <= beam_width:

                    stime = time.time()

                    # # # # # #
                    folder_list = []

                    # exp_stem = f"{expname}_{its}_bw{beam_width}_opt{opt}_{return_sequences}"
                    # migrate this info to a separate log folder, not in the name.
                    exp_stem = f"{expname}"




                    model.load_state_dict(torch.load(
                        f"models/model_{model_name}_epoch_{saved_epoch}.pt"))

                    top_level_deduplicated_original_clauses = []

                    fully_instantiated_list = []
                    too_many_clauses_list = []
                    top_level_clause_lineage_list = []

                    for iteration in range(no_iterations):

                        print(f"Starting iteration {iteration}")
                        print(f"Problems that had too many clauses: {len(too_many_clauses_list)}")
                        print(f"Problems that had no variables left: {len(fully_instantiated_list)}")
                        premsel_accum = [1.0, 0.0, 0.0]

                        premsel_accum = [1.0, 0.0, 0.0]


                        #
                        def update_accum(accum, current):
                            for i, (acc, cur) in enumerate(zip(accum, current)):
                                accum[i] = np.interp(0.1, [0, 1], [acc, cur])


                        def stats_str(stats):
                            if len(stats) == 2:
                                return "loss {:.4f}, acc {:.4f}".format(*stats)
                            else:
                                return "loss {:.4f}, acc {:.4f} ({:.4f} / {:.4f})".format(
                                    stats[0], (stats[1] + stats[2]) / 2, stats[1], stats[2]
                                )


                        # #
                        losses = []
                        #import matplotlib.pyplot as plt
                        # # # #
                        import random
                        #default batch size = 4
                        # batch_size = 1
                        stats = []
                        eval_info = []
                        epoch_losses = []
                        epoch_accuracies = []
                        epoch_finegrained_accuracies = []
                        epoch_finegrained_denoms = []
                        correct_counter = 0
                        full_denominator = 0
                        correct_counter_vector = [0] * its
                        full_denominator_vector = [0] * its
                        epoch_loss = 0

                        assert return_sequences <= beam_width



                        test_data = train_data # unfortunate naming convention by test I mean "nonteacherforcing"

                        # this is where we generate the new folders
                        gen_name = f"{exp_stem}"
                        # cnfwith_folder = f"data/generated_cnfs_with_orig/" + gen_name
                        # cnfwithout_folder = f"data/generated_cnfs_without_orig/" + gen_name

                        predictions_label_format_folder = f"data/predictions_in_label_format/" + gen_name
                        eground_folder = f"data/inst_files/" + gen_name
                        eground_with_orig_folder = f"data/inst_plus_orig_files/" + gen_name
                        # iprover_output_folder = f"data/iprover_runs/" + gen_name
                        auxiliary_clause_info_folder = f"data/clause_info/" + gen_name
                        # folder_list.append((cnfwith_folder, iprover_output_folder))

                        # if not os.path.exists(cnfwith_folder):
                        #     os.makedirs(cnfwith_folder)

                        if not os.path.exists(eground_folder):
                            os.makedirs(eground_folder)

                        if not os.path.exists(eground_with_orig_folder):
                            os.makedirs(eground_with_orig_folder)

                        if not os.path.exists(predictions_label_format_folder):
                            os.makedirs(predictions_label_format_folder)

                        # if not os.path.exists(cnfwithout_folder):
                        #     os.makedirs(cnfwithout_folder)

                        # if not os.path.exists(iprover_output_folder):
                        #     os.makedirs(iprover_output_folder)

                        if not os.path.exists(auxiliary_clause_info_folder):
                            os.makedirs(auxiliary_clause_info_folder)

                        # top_level_orig_list = []

                        top_level_u_list = []
                        top_level_g_list = []
                        top_level_i_list = []
                        top_level_labelstyle_list = []

                        # top_level_specified_list = []
                        top_level_filename_list = []

                        top_level_all_clausid_lists = []
                        top_level_clause_lineages = []
                        new_top_level_deduplicated_original_clauses = []
                        new_top_level_clause_lineage_list = []


                        print(f"Preparation takes {time.time() - stime} seconds.")
                        for j in tqdm(range(0, len(test_data), batch_size)):
                            stime = time.time()
                            # print(
                            #     "Training {}: {} / {}: Premsel {}".format(
                            #         i, j, len(train_data), stats_str(premsel_accum),
                            #     )
                            # )
                            if (j // batch_size) % 10 == 0:
                                print(
                                    "Progress: {} / {}".format(
                                        j, len(test_data)
                                    )
                                )
                            cnf_file_list = train_data_cnf_files
                            batch = test_data[j: j + batch_size]
                            batch_filenames = cnf_file_list[j: j + batch_size]
                            top_level_filename_list += batch_filenames
                            graph, (index_info) = process_cnf_batch(batch)

                            label_info = construct_labels(index_info, -1, training=False)
                            loss_list = []
                            # print(label_info)
                            # assert 2 > 3
                            # TODO: Need to intervene here; what is the most practical output here

                            print(f"Loading the sample takes: {time.time() - stime} seconds")
                            # mock api : predictions = model.test_beam_output(graph[0], label_info, its)
                            # predictions should be
                            stime = time.time()
                            with torch.no_grad():
                                if sampling:
                                    if not random_bsl:
                                        preds = model.test_time_forward_sample(graph[0], label_info, iterations=its,
                                                                               beam_width=beam_width, optimism=opt, random_baseline=random_bsl, temperature=temp)
                                    else:
                                        combined_preds = []

                                        # preds will be list of lentgh amount of clauses?
                                        # so we just have to concat the beamlists per clause

                                        for i in range(spread):
                                            preds = model.test_time_forward_sample(graph[0], label_info, iterations=its,
                                                                                   beam_width=int(beam_width / spread), optimism=opt,
                                                                                   random_baseline=random_bsl, temperature=temp)
                                            # print(preds)
                                            # print(len(preds)) #
                                            # print(len(preds[0]))
                                            # print(preds[0])
                                            # assert 2 > 3
                                            if combined_preds == []:
                                                combined_preds = preds
                                            else:
                                                assert len(preds) == len(combined_preds)

                                                for e, pr in enumerate(preds):
                                                    combined_preds[e] += pr


                                        preds = combined_preds
                                else:
                                    print(f"USING BEAM SEARCH WITH {num_sam} wide beam!")
                                    preds = model.test_time_forward_beam(graph[0], label_info, iterations=its,
                                                                         beam_width=beam_width, optimism=opt)
                                # preds = model.forward(graph[0], label_info, iterations=its)
                                torch.cuda.empty_cache()
                            #
                            # print(preds)
                            # print([len(preds)])
                            # assert 2 > 3
                            print(f"Using the model takes: {time.time() - stime} seconds")

                            stime = time.time()
                            clause_names = []
                            for dic in label_info[2]:
                                clause_names += list(dic.keys())

                            orig_files = []  # [(clause_id, clausestring)]
                            clause_var_dicts = []  # clause:[variables]
                            for file in batch_filenames:

                                if premise_selector:
                                    # need to also read the file without the clauses premise selected out, because I
                                    # need to have unique clause ids.
                                    # with premsel, I may generate clauses, remove them and then regenerate a clause
                                    # with the same ID, which makes the proof extraction assumptions broken.
                                    ids_taken = []
                                    premsel_archive_file = file.replace("eground_plus_orig_files", "no_premsel_copies")
                                    with open(premsel_archive_file, "r") as premsel_archive:
                                        clauses_without_premsel = premsel_archive.readlines()
                                        archive_clause_ids = []

                                        for clwpr in clauses_without_premsel:
                                            typeandid = clwpr.split(",")[0]
                                            # print(cnfclauseid)
                                            ty, aid = typeandid.split("(")
                                            ids_taken.append(aid)

                                with open(file, "r") as f1:
                                    lines = f1.readlines()
                                    # print(lines)

                                    lines_with_clause_id = []
                                    # We need to find out for each clause how many variables there are
                                    clausevardict = {}
                                    for clause in lines:
                                        if "C" in clause:
                                            cnfclauseid = clause.split(",")[0]
                                            # print(cnfclauseid)
                                            _, clause_id = cnfclauseid.split("(")
                                            # print(clause_id)
                                            reg_res = re.findall(f'C{clause_id}_[A-Z]+', clause)

                                            # TODO check if this does not alter variable order!!!!!
                                            # print(reg_res)
                                            vars = sorted(list(set(reg_res)))
                                            # print(vars)
                                            # assert 2 > 3
                                            clausevardict[f"C{clause_id}"] = vars
                                            lines_with_clause_id.append((clause_id, clause))
                                            # print(clausevardict)
                                        else:  # don't forgot the ground clauses
                                            cnfclauseid = clause.split(",")[0]
                                            _, clause_id = cnfclauseid.split("(")
                                            lines_with_clause_id.append((clause_id, clause))

                                    clause_var_dicts.append(clausevardict)
                                    orig_files.append(lines_with_clause_id)
                            # assert 2 > 3
                            # print(clause_var_dicts)
                            # print(len(clause_var_dicts))
                            # print(sum([len(k) for k in clause_var_dicts]))
                            # assert 2 > 3
                            # print(orig_files)
                            #
                            utils_folder = f"utils/"
                            arity_information = []
                            for file in batch_filenames:
                                arity_information.append(extract_arities(file, utils_folder))

                            # print(arity_information)

                            # split the predictions per problem for easy handling
                            # #
                            no_clauses = [len(k) for k in label_info[2]]
                            # print(no_clauses)
                            # pred_split = [[] for k in range(len(batch_filenames))]
                            # print(len(batch_filenames))
                            # print(pred_split)
                            # index_counter = 0
                            # print(len(preds))
                            #
                            #
                            #
                            # for f, no_clause in enumerate(no_clauses):
                            #     for j in range(no_clause):
                            #         print(i, index_counter + j)
                            #         print(f, len(pred_split))
                            #         pred_split[f].append([preds[k][index_counter + j] for k in range(len(preds))])
                            #     print("INDEX_COUNTER")
                            #
                            # print([len(k) for k in pred_split])
                            # index_counter += no_clause
                            #
                            # #
                            label_style_predictions_list = [] # for coverage calculations
                            instantiated_clauses_list = []
                            general_clauses_list = []
                            unchanged_clauses_list = []
                            alphabet = string.ascii_uppercase
                            clause_lineage_dicts = []
                            # # # #
                            top_clause_with_var_indexes = []
                            counter = 0
                            for cvd in clause_var_dicts:
                                cvd_i = {}
                                for clss in cvd:
                                    cvd_i[clss] = counter

                                    counter += 1

                                top_clause_with_var_indexes.append(cvd_i)

                            # print(top_clause_with_var_indexes)
                            # assert 2 > 3

                            for e, (orig_file, symbol_dict, clausevardict, clauses) in enumerate(
                                    zip(orig_files, label_info[1], clause_var_dicts, label_info[2])):
                                # print(clauses)
                                # print(clausevardict)
                                # for e, orig_clauses # #
                                clause_with_var_index = top_clause_with_var_indexes[e]
                                # print(clause_with_var_index)
                                # assert 2  > 3
                                i_clauses = []
                                g_clauses = []
                                u_clauses = []
                                label_style_predictions_prob = []
                                change_list = []
                                list_of_symbols = list(symbol_dict.keys())
                                sig_size = len(list_of_symbols)
                                # print(f"----{e}")
                                # if e == 2:
                                #     assert 2 > 3
                                # print(list_of_symbols)
                                # Figuring out what number the new clauses should get

                                if not premise_selector:
                                    all_clause_ids = [k[0] for k in orig_file]

                                    numbers_taken = [int(k.split("_")[-1]) for k in all_clause_ids]

                                    max_clause_num_taken = max(numbers_taken)
                                    new_clause_num = max_clause_num_taken + 1

                                else:
                                    numbers_taken = [int(k.split("_")[-1]) for k in ids_taken]
                                    max_clause_num_taken = max(numbers_taken)
                                    new_clause_num = max_clause_num_taken + 1

                                clause_lineage_dict = {}
                                for enum_id, (clause_id, clause) in enumerate(orig_file):
                                    # print(enum_id)
                                    instantiations = []
                                    instantiations_probabilities = []
                                    # assert 2 > 3
                                    # does the clause have variables?
                                    if f"B{e}C{clause_id}" in clauses and not do_nothing:
                                        # how many variables does it have?
                                        # vars #
                                        # print(orig_file)
                                        # print(clause_id)
                                        # print(batch_filenames)
                                        # print(clauses)
                                        # print(label_info[2][e])
                                        vars = clausevardict[f"C{clause_id}"]
                                        # where is it in the output:
                                        clause_index = clauses[f"B{e}C{clause_id}"]
                                        # print(vars)
                                        # print("Signature Size: ", sig_size)
                                        # print(clause_index)
                                        orig_clause = copy.deepcopy(clause)

                                        #
                                        # TODO run this in a for loop for the beam-searched_version

                                        # run_iterations = its
                                        # print(preds[clause_with_var_index[f"C{clause_id}"]])

                                        # TODO This filter only makes sense for beam search;
                                        # if not preds[clause_with_var_index[f"C{clause_id}"]][0][1] == [
                                        #     sig_size]:

                                        for beam in range(return_sequences):
                                            i = 0
                                            clause_terminated = False
                                            length_of_ray = len(
                                                preds[clause_with_var_index[f"C{clause_id}"]][beam][1])
                                            instantiations_probabilities.append(
                                                preds[clause_with_var_index[f"C{clause_id}"]][beam][0])

                                            while (
                                                    not clause_terminated and i < length_of_ray):  # TODO: should this be len(preds) instead of 30
                                                # for i in range(5):
                                                #
                                                # print("ACTION")
                                                # TODO fix what happens at the last step

                                                # print(clause_with_var_index[f"C{clause_id}"])

                                                if single_output:
                                                    if len(instantiations) > 0:
                                                        break
                                                chosen_symbol = \
                                                preds[clause_with_var_index[f"C{clause_id}"]][beam][1][i]
                                                # print(torch.argmax(preds[clause_index][i]).item(), "/ ", sig_size)
                                                # print(i, clause_index, len(preds[0]))
                                                # TODO what it the clauses index in this list?
                                                # chosen_symbol = preds[i]).item()
                                                # assert 2 > 3
                                                # print("SYMS / SIG --- ", chosen_symbol, sig_size)

                                                # print(list_of_symbols)
                                                # assert not chosen_symbol > sig_size
                                                if chosen_symbol < sig_size:  # the termination symbol has index 'sig_size'
                                                    # print(list_of_symbols[chosen_symbol])

                                                    # dont even create an instantiation list

                                                    if not i % len(vars) == 0:
                                                        # while we are in middle of vars
                                                        current_instantiation.append(
                                                            list_of_symbols[chosen_symbol])
                                                    else:
                                                        # we are either at place 0 or have already given all vars a symbol
                                                        if not i == 0:

                                                            instantiations.append(current_instantiation)
                                                            current_instantiation = [
                                                                list_of_symbols[chosen_symbol]]
                                                        else:
                                                            current_instantiation = [
                                                                list_of_symbols[chosen_symbol]]

                                                elif chosen_symbol == sig_size:  # termination symbol was chosen
                                                    if i == 0:
                                                        clause_terminated = True
                                                    else:
                                                        if i % len(vars) == 0:
                                                            instantiations.append(current_instantiation)
                                                            clause_terminated = True
                                                elif chosen_symbol > sig_size:
                                                    raise ValueError(
                                                        "This index is bigger than the amount of function symbols you can choose from")

                                                i = i + 1
                                        # else:
                                        #     pass
                                        # print(instantiations)

                                        # Some deduplications
                                        # print(preds[clause_with_var_index[f"C{clause_id}"]])
                                        # print(instantiations)

                                        instantiations = [tuple(k) for k in instantiations]
                                        # print("INSTANTIATIONS BEFORE DEDUP ---------------------------------------------------------------------------------------------------------------")
                                        # print(instantiations)
                                        # TODO deduplicate in a way that does preserve order (?)
                                        instantiations = list(set(instantiations))
                                        prune = True
                                        if prune:
                                            if not num_sam > len(instantiations):

                                                instantiations = random.sample(instantiations, k=num_sam)
                                        # print(
                                        #     "INSTANTIATIONS AFTER DEDUP ---------------------------------------------------------------------------------------------------------------")
                                        # print(instantiations)
                                        # if len(instantiations) == 1:
                                        #     print(instantiations)
                                        #     assert 2> 3

                                        # now to replace the variables with the relevant symbols and new vars

                                        multiple_clause_instantiations = []
                                        label_style_all_insts = []
                                        for inst in instantiations:
                                            current_inst_label_style = f"{clause_id}" + " : "
                                            current_clause = clause
                                            if len(vars) == len(inst):
                                                for var, injected_symbol in zip(vars, inst):

                                                    l_prefix = len(f"B{e}")
                                                    str_sym = injected_symbol[l_prefix:]

                                                    # print(str_sym, " has arity: ",
                                                    #       arity_information[e][str_sym])
                                                    arity = arity_information[e][str_sym]
                                                    if arity > 0:
                                                        new_term = f"{str_sym}("
                                                        for arg in range(arity - 1):
                                                            new_var = var + alphabet[arg]
                                                            new_term += f"{new_var},"
                                                        last_var = var + alphabet[arity - 1]
                                                        new_term += f"{last_var})"
                                                    else:
                                                        new_term = f"{str_sym}"
                                                    # print(new_term)

                                                    current_clause = current_clause.replace(var, new_term)
                                                    # print(var, var.lstrip(f"C{clause_id}"))
                                                    strip_offset = len(f"C{clause_id}")
                                                    current_inst_label_style = current_inst_label_style + " " + var[strip_offset+1:] + " @ " + str_sym + " %"
                                                label_style_all_insts.append(current_inst_label_style)
                                                # print(label_style_all_insts)
                                                # if clause_id == "i_0_57":
                                                #     assert 2 > 3

                                                # print(label_style_predictions_prob)
                                                # assert 2 > 3
                                                multiple_clause_instantiations.append(current_clause)

                                        # assert 2 > 3
                                        # if len(multiple_clause_instantiations) > 1:
                                        #     # TODO check when this occurs
                                        #     assert 4 > 5
                                        #
                                        assert len(label_style_all_insts) == len(multiple_clause_instantiations)
                                        if len(multiple_clause_instantiations) > 0:
                                            change_list.append(orig_clause)
                                            g_clauses.append(orig_clause)
                                            # print("ORIGINAL CLAUSE")
                                            # print(orig_clause)
                                            # Now we need to figure out what ID to give the new clausees
                                            stem_id = "_".join(clause_id.split("_")[:-1])

                                            for printid, instant in enumerate(multiple_clause_instantiations):
                                                relabeled_clause_id = stem_id + "_" + str(new_clause_num)

                                                new_clause_num += 1

                                                instant = instant.replace(clause_id, relabeled_clause_id)
                                                # print(f"INSTANTIATION {printid}")
                                                # print(f"Probability: {instantiations_probabilities[printid]}")
                                                # print(instant)
                                                i_clauses.append(instant)
                                                change_list.append(instant)
                                                clause_lineage_dict[relabeled_clause_id] = clause_id
                                                label_style_all_insts[printid] += f"[{relabeled_clause_id}]"
                                            if not label_style_all_insts == []:
                                                label_style_predictions_prob.append(label_style_all_insts)
                                        else:
                                            u_clauses.append(clause)

                                    else:
                                        u_clauses.append(clause)
                                #
                                clause_lineage_dicts.append(clause_lineage_dict)
                                # assert 2 > 3
                                # # # # # # # #


                                instantiated_clauses_list.append(i_clauses)
                                general_clauses_list.append(g_clauses)
                                unchanged_clauses_list.append(u_clauses)
                                label_style_predictions_list.append(label_style_predictions_prob)
                                # print(len(instantiated_clauses_list))
                                # print(len(general_clauses_list))
                                # print(len(unchanged_clauses_list))
                                # top_level_clause_lineages += clause_lineage_dicts
                                #

                            # if not label_style_predictions_prob == [[]]:
                            #     assert 2 > 3
                            assert len(top_level_labelstyle_list) == len(top_level_i_list)
                            top_level_u_list += unchanged_clauses_list
                            top_level_g_list += general_clauses_list
                            top_level_i_list += instantiated_clauses_list
                            top_level_clause_lineage_list += clause_lineage_dicts
                            top_level_labelstyle_list += label_style_predictions_list
                            # print("UNCHANGED")
                            # print(unchanged_clauses_list)
                            # print("GENERAL")
                            # print(general_clauses_list)
                            # print("INSTANCES")
                            # print(instantiated_clauses_list)
                            # print(label_style_predictions_list)



                            if iteration == 0:
                                # original clauses are u_clauses + g_clauses (unchanged, plus the general versions of the changed ones)
                                deduplicated_original_clauses = [us + gs for us, gs in
                                                                 zip(unchanged_clauses_list,
                                                                     general_clauses_list)]
                                # print(len(deduplicated_original_clauses))
                                # assert 2 > 3
                                top_level_deduplicated_original_clauses += deduplicated_original_clauses

                            else:
                                list_deduplicated_unmodified_clauses = []
                                for kk in range(len(batch)):  # there might be batches smaller than batch_size!
                                    current_full_index = j + kk
                                    set_current_orig_clauses = set(
                                        general_clauses_list[kk] + unchanged_clauses_list[kk])
                                    print(len(top_level_deduplicated_original_clauses), current_full_index)
                                    set_deduplicated_unmodified_clauses = set(
                                        top_level_deduplicated_original_clauses[current_full_index])
                                    set_deduplicated_unmodified_clauses = set_deduplicated_unmodified_clauses.union(
                                        set_current_orig_clauses)

                                    deduplicated_unmodified_clauses = list(set_deduplicated_unmodified_clauses)

                                    new_top_level_deduplicated_original_clauses.append(
                                        deduplicated_unmodified_clauses)

                                for gg in range(len(clause_lineage_dicts)):
                                    current_full_index = j + gg
                                    old_dict = top_level_clause_lineage_list[
                                        current_full_index]
                                    new_dict = clause_lineage_dicts[gg]
                                    updated_dict = copy.deepcopy(old_dict)

                                    for clause in new_dict:
                                        if new_dict[clause] in old_dict:
                                            updated_dict[clause] = old_dict[new_dict[clause]]
                                        else:
                                            #
                                            updated_dict[clause] = new_dict[clause]

                                    # What about

                                    new_top_level_clause_lineage_list.append(updated_dict)

                            print(f"Interpreting the models output takes {time.time() - stime} seconds.")
                        if iteration > 0:
                            top_level_deduplicated_original_clauses = new_top_level_deduplicated_original_clauses
                            top_level_clause_lineage_list = new_top_level_clause_lineage_list
                        # assert 2 > 3
                        # # # #
                        # print(len(top_level_deduplicated_original_clauses), len(test_data))
                        # print(top_level_deduplicated_original_clauses[0])
                        assert len(top_level_deduplicated_original_clauses) == len(test_data)
                        # print(len(top_level_clause_lineage_list))
                        assert len(top_level_clause_lineage_list) == len(test_data)

                        assert len(top_level_filename_list) == len(top_level_u_list) \
                               and len(top_level_filename_list) == len(top_level_g_list) \
                               and len(top_level_filename_list) == len(top_level_i_list)

                        # print(label_style_predictions_list)
                        # print(len(label_style_predictions_list))
                        assert len(top_level_filename_list) == len(top_level_labelstyle_list)
                        #
                        assert len(top_level_filename_list) == len(top_level_clause_lineage_list)

                        for name, labelstylepreds in tqdm(
                                zip(top_level_filename_list, top_level_labelstyle_list)):
                            just_file = name.split("/")[-1]
                            level = name.split("/")[-2]

                            if predict_on == "mizar_level_0":
                                level = "0"
                            # TODO This does not work for double digit levels
                            if level[:-1].endswith("level"):
                                lv_stem, lv_int = level.split("level")
                                lv_int = str(int(lv_int) + 1)

                                recon_lv = lv_stem + "level" + lv_int

                                level = recon_lv
                            # print(labelstylepreds)
                            # print(f"{cnfwith_folder}/" + just_file)

                            level_included_filepath = predictions_label_format_folder + "/" + level + "/"
                            if not os.path.exists(level_included_filepath):
                                os.makedirs(level_included_filepath)
                            with open(level_included_filepath + just_file, "w") as f0:

                                for labli in labelstylepreds:
                                    for insta in labli:
                                        # print(insta)
                                        f0.write(insta)
                                        f0.write(" \n")




                        for name, orig, spec, clause_lineage in tqdm(
                                zip(top_level_filename_list, top_level_deduplicated_original_clauses,
                                    top_level_i_list, top_level_clause_lineage_list)):
                            just_file = name.split("/")[-1]

                            with open(f"{auxiliary_clause_info_folder}/" + just_file, "w") as f2:
                                for new_clause in clause_lineage:
                                    f2.write(f"{new_clause}\t{clause_lineage[new_clause]}")
                                    f2.write("\n")

                        # eground inputs with original clauses for iprover

                        for e, (name, instantiated, unchanged, general) in tqdm(
                                enumerate(zip(top_level_filename_list, top_level_i_list, top_level_u_list, top_level_g_list))):
                            just_file = name.split("/")[-1]
                            lev = name.split("/")[-2]
                            if predict_on == "mizar_level_0":
                                lev = "0"
                            # level_included_filepath = eground_with_orig_folder + "/" + lev + "/"
                            level_included_filepath = eground_with_orig_folder + "/"
                            if not os.path.exists(level_included_filepath):
                                os.makedirs(level_included_filepath)

                            with open(level_included_filepath + just_file, "w") as fi:
                                for specified_clause in instantiated:
                                    fi.write(specified_clause)
                                for general_clause in general:
                                    fi.write(general_clause)
                                for unchanged_clause in unchanged:
                                    fi.write(unchanged_clause)

                        # eground inputs for j

                        for e, (name, instantiated, unchanged) in tqdm(
                                enumerate(zip(top_level_filename_list, top_level_i_list, top_level_u_list))):
                            just_file = name.split("/")[-1]
                            lev = name.split("/")[-2]
                            if predict_on == "mizar_level_0":
                                lev = "0"
                            # level_included_filepath = eground_folder + "/" + lev + "/"
                            level_included_filepath = eground_folder + "/"
                            if not os.path.exists(level_included_filepath):
                                os.makedirs(level_included_filepath)

                            with open(level_included_filepath + just_file, "w") as fz:
                                for specified_clause in instantiated:
                                    fz.write(specified_clause)
                                for unchanged_clause in unchanged:
                                    fz.write(unchanged_clause)




                        indices_that_need_to_be_removed = []
                        number_of_clauses = []


                        top_level_deduplicated_original_clauses_r = copy.deepcopy(
                            top_level_deduplicated_original_clauses)
                        top_level_deduplicated_original_clauses = [top_level_deduplicated_original_clauses_r[k]
                                                                   for k in
                                                                   range(len(
                                                                       top_level_deduplicated_original_clauses_r))
                                                                   if
                                                                   not k in indices_that_need_to_be_removed]

                        # We don't need to print clause lineages for these next time, so we throw out their dicts
                        top_level_clause_lineage_list_r = copy.deepcopy(top_level_clause_lineage_list)
                        top_level_clause_lineage_list = [top_level_clause_lineage_list[k] for k in
                                                         range(len(top_level_clause_lineage_list)) if
                                                         not k in indices_that_need_to_be_removed]

