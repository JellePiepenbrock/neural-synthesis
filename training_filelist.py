import os

os.environ["CUDA_VISIBLE_DEVICES"] = f"0"
import random
import numpy as np
import fcoplib as cop
import torch
import torch.nn as nn
from torch_scatter import segment_csr, scatter_softmax
from graph_data import GraphData
import re
import subprocess
import string
import copy
import glob
import re
import os
import time
from gnn_module import PIEGNN, load_cnf_labels, construct_labels, process_cnf_batch
from inst_config import USE_M2K
import argparse
import glob
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--listfolder', type=str, help='Folder where the list of files to train on')
parser.add_argument('--expname', type=str, help='Name of the experiment; defines the name of the parameter file thats saved')
parser.add_argument('--model', type=str, help='stem of the model name for the model we want to load.')
parser.add_argument('--epoch', type=str, help='epoch checkpoint to take')

args = parser.parse_args()

aa = 0
bb = 0

shuffle_inst = True
# no_layers = 10
no_layers = 10
dimensionality = 64
hidden_dim = 128
num_epochs = 1
batch_size = 16 # perhaps batch size should be smaller?

m2k = USE_M2K
expname = args.expname
listfolder = args.listfolder
model_to_load = args.model
saved_epoch = args.epoch

its = 12


model = PIEGNN(
    (dimensionality, dimensionality, dimensionality),
    (dimensionality, dimensionality, dimensionality),
    device=device,
    hidden_dim=dimensionality,
    layers=no_layers,
    residual=True,
    normalization="layer",
).to(device)

print(f"models/model_{model_to_load}_epoch_{saved_epoch}.pt")
model.load_state_dict(torch.load(
    f"models/model_parameters/model_{model_to_load}_epoch_{saved_epoch}.pt"))

loss = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001) # works with lr=0.00001


with open(f"lists/00all_probs_train_without_devel", "r") as train_split:


    train_theorems = train_split.readlines()
    train_theorems = [k.strip() for k in train_theorems]

if m2k:
    with open("lists/file_lists/M2k_list", "r") as m2k_file:

        restricted_set = m2k_file.readlines()
        restricted_set = [k.strip() for k in restricted_set]

        filtered_predict_theorems = []
        for theorem in train_theorems:

            base_theorem, proof_id = theorem.split("__")
            if base_theorem in restricted_set:
                filtered_predict_theorems.append(theorem)

        train_theorems = filtered_predict_theorems



train_theorems = train_theorems



train_data_cnf_files = []
train_data_label_files = []
train_data = []

with open(f"{listfolder}/cnflist", "r") as c:
    c_lines = c.readlines()
    c_lines = [k.strip() for k in c_lines]

with open(f"{listfolder}/labellist", "r") as l:
    l_lines = l.readlines()
    l_lines = [k.strip() for k in l_lines]

assert len(l_lines) == len(c_lines)

for cnf_file, label_file in zip(c_lines, l_lines):
    if os.path.isfile(cnf_file) and os.path.isfile(label_file):


        if not os.stat(label_file).st_size == 0:
            try:
                training_example = load_cnf_labels(cnf_file, label_file)
                train_data_cnf_files.append(cnf_file)
                train_data_label_files.append(label_file)
                train_data.append(training_example)
            except ArithmeticError as e:
                print("ATTENTION")
                print(e)
                print(cnf_file)
                print(label_file)
                print("---")


print("Taking the levels into account, we have this state of training examples")
print(f"{len(train_data)} training examples")


assert len(train_data) == len(train_data_cnf_files)
assert len(train_data) == len(train_data_label_files)

node_nums_tr = [len(k[0][0].ini_nodes) for k in train_data]


train_data_zip = [(k,l) for (k, l) in zip(train_data, train_data_label_files) if len(k[0][0].ini_nodes) < 600]
train_data, train_data_label_files = zip(*train_data_zip)


premsel_accum = [1.0, 0.0, 0.0]

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


# # #
losses = []

import random

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
# #  #

for i in range(num_epochs):
    correct_counter = 0
    val_correct_counter = 0
    full_denominator = 0
    val_full_denominator = 0
    correct_counter_vector = [0]*its
    val_correct_counter_vector = [0]*its
    full_denominator_vector = [0]*its
    val_full_denominator_vector = [0] * its
    epoch_loss = 0
    val_epoch_loss = 0
    c = list(zip(train_data, train_data_cnf_files, train_data_label_files))
    random.shuffle(c)

    separate_accs_epoch = []
    val_separate_accs_epoch = []

    true_negatives_epoch = 0
    negatives_epoch = 0
    true_positives_epoch = 0
    positives_epoch = 0

    val_true_negatives_epoch = 0
    val_negatives_epoch = 0
    val_true_positives_epoch = 0
    val_positives_epoch = 0


    train_data, train_data_cnf_files, train_data_label_files = zip(*c)
    for j in range(0, len(train_data), batch_size):

        optimizer.zero_grad()
        if (j // batch_size) % 10 == 0:
            print(
                "Training {}: {} / {}: Premsel {}".format(
                    i, j, len(train_data), stats_str(premsel_accum),
                )
            )
        #
        batch, label_list = zip(*train_data[j: j + batch_size])
        print(train_data_label_files[j: j + batch_size])


        graph, (index_info) = process_cnf_batch(batch)

        label_info = construct_labels(index_info, label_list, shuffle_multi=shuffle_inst, shuffle_vars=False, sample_one_multi=False)
        loss_list = []
        #
        preds = model.forward(graph[0], label_info, iterations=its)
        res, correct, denominator, correct_vector, denom_vector, confusion_matrix_info, sample_accuracies = model.calculate_loss(loss, preds, label_info, iterations=its, sample_normalize_loss=True)
        true_positives, all_positives, true_negatives, all_negatives = confusion_matrix_info

        fraction_sample_accuracy = [np.mean(k) for k in sample_accuracies]

        separate_accs_epoch += fraction_sample_accuracy

        true_positives_epoch += true_positives
        positives_epoch += all_positives

        true_negatives_epoch += true_negatives
        negatives_epoch += all_negatives

        accuracy = correct / batch_size


        res.backward()

        optimizer.step()

        epoch_loss += (res.item())
        correct_counter += correct
        full_denominator += denominator
        losses.append(res.item())
        # del res, preds
        correct_counter_vector = [correct_counter_vector[k] + correct_vector[k] for k in range(len(correct_vector))]
        full_denominator_vector = [full_denominator_vector[k] + denom_vector[k] for k in range(len(denom_vector))]
        if j % 20 == 0:
            torch.cuda.empty_cache()


    epoch_losses.append(epoch_loss)
    epoch_acc = correct_counter / float(full_denominator)
    epoch_accuracies.append(epoch_acc)
    list_con = []
    for k in range(len(correct_counter_vector)):
        if full_denominator_vector[k] > 0:
            list_con.append(correct_counter_vector[k] / float(full_denominator_vector[k]))
        else:
            list_con.append(-1)

    # epch
    epoch_finegrained_accuracies.append(list_con)
    epoch_finegrained_denoms.append(full_denominator_vector)
    assert full_denominator == (positives_epoch + negatives_epoch)
    with open(f"data/training_logs/{expname}_log.txt", "a") as logfile:
        logfile.write(f"Training Epoch {i}\n")
        logfile.write(f"Training Loss: {epoch_loss} \n")
        logfile.write(f"Overall Epoch Accuracy {epoch_acc} \n")
        logfile.write(f"Finegrained Epoch Accuracy: {list_con} \n")
        logfile.write(f"Median Fraction of Right Decisions: {np.median(separate_accs_epoch)} \n")
        logfile.write(f"TPR: {true_positives_epoch / float(positives_epoch)} || TNR: {true_negatives_epoch / float(negatives_epoch)} \n")
        logfile.write(f"Total Positives: {positives_epoch} || Total Negatives: {negatives_epoch} \n")


    print(epoch_accuracies)
    print("Epoch Loss")
    print(epoch_loss)

    premsel_stats = []
    premsel_nums = []
    torch.cuda.empty_cache()


    torch.save(
        model.state_dict(),
        f"models/model_{expname}_epoch_{i}.pt"
    )
    torch.cuda.empty_cache()

print(epoch_losses)
print("Accuracies")
print(epoch_accuracies)


