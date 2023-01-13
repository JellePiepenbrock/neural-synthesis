# Construct a replay buffer from a proof directory
#
# Need special case for the original data;

# Proof length is determined by the amount of label files for a theorem.
import random

import os

expname = "demo5"
pretrain_model = "demo_model" # last model from the run on the full dataset
modelepoch = 0
iteration = 0

folders = ["data", "lists", "models"]
for f in folders:
    if not os.path.exists(f):
        os.makedirs(f)


data_path = f"data/"
if not os.path.exists(data_path):
    os.makedirs(data_path)

needed_folders = ["job_scripts", "inst_plus_orig_files", "inst_files", "vampire_proofs", "no_premsel_copies",
                  "clause_info", "proof_statistics", "premsel_transition_in", "premsel_transition_out",
                  "no_premsel_copies", "premsel_comm_logs", "premsel_transition_in_lists", "predictions_in_label_format"]

for nf in needed_folders:
    tentative_path = data_path + "/" + nf
    if not os.path.exists(tentative_path):
        os.makedirs(tentative_path)


os.system(
            f"python -u generate_jobs.py --set test --model {pretrain_model} --epoch {modelepoch} --expname {expname + '_test_' + str(iteration)}")
os.system(f"bash {data_path}/job_scripts/main.sh")

os.system(f"python -u run_ground_solver.py --expname {expname + '_test_' + str(iteration)}")
