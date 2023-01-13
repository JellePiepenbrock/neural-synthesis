from itertools import product
import argparse
from inst_config import CONDASH_LOCATION, LEVELS, NO_GPUS, TEMP, INNER_PARA
#
server = "dgxscratch"

if server == "dgx":
    prefix = "home"
elif server == "air3":
    prefix = "nfs"
elif server == "dgxscratch":
    prefix = "raid/scratch"

parser = argparse.ArgumentParser()
parser.add_argument('--set', type=str, help='Which set to sample from')
parser.add_argument('--model', type=str, help='Model filename stem' )
parser.add_argument('--epoch', type=int, help='Which epoch of the model to use')
parser.add_argument('--expname', type=str, help="Base Name of the current run.")

args = parser.parse_args()

set = args.set
model = args.model
epoch = args.epoch
expname = args.expname

experiment_iteration = int(expname.split("_")[-1])

sets = [set]


if LEVELS == 2:
    num_samples = [[25, 5]]
elif LEVELS == 3:
    num_samples = [[25, 5, 5]]
elif LEVELS == 4:
    if experiment_iteration == 0:
        num_samples = [[25, 5, 5, 5]]
    else:
        num_samples = [[25, 25, 25, 25]]
elif LEVELS == 5:

    num_samples = [[25, 5, 5, 5, 5]]
    # num_samples = [[25, 10, 10, 10, 10]]
elif LEVELS == 10:
    num_samples = [[25, 5, 5, 5, 5, 5, 5, 5, 5, 5]]
elif LEVELS > 10:
    num_samples = [[5 for k in range(LEVELS)]]


temperature = [TEMP]

model = [model]

epochs = [epoch]

keep_clauses = [1]

rnn_its = [12]


num_gpus = NO_GPUS
data_parallel_factors = [NO_GPUS]

dimensions = [sets, num_samples, temperature, model, epochs, keep_clauses, rnn_its, data_parallel_factors]

print(list(product(*dimensions)))

expnames = []
params = []

for k in list(product(*dimensions)):
    params.append(k)
    # expnames.append(f"{expname}_{k[0]}_ns_{'f'.join([str(c) for c in k[1]])}_temp{k[2]}_m_{k[3]}_ep{k[4]}_kc{k[5]}_rnnits{k[6]}_datapara{k[7]}")
    expnames.append(f"{expname}")
print(expnames)

inner_para_factor = INNER_PARA

import random
seed = random.randint(0, 100000)
command_list = []
gpus = [[] for k in range(num_gpus*inner_para_factor)]
for counter, (p, exp) in enumerate(zip(params, expnames)):

    for gpu_chosen in range(p[7]):
        for inner_index in range(inner_para_factor):
            comstring = f"\"python -u controller.py --expname {exp} --seed {seed} --set {p[0]} --num_samples {' '.join([str(c) for c in p[1]])} --temperature {p[2]} --model {p[3]} --epoch {p[4]} --keep_clauses {p[5]} --rnn_its {p[6]} --data_parallel_factor {p[7]} --beam 0 --gpu {gpu_chosen} --inner_index {inner_index} --inner_para_factor {inner_para_factor}\""
            command_list.append(comstring)
            # print(gpus)
            # print(counter % num_gpus)
            gpus[gpu_chosen * inner_para_factor + inner_index].append(comstring)
            # print(comstring)



for e, file_contents in enumerate(gpus):
    file_lines = []
    file_lines.append("import os")

    for line in file_contents:
        file_lines.append(f"os.system({line})")

    with open(f"data/job_scripts/gpu_{e}.py", "w") as f:

        for l in file_lines:
            f.write(l)
            f.write("\n")

# generate masterscript


with open(f"data/job_scripts/main.sh", "w") as f:
    f.write("#!/usr/bin/env bash")
    f.write("\n")
    f.write(f"source {CONDASH_LOCATION}")
    f.write("\n")
    f.write("conda activate ./cenv")
    f.write("\n")
    f.write("export LD_LIBRARY_PATH=`pwd`")
    f.write("\n")
    f.write("ulimit -s unlimited")
    for k in range(num_gpus*inner_para_factor):
        f.write("\n")
        f.write(f"nohup python -u data/jobs_scripts/gpu_{k}.py > data/jobs_scripts/gpu{k}_log &")

        f.write("\n")

    f.write("wait")


