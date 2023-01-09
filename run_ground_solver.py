import os
import glob
import argparse
import multiprocessing as mp
from pathlib import Path
from inst_config import LEVELS, PREMSEL

parser = argparse.ArgumentParser(description="Drive the loop")

parser.add_argument("--expname", type=str, help="Defines the output folder paths")

args = parser.parse_args()
exp = args.expname

levelstrings = ['']
levelstrings = levelstrings + [f'_level{k}' for k in range(1, LEVELS*2)]


reporting_strings = []

# Top level (0)
no_levels = LEVELS

base_prover_input_folder = f"data/inst_plus_orig/{exp}"

parallel_pool = mp.Pool(40)

# run vampire without the clauses that have variables
for intlev, lev in enumerate(levelstrings[:no_levels]):
    if not os.path.exists(f"data/vampire_proofs/{exp + lev}"):

        os.makedirs(f"data/vampire_proofs/{exp + lev}")
    files = glob.glob(f"{base_prover_input_folder}/{exp + lev}/*")
    print(files)
    counter = 0
    reporting_strings.append(f"Vampire was called on {len(files)} files")
    comm_strings = []
    for filen in files:
        counter += 1
        print(counter)
        # os.system(f"cat {filen} | grep -v C | /{prefix}/piepejel/projects/iprover_instantiation/vampire_z3_rel_static_master_4909 -t 30 -acc on --proof tptp --output_axiom_names on -stat full > /{prefix}/piepejel/projects/iprover_instantiation/vampire_proofs/{exp + lev}/{filen.split('/')[-1]}")
        comm_strings.append(f"cat {filen} | grep -v C | ./vampire_z3_rel_static_master_4909 -t 30 -acc on --proof tptp --output_axiom_names on -stat full > data/vampire_proofs/{exp + lev}/{filen.split('/')[-1]}")

    parallel_pool.map(os.system, comm_strings)


# count proofs
reporting_strings.append(f"Files printed: {counter}")

# if it's not proved, delete the files to save space;


# I want some reporting on which things were proved
proved_dict = {}
for lev in levelstrings[:no_levels]:
    output_list = glob.glob(f"data/vampire_proofs/{exp + lev}/*")
    print(f"data/vampire_proofs/{exp + lev}/")
    print(len(output_list))
    count = 0
    pos_count = 0
    print(count)
    for k in output_list:
        count += 1
        with open(k, "r")  as f1:

            contents = f1.read()
            if "SZS status Unsatisfiable" in contents:
                print(k)
                pos_count += 1
                path_k = Path(k)
                if path_k.name in proved_dict:
                    proved_dict[path_k.name] += 1
                else:
                    proved_dict[path_k.name] = 1
            else:
                path_k = Path(k)
                if path_k.name in proved_dict:
                    proved_dict[path_k.name] += 0
                else:
                    proved_dict[path_k.name] = 0
            # else:
                # failed attempt, delete the files
                # path_k = Path(k)





    print(f"Lev: {lev} -- {float(pos_count) / float(count)} -- solved {pos_count} of {count} files found")

    reporting_strings.append(f"Lev: {lev} -- {float(pos_count) / float(count)} -- solved {pos_count} of {count} files found")


delete_not_solved = True
for theorem in proved_dict:
    if proved_dict[theorem] == 0:
        # No proof to be found.
        if delete_not_solved:
            for lev in range(no_levels):
                eg = f"data/inst_files/{exp + levelstrings[lev]}/{theorem}"
                egp = f"data/inst_plus_orig_files/{exp + levelstrings[lev]}/{theorem}"


                if PREMSEL:
                    egcopy = f"data/no_premsel_copies/{exp + levelstrings[lev]}/{theorem}"
                eginfo = f"data/clause_info/{exp + levelstrings[lev]}/{theorem}"

                os.system(f"rm {eg}")
                os.system(f"rm {egp}")

                os.system(f"rm {eginfo}")

num_proved = 0

for theorem in proved_dict:

    if proved_dict[theorem] != 0:
        num_proved += 1

with open(f"data/proof_statistics/{exp}.log", "w") as f:

    for k in reporting_strings:
        f.write(k)
        f.write("\n")
    f.write(f"Union of things proved: {num_proved}")

