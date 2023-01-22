import os
import glob
from pathlib import Path
import re
import copy
import argparse
from inst_config import PREMSEL, LEVELS, KEEP_GENERAL, NOISY_DATA

keep_general_like_inference = KEEP_GENERAL

parser = argparse.ArgumentParser(description="Drive the loop")

parser.add_argument("--expname", type=str, help="Defines the output folder paths")
args = parser.parse_args()
base_folder = f"data/vampire_proofs/"

# LIST OF FILES TO CLEAN UP AT THE END;

cleanup_list = []

premise_sel_list = []

exp = args.expname

experiment_iteration = int(exp.split("_")[-1])
if experiment_iteration == 0:
    premise_selector = False
else:
    premise_selector = PREMSEL

proof_folders = [base_folder + exp]

# MAKE SOME FOLDERS
if not os.path.exists(f"data/constructed_labels/{exp}"):
    os.makedirs(f"data/constructed_labels/{exp}")

if not os.path.exists(f"data/premise_selector_data/{exp}"):
    os.makedirs(f"data/premise_selector_data/{exp}")


# MAKE THE LEVEL STRINGS THAT WE WILL PREPREND AND MAKE A LIST OF THE FOLDERS WE HAVE TO SCAN FOR PROOFS
for k in range(1, LEVELS):

    proof_folders.append(base_folder + exp + "_level" + str(k) )
    if not os.path.exists(f"data/premise_selector_data/{exp + '_level' + str(k) }"):
        os.makedirs(f"data/premise_selector_data/{exp + '_level' + str(k) }")
# print(proof_folders)

proof_folders = [k for k in proof_folders if os.path.exists(k)]

print(proof_folders)

proved_dict = {}
proved_dict_multilevel = {}

# proof_folders are ordered from low to high
for folder in proof_folders:
    all_files = glob.glob(folder + "/*")

    for file in all_files:
        with open(file, "r") as f:
            file_content = f.read()
            print(file)
            if "SZS status Unsatisfiable" in file_content:
                # theorem proved
                print("proved")

                # take the shallowest level where the theorem was proved
                filename = str(Path(file).name)

                # print(filename)
                # if it's not yet in dict, this must be the shallowest proof of it
                if not filename in proved_dict:
                    proved_dict[filename] = file

                    proved_dict_multilevel[filename] = [file]

                    absparentpath = str(Path(file).parents[0])
                    # print(absparentpath)
                    if "level" in absparentpath:
                        integer_level = int(absparentpath.split("level")[-1])
                        print(integer_level)
                        new_path = file
                        for i in range(integer_level, 0, -1):
                            print(i)
                            print("Loop running")

                            if i > 1:
                                new_path = new_path.replace(f"_level{i}", f"_level{i - 1}")
                                print("CASE A")
                            elif i == 1:
                                new_path = new_path.replace(f"_level{i}", "")
                                print("CASE B")
                            print(new_path)
                            # print("replaced")

                            # lowerlevel_path = filename.replace(absparentpath)
                            if new_path in proved_dict_multilevel[filename]:
                                raise ValueError("The assumption that this was the shallowest proof is somehow broken. ")

                            proved_dict_multilevel[filename].append(new_path)

                else:
                    print("seems we already proved it on the previous levels")

# Now we have a list of all the proof files

proof_relevant_instance_dict = {}

for proof in proved_dict_multilevel:

    # We know the first element in the list is the shallowest proved state
    level = proved_dict_multilevel[proof][0]

    proof_relevant_instances_cur = []
    # EXTRACTING THE RELEVANT INSTANCES (THAT NEED TO BE CONSTRUCTED) FROM THE VAMPIRE PROOFS
    with open(level, "r") as f:
        file_content = f.readlines()
        # print(level)
        for line in file_content:
            if "file(" in line:
                # print(line)
                clause_id = line.split(",")[1].split(")")[0]
                proof_relevant_instances_cur.append(clause_id)

    proof_relevant_instance_dict[proof] = proof_relevant_instances_cur

# Now we have the proof relevant instances and we know the amount of levels.

# First trace all the info in clause info.


for proof in proof_relevant_instance_dict:


    proof_graph = {}


    necessary_clauses_multilevel = []
    necessary_labels_multilevel = []
    # ATTENTION: the proofs are in proved_dict

    proof_relevant_instances_copy = copy.deepcopy(proof_relevant_instance_dict[proof])

    print("We need to explain the following clauses [they need to be reachable from the base clauses in the proof graph]")
    for k in proof_relevant_instances_copy:
        print(k)
    print(proved_dict_multilevel[proof])
    # First one is the last level
    for e, level in enumerate(proved_dict_multilevel[proof]):

        exp_level = Path(level).parents[0].name
        necessary_clauses_cur = []
        necessary_labels_cur = []
        clause_info_loc = level.replace("/vampire_proofs/", "/clause_info/")


        # TODO take care of these paths

        # print(len(proved_dict_multilevel[proof]))

        if e == len(proved_dict_multilevel[proof]) - 1:
            pred_format_loc_component = "/predictions_in_label_format/"
            # ground level
            pred_format_loc_component = pred_format_loc_component + exp_level + "/cnf2_var_renamed_140621/"
            pred_format_loc = level.replace("/vampire_proofs/", pred_format_loc_component)
            pred_format_loc_p = Path(pred_format_loc)

            pred_format_loc = str(pred_format_loc_p.parents[0].parents[0].absolute()) + "/" + pred_format_loc_p.name
        elif e == len(proved_dict_multilevel[proof]) - 2:
            # print(level)
            # print(pred_format_loc_component)
            pred_format_loc_component = "/predictions_in_label_format/"
            pred_format_loc_component = pred_format_loc_component + exp_level + "/" + exp + "/"
            # print(pred_format_loc_component)
            pred_format_loc = level.replace("/vampire_proofs/" + exp_level + "/", pred_format_loc_component)
            pred_format_loc_p = Path(pred_format_loc)
            # print(pred_format_loc_p)

            # assert 2  > 3
        else:
            pred_format_loc_component = "/predictions_in_label_format/"
            pred_format_loc_component = pred_format_loc_component + exp_level + "/" + exp_level + "/"
            # print(pred_format_loc_component)
            pred_format_loc = level.replace("/vampire_proofs/" + exp_level + "/", pred_format_loc_component)
            pred_format_loc_p = Path(pred_format_loc)

        with open(clause_info_loc, "r") as clf:
            # print(clause_info_loc)
            lines = clf.readlines()

            lines = [k.strip() for k in lines]
            with open(pred_format_loc, "r") as predform:

                predlines = predform.readlines()
                predlines = [k.strip() for k in predlines]
                predlinedict = {}
                # for k in predlines:
                    # print(k)
                for k in predlines:

                    # IDENTIFIER IS THE ID OF THE CLAUSE THAT WAS GENERATED BY THIS PARTICULAR INSTANTIATION
                    content, identifier = k.split("[")
                    identifier = identifier.strip().rstrip("]")

                    predlinedict[identifier] = content


                # LINES ARE THE CLAUSE INFOS
                # COLLECT THE PARENTS OF CLAUSES THAT WERE NEEDED BY VAMPIRE
                # AND THEN GO TO THE NEXT LEVEL AND DO THE SAME ...
                for k in lines:
                    child, parent = k.split("\t")
                    parent.rstrip()

                    if child in proof_relevant_instances_copy:
                        # print(child, parent)
                        # necessary_clauses_cur.append(child)
                        # print(f"This was needed: {parent} leads to {child}")
                        necessary_clauses_cur.append(parent)
                        if parent in proof_graph:
                            proof_graph[parent] += [child]
                        else:
                            proof_graph[parent] = [child]
                        # This will fail if we find a relevant clause from a different level - which should not happen

                        # print(predlinedict[child])
                        # predlinedict contains a map {NEXTLEVELID -> PREDFORMATINSTANCE}
                        necessary_labels_cur.append((child, predlinedict[child]))

                # WE ADD THE PARENTS TO THE SET OF NECESSARY CLAUSES
                for k in necessary_clauses_cur:
                    if not k in proof_relevant_instances_copy:
                        proof_relevant_instances_copy.append(k)

                necessary_clauses_multilevel.append(necessary_clauses_cur)
                necessary_labels_multilevel.append(necessary_labels_cur)


    # now that I know the relevant proof instances, I need to copy them into cnfs and labels



    # now we open the original CNFs
    with open(f"data/base_data/cnf2_var_renamed_140621/{proof}", "r") as cnffile:
        cnf_lines = cnffile.readlines()
        cnf_lines = [k.strip() for k in cnf_lines]
        orig_clause_ids = []
        for k in cnf_lines:
            orig_clause_ids.append(k.split(",")[0].split("(")[1])
        # print(orig_clause_ids)

    flatted_labels = []
    for k in necessary_labels_multilevel:
        flatted_labels += k
    # print(flatted_labels)
    if len(flatted_labels) == 0:

        continue
    tracked_clauses, tracks = zip(*flatted_labels)

    for parent in proof_graph:

        for resulting_clause in proof_graph[parent]:
            # resulting_clause = inference
            # print(resulting_clause)

            assert resulting_clause in tracked_clauses



    #
    first_level_labels = []

    # The original clauses we don't have to additionally explain.
    clauses_reached = copy.deepcopy(orig_clause_ids)


    # This whole section creates the ideal training data, with every clause being there at the earliest moment
    # that it can be there

    input_levels = [orig_clause_ids]

    input_level = 0

    # print(necessary_clauses_multilevel)
    to_construct = []
    for k in proof_graph:
        to_construct += proof_graph[k]
    # to_construct = [proof_graph[k] for k in proof_graph]
    # print(to_construct)
    print(proof_graph)

    while (to_construct):
        # print(to_construct)
        new_input_level = []
        for parent_of_inference in proof_graph:

            if parent_of_inference in input_levels[input_level]:
                for resulting_clause in proof_graph[parent_of_inference]:

                    # print(parent_of_inference, "--->" , resulting_clause)

                    new_input_level.append(resulting_clause)
                    to_construct.remove(resulting_clause)

        input_levels.append(new_input_level)
        input_level += 1

        if input_level > LEVELS * 3:
            raise ValueError("We cannot construct the labels!")

        assert input_level <= LEVELS # if we get a longer proof than we tried, that'd be weird.
    print(input_levels)

    # put cnf lines and labels into lists and print the new


    # --------------------
    # Generate premsel data

    # If we run with premsel, the number of clauses is not monotone anymore: we might have deleted some.
    # Therefore we have the no_premsel_copies, which consist of the files before premise selection was done.

    if premise_selector:
        last_level_cnf = proved_dict_multilevel[proof][0].replace("vampire_proofs", "no_premsel_copies")

        all_cnfs = [proved_dict_multilevel[proof][k].replace("vampire_proofs", "no_premsel_copies") for k in range(len(proved_dict_multilevel[proof]))]

    else:
        last_level_cnf = proved_dict_multilevel[proof][0].replace("vampire_proofs", "inst_plus_orig_files")

        all_cnfs = [proved_dict_multilevel[proof][k].replace("vampire_proofs", "inst_plus_orig_files") for k in
                    range(len(proved_dict_multilevel[proof]))]
        # a
    print(all_cnfs)
    # assert 2 > 3


    if premise_selector:
        all_clauses_created = {}
        for cnf_file in all_cnfs:
            # Open the deepest file
            with open(cnf_file, "r") as partial_cnf:
                partial_clauses_created = partial_cnf.readlines()
                partial_clauses_created = [k.strip() for k in partial_clauses_created]
            # Make a dictionary where the clauses that are in the last files are set to 0.

            for part in partial_clauses_created:
                if part not in all_clauses_created:
                    all_clauses_created[part] = 0

        all_clauses_created = list(all_clauses_created.keys())
    else:
        all_clauses_created = {}
        for cnf_file in all_cnfs:
            # Open the deepest file
            with open(last_level_cnf, "r") as partial_cnf:
                partial_clauses_created = partial_cnf.readlines()
                partial_clauses_created = [k.strip() for k in partial_clauses_created]
            # Make a dictionary where the clauses that are in the last files are set to 0.

            for part in partial_clauses_created:
                if part not in all_clauses_created:
                    all_clauses_created[part] = 0

        all_clauses_created = list(all_clauses_created.keys())

    ###################### PREMSEL DATA RELATED #################################

    all_proof_clauses = []
    for node in proof_graph:
        all_proof_clauses.append(node)
        all_proof_clauses += proof_graph[node]
    # print(all_proof_clauses)

     if premise_selector:
        for cn in all_cnfs:
            with open(cn, "r") as cnfile:
                cnlines = cnfile.readlines()
                cnlines = [k.strip() for k in cnlines]

                generated_premsel_train_file = cn.replace("no_premsel_copies", "premise_selector_data")

                premise_sel_list.append(generated_premsel_train_file)
                with open(generated_premsel_train_file, "w") as premise_data:
                    premise_data_clauses = []
                    for line in cnlines:
                        line_id = line.split(",")[0].split("(")[1]
                        # print(line)
                        clausetype = line.split(",")[1]
                        if line_id in all_proof_clauses:

                            ax = line.replace(f",{clausetype},", ",axiom_useful,")
                            # print(ax)
                            premise_data_clauses.append(ax)

                        else:
                            ax = line.replace(f",{clausetype},", ",axiom_redundant,")
                            premise_data_clauses.append(ax)

                    # print(premise_data_clauses)
                    for clause in premise_data_clauses:
                        premise_data.write(clause)
                        premise_data.write("\n")


    ######################### END OF PREMSEL #############################################

    # Orig to concat stuff to
    with open(f"data/base_data/cnf2_var_renamed_140621/{proof}", "r") as cnffile:
        cnf_lines = cnffile.readlines()
        cnf_lines = [k.strip() for k in cnf_lines]

    cnf_levels = [cnf_lines] # first level is alway the original cnf.
    label_levels = []


    for inputs in input_levels[1:]: #skip the first level, those will be in the original cnf anyway
        new_label_level = []
        new_cnf_level = []
        for input_clause in inputs:
            for available_clause in all_clauses_created:
                # print(available_clause)
                id = available_clause.split(",")[0].split("(")[1].rstrip().lstrip()

                if input_clause == id:

                    new_clause_nonnormalized = copy.copy(available_clause)
                    new_cnf_level.append(available_clause)

            for inference in flatted_labels:
                if inference[0] == input_clause:
                    new_label_nonnormalized = copy.copy(inference)
                    new_label_level.append(inference[1])



        label_levels.append(new_label_level)

        cnf_levels.append(new_cnf_level)

    # now print them to files.
    if not os.path.exists(f"data/constructed_labels/{exp}/cnfs/"):
        os.makedirs(f"data/constructed_labels/{exp}/cnfs/")

    if not os.path.exists(f"data/constructed_labels/{exp}/labels/"):
        os.makedirs(f"data/constructed_labels/{exp}/labels/")


    if not os.path.exists(f"data/constructed_labels/{exp}/cnfs/level0"):
        os.makedirs(f"data/constructed_labels/{exp}/cnfs/level0")
    with open(f"data/constructed_labels/{exp}/cnfs/level0/{proof}", "w") as cnf_file:

        for line_to_print in cnf_levels[0]:
            cnf_file.write(line_to_print)
            cnf_file.write("\n")

    level = 0

    assert len(label_levels) == len(cnf_levels) - 1
    prev_level_clauses = cnf_levels[0]
    for i in range(1, len(cnf_levels)):

        # cnf_entries = cnf_levels[i]
        label_entries = label_levels[i-1]



        if keep_general_like_inference:
            # If we don't delete them, make sure we don't duplicate them

            notdupclauses = []
            for cl in cnf_levels[i]:
                if cl not in prev_level_clauses:
                    notdupclauses.append(cl)
            this_level_cnf_clauses = prev_level_clauses + notdupclauses
        else:
            this_level_cnf_clauses = prev_level_clauses + cnf_levels[i]
        # print(this_level_cnf_clauses)
        filtered_this_level_cnf_clauses = []
        clauses_to_remove = []

        #label entries contains the pred in label format string
        # keep_general_like_inference = False
        # if not keep_general_like_inference:


        if keep_general_like_inference:

            print("Original clauses kept")

        else:

            for label in label_entries:
                clause_to_remove = label.split(":")[0].rstrip().lstrip()
                clauses_to_remove.append(clause_to_remove)

            # this is a choice; we remove the clauses that were instantiated; this could be slightly different than the actual inference setting!

            for cnf_clause in this_level_cnf_clauses:
                id = cnf_clause.split(",")[0].split("(")[1].rstrip().lstrip()
                if not id in clauses_to_remove:

                    filtered_this_level_cnf_clauses.append(cnf_clause)
                else:
                    print(f"{id} was removed")

        # print(len(this_level_cnf_clauses), len(filtered_this_level_cnf_clauses))
        # print(label_entries)

        if not os.path.exists(f"data/constructed_labels/{exp}/labels/level{i-1}/"):
            os.makedirs(f"data/constructed_labels/{exp}/labels/level{i-1}/")
        with open(f"data/constructed_labels/{exp}/labels/level{i-1}/{proof}",
                  "w") as lab_file:
            print(f"data/constructed_labels/{exp}/labels/level{i-1}/{proof}")
            for line_to_print in label_entries:
                lab_file.write(line_to_print)
                lab_file.write("\n")


        if i < len(cnf_levels) - 1:
            if not os.path.exists(f"data/constructed_labels/{exp}/cnfs/level{i}/"):
                os.makedirs(f"data/constructed_labels/{exp}/cnfs/level{i}/")


            with open(f"/data/constructed_labels/{exp}/cnfs/level{i}/{proof}",
                      "w") as cnf_file:

                for line_to_print in filtered_this_level_cnf_clauses:
                    cnf_file.write(line_to_print)
                    cnf_file.write("\n")

        prev_level_clauses = filtered_this_level_cnf_clauses

    with open(f"data/premise_selector_data_lists/{exp}", "w") as prl:
        for l in premise_sel_list:
            prl.write(l)
            prl.write("\n")

    print("listfile written")
    # For the premise selection, take all the clauses in proof_graph as positive and the rest as negatives


    # Generating noisy training data as well:

    # for proof in proof_relevant_instance_dict:

    if NOISY_DATA:

        if not os.path.exists(f"data/noisy_data/{exp}"):
            os.makedirs(f"data/noisy_data/{exp}")

        if not os.path.exists(f"data/noisy_data_labels/{exp}"):
            os.makedirs(f"data/noisy_data_labels/{exp}")

        for levelint in range(LEVELS + 2):
            if not os.path.exists(f"data/noisy_data/{exp}/level{levelint}/"):
                os.makedirs(f"data/noisy_data/{exp}/level{levelint}")

            if not os.path.exists(f"data/noisy_data_labels/{exp}/level{levelint}/"):
                os.makedirs(f"data/noisy_data_labels/{exp}/level{levelint}")


        all_cnfs = list(reversed([proved_dict_multilevel[proof][k].replace("vampire_proofs", "inst_plus_orig_files") for k in
                    range(len(proved_dict_multilevel[proof]))]))

        # Take each of these cnfs and get
        all_labels_needed = {}
        resulting_ids = {}

        for k in flatted_labels:
            id = k[1].split(":")[0].strip() # clause id to connect to

            all_labels_needed[id] = k[1]
            resulting_ids[id] = k[0]

        print("ALL LABELS NEEDED")
        print(all_labels_needed)
        associated_label_list = []
        # remember that the first cnf here is NOT the starting data, but already the first output;
        for e, noisy_cnf in enumerate(all_cnfs):

            associated_data = []

            new_noisy_cnf_path = noisy_cnf.replace("inst_plus_orig_files", "noisy_data")
            new_noisy_label_path = noisy_cnf.replace("inst_plus_orig_files", "noisy_data_labels")
            print(new_noisy_cnf_path)

            with open(noisy_cnf, "r") as noise_cnf_file:

                noisy_cnf = noise_cnf_file.readlines()
                noisy_cnf = [k.strip() for k in noisy_cnf]
                print(noisy_cnf[:2])
            noisy_clause_ids = []
            for k in noisy_cnf:
                noisy_clause_ids.append(k.split(",")[0].split("(")[1])

            for noise_clause in noisy_cnf:
                # print(available_clause)
                id = noise_clause.split(",")[0].split("(")[1].rstrip().lstrip()
                print(id)
                # The clause created by the instantiation is neccessary and not there yet.
                if id in all_labels_needed and not resulting_ids[id] in noisy_clause_ids:
                    print("FOUND IT")
                    associated_data.append(all_labels_needed[id])
                elif id in all_labels_needed and resulting_ids[id] in noisy_clause_ids:
                    print("WE ALREADY HAVE THIS ONE")

            thm_name = all_cnfs[0].split("/")[-1]

            if len(associated_data) > 0:

                with open(f"data/noisy_data/{exp}/level{e}/{thm_name}", "w") as noisy_cnf_copy:

                    for line in noisy_cnf:

                        noisy_cnf_copy.write(line)
                        noisy_cnf_copy.write("\n")

                with open(f"data/noisy_data_labels/{exp}/level{e}/{thm_name}",
                          "w") as noisy_label_file:

                    for line in associated_data:
                        noisy_label_file.write(line)
                        noisy_label_file.write("\n")


            associated_label_list.append(associated_data)





