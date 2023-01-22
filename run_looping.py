
import random
from gnn_module import PIEGNN, load_cnf_labels, construct_labels, process_cnf_batch
from replay_buffer import ReplayBuffer, scan_folder_for_proofs, scan_noisy_data_folder
from inst_config import PREMSEL, USE_M2K, BIAS_TRAINING_SAMPLES, PRETRAIN_NUM_EPOCHS, EVAL_INTERVAL, PREMSEL_TRAIN_EPOCHS, NOISY_DATA
import os

start_iter = 0

expname = "looping"

forbidden_words = ['level', 'val', 'test', 'noisy_data'] # these words cannot be used in the expname because they serve special purposes
for fw in forbidden_words:
    if fw in expname:
        raise ValueError(f"Used the forbidden word {fw}; this word is reserved for internal operations in the pipeline")


report_file = f"data/main_loop_reporting/{expname}"

# modes = ['clean', 'resume', 'from_buffer', 'resume_under_alias']
mode = "clean"

if mode == "clean":
    orig_data_dict = scan_folder_for_proofs("original_data")
    buffer = ReplayBuffer()
    buffer.add_samples(orig_data_dict)
    with open(report_file, "a") as buffer_size_report:
        buffer_size_report.write(str(len(buffer.samples)))
        buffer_size_report.write("\n")
        buffer_size_report.write("----------------\n")

    if USE_M2K:
        pretrain_model = "demo"
        modelepoch = 0

        training_samples_per_iter = 1000
    else:

        pretrain_model = "demo"
        modelepoch = 0

        training_samples_per_iter = 10000


# expname might need to contain iteration number
for iteration in range(start_iter, 1000):

    if iteration == 0:
        # use may22_c2i_gnn_refactor_dataparallel_pred

        os.system(f"python -u generate_jobs.py --set train --model {pretrain_model} --epoch {modelepoch} --expname {expname+ '_' +str(iteration)}")
        os.system(f"bash data/job_scripts/main.sh")
        # assert 2 > 3
        # We have some predictions
        # We need to run vampire
        os.system(f"python -u run_ground_solver.py --expname {expname+ '_' +str(iteration)}")

        # Now extract the proofs
        with open(f"data/proof_ex_comm/{expname + '_' + str(iteration)}", "w") as comexfile:
            comexfile.write(f"python -u proof_extraction.py --expname {expname+ '_' +str(iteration)} &> data/proof_extraction_logs/{expname + '_' + str(iteration)}\n")

        os.system(
            f"python -u proof_extraction.py --expname {expname + '_' + str(iteration)}")

        # assert 2 > 3
        new_samples_dict = scan_folder_for_proofs(expname + '_' + str(iteration))
        print(len(buffer.samples))
        buffer.add_samples(new_samples_dict)
        print(len(buffer.samples))
        with open(report_file, "a") as buffer_size_report:
            buffer_size_report.write(str(len(buffer.samples)))
            buffer_size_report.write("\n")

        if not BIAS_TRAINING_SAMPLES:
            proofs_for_now = buffer.produce_sample(training_samples_per_iter)
        else:
            proofs_for_now = buffer.produce_sample_biased(training_samples_per_iter)

        listpath = f"data/replay_lists/{expname}_{iteration}/"
        if not os.path.exists(listpath):
            os.makedirs(listpath)
        with open(f"{listpath}cnflist", "w") as cnflist:
            with open(f"{listpath}labellist", "w") as labellist:

                for proof_to_learn in proofs_for_now:
                    for step in proof_to_learn:
                        print(step)
                        cnflist.write(step[0])
                        cnflist.write("\n")
                        labellist.write(step[1])
                        labellist.write("\n")

        if NOISY_DATA:
            noisy_cnf_list, noisy_label_list = scan_noisy_data_folder(expname + '_' + str(iteration))
            print(noisy_cnf_list)
            print(len(noisy_cnf_list), len(noisy_label_list))
            # assert 2 > 3
            with open(f"{listpath}cnflist", "a") as cnflist:
                with open(f"{listpath}labellist", "a") as labellist:

                    for cnf, label in zip(noisy_cnf_list, noisy_label_list):

                        cnflist.write(cnf)
                        cnflist.write("\n")
                        labellist.write(label)
                        labellist.write("\n")


        # So we have an idea of the performance before the loop.
        with open(f"data/main_controller_logs/{expname}", "a") as mclog:
            mclog.write(f"python -u generate_jobs.py --set val --model {pretrain_model} --epoch {modelepoch} --expname {expname + '_val_' + str(iteration)}")
            mclog.write("\n")
        os.system(
            f"python -u generate_jobs.py --set val --model {pretrain_model} --epoch {modelepoch} --expname {expname + '_val_' + str(iteration)}")
        os.system(f"bash data/job_scripts/main.sh")

        os.system(f"python -u run_ground_solver.py --expname {expname + '_val_' + str(iteration)}")

        os.system(
            f"python -u generate_jobs --set test --model {pretrain_model} --epoch {modelepoch} --expname {expname + '_test_' + str(iteration)}")
        os.system(f"bash data/job_scripts/main.sh")

        os.system(f"python -u run_ground_solver.py --expname {expname + '_test_' + str(iteration)}")

        # Train on the old data plus the new proofs we got.
        os.system(f"python training_filelist.py --listfolder {listpath} --expname {expname + '_' + str(iteration)} --model {pretrain_model} --epoch {modelepoch}")

        # Train premsel here;
        if PREMSEL:
            premsel_generated_list = f"data/premise_selector_data_lists/{expname + '_' + str(iteration)}"
            print(f"python premise_selector.py --mode train_full_data_balancing --model 0 --expname {expname + '_' + str(iteration)} --input_filelist {premsel_generated_list} --input_folder 0 --epoch 0 --first_iter 1")

            os.system(f"python premise_selector.py --mode train_full_data_balancing --model 0 --expname {expname + '_' + str(iteration)} --input_filelist {premsel_generated_list} --input_folder 0 --epoch 0 --first_iter 1")

        # assert 2 > 3
    # A model will be saved in model_parameters/model_expname_epoch_0 (or more epochs)
    # I can loop the buffer training if I want.
    elif iteration > 0:
        os.system(
            f"python -u generate_jobs.py --set train --model {expname+ '_' + str(iteration-1)} --epoch 0 --expname {expname + '_' + str(iteration)}")


        os.system(f"bash data/job_scripts/main.sh")



        os.system(f"python -u run_ground_solver.py --expname {expname + '_' + str(iteration)} > data/vampire_run_logs/{expname + '_' + str(iteration)}")

        with open(f"data/proof_ex_comm/{expname + '_' + str(iteration)}", "w") as comexfile:
            comexfile.write(f"python -u proof_extraction.py --expname {expname+ '_' +str(iteration)} > data/proof_extraction_logs/{expname + '_' + str(iteration)}\n")
        os.system(f"python -u proof_extraction.py --expname {expname + '_' + str(iteration)} > data/proof_extraction_logs/{expname + '_' + str(iteration)}")

        new_samples_dict = scan_folder_for_proofs(expname + '_' + str(iteration))
        print(len(buffer.samples))
        buffer.add_samples(new_samples_dict)
        print(len(buffer.samples))
        with open(report_file, "a") as buffer_size_report:
            buffer_size_report.write(str(len(buffer.samples)))
            buffer_size_report.write("\n")

        if not BIAS_TRAINING_SAMPLES:
            proofs_for_now = buffer.produce_sample(training_samples_per_iter)
        else:
            proofs_for_now = buffer.produce_sample_biased(training_samples_per_iter)

        listpath = f"data/replay_lists/{expname}_{iteration}/"
        if not os.path.exists(listpath):
            os.makedirs(listpath)
        with open(f"{listpath}cnflist", "w") as cnflist:
            with open(f"{listpath}labellist", "w") as labellist:

                for proof_to_learn in proofs_for_now:
                    for step in proof_to_learn:
                        print(step)
                        cnflist.write(step[0])
                        cnflist.write("\n")
                        labellist.write(step[1])
                        labellist.write("\n")
        if NOISY_DATA:
            noisy_cnf_list, noisy_label_list = scan_noisy_data_folder(expname + '_' + str(iteration))
            with open(f"{listpath}cnflist", "a") as cnflist:
                with open(f"{listpath}labellist", "a") as labellist:

                    for cnf, label in zip(noisy_cnf_list, noisy_label_list):

                        cnflist.write(cnf)
                        cnflist.write("\n")
                        labellist.write(label)
                        labellist.write("\n")

        os.system(
            f"python training_filelist.py --listfolder {listpath} --expname {expname + '_' + str(iteration)} --model {expname + '_' + str(iteration-1)} --epoch 0 > data/training_output/{expname + '_' + str(iteration)}")

        if PREMSEL:
            premsel_generated_list = f"data/premise_selector_data_lists/{expname + '_' + str(iteration)}"


            os.system(f"python premise_selector.py --mode train_full_data_balancing --input_filelist {premsel_generated_list} --input_folder 0  --expname {expname + '_' + str(iteration)} --model {expname + '_' + str(iteration-1)} --epoch {PREMSEL_TRAIN_EPOCHS - 1} --first_iter 0 > data/premsel_train_logs/{expname + '_' + str(iteration)}")

        # assert 2 > 3
    buffer.save_buffer(f"{expname}_{iteration}")

    if iteration % EVAL_INTERVAL == 0 and iteration != 0:
        os.system(
            f"python -u generate_jobs.py --set val --model {expname + '_' + str(iteration - 1)} --epoch 0 --expname {expname + '_val_' + str(iteration)}")
        with open(f"data/main_controller_logs/{expname}", "a") as mclog:
            mclog.write(f"python -u generate_jobs.py --set val --model {expname + '_' + str(iteration - 1)} --epoch 0 --expname {expname + '_val_' + str(iteration)}")
            mclog.write("\n")
        os.system(f"bash data/job_scripts/main.sh")

        os.system(f"python -u run_ground_solver.py --expname {expname + '_val_' + str(iteration)}")
        with open(f"data/main_controller_logs/{expname}", "a") as mclog:
            mclog.write( f"python -u generate_jobs.py --set test --model {expname + '_' + str(iteration - 1)} --epoch 0 --expname {expname + '_test_' + str(iteration)}")
            mclog.write("\n")
        os.system(
            f"python -u generate_jobs.py --set test --model {expname + '_' + str(iteration - 1)} --epoch 0 --expname {expname + '_test_' + str(iteration)}")
        os.system(f"bash data/job_scripts/main.sh")

        os.system(f"python -u run_ground_solver.py --expname {expname + '_test_' + str(iteration)}")

