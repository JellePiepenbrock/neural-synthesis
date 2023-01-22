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

aa = 0
bb = 0
cc = 0
def load_labels_conditional(file_location, sample_multi=-1):
    # print(f"Conditional label loading from: {file_location}")

    with open(file_location, "r") as f1:
        lines = f1.readlines()
        lines = [k.strip() for k in lines]
        # dictionary from VAR -> [TERM]
        clause_instantiation_dictionary = {}
        for line in lines:
            clause_identifier, instances = line.split(":")
            clause_identifier = clause_identifier.strip()
            clause_identifier = "C" + clause_identifier

            # print(clause_identifier)
            instances = instances.strip()
            instances = instances.split("%")
            instances = [k for k in instances if not k == '']
            instances = [k.lstrip().rstrip() for k in instances]
            inst_dict = {}
            for instance in instances:
                variable, term = instance.split("@")
                variable = variable.lstrip().rstrip()
                term = term.lstrip().rstrip()
                # print(clause_identifier + "_" + variable, term)

                variable_identifier = clause_identifier + "_" + variable
                if variable_identifier not in inst_dict:
                    inst_dict[variable_identifier] = term
                else:
                    assert 2 > 3

            if clause_identifier not in clause_instantiation_dictionary:
                clause_instantiation_dictionary[clause_identifier] = [inst_dict]

            else:

                clause_instantiation_dictionary[clause_identifier] += [inst_dict]

        single_inst_exp = True
        new_clause_dict = {}
        if not sample_multi == -1:

            for clause_labels in clause_instantiation_dictionary:
                if single_inst_exp:
                    if clause_instantiation_dictionary[clause_labels] > 1:

                        raise ValueError(f"More than 1 label per clause found in this problem : {file_location}")
                real_sample_size = min(len(clause_instantiation_dictionary[clause_labels]), sample_multi)
                new_clause_dict[clause_labels] = random.sample(clause_instantiation_dictionary[clause_labels], k=real_sample_size)

            clause_instantiation_dictionary = new_clause_dict

    if len(clause_instantiation_dictionary) == 0:
        print(file_location)
        assert 2 > 3
    return clause_instantiation_dictionary

def load_label_file(file_location):

    print("ARE WE GOING TO READ IN THE LABELS?")
    print(f"Yes, from {file_location}")
    with open(file_location, "r") as f1:
        lines = f1.readlines()
        lines = [k.strip() for k in lines]
        # dictionary from VAR -> [TERM]
        var_to_ground_term_dictionary = {}
        for line in lines:
            clause_identifier, instances = line.split(":")
            clause_identifier = clause_identifier.strip()
            clause_identifier = "Ci_0_" + clause_identifier

            print(clause_identifier)
            instances = instances.strip()
            instances = instances.split("%")
            instances = [k for k in instances if not k == '']
            instances = [k.lstrip().rstrip() for k in instances]
            for instance in instances:
                variable, term = instance.split("@")
                variable = variable.lstrip().rstrip()
                term = term.lstrip().rstrip()
                # print(clause_identifier + "_" + variable, term)
                variable_identifier = clause_identifier + "_" + variable
                if variable_identifier not in var_to_ground_term_dictionary:
                    var_to_ground_term_dictionary[variable_identifier] = [term]
                else:
                    var_to_ground_term_dictionary[variable_identifier] += [term]

            print("Instances")
            # print(instances)
        # print(var_to_ground_term_dictionary)
        # print(lines)
        # assert 2 > 3
    # We now know what ground terms should be put into what variable
    return var_to_ground_term_dictionary

def get_clausename(clause):
    s = clause.split(",")[0]

    j = s.split("(")[1]
    return j


def relabel_clauses(inputfolder, outputfolder):
    filelist = glob.glob(f"{inputfolder}/*")
    if not os.path.isdir(outputfolder):
        os.makedirs(outputfolder)

    for k in filelist:
        with open(k, "r") as f1:
            filename = k.split("/")[-1]
            lines = f1.readlines()
            lines = [k.strip() for k in lines]
            # print(lines)
            newlines = []
            for line in lines:
                clausename = get_clausename(line)
                clause_prefix = "C" + clausename
                text_after = re.sub('[A-Z]+[a-z]*[0-9]*', rf'{clause_prefix}_' + '\g<0>', line)

                # print(text_after)
                newlines.append(text_after)

            with open(outputfolder + filename, "w") as f2:
                # print(outputfolder + filename)
                for line in newlines:
                    f2.write(line)
                    f2.write("\n")


def extract_arities(input_file, utilsfolder):

    # results = os.system(f"{utilsfolder}GetSymbols {input_file} | {utilsfolder}extractsyms_ar.pl ")
    # proc = subprocess.Popen([f"{utilsfolder}GetSymbols", f"{input_file}", "|", f"{utilsfolder}extractsyms_ar.pl"], stdout=subprocess.PIPE, shell=True)
    proc = subprocess.check_output(f"{utilsfolder}GetSymbols {input_file} | {utilsfolder}extractsyms_ar.pl ",
                             shell=True)

    linelist = proc.decode().split("\n")
    for e, l in enumerate(linelist):
        if l == "FUNCS":
            linelist_end = linelist[e:]
            break

    functions = [k for k  in linelist_end if not k in ["FUNCS", "CONSTS", '']]

    # print(linelist_end)
    # print(functions)
    function_arities = {}
    for function in functions:
        # print(function)

        index_last_occ = function[::-1].find("__")
        # print(index_last_occ)
        symbol = function[:len(function)-index_last_occ-2]
        arity = function[::-1][:index_last_occ]
        function_arities[symbol] = int(arity)
        if int(arity) > 25:
            raise ValueError("Symbol with too high arity")
    # print(proc)
    # print(linelist)
    # print(function_arities)


    return function_arities

def load_label_file_symbols(file_location):
    print(file_location)
    with open(file_location, "r") as f1:
        lines = f1.readlines()
        lines = [k.strip() for k in lines]
        # dictionary from VAR -> SYMBOL at the head of the term
        var_to_symbol_dictionary = {}
        for line in lines:

            clause_identifier, symbols = line.split(":")
            # print(clause_identifier)
            if not len(symbols) == 0:
                clause_identifier = clause_identifier.strip()
                clause_identifier = "C" + clause_identifier

                # print(clause_identifier)
                symbols = symbols.strip()
                symbols = symbols.split("%")

                symbols = [k for k in symbols if not k == '']
                symbols = [k.lstrip().rstrip() for k in symbols]
                # print(symbols)
                # Symbol meaning the labeled right answers from solutions
                for symbol in symbols:
                    variable, head = symbol.split("@")
                    variable = variable.lstrip().rstrip()
                    head = head.lstrip().rstrip()
                    # print(clause_identifier + "_" + variable, head)
                    variable_identifier = clause_identifier + "_" + variable
                    if variable_identifier not in var_to_symbol_dictionary:
                        var_to_symbol_dictionary[variable_identifier] = [head]
                    else:
                        var_to_symbol_dictionary[variable_identifier] += [head]
    # print(var_to_symbol_dictionary)

    return var_to_symbol_dictionary

def load_data_symbols_individual(cnf_file, label_file):

    graph, (clausetypes, term_names, symbol_names) = cop.load_cnfpremsel(cnf_file)
    var_to_sym_dict = load_label_file_symbols(label_file)
    print(cnf_file)
    print(label_file)
    print(var_to_sym_dict)
    # assert 2 > 3

    x = GraphData(graph)
    x.ini_clauses = np.array(clausetypes)

    return (x, (clausetypes, term_names, symbol_names, var_to_sym_dict))

def load_cnf_only(file_location, return_sig_size=False):
    print(file_location)
    graph, (clausetypes, term_names, symbol_names) = cop.load_cnfpremsel(file_location)
    print(clausetypes)
    print(term_names)
    print(symbol_names) #
    print(graph) #
    # print(len(symbol_names[0]))
    x = GraphData(graph)

    x.ini_clauses = np.array(clausetypes)

    # I also need a list of clause names (these are only clauses with variables in the body).
    clause_names = []
    with open(file_location, "r") as f1:
        lines = f1.readlines()
        # print(lines)
        # assert 2 > 3
        for line in lines:
            cc = line.split(",")[0]
            cc = cc.split("(")[1]
            if "C" in line:
                # print(line)
                clause_names.append(cc)
            # print(cc)
    if len(set(clause_names)) != len(clause_names):
        raise ArithmeticError("There are multiple clauses with the same name!!! Very bad!")
    # assert 2 > 3
    if len(clause_names) == 0:
        print(file_location)
        raise ValueError("This problem has no clauses with variables.")

    if return_sig_size:
        return (x, (term_names, symbol_names, clause_names), len(symbol_names[0]))
    else:
        return (x, (term_names, symbol_names, clause_names))

signature_sizes = []

def process_cnf_batch(batch):

    """"We need some function that takes the output of cnf_data and gives us a two dictionaries: the indices for each
    variable and the indices for each functional symbol.
    """

    (graphs, term_names, symbol_names, clauses) = zip(*batch)

    big_graph = combine_graphs(graphs)

    new_term_list = []
    new_symbol_list = []

    total_term_index = {}
    total_var_index = {}
    term_index = 0
    for e, terms in enumerate(term_names):
        for term in terms:
            new_term_name = f"B{e}" + term
            new_term_list.append(new_term_name)
            total_term_index[new_term_name] = term_index
            term_index += 1
            if term.startswith(
                    "C") and not "=" in term:  # note that this is the OLD name, so if if started with C it was a variable
                # condition now catches the condition where a term is VAR1 != VAR2
                total_var_index[new_term_name] = term_index


    for e, symbols in enumerate(symbol_names):
        fsyms = []
        rsyms = []
        for functional_symbol in symbols[0]:

            fsyms.append(f"B{e}" + functional_symbol)

        for relational_symbol in symbols[1]:

            rsyms.append(f"B{e}" + relational_symbol)

        new_symbol_list.append((fsyms, rsyms))

    symbol_index_list = []
    global signature_sizes
    signature_sizes.append(len(fsyms))
    symbol_index = 0
    for problem in new_symbol_list:
        symbol_indices = {}
        for functional_symbol in problem[0]:
            symbol_indices[functional_symbol] = symbol_index
            symbol_index += 1

        # Now add the number of relational symbols
        symbol_index += len(problem[1])
        symbol_index_list.append(symbol_indices)
    clause_index_list = []

    clause_index = 0
    # WARNING: this is just clauses with variables; clauses without variables don't show up here (see load_cnf_only function)
    for e, problem_clauses in enumerate(clauses):
        clause_indices = {}
        for pc in problem_clauses:

            clause_indices[f"B{e}C" + pc] = clause_index
            clause_index += 1

        clause_index_list.append(clause_indices)

    clause_var_index_list = []
    for e, problem_clauses in enumerate(clause_index_list):
        clause_var_index = {}
        for pc in problem_clauses:

            var_list = []
            for var in total_var_index:

                if var.startswith(pc + "_"): # _ is to exlude vars from claues 21, 22, 23, ending up in clause 2 etc.
                    var_list.append(var)

            clause_var_index[pc] = var_list

        clause_var_index_list.append(clause_var_index)

    return big_graph, (total_var_index, symbol_index_list, clause_index_list, clause_var_index_list)



def load_data_symbols(file_location_list, label_folder, training=True):
    data_list = []
    if training:


        for file_location in file_location_list:

            graph, (clausetypes, term_names, symbol_names) = cop.load_cnfpremsel(file_location)
            # Just the filename so we can look up the corresponding label file
            filename = file_location.split("/")[-1]

            # Gather the labels
            var_to_sym_dict = load_label_file_symbols(label_folder + filename)

            x = GraphData(graph)

            x.ini_clauses = np.array(clausetypes)

            data_list.append((x, (clausetypes, term_names, symbol_names, var_to_sym_dict)))
    else:

        for file_location in file_location_list:
            graph, (clausetypes, term_names, symbol_names) = cop.load_cnfpremsel(file_location)
            x = GraphData(graph)

            x.ini_clauses = np.array(clausetypes)

            data_list.append((x, (clausetypes, term_names, symbol_names, -1)))

    return data_list

def load_data(datadir, sample=-1):
    fnames = os.listdir(datadir)
    import random

    if not sample == -1:
        if sample > len(fnames):
            sample = len(fnames)
        fnames = random.sample(fnames, sample)

    print(f"Number of files to load: {len(fnames)}")
    test = False
    if not test:
        random.shuffle(fnames)

    data_list = []
    for fname in fnames:
        data, (lens, labels, symbols) = cop.load_premsel(os.path.join(datadir, fname))
        # print(symbols)
        data_list.append((GraphData(data), (lens, labels)))

    return data_list

# start of neural network code

def segment_reduce(data, agg_segments, device):
    # We add as a last number to the segments, the length of the data (segment_csr is different from the tensorflow
    # segment operation
    agg_segments = torch.cat(
        (agg_segments, torch.tensor(data.shape[0], device=device).reshape(1))
    )

    reduced_max = segment_csr(data, agg_segments, reduce="max")
    reduced_mean = segment_csr(data, agg_segments, reduce="mean")

    cat = torch.cat((reduced_max, reduced_mean), axis=1)

    return cat


def segment_reduce_prime(data, agg_segments, device):

    agg_segments = torch.cat(
        (agg_segments, torch.tensor(data.shape[0], device=device).reshape(1))
    )

    reduced_max = segment_csr(data, agg_segments, reduce="max")
    reduced_min = segment_csr(data, agg_segments, reduce="min")
    reduced_mean = segment_csr(data, agg_segments, reduce="mean")

    cat = torch.cat((reduced_mean, reduced_max + reduced_min), axis=1)

    return cat


def gather_opt(vectors, indices, device):
    """"Prepend another zero vector to the node embeddings, so the index 0 will refer to that 0 embedding"""

    zeros = torch.zeros(vectors.shape[1:], device=device)  # we are getting a batch of vectors so 0 dim is batch dim
    vectors = torch.cat((zeros.unsqueeze(0), vectors))

    # Remember that we gave non-existent nodes the label -1, so they will index to the 0 element
    # print(indices)
    # print(type(indices))
    # print("MAXIMUM INDEX: ")
    # print(torch.max(indices))
    # if isinstance(indices, ):
    #     indices = indices.long()
    if indices.dtype == torch.float64:
        indices = indices.long()
    return vectors[indices + 1]


def exclusive_cumulative_sum(tens):
    """PyTorch does not have exclusive cumulative sum, so implementing it by cumsum and roll"""

    cumulative_sum = tens.cumsum(0)
    rolled_cumulative_sum = cumulative_sum.roll(
        1, 0
    )  # roll 1 position to the right on axis 0)
    ex_cumsum = rolled_cumulative_sum.clone()
    ex_cumsum[0] = 0
    return ex_cumsum


def prepare_segments(lens):

    # Remember where the nonzero segments (nodes with no incoming messages) are so we know where to pad with empty
    # messages later

    # host device synchonization
    # https://pytorch.org/docs/stable/generated/torch.nonzero.html
    nonzero_indices = torch.nonzero(lens).reshape(-1)

    # Remove the empty segments (nodes with no incoming messages)
    nonzero_lens = lens[nonzero_indices].reshape(-1)

    # Get the segment divider indices for the segment_mean/max ops
    segments = exclusive_cumulative_sum(nonzero_lens)

    return nonzero_indices, nonzero_lens, segments


def add_zeros(data, nonzero_indices, full_shape, device):
    """We add a 0 vector at places where there is a node that got no incoming messages"""

    # Make zero vector
    zero_base_tensor = torch.zeros(full_shape, device=device)

    # Put the non-zero data in the right place, replacing the zero vectors
    zero_base_tensor.index_copy_(0, nonzero_indices, data)

    # We have added 0 for the nodes that did not have inputs (so the shape will stay the same)
    return zero_base_tensor


def combine_graphs(batch):
    """Takes the functionality originally implemented in GraphPlaceholder's feed function"""
    # print("BATCH OF GRAPHS")
    # print(batch)
    batch = [g.clone() for g in batch]
    node_nums = [g.num_nodes for g in batch]
    symbol_nums = [g.num_symbols for g in batch]
    clause_nums = [g.num_clauses for g in batch]

    data = GraphData.ini_list()
    for g in batch:
        data.append(g)
    data.flatten()

    return data, (node_nums, symbol_nums, clause_nums)




def combine_term_names(batch):
    """Put a prefix before every term so that we are sure we can know where a variable / term came from (from which
    problem in the batch)"""

    # Get the info in separate varsz
    (clause_types, term_names, inst_labels) = zip(*batch)
    # print("We have arrived in the combine term names function")
    # print("Clause types")
    # print(clause_types)
    # print("Term names")
    # print(term_names)
    # print("Instantiation Labels")
    # print(inst_labels)


    new_term_list = []
    new_inst_label_dict = {}
    new_var_list = []
    ground_term_list = []

    # print("Lenght of term names")
    # print(len(term_names))
    # print(len(inst_labels))



    for e, (tn, ild) in enumerate(zip(term_names, inst_labels)): #term names, inst label dict
        for term in tn:
            new_term_list.append(f"B{e}" + term)


        # print(f"This has {len(new_term_list)} terms")
        ground_term_list_partial = [k for k in new_term_list if not "C" in k] # This is wrong but also right -- it's slightly
        # ugly because there are two reasons a C shows up: my clause labeling and the C from the variable name. It comes
        # out to the desired thing, but be aware.
        ground_term_list += ground_term_list_partial
        # print(f"This has {len(ground_term_list)} ground terms")
        # print(f"This has {len(ild)} variables")

        for var in ild:

            new_gt_list = []
            for gt in ild[var]:
                new_gt_list.append(f"B{e}" + gt)
            new_inst_label_dict[f"B{e}" + var] = new_gt_list




    clause_types_catted = torch.cat([torch.tensor(k) for k in clause_types])

    return (clause_types_catted, new_term_list, new_inst_label_dict, ground_term_list)


def combine_term_names_symbols(batch, training=True):
    """Put a prefix before every term and symbol so that we are sure we can know where a variable / term came from (from which
    problem in the batch)"""

    # Get the info in separate varsz
    (clause_types, term_names, symbol_names, sym_labels) = zip(*batch)
    # print("We have arrived in the combine term names function")
    # print("Clause types")
    # print(clause_types)
    # print("Term names")
    # print(term_names)
    # print("Symbol Names")
    # print(symbol_names)
    # print([(len(k[0]), len(k[1])) for k in symbol_names])
    # print("Instantiation Labels")
    # print(sym_labels)

    # flattened_symbol_list = []
    # symbol_indices = {}
    # symbol_index = 0
    # for problem in symbol_names:
    #     for functional_symbol in problem[0]:
    #         symbol_indices[functional_symbol] = symbol_index
    #         symbol_index += 1
    #
    #     # Now add the number of relational symbols
    #     symbol_index += len(problem[1])

    # print(symbol_indices)

    # print("FLATTENED SYMBOLS")
    # print(flattened_symbol_list)

    new_term_list = []
    new_symbol_list = []
    new_func_symbol_list = []
    new_rel_symbol_list = []
    new_inst_label_dict = {}

    new_var_list = []

    ground_term_list = []

    # print("Lenght of term names")
    # print(len(term_names))
    # print(len(sym_labels))
    # I need a way to disregard all the relational symbols but still know the indices into the symbol vector
    # Just concat all of them to make indices, then make a dict
    # print(len(term_names), len(symbol_names), len(sym_labels))
    flattened_list = []

    total_term_index = {}
    total_var_index = {}
    term_index = 0
    for e, terms in enumerate(term_names):
        for term in terms:
            new_term_name = f"B{e}" + term
            new_term_list.append(new_term_name)
            total_term_index[new_term_name] = term_index
            term_index += 1
            if term.startswith("C") and not "=" in term: # note that this is the OLD name, so if if started with C it was a variable
                # condition now catches the condition where a term is VAR1 != VAR2
                total_var_index[new_term_name] = term_index

    # print(total_term_index)
    # print(total_var_index)

    for e, symbols in enumerate(symbol_names):
        fsyms = []
        rsyms = []
        for functional_symbol in symbols[0]:
            flattened_list.append(f"B{e}" + functional_symbol)
            fsyms.append(f"B{e}" + functional_symbol)

        for relational_symbol in symbols[1]:
            flattened_list.append(f"B{e}" + relational_symbol)
            rsyms.append(f"B{e}" + relational_symbol)

        new_symbol_list.append((fsyms, rsyms))

    # print([(e, k) for e, k in enumerate(flattened_list)])
    # print("NEW SYMBOL LIST")
    # print(new_symbol_list)
    symbol_indices = {}
    symbol_index = 0
    for problem in new_symbol_list:
        for functional_symbol in problem[0]:
            symbol_indices[functional_symbol] = symbol_index
            symbol_index += 1

        # Now add the number of relational symbols
        symbol_index += len(problem[1])
    # print("SYMBOL INDEX")
    # print(symbol_indices)


    if training:
        total_sym_label_dict = {}
        for e, sym_dict in enumerate(sym_labels):
            for variable in sym_dict:
                varname = f"B{e}" + variable

                suggested_symbols = sym_dict[variable]
                new_suggested_symbols = [f"B{e}" + k for k in suggested_symbols]
                total_sym_label_dict[varname] = new_suggested_symbols
        # print("SYMBOL LABELS")
        # print(sym_labels)
        # print("NEW SYMBOL LABELS")
        # print(total_sym_label_dict)


        for key in total_sym_label_dict:
            answers = total_sym_label_dict[key]
            # print("KEY:", key)
            for a in answers:
                # print(a)
                assert a in symbol_indices
        # print(f"This has {len(new_func_symbol_list)} function symbols and {len(new_rel_symbol_list)} relational symbols")

        # print(f"There are {len(total_var_index)} variables and {len(total_sym_label_dict)} of them are mentioned in the labels")
        for key in total_sym_label_dict:
            # print(total_var_index)
            # print(total_sym_label_dict)
            # print(total_var_index[key])
            # print(key)
            assert key in total_var_index # if this passed I know the indices of all the relevant variables

        # clause_types_catted = torch.cat([torch.tensor(k) for k in clause_types])

        # I also need indices for the variables

        # print("New term list")
        # print(len(new_term_list))

        return (total_sym_label_dict, symbol_indices, total_var_index)

    else:
        return (-1, symbol_indices, total_var_index)



def batch_graphs(data):
    graphs, label_infos = zip(*data)

    # Make all of the graphs 1 big graph
    big_graph, segments = combine_graphs(graphs)



    return (big_graph, segments), label_infos

def batch_graphs_instantiation_dev(data, training=True):
    import copy


    # print("----------------------------")


    graph, label_infos = zip(*data)
    big_graph = combine_graphs(graph)
    # print(big_graph)
    # print(label_infos)

    # big_labels = combine_term_names(label_infos)
    big_labels = get_labels_symbols(label_infos, training=training)


    return big_graph, big_labels
    # Need to modify combine graph to get a prefix for the term names (i.e. B01_{term_name}) so that we know
    # for sure where a term came from

def batch_input(data):

    graphs, term_information = zip(*data)
    big_graph = combine_graphs(graphs)

class PIEGNN(nn.Module):
    """PyTorch implementation of the network from Property Invariant Embedding (PIE) for Automated Reasoning"""

    def __init__(
        self,
        start_dims,
        next_dims,
        device="cpu",
        hidden_dim=8,
        layers=2,
        repeat_layer=False,
        residual=True,
        normalization="batch",
    ):

        super(PIEGNN, self).__init__()

        # Whether we use one layer that we repeat or separate parameters per layer / hop
        self.repeat_layer = repeat_layer
        self.device = device
        self.residual = residual
        self.num_layers = layers
        self.normalization = normalization

        self.hidden_dim = hidden_dim
        self.node_dim, self.symbol_dim, self.clause_dim = start_dims
        self.node_dim_next, self.symbol_dim_next, self.clause_dim_next = next_dims

        # Initialize embeddings

        from torch.nn.init import xavier_uniform_

        dim_nodes, dim_symbols, dim_clauses = (
            self.node_dim,
            self.symbol_dim,
            self.clause_dim,
        )
        #
        # Default tensorflow initialization is glorot uniform, so we use that too
        # TODO: make this nn.Parameter!!!!!!!
        # self.node_emb = xavier_uniform_(torch.empty(4, dim_nodes)).to(device)
        # # self.symbol_emb = torch.zeros((2, dim_symbols)).to(device)
        # # TODO Make this a setting
        # self.symbol_emb = xavier_uniform_(torch.empty((2, dim_clauses))).to(device)
        # self.clause_emb = xavier_uniform_(torch.empty((3, dim_clauses))).to(device)

        self.node_emb = nn.Parameter(xavier_uniform_(torch.empty(4, dim_nodes)).to(self.device), requires_grad=True)
        # self.symbol_emb = torch.zeros((2, dim_symbols)).to(device)
        # TODO Make this a setting
        self.symbol_emb = nn.Parameter(xavier_uniform_(torch.empty((2, dim_symbols))).to(self.device), requires_grad=True)
        self.clause_emb = nn.Parameter(xavier_uniform_(torch.empty((3, dim_clauses))).to(self.device), requires_grad=True)

        # Several normalizations
        if self.normalization == "batch":
            self.clause_bn = nn.BatchNorm1d(self.clause_dim_next)
            self.node_bn = nn.BatchNorm1d(self.node_dim_next)
            self.symbol_bn = nn.BatchNorm1d(self.symbol_dim_next)
        elif self.normalization == "layer":
            self.clause_ln = nn.LayerNorm(self.clause_dim_next)
            self.node_ln = nn.LayerNorm(self.node_dim_next)
            self.symbol_ln = nn.LayerNorm(self.symbol_dim_next)

        # We need to specify modulelists so we can easily find our multiple layers

        self.mc_list = nn.ModuleList()
        self.mct_list = nn.ModuleList()
        self.mts_123_list = nn.ModuleList()
        self.ms_list = nn.ModuleList()
        self.mts_bold_list = nn.ModuleList()
        self.bst_list = nn.ParameterList()

        self.node_processor_layer_list = (
            nn.ModuleList()
        )  # double list, with n-layer elements with sublist of size 3
        self.symbol_processor_layer_list = nn.ModuleList()  # same but for the symbols
        self.y_aggregator_list = (
            nn.ModuleList()
        )  # double list again with 3 in each layer

        self.mtc_list = nn.ModuleList()
        self.mt_list = nn.ModuleList()

        for layer in range(0, self.num_layers):
            # First layer (initial embedding) can have different size but afterwards all need to be the same for residuals
            if layer == 0:
                clause_in = self.clause_dim
                clause_out = self.clause_dim_next

                node_in = self.node_dim
                node_out = self.node_dim_next

                symbol_in = self.symbol_dim
                symbol_out = self.symbol_dim_next

            else:
                clause_in = self.clause_dim_next
                clause_out = self.clause_dim_next

                node_in = self.node_dim_next
                node_out = self.node_dim_next

                symbol_in = self.symbol_dim_next
                symbol_out = self.symbol_dim_next

            self.mc_list.append(nn.Linear(clause_in, clause_out))

            self.mct_list.append(nn.Linear(2 * node_in, clause_out, bias=False))
            self.mts_123_list.append(nn.Linear(3 * node_in, symbol_out, bias=True))
            self.ms_list.append(nn.Linear(symbol_in, symbol_out, bias=False))
            self.mts_bold_list.append(nn.Linear(2 * symbol_out, symbol_out, bias=False))
            self.bst_list.append(nn.Parameter(torch.randn(size=(1, node_out))))

            # We will add a processorlist for each layer
            self.node_processor_layer_list.append(nn.ModuleList())
            # Then we will add three networks to this modulelist
            for i in range(3):
                self.node_processor_layer_list[layer].append(
                    nn.Linear(2 * node_in, node_out, bias=False)
                )

            # same idea as above:
            self.symbol_processor_layer_list.append(nn.ModuleList())
            for i in range(3):
                self.symbol_processor_layer_list[layer].append(
                    nn.Linear(symbol_in, node_out, bias=False)
                )

            self.y_aggregator_list.append(nn.ModuleList())
            for i in range(3):
                self.y_aggregator_list[layer].append(
                    nn.Linear(2 * node_out, node_out, bias=False)
                )

            self.mtc_list.append(nn.Linear(2 * clause_in, node_out, bias=False))

            self.mt_list.append(nn.Linear(node_in, node_out))

        # 4 times the clause dimension as input because we concat the conjecture to each clause
        # And for both of them we use both max and mean
        # There is only 1 of these, not 1 for every layer
        self.clause_decider = nn.Sequential(
            nn.Linear(4 * self.clause_dim_next, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

        self.symbol_projector = nn.Sequential(
            nn.Linear(self.symbol_dim_next, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.symbol_dim_next),
        )
        #
        self.variable_projector = nn.Sequential(
            nn.Linear(self.node_dim_next, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.node_dim_next),
        )

        self.cls_token = nn.Parameter(torch.randn(self.node_dim_next), requires_grad=True)
        self.termination_symbol = nn.Parameter(torch.randn(self.node_dim_next), requires_grad=True)

        self.empty_symbol = nn.Parameter(torch.randn(self.node_dim_next), requires_grad=True)
        self.generator = nn.Sequential(nn.Linear(3*self.node_dim_next, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.node_dim_next))
        # self.generator = nn.Sequential(nn.Linear(3*self.node_dim_next, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.node_dim_next))
        self.rnn_ln = nn.LayerNorm(self.symbol_dim_next)

        # self.lstm_generator = nn.LSTMCell(self.node_dim_next, self.node_dim_next)
        # self.lstm_ln = nn.LayerNorm(self.symbol_dim_next)
        # self.cx_init = torch.randn(self.node_dim_next)
        # self.hx_init = torch.randn(self.node_dim_next)



    def graph_start(self, graph_object):

        # TDOO move to the right GPU immediately!
        ini_nodes = torch.tensor(graph_object.ini_nodes, device=self.device)
        ini_symbols = torch.tensor(graph_object.ini_symbols, device=self.device)
        ini_clauses = torch.tensor(graph_object.ini_clauses, device=self.device)

        nodes = self.node_emb[ini_nodes]
        symbols = self.symbol_emb[ini_symbols]
        clauses = self.clause_emb[ini_clauses]

        return nodes, symbols, clauses

    def graph_conv(self, node_vectors, graph_structure, layer):

        nodes, symbols, clauses = node_vectors

        in_nodes = []

        for e, n in enumerate(graph_structure.node_inputs):

            if not len(n.nodes) == 0:
                vector_representations = gather_opt(nodes, torch.tensor(n.nodes, device=self.device), self.device)

                dim = vector_representations.shape[1] * vector_representations.shape[2]
                vector_representations = torch.reshape(
                    vector_representations, [-1, dim]
                )
                xn = self.node_processor_layer_list[layer][e].forward(
                    vector_representations
                )
                # print(torch.tensor(n.symbols))
                # print(type(torch.tensor(n.symbols)))
                xs = symbols[torch.tensor(n.symbols, dtype=torch.long, device=self.device)]
                # print(type(xs))
                xs = xs * torch.tensor(n.sgn, device=self.device).unsqueeze(1)
                # print(type(xs))
                xs = self.symbol_processor_layer_list[layer][e].forward(torch.tensor(xs, dtype=torch.float32, device=self.device)) # Some very rare circumstances cause pytorch to infer a float64 type, which crashes the code, so we cast it

                nonzero_indices, nonzero_lens, segments = prepare_segments(
                    torch.tensor(n.lens, device=self.device)
                )

                zijd = segment_reduce(
                    torch.relu(xn + xs + self.bst_list[layer]), segments.to(self.device), self.device
                )



                zijd = add_zeros(
                    zijd, nonzero_indices.to(self.device), (nodes.shape[0], zijd.shape[1]), self.device
                )
                zijd = self.y_aggregator_list[layer][e].forward(zijd)
                in_nodes.append(zijd)

        # Add tij
        node_representations = self.mt_list[layer](nodes)

        in_nodes.append(node_representations)

        # out_nodes <- clauses

        nc = graph_structure.node_c_inputs
        x = clauses[torch.tensor(nc.data, device=self.device)]

        nonzero_indices, nonzero_lens, segments = prepare_segments(
            torch.tensor(nc.lens, device=self.device)
        )

        x = segment_reduce(x, segments.to(self.device), self.device)

        x = add_zeros(x, nonzero_indices.to(self.device), (nodes.shape[0], x.shape[1]), self.device)
        vij = self.mtc_list[layer](x)

        in_nodes.append(vij)

        # out_symbols <- symbols, nodes
        sy = graph_structure.symbol_inputs
        x = gather_opt(nodes, torch.tensor(sy.nodes, device=self.device), self.device)
        dim = x.shape[1] * x.shape[2]
        x = torch.reshape(x, [-1, dim])
        x = self.mts_123_list[layer](x)

        x = x * torch.tensor(sy.sgn, device=self.device).unsqueeze(1).to(self.device)
        segments = exclusive_cumulative_sum(torch.tensor(sy.lens, device=self.device))
        x = segment_reduce_prime(x, segments.to(self.device), self.device)

        symbol_next_first_term = self.ms_list[layer](symbols)

        symbol_next_second_term = self.mts_bold_list[layer](x)

        out_symbols = torch.tanh(symbol_next_first_term + symbol_next_second_term)

        # out_clauses <- nodes, clauses

        c = graph_structure.clause_inputs
        x = nodes[torch.tensor(c.data, device=self.device)]
        segments = exclusive_cumulative_sum(torch.tensor(c.lens, device=self.device))
        x = segment_reduce(x, segments.to(self.device), self.device)

        clause_next_first_term = self.mc_list[layer](clauses)

        clause_next_second_term = self.mct_list[layer](x)

        out_clauses = torch.relu(clause_next_first_term + clause_next_second_term)

        out_nodes = in_nodes[0]
        for term in in_nodes[1:]:
            out_nodes = out_nodes + term

        return out_nodes, out_symbols, out_clauses

    # def test_time_forward_random(self,
    #                              graph_structure,
    #                              label_info,
    #                              iterations=6,
    #                              beam_width=6,
    #                              optimism=1):
    #
    #     """"Baseline that does not involve too much GPU-ing"""
    #     vector_representations = model.graph_start(graph_structure)
    #
    #     var_i, sym_i, clause_i, clause_var_index, _ = label_info





    def test_time_forward_beam(
            self,
            graph_structure,
            label_info, iterations=6, beam_width=6, optimism=1
    ):

        # Initialize all the embeddings
        vector_representations = self.graph_start(graph_structure)

        for i in range(self.num_layers):
            if i == 0:
                nodes, symbols, clauses = self.graph_conv(
                    vector_representations, graph_structure, i
                )

            else:
                if self.repeat_layer:
                    new_nodes, new_symbols, new_clauses = self.graph_conv(
                        (nodes, symbols, clauses), graph_structure, 0
                    )

                else:
                    new_nodes, new_symbols, new_clauses = self.graph_conv(
                        (nodes, symbols, clauses), graph_structure, i
                    )

                # residual_nodes, residual_symbols, residual_clauses = vector_representations
                if self.residual:
                    nodes = nodes + new_nodes
                    symbols = symbols + new_symbols
                    clauses = clauses + new_clauses
                # elif not self.residual and self.normalize()
                else:
                    nodes = new_nodes
                    symbols = new_symbols
                    clauses = new_clauses

                if self.normalization == "batch":
                    nodes = self.node_bn(nodes)
                    symbols = self.symbol_bn(symbols)
                    clauses = self.clause_bn(clauses)
                elif self.normalization == "layer":
                    nodes = self.node_ln(nodes)
                    symbols = self.symbol_ln(symbols)
                    clauses = self.clause_ln(clauses)

        # for each clause, construct:

        # [cls, clause, VARA]
        # [cls, clause, VARA + SYMA, VARB]
        # [cls, clause, VARA + SYMA, VARB + SYMB]

        # construct VARA
        # we need to forward somehow related to the amount of variables in each clause.

        # print(label_info)
        # assert 2 > 3
        # var_index, symbol_index, clause_with_vars_index, label_sequences
        # NOTE THAT CLAUSE i is NOT the index into the clause vectors!!
        var_i, sym_i, clause_i, clause_var_index, _ = label_info

        # assert 2 > 3
        vars_to_query = []
        # print(clause_var_index)
        # print(clause_i)
        # #
        var_inputs = []
        for e, cl_index in enumerate(clause_i):
            cl_2var_vector = {}
            # associated vars
            for cl in cl_index:
                assoc_vars = clause_var_index[e][cl]

                assoc_vars_i = [var_i[k] for k in assoc_vars]

                assoc_vars_vecs = [nodes[k] for k in assoc_vars_i]
                # print(cl)
                # print(assoc_vars_i)
                # print(assoc_vars_vecs)
                cl_2var_vector[cl] = assoc_vars_vecs

            var_inputs.append(cl_2var_vector)

        # print(var_inputs)

        # print(sym_i)

        # Here we construct the lists that contain the symbol vectors for each problem in the batch
        # We  also add a termination symbol
        sym_inputs = []
        for e, signature in enumerate(sym_i):
            signature_list = []
            for key in signature:
                signature_list.append(symbols[signature[key]])

            signature_list.append(self.termination_symbol)
            sym_inputs.append(torch.stack(signature_list))
        # print(sym_inputs)

        # assert 2 > 3
        # TODO:
        # Construct for each clause:
        # Two lists of length len(labels) for that clause
        # One with the VARS (use the modulo construction)
        # One with the symbols (just index into the sym_input lists)
        var_vector_list = []
        sym_vector_list = []
        # assert 2 > 3

        clause_segments = []
        for e, problem_clauses in enumerate(var_inputs):
            clause_segments.append(len(problem_clauses))

        num_var_list = []
        for e, problem_clauses in enumerate(var_inputs):

            for f, clause in enumerate(problem_clauses):
                clause_variable_list = []
                clause_symbol_list = []
                num_vars = len(problem_clauses[clause])
                num_var_list.append(num_vars)

                for step in range(iterations):

                    current_vector = problem_clauses[clause][step % num_vars]

                    clause_variable_list.append(current_vector)


                var_vector_list.append(clause_variable_list)

        # print(len(labels))

        which_problem_indicator = []
        for e, k in enumerate(clause_segments):
            for i in range(k):
                which_problem_indicator.append(e)
        # print(clause_segments)
        # print(which_problem_indicator)
        signature_index = 0
        hidden_cache = []
        beamlist = [] # where we store all the beams, with their log probability
        # iter = 0
        #

        optimism_factor = optimism # https://www.mdpi.com/2078-2489/12/9/355/htm section 3.4


        for iter in range(iterations):
            if iter == 0:
                which_problem_indicator = []
                for e, k in enumerate(clause_segments):
                    for i in range(k):
                        which_problem_indicator.append(e)

            beam_search_input_stack = []
            symbol_signature_stack = []

            query_input_stack = []
            for e, single_clause in enumerate(var_vector_list):
                signature_index = which_problem_indicator[e]

                if iter == 0:
                    ss = torch.cat((self.cls_token, var_vector_list[e][iter], self.empty_symbol))
                    beam_search_input_stack.append(ss)

                    symbol_signature_stack.append(sym_inputs[signature_index])

                elif iter == 1:

                    for path in range(beam_width):

                        print(beamlist[e][path][1][0])
                        prefix = torch.cat((self.cls_token, var_vector_list[e][iter - 1], sym_inputs[signature_index][beamlist[e][path][1][0]]))
                        query = torch.cat((var_vector_list[e][iter], self.empty_symbol))
                        beam_search_input_stack.append(prefix)
                        query_input_stack.append(query)
                        symbol_signature_stack.append(sym_inputs[signature_index])

                elif iter > 1:

                    # print(prev_hidden_index)
                    # print("STARTING ITERATION 2")
                    for path in range(beam_width):
                        # print(beamlist[e][path])
                        absolute_index = beam_width * e + path
                        absolute_index_prev_state = beam_width * e + prev_hidden_index[absolute_index]
                        # print(prev_hidden_index[absolute_index])
                        # print(absolute_index_prev_state)
                        # print(beamlist[e])
                        prefix = torch.cat((hidden_cache[-1][absolute_index_prev_state], var_vector_list[e][iter - 1],
                                            sym_inputs[signature_index][beamlist[e][path][1][-1]]))
                        query = torch.cat((var_vector_list[e][iter], self.empty_symbol))
                        beam_search_input_stack.append(prefix)
                        query_input_stack.append(query)
                        symbol_signature_stack.append(sym_inputs[signature_index])


                # elif iter > 1:
                #     for path in range(beam_width):
                #         absolute_index = beam_width * e + path
                #         var_queries = torch.cat((self.cls_token, var_vector_list[e][iter - 1],
                #                                  sym_inputs[signature_index][beamlist[e][path][1][-1]])) # take the last
                # elif iter > 0:
                #     # TODO construct a beam _vector list
                #
                #     ss =
            # what to do if iter != 0

            # if iter == 2:
            #     assert 2 > 3

            # print(len(teacher_force_input_stack))
            # print(float(len(teacher_force_input_stack)) / len(labels))
            cc = 0

            if iter == 0:

                beam_search_input_stack = torch.stack(beam_search_input_stack)

                h_final = self.rnn_ln(self.generator(beam_search_input_stack))

                symbol_requests = self.symbol_projector(h_final)

                repeat_recipe = torch.tensor([k.shape[0] for k in symbol_signature_stack], device=self.device)

                prepare_vectors = torch.repeat_interleave(symbol_requests, repeat_recipe, dim=0)

                symbol_signature_stack = torch.cat(symbol_signature_stack, dim=0)

                assert sum(clause_segments) == len(var_vector_list)

                # now we just have to do a dot product between symbol_requests and the signature
                # we need to repeat the symbol_requests enough times
                similarities = torch.einsum('in, in->i', prepare_vectors, symbol_signature_stack)


                indices = []
                segment_counter = 0
                for e, number_of_symbols in enumerate(repeat_recipe):
                    indices += [segment_counter] * number_of_symbols
                    segment_counter += 1

                indices = torch.tensor(indices, device=self.device)
                softmaxed_similarities = scatter_softmax(similarities, indices)

                split_softmaxes = torch.split(softmaxed_similarities, repeat_recipe.tolist())

                predictions = split_softmaxes
                if optimism_factor != 1:
                    new_preds = []
                    for aa, k in enumerate(predictions):

                        new_k = torch.clone(k)
                        new_k[-1] = new_k[-1] / optimism_factor
                        new_preds.append(new_k)
                    predictions = new_preds



                topk = []
                for e, f in enumerate(predictions):
                    print(f.shape, beam_width, sym_inputs[which_problem_indicator[e]].shape)

                    topk.append(torch.topk(f, beam_width))

                for probs, actions in topk:

                    beamlist.append([[pr, [a]] for (pr, a) in zip(probs.tolist(), actions.tolist())])

            elif iter > 0:
                prev_hidden_index = []
                beam_search_input_stack = torch.stack(beam_search_input_stack)
                query_input_stack = torch.stack(query_input_stack)

                prefix_hidden = self.rnn_ln(self.generator(beam_search_input_stack))

                hidden_cache.append(prefix_hidden)

                next_step_input = torch.cat((prefix_hidden, query_input_stack), dim=1)


                h_final = self.rnn_ln(self.generator(next_step_input))
                symbol_requests = self.symbol_projector(h_final)

                repeat_recipe = torch.tensor([k.shape[0] for k in symbol_signature_stack], device=self.device)



                prepare_vectors = torch.repeat_interleave(symbol_requests, repeat_recipe, dim=0)

                symbol_signature_stack = torch.cat(symbol_signature_stack, dim=0)

                assert sum(clause_segments) == len(var_vector_list)

                similarities = torch.einsum('in, in->i', prepare_vectors, symbol_signature_stack)

                indices = []
                segment_counter = 0
                for e, number_of_symbols in enumerate(repeat_recipe):
                    indices += [segment_counter] * number_of_symbols
                    segment_counter += 1
                #
                # print(indices)
                # print(len(indices))
                indices = torch.tensor(indices, device=self.device)
                softmaxed_similarities = scatter_softmax(similarities, indices)

                split_softmaxes = torch.split(softmaxed_similarities, repeat_recipe.tolist())
                # print([k.shape for k in split_softmaxes])
                predictions = split_softmaxes
                if optimism_factor != 1:
                    new_preds = []
                    for aa, k in enumerate(predictions):

                        new_k = torch.clone(k)
                        new_k[-1] = new_k[-1] / optimism_factor
                        new_preds.append(new_k)
                    predictions = new_preds

                distributions = [(e // beam_width, f) for (e, f) in enumerate(predictions)]
                new_beamlist = []


                for cn, clause_beams in enumerate(beamlist):
                    new_clause_beams = []

                    signature_index = which_problem_indicator[cn]
                    sig_s = len(sym_inputs[signature_index])


                    new_possible_rays_clause = []
                    new_possible_rays_probs = []
                    new_possible_rays_hidden_indices = []
                    for ray in range(beam_width):
                        absolute_index = cn * beam_width + path

                        possible_continuations = distributions[absolute_index]

                        assert possible_continuations[0] == cn

                        assert len(possible_continuations[1]) == sig_s
                        if clause_beams[ray][1][-1] == (sig_s - 1): # if it ended in termination

                            new_possible_ray = copy.deepcopy(clause_beams[ray][1])
                            new_possible_rays_clause.append(new_possible_ray)

                            new_possible_ray_probability = clause_beams[ray][0]
                            new_possible_rays_probs.append(new_possible_ray_probability)
                            # print(new_possible_ray_probability)
                            new_possible_rays_hidden_indices.append(ray) # This is strange (points to a newer list in the hidden_cache), but will have no effects, as we will never continue this beam.

                        else: # some function symbol was chosen in the last step
                            prefix_probability = clause_beams[ray][0]
                            for i in range(len(possible_continuations[1])):
                                new_possible_ray = copy.deepcopy(clause_beams[ray][1])
                                new_possible_ray.append(i)
                                new_possible_rays_clause.append(new_possible_ray)

                                new_possible_ray_probability = prefix_probability * possible_continuations[1][i]
                                new_possible_rays_probs.append(new_possible_ray_probability)

                                new_possible_rays_hidden_indices.append(ray)

                    assert len(new_possible_rays_clause) == len(new_possible_rays_probs) and len(new_possible_rays_clause) == len(new_possible_rays_hidden_indices)


                    top_k_options = torch.topk(torch.tensor(new_possible_rays_probs, device=self.device), k = beam_width)
                    topk_values, topk_indices = top_k_options
                    # print(topk_values)
                    # print(topk_indices)
                    # Does ray order matter?
                    #
                    new_clause_beams = [(new_possible_rays_probs[k], new_possible_rays_clause[k]) for k in topk_indices.tolist()]
                    new_clause_hidden_indices = [new_possible_rays_hidden_indices[k] for k in topk_indices.tolist()]
                    prev_hidden_index += new_clause_hidden_indices
                    new_beamlist.append(new_clause_beams)

                # print(len(prev_hidden_index))
                # print(sum(clause_segments) * beam_width)
                # print(sum(clause_segments))

                assert len(prev_hidden_index) == sum(clause_segments) * beam_width
                beamlist = new_beamlist


        return beamlist

    def test_time_forward_sample(
            self,
            graph_structure,
            label_info, iterations=6, beam_width=6, optimism=1, random_baseline=False, temperature=1.0
    ):
        stime = time.time()
        temperature = torch.tensor(temperature, device=self.device)
        # Initialize all the embeddings
        vector_representations = self.graph_start(graph_structure)

        for i in range(self.num_layers):
            if i == 0:
                nodes, symbols, clauses = self.graph_conv(
                    vector_representations, graph_structure, i
                )

            else:
                if self.repeat_layer:
                    new_nodes, new_symbols, new_clauses = self.graph_conv(
                        (nodes, symbols, clauses), graph_structure, 0
                    )

                else:
                    new_nodes, new_symbols, new_clauses = self.graph_conv(
                        (nodes, symbols, clauses), graph_structure, i
                    )

                # residual_nodes, residual_symbols, residual_clauses = vector_representations
                if self.residual:
                    nodes = nodes + new_nodes
                    symbols = symbols + new_symbols
                    clauses = clauses + new_clauses
                # elif not self.residual and self.normalize()
                else:
                    nodes = new_nodes
                    symbols = new_symbols
                    clauses = new_clauses

                if self.normalization == "batch":
                    nodes = self.node_bn(nodes)
                    symbols = self.symbol_bn(symbols)
                    clauses = self.clause_bn(clauses)
                elif self.normalization == "layer":
                    nodes = self.node_ln(nodes)
                    symbols = self.symbol_ln(symbols)
                    clauses = self.clause_ln(clauses)

        print(f"GNN Forward Pass takes {time.time() - stime} seconds")
        stime = time.time()
        # for each clause, construct:

        # [cls, clause, VARA]
        # [cls, clause, VARA + SYMA, VARB]
        # [cls, clause, VARA + SYMA, VARB + SYMB]

        # construct VARA
        # we need to forward somehow related to the amount of variables in each clause.

        # print(label_info)
        # assert 2 > 3
        # var_index, symbol_index, clause_with_vars_index, label_sequences
        # NOTE THAT CLAUSE i is NOT the index into the clause vectors!!

        # It (clause_i) should just count clauses with variables
        var_i, sym_i, clause_i, clause_var_index, _ = label_info

        # assert 2 > 3
        vars_to_query = []
        # print(clause_var_index)
        # print(clause_i)
        # #
        var_inputs = []
        for e, cl_index in enumerate(clause_i):
            # print(cl_index)

            cl_2var_vector = {}
            # associated vars
            for cl in cl_index:
                # print(cl)
                # print(clause_var_index[e])
                # print(var_i)
                # assert 2 > 3
                assoc_vars = clause_var_index[e][cl]

                assoc_vars_i = [var_i[k] for k in assoc_vars]

                assoc_vars_vecs = [nodes[k] for k in assoc_vars_i]
                # print(cl)
                # print(assoc_vars_i)
                # print(assoc_vars_vecs)
                cl_2var_vector[cl] = assoc_vars_vecs

            var_inputs.append(cl_2var_vector)

        # print(var_inputs)

        # print(sym_i)

        # Here we construct the lists that contain the symbol vectors for each problem in the batch
        # We  also add a termination symbol
        sym_inputs = []
        for e, signature in enumerate(sym_i):
            signature_list = []
            for key in signature:
                signature_list.append(symbols[signature[key]])

            signature_list.append(self.termination_symbol)
            sym_inputs.append(torch.stack(signature_list))
        # print(sym_inputs)

        # assert 2 > 3
        # TODO:
        # Construct for each clause:
        # Two lists of length len(labels) for that clause
        # One with the VARS (use the modulo construction)
        # One with the symbols (just index into the sym_input lists)
        var_vector_list = []
        sym_vector_list = []
        # assert 2 > 3

        clause_segments = []
        for e, problem_clauses in enumerate(var_inputs):
            clause_segments.append(len(problem_clauses))

        num_var_list = []
        for e, problem_clauses in enumerate(var_inputs):

            for f, clause in enumerate(problem_clauses):
                clause_variable_list = []
                clause_symbol_list = []
                num_vars = len(problem_clauses[clause])
                num_var_list.append(num_vars)

                for step in range(iterations):

                    current_vector = problem_clauses[clause][step % num_vars]

                    clause_variable_list.append(current_vector)


                var_vector_list.append(clause_variable_list)

        # print(len(labels)) £ £

        which_problem_indicator = []
        for e, k in enumerate(clause_segments):
            for i in range(k):
                which_problem_indicator.append(e)
        # print(clause_segments)
        # print(which_problem_indicator)
        # print([len(k) for k in var_vector_list])
        assert sum(clause_segments) == len(which_problem_indicator)


        hidden_cache = []
        beamlist = [] # where we store all the beams, with their log probability
        # iter = 0
        #
        # beam_width = 10
        optimism_factor = optimism # https://www.mdpi.com/2078-2489/12/9/355/htm section 3.4
        prev_hidden_index = []
        print(f"Preparing the RNN takes {time.time() - stime} seconds")


        for iter in range(iterations):

            stime = time.time()
            # if iter == 0:
            #     which_problem_indicator = []
            #     for e, k in enumerate(clause_segments):
            #         for i in range(k):
            #             which_problem_indicator.append(e)

            sampling_input_stack = []
            symbol_signature_stack = []

            query_input_stack = []
            ## CONSTRUCTING THE INPUT STACKS
            for e, single_clause in enumerate(var_vector_list):
                signature_index = which_problem_indicator[e]
                # print(which_problem_indicator, signature_index)
                # assert 2 > 3
                if iter == 0:
                    ss = torch.cat((self.cls_token, var_vector_list[e][iter], self.empty_symbol))
                    sampling_input_stack.append(ss)

                    symbol_signature_stack.append(sym_inputs[signature_index])

                elif iter == 1:
                    for path in range(beam_width):

                        # print(beamlist[e][path][1][0])
                        prefix = torch.cat((self.cls_token, var_vector_list[e][iter - 1], sym_inputs[signature_index][beamlist[e][path][1][0]]))
                        query = torch.cat((var_vector_list[e][iter], self.empty_symbol))
                        sampling_input_stack.append(prefix)
                        query_input_stack.append(query)
                        symbol_signature_stack.append(sym_inputs[signature_index])

                elif iter > 1:
                    # print(prev_hidden_index)
                    # print("STARTING ITERATION 2")
                    for path in range(beam_width):
                        # print(beamlist[e][path])
                        absolute_index = beam_width * e + path
                        absolute_index_prev_state = beam_width * e + prev_hidden_index[absolute_index]
                        # print(prev_hidden_index[absolute_index])
                        # print(absolute_index_prev_state)

                        prefix = torch.cat((hidden_cache[-1][absolute_index_prev_state], var_vector_list[e][iter - 1],
                                            sym_inputs[signature_index][beamlist[e][path][1][-1]]))
                        query = torch.cat((var_vector_list[e][iter], self.empty_symbol))
                        sampling_input_stack.append(prefix)
                        query_input_stack.append(query)
                        symbol_signature_stack.append(sym_inputs[signature_index])



            cc = 0
            print(f"Constructing the input stack takes {time.time() - stime} seconds")
            # EXPANDING THE SEQUENCES
            stime = time.time()
            if iter == 0:

                sampling_input_stack = torch.stack(sampling_input_stack)
                # print(sampling_input_stack.shape)

                # assert 2> 3
                #
                h_final = self.rnn_ln(self.generator(sampling_input_stack))
                # print(h_final.shape)
                # assert 2 > 3
                # if iter == 0 we actually don't need the cache
                # hidden_cache.append(h_final)
                symbol_requests = self.symbol_projector(h_final)
                # print(symbol_requests.shape)
                # print(clause_segments)
                # print(len(symbol_signature_stack))
                # print(symbol_signature_stack[0].shape)

                # TODO make tensor on the correct device immediately [Done]
                repeat_recipe = torch.tensor([k.shape[0] for k in symbol_signature_stack], device=self.device)
                # aaaa = [k.shape for k in sym_inputs]


                # print(repeat_recipe)
                # print(clause_segments)
                # print(repeat_recipe[75])
                # print(repeat_recipe[76])

                # assert 2 > 3
                # check = [k.shape for k in sym_inputs]
                # print(check)

                prepare_vectors = torch.repeat_interleave(symbol_requests, repeat_recipe, dim=0)

                symbol_signature_stack = torch.cat(symbol_signature_stack, dim=0)

                # print(prepare_vectors.shape)
                # print(sampling_input_stack.shape)
                # print(symbol_signature_stack.shape)
                assert sum(clause_segments) == len(var_vector_list)
                #
                # now we just have to do a dot product between symbol_requests and the signature
                # we need to repeat the symbol_requests enough times
                similarities = torch.einsum('in, in->i', prepare_vectors, symbol_signature_stack)
                # print(similarities.shape)


                # TODO: Is this the culprit?
                indices = []
                segment_counter = 0
                for e, number_of_symbols in enumerate(repeat_recipe):
                    indices += [segment_counter] * number_of_symbols
                    segment_counter += 1
                #
                # print(indices) #
                # assert 2 > 3
                # print(len(indices))
                # TODO make tensor on the correct device immediately [done]
                indices = torch.tensor(indices, device=self.device)
                # TODO: implement temperature here
                # print(similarities)

                if not temperature.item() == 1.0:
                    similarities = similarities / temperature
                # print(similarities)
                softmaxed_similarities = scatter_softmax(similarities, indices)
                # print(softmaxed_similarities.shape)
                split_softmaxes = torch.split(softmaxed_similarities, repeat_recipe.tolist())
                if random_baseline:
                    new_split_softmaxes = []
                    for ds in split_softmaxes:
                        num_items = len(ds)
                        # TODO make tensor on the correct device immediately
                        new_ds = torch.full(ds.shape, torch.tensor(1 / float(num_items), device=self.device))

                        new_split_softmaxes.append(new_ds)

                    split_softmaxes = new_split_softmaxes
                # print([k.shape for k in split_softmaxes])
                # assert 2 > 3
                # print([k.shape for k in split_softmaxes])
                predictions = split_softmaxes
                if optimism_factor != 1:
                    new_preds = []
                    for aa, k in enumerate(predictions):
                        # print(k)
                        new_k = torch.clone(k)
                        new_k[-1] = new_k[-1] / optimism_factor

                        # renormalize (necessary to make the sampling later on make sense)
                        new_k = new_k / torch.sum(new_k)
                        new_preds.append(new_k)
                    predictions = new_preds
                # print(predictions)
                # print(len(predictions))
                # print([torch.topk(f, 3) for f in predictions])
                # print([len(k) for k in var_vector_list])
                # print(len(var_vector_list))

                # TODO For sampling I need to sample here
                top_k_sampling = False
                if top_k_sampling:
                    topk = [torch.topk(f, beam_width) for f in predictions]
                    for probs, actions in topk:
                        # print(actions.tolist())
                        beamlist.append([[pr, [a]] for (pr, a) in zip(probs.tolist(), actions.tolist())])
                        # print(torch.log(probs).tolist())
                    # print(beamlist)
                    # print(len(beamlist))

                else:
                    # Just sample the whole distribution

                    distributions = [torch.distributions.Categorical(f) for f in predictions]

                    # ----
                    # dist_lens = [torch.log(torch.tensor(float(len(k)))) for k in predictions]
                    # ent = torch.mean(torch.tensor([k.entropy().item() for k in distributions]))
                    # norm_ent = torch.mean(torch.tensor([k.entropy().item() / lgn for (k, lgn) in zip(distributions, dist_lens)]))
                    # print("First step entropy")
                    # print(ent)
                    # print("Normalized first step entropy")
                    # print(norm_ent)
                    # ----
                    for distribution, probabilities in zip(distributions, predictions):
                        clause_samples = []
                        for sam_num in range(beam_width):
                            choice = distribution.sample().item()
                            prob = probabilities[choice].item()
                            clause_samples.append([prob, [choice]])
                        beamlist.append(clause_samples)

            elif iter > 0:

                ctime = time.time()
                sampling_input_stack = torch.stack(sampling_input_stack)
                query_input_stack = torch.stack(query_input_stack)
                # print(sampling_input_stack.shape)

                # assert 2 > 3
                #
                prefix_hidden = self.rnn_ln(self.generator(sampling_input_stack))
                # print(h_final.shape)
                hidden_cache.append(prefix_hidden)
                # print(prefix_hidden.shape, query_input_stack.shape)
                next_step_input = torch.cat((prefix_hidden, query_input_stack), dim=1)
                # print(next_step_input.shape)
                # assert 2 > 3
                h_final = self.rnn_ln(self.generator(next_step_input))
                symbol_requests = self.symbol_projector(h_final)
                # print(symbol_requests.shape)
                # print(clause_segments)
                # print(len(symbol_signature_stack))
                # print(symbol_signature_stack[0].shape)
                # TODO make tensor on the correct device immediately [Done]
                repeat_recipe = torch.tensor([k.shape[0] for k in symbol_signature_stack], device=self.device)
                # print([k.shape for k in sym_inputs])
                # print(repeat_recipe)
                # print(repeat_recipe.shape)

                # print(repeat_recipe)
                # print(clause_segments)
                # print(repeat_recipe[75])
                # print(repeat_recipe[76])
                # print(repeat_recipe[77])
                # print(repeat_recipe[75+76])
                # print(repeat_recipe[75 + 77])
                # assert 2 > 3
                # check = [k.shape for k in sym_inputs]
                # print(check)

                prepare_vectors = torch.repeat_interleave(symbol_requests, repeat_recipe, dim=0)

                symbol_signature_stack = torch.cat(symbol_signature_stack, dim=0)
                # print(prepare_vectors.shape)
                # print(sampling_input_stack.shape)
                # print(symbol_signature_stack.shape)
                # assert 2 > 3
                assert sum(clause_segments) == len(var_vector_list)
                #
                # now we just have to do a dot product between symbol_requests and the signature
                # we need to repeat the symbol_requests enough times
                similarities = torch.einsum('in, in->i', prepare_vectors, symbol_signature_stack)
                # TODO: implement tempterature here

                # print(similarities)

                if not temperature.item() == 1.0:
                    similarities = similarities / temperature
                # print(similarities)
                # print(similarities.shape)
                # if iter == 2:
                #     #
                #     torch.set_printoptions(threshold=1000000)
                #     print(similarities)
                #     # #
                #     assert 2 > 3
                indices = []
                segment_counter = 0

                for e, number_of_symbols in enumerate(repeat_recipe):
                    indices += [segment_counter] * number_of_symbols
                    segment_counter += 1
                #
                # print(indices)
                # print(len(indices))
                # TODO make tensor on the correct device immediately [done]
                indices = torch.tensor(indices, device=self.device)
                softmaxed_similarities = scatter_softmax(similarities, indices)
                # print(softmaxed_similarities.shape)


                split_softmaxes = torch.split(softmaxed_similarities, repeat_recipe.tolist())
                # print(split_softmaxes)
                if random_baseline:
                    new_split_softmaxes = []
                    for ds in split_softmaxes:
                        num_items = len(ds)
                        # TODO make tensor on the correct device immediately
                        new_ds = torch.full(ds.shape, torch.tensor(1/float(num_items), device=self.device))

                        new_split_softmaxes.append(new_ds)

                    split_softmaxes = new_split_softmaxes

                # print([k.shape for k in split_softmaxes])
                predictions = split_softmaxes

                # The optimism factor reduces the logit for the termination symbol; this turns down the premise
                # selection capabilities.
                if optimism_factor != 1:
                    new_preds = []
                    for aa, k in enumerate(predictions):
                        # print(k)
                        new_k = torch.clone(k)
                        new_k[-1] = new_k[-1] / optimism_factor
                        # renormalize (necessary to make the sampling later on make sense)
                        new_k = new_k / torch.sum(new_k)
                        new_preds.append(new_k)
                    predictions = new_preds
                # elif sampling_noise != 0:
                #     new_preds = []
                #
                #     for aa, pr in enumerate(predictions):
                #
                #         new_k = torch.clone(k)
                #         new_k = new_k / temperature



                # print(predictions)
                # print(len(predictions))
                # print(predictions[:50])
                # assert 2 > 3
                # print([torch.topk(f, 3) for f in predictions])
                # print([len(k) for k in var_vector_list])
                # print(len(var_vector_list))
                distributions = [(e // beam_width, f) for (e, f) in enumerate(predictions)]
                # print(topk)
                # #
                # -----------
                # dist_ents  = [torch.distributions.Categorical(k[1]).entropy().item() for k in distributions]
                # ent = torch.mean(torch.tensor(dist_ents))
                # print("Non-first step entropy")
                # print(ent)
                # dist_lens = [torch.log(torch.tensor(float(len(k[1])))) for k in distributions]
                # ent = torch.mean(torch.tensor(dist_ents))
                # norm_ent = torch.mean(torch.tensor([e / l for (e, l) in zip(dist_ents, dist_lens)]))
                # print("Normalized non-first step entropy")
                # print(norm_ent)
                # ------------
                print(f"Getting Predictions takes {time.time() - ctime} seconds")
                ctime = time.time()
                new_beamlist = []

                time_deepcopy = 0.0
                time_dist = 0.0
                time_cat = 0.0
                time_sam = 0.0
                time_inner_samp = 0.0
                time_item = 0.0

                # TODO: I can stack the categorical.
                # For each signature size, I can stack easily.
                # TODO put a check that checks whether stacked_dist is the size of sig_size.
                dividers = [k*beam_width for k in clause_segments] # for each clause we have k samples

                # print(sum(dividers), len(distributions))
                assert sum(dividers) == len(distributions)

                samples_list = []
                probs_list = []
                start_index = 0
                for block in dividers:

                    block_dists = distributions[start_index:start_index + block]
                    block_dists = [k[1] for k in block_dists]

                    stacked_dist = torch.stack(block_dists)

                    # print(stacked_dist.shape[1])
                    # print(len(sym_inputs[signature_index]))
                    assert stacked_dist.shape[1] == len(sym_inputs[signature_index])
                    # print(stacked_dist)
                    stacked_cat = torch.distributions.Categorical(stacked_dist)
                    step_index = torch.arange(start=0, end=block, step=1, device=self.device)

                    samples = stacked_cat.sample()
                    # print(samples)
                    # print(samples.shape)
                    # # prob_index = torch.stack((step_index, samples))
                    # print(prob_index)
                    probs = stacked_dist[step_index, samples] # multidimensional indexing
                    # print(probs)
                    # print(stacked_dist.shape)
                    # print(prob_index.shape)
                    # print(probs.shape)
                    samples_list.append(samples)
                    probs_list.append(probs)
                    # print(samples.shape)
                    start_index = start_index + block

                all_samples = torch.cat(samples_list).tolist()
                all_probs = torch.cat(probs_list).tolist()
                # print(all_samples.shape)
                # print(sum(dividers))
                # print(which_problem_indicator)
                assert len(all_samples) == sum(dividers)

                for cn, prefix_sequence in enumerate(beamlist):

                    signature_index = which_problem_indicator[cn]
                    sig_s = len(sym_inputs[signature_index])
                    # print("SIGLEN", sig_s)
                    original_signature_map = list(range(sig_s)) * beam_width

                    new_beams_clause = []
                    # TODO make this sampling
                    for path in range(beam_width):
                        absolute_index = cn * beam_width + path
                        # possible_continuations = distributions[absolute_index]
                        prefix_probability = prefix_sequence[path][0]

                        choice = all_samples[absolute_index]
                        proba = all_probs[absolute_index]


                        old_prefix_beam = copy.deepcopy(beamlist[cn][path][1])
                        # after_dc = time.time()
                        # time_deepcopy += (after_dc - before_dc)


                        prev_hidden_index.append(path)
                        old_prefix_beam.append(choice)
                        new_beams_clause.append([proba, old_prefix_beam])
                    new_beamlist.append(new_beams_clause)


                beamlist = new_beamlist
                print(f"Doing the administration takes {time.time() - ctime} seconds")
                # print(f"Of that, {time_deepcopy} seconds was spent in deepcopy.")
                # print(f"Of that, {time_dist} seconds was spent in distribution.")
                # print(f"Of that, {time_cat} seconds was spent in initializing distribution")
                # print(f"Of that, {time_sam} seconds was spent in sampling from the distribution")
                # print(f"Of that, {time_inner_samp} seconds was spent in inner sample")
                # print(f"Of that, {time_item} seconds was spent in itemizing")
                # assert 2 > 3
            print(f"Iteration {iter} takes {time.time() - stime} seconds")


        return beamlist


    def forward(
        self,
        graph_structure,
        label_info, iterations=11
    ):

        # Initialize all the embeddings
        vector_representations = self.graph_start(graph_structure)

        for i in range(self.num_layers):
            if i == 0:
                nodes, symbols, clauses = self.graph_conv(
                    vector_representations, graph_structure, i
                )

            else:
                if self.repeat_layer:
                    new_nodes, new_symbols, new_clauses = self.graph_conv(
                        (nodes, symbols, clauses), graph_structure, 0
                    )

                else:
                    new_nodes, new_symbols, new_clauses = self.graph_conv(
                        (nodes, symbols, clauses), graph_structure, i
                    )

                # residual_nodes, residual_symbols, residual_clauses = vector_representations
                if self.residual:
                    nodes = nodes + new_nodes
                    symbols = symbols + new_symbols
                    clauses = clauses + new_clauses
                # elif not self.residual and self.normalize()
                else:
                    nodes = new_nodes
                    symbols = new_symbols
                    clauses = new_clauses

                if self.normalization == "batch":
                    nodes = self.node_bn(nodes)
                    symbols = self.symbol_bn(symbols)
                    clauses = self.clause_bn(clauses)
                elif self.normalization == "layer":
                    nodes = self.node_ln(nodes)
                    symbols = self.symbol_ln(symbols)
                    clauses = self.clause_ln(clauses)


        # for each clause, construct:

        # [cls, clause, VARA]
        # [cls, clause, VARA + SYMA, VARB]
        # [cls, clause, VARA + SYMA, VARB + SYMB]

        # construct VARA
        # we need to forward somehow related to the amount of variables in each clause.

        # print(label_info)
        # assert 2 > 3
        # var_index, symbol_index, clause_with_vars_index, label_sequences
        # NOTE THAT CLAUSE i is NOT the index into the clause vectors!!
        var_i, sym_i, clause_i, clause_var_index, labels = label_info

        # assert 2 > 3
        vars_to_query = []
        # print(clause_var_index)
        # print(clause_i)
        # #

        # TODO: If we shuffle clause_var_index' entries, it can't overfit on the variable order anymore
        var_inputs = []
        for e, cl_index in enumerate(clause_i):
            cl_2var_vector = {}
            # associated vars
            for cl in cl_index:
                assoc_vars = clause_var_index[e][cl]

                assoc_vars_i = [var_i[k] for k in assoc_vars]

                assoc_vars_vecs = [nodes[k] for k in assoc_vars_i]
                # print(cl)
                # print(assoc_vars_i)
                # print(assoc_vars_vecs)
                cl_2var_vector[cl] = assoc_vars_vecs

            var_inputs.append(cl_2var_vector)

        # print(var_inputs) # # #
        # print(sym_i)

        # Here we construct the lists that contain the symbol vectors for each problem in the batch
        # We  also add a termination symbol
        sym_inputs = []
        for e, signature in enumerate(sym_i):
            signature_list = []
            for key in signature:
                signature_list.append(symbols[signature[key]])

            signature_list.append(self.termination_symbol)
            sym_inputs.append(torch.stack(signature_list))

        # TODO:
        # Construct for each clause:
        # Two lists of length len(labels) for that clause
        # One with the VARS (use the modulo construction)
        # One with the symbols (just index into the sym_input lists)
        var_vector_list = []
        sym_vector_list = []
        # assert 2 > 3

        clause_segments = []
        for e, problem_clauses in enumerate(var_inputs):
            clause_segments.append(len(problem_clauses))

        for e, problem_clauses in enumerate(var_inputs):

            for f, clause in enumerate(problem_clauses):
                clause_variable_list = []
                clause_symbol_list = []
                num_vars = len(problem_clauses[clause])

                true_symbols = labels[clause]
                # print(clause)
                # print(true_symbols)
                # print("Num VARS: ", num_vars)
                for step in range(iterations):

                    current_vector = problem_clauses[clause][step % num_vars]
                    # print(step % num_vars)
                    clause_variable_list.append(current_vector)
                    if step < len(true_symbols) - 1:
                        # print(step)
                        # print(sym_inputs[e][true_symbols[step]])
                        clause_symbol_list.append(sym_inputs[e][true_symbols[step]])



                var_vector_list.append(clause_variable_list)
                sym_vector_list.append(clause_symbol_list)

        # Now we need to run the RNN in a teacher forcing fashion and cache the hidden states
        # First input (init, VARA, fakesym)
        # Second input (init, VARA, SYMA)
        # Third input (h1, VARB, SYMB) <- notably, all info about previous symbol choices is in h1, so we can cleanly
        # condition P(SYMB1@VARB | previous) vs.  P(SYMB2@VARB | previous)

        # first construct the cache


        # for e, (varr, symm) in enumerate(zip(var_vector_list, sym_vector_list)):
        #
        #     if e == 0:
        #         h0 = self.cls_token

        input_lists = []

        # print(labels)

        # print(len(var_vector_list))
        hidden_cache = []
        for step in range(iterations):
            clause_inputs = []
            if step == 0:
                for cls_num, (var_vectors, sym_vectors) in enumerate(zip(var_vector_list, sym_vector_list)):

                    if not step > len(sym_vectors) - 1: # is the label list long enough
                        # print(var_vectors[step])
                        # print(sym_vectors[step])
                        input_cat = torch.cat((self.cls_token, var_vectors[step], sym_vectors[step]))
                        clause_inputs.append(input_cat)
                    else:
                        # some dummy calculations to make my indexing life easier (probably the next step (predicting new symbols)
                        #  is always
                        # more memory intensive than this one (by a factor of #avg vars in clause), so optimizing here is of marginal value)
                        input_cat = torch.cat((self.cls_token, var_vectors[step], var_vectors[step]))
                        clause_inputs.append(input_cat)
            else:
                for cls_num, (var_vectors, sym_vectors) in enumerate(zip(var_vector_list, sym_vector_list)):

                    if not step > len(sym_vectors) - 1:

                        input_cat = torch.cat((hidden[cls_num], var_vectors[step], sym_vectors[step]))
                        clause_inputs.append(input_cat)
                    else:
                        input_cat = torch.cat((hidden[cls_num], var_vectors[step], var_vectors[step]))
                        clause_inputs.append(input_cat)

            all_inputs = torch.stack(clause_inputs)
            hidden = self.rnn_ln(self.generator(all_inputs))
            hidden_cache.append(hidden)
            # print(hidden.shape)
            # print(len(hidden_cache))
        # print(var_vector_list)
        # print(sym_vector_list)
        # print(labels)
        # print(sym_inputs)

        #
        # print([len(labels[k]) for k in labels])
        # print(np.mean([len(labels[k]) for k in labels]))
        # print(np.mean([1 if len(labels[k]) == 1 else 0 for k in labels]))
        # print(sum([len(labels[k]) for k in labels]) - 100*np.mean([1 if len(labels[k]) == 1 else 0 for k in labels]))
        #
        teacher_force_input_stack = []
        symbol_signature_stack = []
        # print(len(labels))

        which_problem_indicator = []
        for e, k in enumerate(clause_segments):
            for i in range(k):
                which_problem_indicator.append(e)

        signature_index = 0

        for e, clause in enumerate(labels):

            signature_index = which_problem_indicator[e] # ensure we add the signature symbols for the correct problem
            for f, label in enumerate(labels[clause]):
                if f < iterations:
                    if f == 0:

                        ss = torch.cat((self.cls_token, var_vector_list[e][f], self.empty_symbol))
                        teacher_force_input_stack.append(ss)

                    else:
                        # print(f)
                        # print(hidden_cache[f - 1][e])
                        # print(len(var_vector_list))
                        # print(len(var_vector_list[e]))
                        # print(len(var_vector_list[e][f]))
                        ss = torch.cat((hidden_cache[f - 1][e], var_vector_list[e][f], self.empty_symbol))
                        teacher_force_input_stack.append(ss)



                    symbol_signature_stack.append(sym_inputs[signature_index])



        # print(len(teacher_force_input_stack))
        # print(float(len(teacher_force_input_stack)) / len(labels))
        cc = 0
        for e, clause in enumerate(labels):
            for f, label in enumerate(labels[clause]):
                if f < iterations:
                    cc += 1
        # print(cc)

        teacher_force_input_stack = torch.stack(teacher_force_input_stack)
        # print(teacher_force_input_stack.shape)

        h_final = self.rnn_ln(self.generator(teacher_force_input_stack))
        # print(h_final.shape)
        symbol_requests = self.symbol_projector(h_final)
        # print(symbol_requests.shape)
        # print(clause_segments)
        # print(len(symbol_signature_stack))
        # print(symbol_signature_stack[0].shape)
        repeat_recipe = torch.tensor([k.shape[0] for k in symbol_signature_stack], device=self.device)
        # print(repeat_recipe)
        # check = [k.shape for k in sym_inputs]
        # print(check)

        prepare_vectors = torch.repeat_interleave(symbol_requests, repeat_recipe, dim=0)

        symbol_signature_stack = torch.cat(symbol_signature_stack, dim=0)
        # print(prepare_vectors.shape)
        # print(symbol_signature_stack.shape)
        assert sum(clause_segments) == len(var_vector_list)
        #
        # now we just have to do a dot product between symbol_requests and the signature
        # we need to repeat the symbol_requests enough times
        similarities = torch.einsum('in, in->i', prepare_vectors, symbol_signature_stack)
        # print(similarities.shape)
        #
        indices = []
        segment_counter = 0
        for e, number_of_symbols in enumerate(repeat_recipe):
            indices += [segment_counter] * number_of_symbols
            segment_counter += 1

        # print(indices)
        # print(len(indices))
        # indices = torch.tensor(indices).to(device)
        # softmaxed_similarities = scatter_softmax(similarities, indices)
        # print(softmaxed_similarities.shape)
        split_logits = torch.split(similarities, repeat_recipe.tolist())
        # print([k.shape for k in split_softmaxes])
        predictions = split_logits

        return predictions


    def calculate_loss(self, loss, preds, label_info, iterations=11, sample_normalize_loss = True):

        """
        Calculate a loss - preds is split into softmax outputs, use no reduction and sum all of the losses
        """

        boost_positives = False
        # print("These are the predictions")
        # print([k.shape for k in preds])
        var_i, sym_i, clause_i, clause_var_index, labels_seqs = label_info
        # print(labels_seqs)
        problem_segments_clauses = [len(k) for k in clause_i]
        problem_segments_symbols = [len(k) + 1 for k in sym_i]  # add 1 for termination symbol
        # print(sym_i)
        # print(problem_segments_clauses)
        # print(problem_segments_symbols)

        # relabel -1 in label to last of signature #

        # Count how many decision points are in each thing in the batch.



        batch_weights = [0 for k in range (16)] # Here we count how many decision points are in each sample
        batch_sample_indices = [] # Here we keep indices that tell us which losses to sum.



        new_labels_seqs = {}
        for key in labels_seqs:
            batch_index = int(key[1:].split("C")[0])  # B0Ci_0_20 -> 0
            new_seq = []
            batch_weights[batch_index] += len(labels_seqs[key])

            for symbol_chosen in labels_seqs[key]:
                batch_sample_indices.append(batch_index)
                if symbol_chosen == -1:
                    new_seq.append(problem_segments_symbols[batch_index] - 1)
                else:
                    new_seq.append(symbol_chosen)

            new_labels_seqs[key] = new_seq

        # print(new_labels_seqs)
        #
        # flattened labels
        total_loss = 0
        flat_labels = []
        position_annotation = []


        # This seems to be quadratic over the full batch, not per graph; could be bad
        for clause in new_labels_seqs:
            for f, lab in enumerate(new_labels_seqs[clause]):
                if f < iterations:
                    flat_labels.append(lab)
                    position_annotation.append(f)

        flat_labels_orig = []

        for clause in labels_seqs:
            for f, lab in enumerate(labels_seqs[clause]):
                if f < iterations:
                    # print(lab)
                    flat_labels_orig.append(lab)

        # assert 2 > 3
        #
        exactly_right = 0
        denom = 0
        exactly_right_vector = [0] * iterations
        denom_vector = [0] * iterations

        true_positives = 0
        all_positives = 0

        true_negatives = 0
        all_negatives = 0

        separate_accuracies = [[] for k in range(16)]
        if sample_normalize_loss:
            separate_losses = [[] for k in range(16)]

        for y_hat, true_choice, pos, orig_true, batch_ind in zip(preds, flat_labels, position_annotation, flat_labels_orig, batch_sample_indices):
            ## TODO reweigth this if label == -1 and pos = 0 ; loss might be overwhelmed by trying to prevent "false positives", but those are actually super cheap for us (SAT solver)
            # print(y_hat.reshape(1, -1))
            # print(torch.tensor(true_choice).reshape(1))
            l = loss(y_hat.reshape(1, -1), torch.tensor(true_choice, device=self.device).reshape(1))


            num_choices = len(y_hat)
            # print(y_hat.shape, len(y_hat))
            model_choice = torch.argmax(y_hat).item()
            if model_choice == true_choice:
                exactly_right += 1
                exactly_right_vector[pos] += 1

                separate_accuracies[batch_ind].append(1)
            else:
                separate_accuracies[batch_ind].append(0)
            # negatives
            # print(true_choice, orig_true)
            if orig_true == -1:
                if model_choice == num_choices - 1:
                    # True negative
                    # print("TRUE NEGATIVES")
                    true_negatives += 1
                    all_negatives += 1
                else:
                    # False positive
                    # print("FALSE POSITIVE")
                    all_negatives += 1
            # positives
            else:
                if boost_positives:
                    l = l * 100
                if model_choice == num_choices - 1:
                    # False negative
                    # print("FALSE NEGATIVE")
                    all_positives += 1
                else:
                    # True positive
                    # print("TRUE POSITIVE")
                    true_positives += 1
                    all_positives += 1

            if sample_normalize_loss:
                separate_losses[batch_ind] += [l]
            else:
                total_loss += l

            denom += 1
            denom_vector[pos] += 1


        if sample_normalize_loss:
            for sample_loss_list in separate_losses:
                sample_loss = torch.tensor(0, device=self.device)
                num_decision_points = len(sample_loss_list)
                # print(len(sample_loss_list))

                for loss_point in sample_loss_list:
                    sample_loss = sample_loss + loss_point

                if not num_decision_points == 0:
                    sample_loss = sample_loss / torch.tensor(num_decision_points, device=self.device)
                # print("Using sample_weighted Loss")
                # print(sample_loss)
                total_loss = total_loss + sample_loss

        # print("total loss: ")
        # print(total_loss.reshape(1))
        # print(f"Fractions right:")
        # fr = [np.mean(k) for k in separate_accuracies if not len(k) == 0]

        # print(fr)
        # print(np.median(fr))
        separate_accuracies = [k for k in separate_accuracies if not len(k) == 0]

        return total_loss.reshape(1), exactly_right, denom, exactly_right_vector, denom_vector, (
        true_positives, all_positives, true_negatives, all_negatives), separate_accuracies

    def train_step(self, batch, optimizer, loss_function, balanced_loss=True):

        graph_info, label_info = batch

        # Structure of this data:
        # Labels[0] contain length of segments for the premises and conjecture! These CAN be more than
        # one "clause" long so you have to collapse them
        # Labels[1] contains the label of each premise!

        graph_structure, lens = graph_info

        # labels = [torch.tensor(k[1], dtype=torch.float) for k in output_info]
        # clause_segments = [torch.tensor(k[0], dtype=torch.long) for k in output_info]

        # labels = torch.cat(labels)

        optimizer.zero_grad()
        y_hat = self.forward(graph_structure, (label_info[1], label_info[2]))
        # print("Y_HAT_SHAPE")
        # print(y_hat.shape)
        # print(label_info[1])
        # print(label_info[2])
        # torch.set_printoptions(threshold=10_000)
        # print(label_info[0])
        # print(y_hat)
        # TODO cut out the zeros, they should not show up in the loss
        # print([len(k) for k in label_info[1]])
        # print([len(k) for k in label_info[2]])

        # Now cut up these tensors

        tensor_list_results = []
        tensor_list_labels = []

        for e, (vardict, symdict) in enumerate(zip(label_info[1], label_info[2])):
            # print(len(vardict), len(symdict))
            # results_snipped = y_hat[e, :len(vardict), :len(symdict)]
            labels_snipped = label_info[0][e, :len(vardict), :len(symdict)]
            # print(results_snipped.shape)
            # print(labels_snipped.shape)

            # tensor_list_results.append(results_snipped.reshape(-1))
            tensor_list_results.append(y_hat[e].reshape(-1))
            tensor_list_labels.append(labels_snipped.reshape(-1))

        # print(tensor_list_results)
        # print(tensor_list_labels)

        results = torch.cat(tensor_list_results)
        labels = torch.cat(tensor_list_labels).to(self.device)

        labels_bool = torch.as_tensor(labels, dtype=torch.bool)
        pos_mask = labels_bool == True
        neg_mask = labels_bool == False


        if not balanced_loss:

            loss = loss_function(results, labels)
        else:

            num_pos = sum(pos_mask).item()
            num_neg = sum(neg_mask).item()
            # print(num_pos, num_neg)
            # print(pos_mask)
            # print(neg_mask)
            # print(y_hat.shape)
            # print(labels.shape)
            loss = loss_function(results, labels)
            # print(loss.shape, pos_mask.shape, neg_mask.shape)
            if not num_pos == 0:
                loss_on_pos = loss[pos_mask].mean()
            else:
                loss_on_pos = 0

            if not num_neg == 0:
                loss_on_neg = loss[neg_mask].mean()
            else:
                loss_on_neg = 0

            # print(loss_on_pos, loss_on_neg)
            loss = (loss_on_pos + loss_on_neg) / 2.0

        tpr_mask = results[pos_mask] > 0
        # print(tpr_mask
        if not torch.all(~tpr_mask):
            tpr = torch.as_tensor(tpr_mask, dtype=torch.float).mean().item()
        else:
            tpr = 0.0
        # print(tpr)
        tnr_mask = results[neg_mask] < 0
        if not torch.all(~tnr_mask):
            # if not all predictions are positive
            tnr = torch.as_tensor(tnr_mask, dtype=torch.float).mean().item()
        else:
            # if all preds were positive, there are no true negatives that were caught
            tnr = 0.0
        #
        loss.backward()
        optimizer.step()

        return (loss.item(), tpr, tnr)


    def evaluation(self, batch, loss_function):
        """"This function is supposed to give back the tpr, tnr, the mean of those and the loss on a validation set (5%)
        Without doing any backprop or parameter changes
        """

        balanced_loss = True

        graph_info, label_info = batch

        # Structure of this data:
        # Labels[0] contain length of segments for the premises and conjecture! These CAN be more than
        # one "clause" long so you have to collapse them
        # Labels[1] contains the label of each premise!

        graph_structure, lens = graph_info


        y_hat = self.forward(graph_structure, (label_info[1], label_info[2]))
        # labels = [torch.tensor(k[1], dtype=torch.float) for k in output_info]
        # clause_segments = [torch.tensor(k[0], dtype=torch.long) for k in output_info]


        # clause_segments = torch.cat(clause_segments)
        tensor_list_results = []
        tensor_list_labels = []

        for e, (vardict, symdict) in enumerate(zip(label_info[1], label_info[2])):
            # print(len(vardict), len(symdict))
            # results_snipped = y_hat[e, :len(vardict), :len(symdict)]
            labels_snipped = label_info[0][e, :len(vardict), :len(symdict)]
            # print(results_snipped.shape)
            # print(labels_snipped.shape)

            tensor_list_results.append(y_hat[e].reshape(-1))
            tensor_list_labels.append(labels_snipped.reshape(-1))

        # print(tensor_list_results)
        # print(tensor_list_labels)
        results = torch.cat(tensor_list_results)
        labels = torch.cat(tensor_list_labels).to(self.device)

        labels_bool = torch.as_tensor(labels, dtype=torch.bool)
        pos_mask = labels_bool == True
        neg_mask = labels_bool == False

        if not balanced_loss:

            loss = loss_function(results, labels)
        else:

            num_pos = sum(pos_mask).item()
            num_neg = sum(neg_mask).item()

            loss = loss_function(results, labels)
            if not num_pos == 0:
                loss_on_pos = loss[pos_mask].mean()
            else:
                loss_on_pos = 0

            if not num_neg == 0:
                loss_on_neg = loss[neg_mask].mean()
            else:
                loss_on_neg = 0

            loss = (loss_on_pos + loss_on_neg) / 2.0

            # Now calculate the tpr and tnr
            # How many preds for the pos were > 0?
        tpr_mask = results[pos_mask] > 0
        # print(tpr_mask)
        if not torch.all(~tpr_mask):
            tpr = torch.as_tensor(tpr_mask, dtype=torch.float).mean().item()
        else:
            tpr = 0.0
        # print(tpr)
        tnr_mask = results[neg_mask] < 0
        if not torch.all(~tnr_mask):
            # if not all predictions are positive
            tnr = torch.as_tensor(tnr_mask, dtype=torch.float).mean().item()
        else:
            # if all preds were positive, there are no true negatives that were caught
            tnr = 0.0
        # print(tnr_mask)
        # print(tnr)

        # print(f"YHAT SHAPE: {y_hat.shape}, {len(y_hat)}")
        return (loss.item(), tpr, tnr), len(y_hat)


def construct_labels(batched_indices, batch_of_labels, training=True, shuffle_multi=False, shuffle_vars=False, sample_one_multi=False):
    # print("--------------------")
    # print(batched_indices)
    # print(batch_of_labels)
    # print(len(batch_of_labels))
    # var index and sym index are already prefixed
    var_index, symbol_index, clause_index, clause_var_index = batched_indices


    # print(clause_index)
    # print(var_index)
    # print(clause_var_index)
    # print(batch_of_labels)
    # assert 2> 3
    # local_symbol_index
    # print(symbol_index)

    if shuffle_vars:
        # TODO make a new version o clause_index, that has the variables in different order. Then use this to construct a new int_label_dict, that has the answers in the correct, new order.
        shuffled_clause_var_index = []

        for problem_clausevardict in clause_var_index:
            shuffled_problem_clause_var_dict = {}
            for clause in problem_clausevardict:
                varlist = problem_clausevardict[clause]
                if len(varlist) == 1:
                    # only 1 variable so just copy
                    shuffled_problem_clause_var_dict[clause] = copy.deepcopy(varlist)
                elif len(varlist) > 1:
                    shuffled_problem_clause_var_dict[clause] = copy.deepcopy(varlist)
                    random.shuffle(shuffled_problem_clause_var_dict[clause])
                else:
                    raise ValueError("clause in var_index but has no vars")
            shuffled_clause_var_index.append(shuffled_problem_clause_var_dict)
            assert len(shuffled_problem_clause_var_dict) == len(problem_clausevardict)

    # print(shuffled_clause_var_index)
    # assert 2 >3
    if training:
        local_symdict = [{k:e for (e, k) in enumerate(ko.keys())} for ko in symbol_index]
        # print(local_symdict)
        # assert 2 > 3
        if not shuffle_vars:
            new_label_dict_list = []
            new_label_int_dict_list = []
            for e, label_dict in enumerate(batch_of_labels):

                prefix = f"B{e}"
                new_label_dict = {}
                new_label_int_dict = {}
                for key in label_dict:
                    new_instantiation_list = []
                    new_instantiation_int_list = []
                    for instantiation in label_dict[key]:
                        new_instantiation = {}
                        new_int_instantiation = {}
                        for variable in instantiation:

                            new_instantiation[prefix+variable] = prefix + instantiation[variable]
                            new_int_instantiation[prefix+variable] = local_symdict[e][prefix+instantiation[variable]]

                        new_instantiation_list.append(new_instantiation)
                        new_instantiation_int_list.append(new_int_instantiation)

                    new_label_dict[prefix + key] = new_instantiation_list
                    new_label_int_dict[prefix + key] = new_instantiation_int_list
                new_label_dict_list.append(new_label_dict)
                new_label_int_dict_list.append(new_label_int_dict)
            # print(new_label_dict_list)
            # print(new_label_int_dict_list)
        else:
            new_label_dict_list = []
            new_label_int_dict_list = []
            for e, label_dict in enumerate(batch_of_labels):
                # print("--------------")
                # print(e) #
                # print(label_dict)
                prefix = f"B{e}"
                new_label_dict = {}
                new_label_int_dict = {}
                for clause_di in label_dict:
                    new_instantiation_list = []
                    new_instantiation_int_list = []
                    for instantiation in label_dict[clause_di]:
                        new_instantiation = {}
                        new_int_instantiation = {}
                        # # #
                        # print(local_symdict[e])
                        # print(instantiation)
                        if clause_di.startswith("Cc"):
                            print(e)
                            assert 2 > 3
                        for variable in shuffled_clause_var_index[e][prefix+clause_di]:
                            # print(shuffled_clause_var_index)
                            print(variable)
                            # # # #

                            print(instantiation)
                            new_instantiation[variable] = instantiation[variable.lstrip(prefix)]
                            new_int_instantiation[variable] = local_symdict[e][prefix + instantiation[variable.lstrip(prefix)]]

                        new_instantiation_list.append(new_instantiation)
                        new_instantiation_int_list.append(new_int_instantiation)
                        # print(new_instantiation)
                        # print(new_int_instantiation)
                    new_label_dict[prefix + clause_di] = new_instantiation_list
                    new_label_int_dict[prefix + clause_di] = new_instantiation_int_list
                    # print(new_instantiation_list)
                    # print(new_instantiation_int_list)
                new_label_dict_list.append(new_label_dict)
                new_label_int_dict_list.append(new_label_int_dict)

        # print("------------------------")
        # print(shuffled_clause_var_index)
        # print(new_label_dict_list)
        # print(new_label_int_dict_list)
        # assert 2 > 3
        # Okay, now we have to set up the labels.
        # Remember that we might need a shuffle function for the order in case of multi-instantiation clauses.
        # But the labels for the non-instantiated clauses should immediately go to the termination signal.
        # print(new_label_dict_list)
        # print(new_label_int_dict_list)
        # print(var_index)
        # print(clause_index)
        # assert 2 > 3
        # For every clause, we need to construct the symbol labels (let's assume A, B, C ordering for now)
        # If we don't assume A, B, C ordering:
        # We need to

        # batch_labels = {k:[] for k in ci for ci in clause_index}
        batch_labels = {}
        for ci in clause_index:
            for k in ci:
                batch_labels[k] = []

        # Some info generation
        multi_clauses = 0
        single_clauses = 0

        garbage_clauses = 0

        multi_inst_len_list = []
        single_clause_num_var_list = []
        multi_clause_num_var_list = []
        num_var_in_single_total = 0
        num_var_in_multi_total = 0

        # making sequences

        # Want to count how many decision points there are in each sample in the batch

        decision_points = []
        for e, problem_clauses in enumerate(clause_index):
            batch_sample_dec_points = 0
            for clause in problem_clauses:
                if clause not in new_label_int_dict_list[e]:
                    batch_labels[clause] += [-1]
                    garbage_clauses += 1
                else:
                    # if multiple instantiations
                    if len(new_label_int_dict_list[e][clause]) > 1:
                        # print(new_label_int_dict[e][clause])
                        # assert 2 > 3 # put in to enforce single inst exp
                        multi_clauses += 1
                        multi_inst_len_list.append(len(new_label_int_dict_list[e][clause]))
                        all_clause_instantiations = new_label_int_dict_list[e][clause]
                        # print(all_clause_instantiations)
                        if shuffle_multi:
                            random.shuffle(all_clause_instantiations)
                        sample_one_multi = False
                        if sample_one_multi:
                            all_clause_instantiations = random.sample(all_clause_instantiations, k=1)
                        multi_clause_num_var_list.append(len(all_clause_instantiations[0]))
                        num_var_in_multi_total += (len(all_clause_instantiations[0]) * len(all_clause_instantiations))
                        for instantiation in all_clause_instantiations:

                            for variable in instantiation:
                                batch_labels[clause] += [instantiation[variable]]

                        batch_labels[clause] += [-1]
                    else:
                        # only 1 instantiation
                        single_clauses += 1

                        instantiation = new_label_int_dict_list[e][clause][0]
                        single_clause_num_var_list.append(len(instantiation))
                        num_var_in_single_total += len(instantiation)
                        for variable in instantiation:
                            # print(instantiation)
                            # print(variable)
                            # assert 2 > 3
                            batch_labels[clause] += [instantiation[variable]]

                        batch_labels[clause] += [-1]

                batch_sample_dec_points += len(batch_labels[clause])
            decision_points.append(batch_sample_dec_points)
        global aa
        global bb
        global cc
        aa += num_var_in_multi_total
        bb += num_var_in_single_total
        cc += garbage_clauses


        print("Distribution of decision points:")
        print(decision_points)
        print(f"> 1 inst: {multi_clauses} || 1 inst: {single_clauses} || 0 inst: {garbage_clauses}")
        print(f"Mean # of > 1 insts {np.mean(multi_inst_len_list)} || Mean # of vars in 1 inst: {np.mean(single_clause_num_var_list)}")
        print(f"Mean # of vars in multi inst: {np.mean(multi_clause_num_var_list)}")
        print(f"# LP in multi: {num_var_in_multi_total} || # LP in single total: {num_var_in_single_total} || # LP in 0 total: {garbage_clauses}")
        print(f"Running fraction of decisions happening in multi-inst: {aa / (float(aa + bb))}")
        print(f"Total LP Multi: {aa} || Single: {bb} || Zero: {cc}")

        if shuffle_vars:
            return var_index, symbol_index, clause_index, shuffled_clause_var_index, batch_labels
        else:
            return var_index, symbol_index, clause_index, clause_var_index, batch_labels
    else:
        return var_index, symbol_index, clause_index, clause_var_index, -1


def load_cnf_labels(cnf_loc, label_loc):
    # print(cnf_loc)
    x,  (vars, syms, clauses) = load_cnf_only(cnf_loc)
    # print(clauses)
    # print(len(clauses))
    # assert 2 > 3
    res = load_labels_conditional(label_loc)

    return ((x, vars, syms, clauses), res)

# def beam_search(graph, model, iterationsbeam_width=2, return_sequences=2):
#     # TODO think about how to cache
import gc
def debug_gpu():
    # Debug out of memory bugs.
    tensor_list = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tensor_list.append(obj)
        except:
            pass
    print(f'Count of tensors = {len(tensor_list)}.')

