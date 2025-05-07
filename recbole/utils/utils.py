# -*- coding: utf-8 -*-
# @Time   : 2020/7/17
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2021/3/8, 2022/7/12, 2023/2/11
# @Author : Jiawei Guan, Lei Wang, Gaowei Zhang
# @Email  : guanjw@ruc.edu.cn, zxcptss@gmail.com, zgw2022101006@ruc.edu.cn

"""
recbole.utils.utils
################################
"""

import h5py
import datetime
import importlib
import os
import random
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from texttable import Texttable


from recbole.utils.enum_type import ModelType


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def ensure_dir(dir_path):
    r"""Make sure the directory exists, if it does not exist, create it

    Args:
        dir_path (str): directory path

    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_model(model_name):
    r"""Automatically select model class based on model name

    Args:
        model_name (str): model name

    Returns:
        Recommender: model class
    """
    model_submodule = [
        "general_recommender",
        "context_aware_recommender",
        "sequential_recommender",
        "knowledge_aware_recommender",
        "exlib_recommender",
    ]

    model_file_name = model_name.lower()
    model_module = None
    for submodule in model_submodule:
        module_path = ".".join(["recbole.model", submodule, model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            break

    if model_module is None:
        raise ValueError(
            "`model_name` [{}] is not the name of an existing model.".format(model_name)
        )
    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer(model_type, model_name):
    r"""Automatically select trainer class based on model type and model name

    Args:
        model_type (ModelType): model type
        model_name (str): model name

    Returns:
        Trainer: trainer class
    """
    try:
        return getattr(
            importlib.import_module("recbole.trainer"), model_name + "Trainer"
        )
    except AttributeError:
        if model_type == ModelType.KNOWLEDGE:
            return getattr(importlib.import_module("recbole.trainer"), "KGTrainer")
        elif model_type == ModelType.TRADITIONAL:
            return getattr(
                importlib.import_module("recbole.trainer"), "TraditionalTrainer"
            )
        else:
            return getattr(importlib.import_module("recbole.trainer"), "Trainer")


def early_stopping(value, best, cur_step, max_step, bigger=True, epoch_idx=None):
    r"""validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value >= best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value <= best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    if epoch_idx < 50:
        stop_flag = False
    return best, cur_step, stop_flag, update_flag



def calculate_valid_score(valid_result, valid_metric=None):
    r"""return valid score from valid result

    Args:
        valid_result (dict): valid result
        valid_metric (str, optional): the selected metric in valid result for valid score

    Returns:
        float: valid score
    """
    if valid_metric:
        if valid_metric == 'loss':
            return (valid_result[valid_metric] * -1)
        return valid_result[valid_metric]
    else:
        return valid_result["Recall@10"]
    


def dict2str(result_dict):
    r"""convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    return "    ".join(
        [str(metric) + " : " + str(value) for metric, value in result_dict.items()]
    )


def init_seed(seed, reproducibility):
    r"""init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_tensorboard(logger):
    r"""Creates a SummaryWriter of Tensorboard that can log PyTorch models and metrics into a directory for
    visualization within the TensorBoard UI.
    For the convenience of the user, the naming rule of the SummaryWriter's log_dir is the same as the logger.

    Args:
        logger: its output filename is used to name the SummaryWriter's log_dir.
                If the filename is not available, we will name the log_dir according to the current time.

    Returns:
        SummaryWriter: it will write out events and summaries to the event file.
    """
    base_path = "log_tensorboard"

    dir_name = None
    for handler in logger.handlers:
        if hasattr(handler, "baseFilename"):
            dir_name = os.path.basename(getattr(handler, "baseFilename")).split(".")[0]
            break
    if dir_name is None:
        dir_name = "{}-{}".format("model", get_local_time())

    dir_path = os.path.join(base_path, dir_name)
    writer = SummaryWriter(dir_path)
    return writer


def get_gpu_usage(device=None):
    r"""Return the reserved memory and total memory of given device in a string.
    Args:
        device: cuda.device. It is the device that the model run on.

    Returns:
        str: it contains the info about reserved memory and total memory of given device.
    """

    reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3

    return "{:.2f} G/{:.2f} G".format(reserved, total)


def get_flops(model, dataset, device, logger, transform, verbose=False):
    r"""Given a model and dataset to the model, compute the per-operator flops
    of the given model.
    Args:
        model: the model to compute flop counts.
        dataset: dataset that are passed to `model` to count flops.
        device: cuda.device. It is the device that the model run on.
        verbose: whether to print information of modules.

    Returns:
        total_ops: the number of flops for each operation.
    """
    if model.type == ModelType.DECISIONTREE:
        return 1
    if model.__class__.__name__ == "Pop":
        return 1

    import copy

    model = copy.deepcopy(model)

    def count_normalization(m, x, y):
        x = x[0]
        flops = torch.DoubleTensor([2 * x.numel()])
        m.total_ops += flops

    def count_embedding(m, x, y):
        x = x[0]
        nelements = x.numel()
        hiddensize = y.shape[-1]
        m.total_ops += nelements * hiddensize

    class TracingAdapter(torch.nn.Module):
        def __init__(self, rec_model):
            super().__init__()
            self.model = rec_model

        def forward(self, interaction):
            return self.model.predict(interaction)

    custom_ops = {
        torch.nn.Embedding: count_embedding,
        torch.nn.LayerNorm: count_normalization,
    }
    wrapper = TracingAdapter(model)
    inter = dataset[torch.tensor([1])].to(device)
    inter = transform(dataset, inter)
    inputs = (inter,)
    from thop.profile import register_hooks
    from thop.vision.basic_hooks import count_parameters

    handler_collection = {}
    fn_handles = []
    params_handles = []
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m: nn.Module):
        m.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64))
        m.register_buffer("total_params", torch.zeros(1, dtype=torch.float64))

        m_type = type(m)

        fn = None
        if m_type in custom_ops:
            fn = custom_ops[m_type]
            if m_type not in types_collection and verbose:
                logger.info("Customize rule %s() %s." % (fn.__qualname__, m_type))
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
            if m_type not in types_collection and verbose:
                logger.info("Register %s() for %s." % (fn.__qualname__, m_type))
        else:
            if m_type not in types_collection and verbose:
                logger.warning(
                    "[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params."
                    % m_type
                )

        if fn is not None:
            handle_fn = m.register_forward_hook(fn)
            handle_paras = m.register_forward_hook(count_parameters)
            handler_collection[m] = (
                handle_fn,
                handle_paras,
            )
            fn_handles.append(handle_fn)
            params_handles.append(handle_paras)
        types_collection.add(m_type)

    prev_training_status = wrapper.training

    wrapper.eval()
    wrapper.apply(add_hooks)

    with torch.no_grad():
        wrapper(*inputs)

    def dfs_count(module: nn.Module, prefix="\t"):
        total_ops, total_params = module.total_ops.item(), 0
        ret_dict = {}
        for n, m in module.named_children():
            next_dict = {}
            if m in handler_collection and not isinstance(
                m, (nn.Sequential, nn.ModuleList)
            ):
                m_ops, m_params = m.total_ops.item(), m.total_params.item()
            else:
                m_ops, m_params, next_dict = dfs_count(m, prefix=prefix + "\t")
            ret_dict[n] = (m_ops, m_params, next_dict)
            total_ops += m_ops
            total_params += m_params

        return total_ops, total_params, ret_dict

    total_ops, total_params, ret_dict = dfs_count(wrapper)

    # reset wrapper to original status
    wrapper.train(prev_training_status)
    for m, (op_handler, params_handler) in handler_collection.items():
        m._buffers.pop("total_ops")
        m._buffers.pop("total_params")
    for i in range(len(fn_handles)):
        fn_handles[i].remove()
        params_handles[i].remove()

    return total_ops


def list_to_latex(convert_list, bigger_flag=True, subset_columns=[]):
    result = {}
    for d in convert_list:
        for key, value in d.items():
            if key in result:
                result[key].append(value)
            else:
                result[key] = [value]

    df = pd.DataFrame.from_dict(result, orient="index").T

    if len(subset_columns) == 0:
        tex = df.to_latex(index=False)
        return df, tex

    def bold_func(x, bigger_flag):
        if bigger_flag:
            return np.where(x == np.max(x.to_numpy()), "font-weight:bold", None)
        else:
            return np.where(x == np.min(x.to_numpy()), "font-weight:bold", None)

    style = df.style
    style.apply(bold_func, bigger_flag=bigger_flag, subset=subset_columns)
    style.format(precision=4)

    num_column = len(df.columns)
    column_format = "c" * num_column
    tex = style.hide(axis="index").to_latex(
        caption="Result Table",
        label="Result Table",
        convert_css=True,
        hrules=True,
        column_format=column_format,
    )

    return df, tex


def get_environment(config):
    gpu_usage = (
        get_gpu_usage(config["device"])
        if torch.cuda.is_available() and config["use_gpu"]
        else "0.0 / 0.0"
    )

    import psutil

    memory_used = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    memory_total = psutil.virtual_memory()[0] / 1024**3
    memory_usage = "{:.2f} G/{:.2f} G".format(memory_used, memory_total)
    cpu_usage = "{:.2f} %".format(psutil.cpu_percent(interval=1))
    """environment_data = [
        {"Environment": "CPU", "Usage": cpu_usage,},
        {"Environment": "GPU", "Usage": gpu_usage, },
        {"Environment": "Memory", "Usage": memory_usage, },
    ]"""

    table = Texttable()
    table.set_cols_align(["l", "c"])
    table.set_cols_valign(["m", "m"])
    table.add_rows(
        [
            ["Environment", "Usage"],
            ["CPU", cpu_usage],
            ["GPU", gpu_usage],
            ["Memory", memory_usage],
        ]
    )

    return table


def get_item_titles(tensor, df):
    """
    Given a tensor of any shape, return the titles corresponding to the IDs
    in 'tensor', skipping any elements that are 0. For IDs not found in the 
    DataFrame, return "undefined". The shape of the returned list of lists matches
    the shape of the input tensor.
    
    :param tensor: The tensor (PyTorch, NumPy array, etc.) containing item IDs.
    :param df: The DataFrame containing the 'item_id:token' and 'movie_title:token' columns.
    :return: A nested list of movie titles corresponding to all IDs in 'tensor'.
    """
    # Convert tensor to list of lists to preserve shape
    tensor_as_list = tensor.tolist()
    # Create a dictionary for quick lookup of item titles
    id_to_title = dict(zip(df['item_id:token'], df['movie_title:token_seq']))

    # Replace IDs with movie titles, preserving the shape
    result = []
    if tensor.dim() == 1:
        for id in tensor_as_list:
            result.append(id_to_title.get(id, "undefined"))
    else:
        for row in tensor_as_list:
            # Use `.get(item_id, "undefined")` to handle missing IDs
            titles_row = [id_to_title.get(item_id, "undefined") for item_id in row if item_id != 0]
            result.append(titles_row)
        
    return result



def label_popular_items():
    # Load the data
    data = pd.read_csv(r'./dataset/ml-1m/interactions_remapped.csv', encoding='latin1')
    titles_data = pd.read_csv(r'./dataset/ml-1m/items_remapped.csv', encoding='latin1')

    # Calculate interaction counts per item
    item_interactions = data['item_id:token'].value_counts().reset_index()
    item_interactions.columns = ['item_id:token', 'interaction_count']

    # Sort items by interaction count in descending order
    item_interactions = item_interactions.sort_values(by='interaction_count', ascending=False)

    # Compute cumulative interaction percentage
    item_interactions['cumulative_interaction'] = item_interactions['interaction_count'].cumsum()
    total_interactions = item_interactions['interaction_count'].sum()
    item_interactions['cumulative_percentage'] = item_interactions['cumulative_interaction'] / total_interactions

    # Determine popularity labels based on cumulative percentage
    item_interactions['popularity_label'] = 0
    item_interactions.loc[item_interactions['cumulative_percentage'] <= 0.2, 'popularity_label'] = 1  # Top 20%
    item_interactions.loc[item_interactions['cumulative_percentage'] >= 0.8, 'popularity_label'] = -1  # Bottom 20%

    # Merge with item titles data
    output_df = pd.merge(item_interactions, titles_data, how='left', on='item_id:token')

    # Select relevant columns
    output_df = output_df[['item_id:token', 'movie_title:token_seq', 'popularity_label', 'interaction_count']]

    # Sort by popularity label and interaction count
    output_df = output_df.sort_values(by=['popularity_label', 'interaction_count'], ascending=[False, False])

    # Save the output to a CSV file
    output_df.to_csv(r"./dataset/ml-1m/item_popularity_labels_with_titles.csv", index=False)

    print("Popularity labels with titles saved to 'item_popularity_labels_with_titles.csv'")



import math
def save_user_popularity_score(alpha, user_ids, sequences):
    sequences = [seq.cpu().numpy().tolist() for seq in sequences]
    user_ids = [id.item() for id in user_ids]
    
    """
    Updates an HDF5 file with new user IDs and sequences.
    
    Args:
        file_path (str): Path to the HDF5 file.
        user_ids (list): List of user IDs (length [N]).
        sequences (list or numpy array): List or array of sequences (shape [N, 50]).
    """
    # Ensure sequences are lists if they are tensors (PyTorch or NumPy)
    
    item_labels = pd.read_csv("./dataset/ml-1m/item_popularity_labels_with_titles.csv")
    # Open the HDF5 file in append mode (create if it doesn't exist)
    with h5py.File('user_popularity_scores.h5', "a") as f:
        for user_id, user_sequences in zip(user_ids, sequences):
            # Ensure the group for the user exists
            if str(user_id) not in f:
                f.create_group(str(user_id))
            user_group = f[str(user_id)]
            filtered_seq = [x for x in user_sequences if x != 0]
            seq_key = str(filtered_seq)  # Convert sequence to a string key for checking
            total_weight = 0
            if seq_key not in user_group:                    
                # Calculate total_score
                total_pop_score = 0
                total_unpop_score = 0
                for idx, item in enumerate(reversed(filtered_seq)):
                    # Get the popularity label for the item
                    popularity_label = item_labels.loc[item_labels['item_id:token'] == item, 'popularity_label']
                    weight = math.pow(alpha, idx)
                    total_weight += weight
                    pop_label = popularity_label.iloc[0]
                    if pop_label == 1:   
                        total_pop_score += weight
                    elif pop_label == -1:
                        total_unpop_score += weight
                # Save the sequence and total_score as a dataset and attribute
                dataset_name = f"seq_{len(user_group)}"
                dataset = user_group.create_dataset(dataset_name, data=filtered_seq)
                dataset.attrs["total_pop_score"] = total_pop_score / total_weight
                dataset.attrs["total_unpop_score"] = total_unpop_score / total_weight



def fetch_user_popularity_score(user_ids, sequences):
    """
    Fetches the total scores for the given user IDs and their sequences from the HDF5 file.

    Args:
        user_ids (list): List of user IDs to fetch.
        sequences (list): List of sequences corresponding to the user IDs.

    Returns:
        list: A list of total scores for the provided sequences.
    """
    # Convert sequences and user_ids from tensor to native Python types.
    sequences = [seq.cpu().numpy() for seq in sequences]
    user_ids = [uid.item() for uid in user_ids]

    total_pop_scores = []
    total_unpop_scores = []
    file_path = r"./dataset/ml-1m/user_popularity_scores.h5"

    with h5py.File(file_path, "r") as f:
        for uid, seq in zip(user_ids, sequences):
            uid_str = str(uid)
            if uid_str not in f:
                print(f"User ID {uid_str} not found in the HDF5 file.")
                continue

            user_group = f[uid_str]
            # Compute the filtered sequence once for the user (removing zeros)
            filtered_seq = np.array([x for x in seq if x != 0])

            # Option 1: Iterate through datasets and compare using numpy
            found = False
            for ds_name in user_group:
                stored_seq = user_group[ds_name][:]
                # First check the shape before doing an element-wise comparison
                if stored_seq.shape == filtered_seq.shape and np.array_equal(stored_seq, filtered_seq):
                    total_pop_score = user_group[ds_name].attrs.get("total_pop_score", None)
                    total_pop_scores.append(total_pop_score)
                    total_unpop_score = user_group[ds_name].attrs.get("total_unpop_score", None)
                    total_unpop_scores.append(total_unpop_score)
                    found = True
                    break

            if not found:
                print(f"Sequence {filtered_seq.tolist()} not found for user {uid_str}.")
                
    print(f"{len(total_pop_scores)} ' ' {len(total_unpop_scores)}")
    return total_pop_scores, total_unpop_scores



def save_batch_activations(bulk_data, neuron_count, dataset, popular):
    if popular:
        file_path = rf"./dataset/{dataset}/neuron_activations_sasrecsae_final_pop.h5"
    if popular == False:
        file_path = rf"./dataset/{dataset}/neuron_activations_sasrecsae_final_unpop.h5"
        
    bulk_data = bulk_data.permute(1, 0).detach().cpu().numpy()  # [neuron_count, batch_size]
    real_batch_size = bulk_data.shape[1]  # Might be < batch_size in final step

    if not os.path.exists(file_path):
        with h5py.File(file_path, "w") as f:
            max_shape = (neuron_count, None)
            f.create_dataset(
                "dataset",
                data=bulk_data,
                maxshape=max_shape,
                chunks=(neuron_count, real_batch_size),
                dtype="float32",
            )
    else:
        with h5py.File(file_path, "a") as f:
            dset = f["dataset"]
            current_cols = dset.shape[1]
            new_cols = current_cols + real_batch_size
            dset.resize((neuron_count, new_cols))
            dset[:, current_cols:new_cols] = bulk_data
            


def save_batch_user_popularities(bulk_data_pop, bulk_data_unpop):
    """
    Saves two bulk data lists to separate HDF5 files, appending them to 1D datasets.

    Args:
        bulk_data_pop (list or numpy.ndarray): A 1D list or array of values to append to the first file.
        bulk_data_unpop (list or numpy.ndarray): A 1D list or array of values to append to the second file.
    """
    
    def save_to_h5(file_path, bulk_data):
        bulk_data = np.array(bulk_data, dtype=np.float32)  # Convert to NumPy array
        with h5py.File(file_path, "a") as f:
            if "dataset" not in f:
                # Create the dataset if it doesn't exist
                max_shape = (1100000,)  # Predefined max length
                f.create_dataset(
                    "dataset",
                    data=bulk_data,
                    maxshape=max_shape,
                    chunks=(len(bulk_data),),  # Chunk size equals bulk size
                    dtype="float32"
                )
            else:
                # Resize and append to the existing dataset
                dataset = f["dataset"]
                current_size = dataset.shape[0]
                new_size = current_size + len(bulk_data)
                dataset.resize((new_size,))
                dataset[current_size:new_size] = bulk_data

    # Save first bulk data
    save_to_h5(r"./dataset/ml-1m/user_scores_pop.h5", bulk_data_pop)
    
    # Save second bulk data
    save_to_h5(r"./dataset/ml-1m/user_scores_unpop.h5", bulk_data_unpop)

    print("Both datasets have been successfully saved.")
    
    

def calculate_pearson_correlation(file2_path, output_csv_path):
    """
    Calculates the Pearson correlation of each row in a dataset with another dataset and saves the result to a CSV file.

    Args:
        file1_path (str): Path to the first HDF5 file (shape (N, F)).
        file2_path (str): Path to the second HDF5 file (shape (F,)).
        output_csv_path (str): Path to save the resultant CSV file (shape (N,)).
    """
    
    file1_path = r"./dataset/ml-1m/neuron_activations.h5"
    # file2_path = r"./dataset/ml-1m/user_scores.h5"
    # Load the data from the HDF5 files
    with h5py.File(file1_path, "r") as f1, h5py.File(file2_path, "r") as f2:
        dataset2 = f2["dataset"][:]  # Shape (F,)
        dataset1 = f1["dataset"][:]  # Shape (F

    # Validate the shapes
    if dataset1.shape[1] != dataset2.shape[0]:
        raise ValueError("The number of features (F) in file1 must match the length of the dataset in file2.")
    # Calculate Pearson correlation for each row in dataset1 with dataset2
    correlations = []
    for row in dataset1:
        correlation = np.corrcoef(row, dataset2)[0, 1]  # Pearson correlation
        correlations.append(correlation)

    # Save the results to a CSV file
    df = pd.DataFrame({"Pearson_Correlation": correlations})
    df.to_csv(output_csv_path, index=False)
    print(f"Pearson correlations saved to {output_csv_path}")
    return correlations


import matplotlib.pyplot as plt


def get_extreme_correlations2(file_name: str, n: int, unpopular_only: bool):
    """
    Retrieves the highest and lowest correlation indexes and their values.
    
    Parameters:
    file_name (str): CSV file name containing correlation values.
    n (int): Number of extreme values to retrieve.
    unpopular_only (bool): Whether to return only the lowest values.
    
    Returns:
    list or tuple: If unpopular_only is True, returns a list of lowest indexes and their values.
                   Otherwise, returns a tuple of (highest_indexes, highest_values, lowest_indexes, lowest_values).
    """
    file_path = r"./dataset/ml-1m/" + file_name
    df = pd.read_csv(file_path)
    
    # Assuming the column name is unknown, take the first column
    column_name = df.columns[0]
    values = df[column_name]
    
    # Get indexes and values of highest and lowest n/2 values
    highest = values.nlargest(n)
    lowest = values.nsmallest(n)

    highest_indexes = highest.index.tolist()
    highest_values = highest.tolist()
    lowest_indexes = lowest.index.tolist()
    lowest_values = lowest.tolist()
    
    if unpopular_only:
        return list(zip(lowest_indexes, lowest_values))
    
    return (list(zip(highest_indexes, highest_values)), list(zip(lowest_indexes, lowest_values)))



def count():

    df_inter = pd.read_csv(r'./dataset/lfm1b-artists/lfm1b-artists_unsparsed.inter', sep='\t')

    # ============================================================
    # 2. KEEP ONLY THE LAST 75 INTERACTIONS PER USER
    # ============================================================
    df_inter.sort_values(by=['user_id:token', 'timestamp:float'], inplace=True)

    # Group by user_id and keep only the last 75 rows in each group.
    df_inter = df_inter.groupby('user_id:token', group_keys=False).tail(60)

    # ============================================================
    # 3. READ THE ITEM FILE
    # ============================================================
    df_item = pd.read_csv(r'./dataset/lfm1b-artists/lfm1b-artists_unsparsed.item', sep='\t')

    # ============================================================
    # 4. REMOVE ITEMS THAT NO LONGER HAVE ANY INTERACTIONS
    # ============================================================
    valid_items = df_inter['item_id:token'].unique()
    df_item = df_item[df_item['item_id:token'].isin(valid_items)]
    df_inter = df_inter[["item_id:token", "user_id:token", "timestamp:float"]]
    # ============================================================
    # 5. SAVE THE RESULTS
    # ============================================================
    df_inter.to_csv(r'./dataset/lfm1b-artists/lfm1b-artists-filtered.inter', sep='\t', index=False)
    df_item.to_csv(r'./dataset/lfm1b-artists/lfm1b-artists-filtered.item', sep='\t', index=False)
    

def compute_averages(output_file="output_averages.csv"): 
    """
    Computes the average values for all 4096 elements in A based on labels in C,
    and appends the results to a CSV file. If the file does not exist, it is created.
    
    Parameters:
    output_file (str): Path to save the resulting CSV file.
    """
    file1_path = r"./dataset/ml-1m/neuron_activations.h5"
    file2_path = r"./dataset/ml-1m/user_scores.h5"
    
    with h5py.File(file1_path, "r") as f1, h5py.File(file2_path, "r") as f2:
        dataset1 = f1["dataset"][:]  # Shape (4096, N)
        dataset2 = f2["dataset"][:]  # Shape (N,)
    
    # Step 1: Create list D based on conditions
    D = np.where(dataset2 < 0.3, -1, np.where(dataset2 > 0.7, 1, 0))
    
    # Step 2: Compute averages for each row in dataset1 based on D values
    valid_neg = D == -1
    valid_pos = D == 1
    
    avg_neg = np.where(np.any(valid_neg, axis=0), np.mean(dataset1[:, valid_neg], axis=1), 0)
    avg_pos = np.where(np.any(valid_pos, axis=0), np.mean(dataset1[:, valid_pos], axis=1), 0)
    diff = avg_pos - avg_neg
    
    # Combine results
    Y = np.column_stack((avg_neg, avg_pos, diff))
    
    # Convert to DataFrame
    df_Y = pd.DataFrame(Y, columns=["Avg (-1)", "Avg (1)", "Difference"])
    
    # Append to CSV file, creating it if it does not exist
    df_Y.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
    
    print(f"Results appended to {output_file}")


def get_difference_values(indexes, csv_file="output_averages.csv"):
    """
    Given an array of indexes, returns an array with 'Difference' values for the given indexes.
    
    Parameters:
    indexes (np.ndarray): Array of indexes to retrieve values for.
    csv_file (str): Path to the CSV file containing precomputed values.
    
    Returns:
    np.ndarray: Array of 'Difference' values corresponding to the given indexes.
    """
    df = pd.read_csv(csv_file)
    return df.iloc[indexes]["Difference"].values


def remove_sparse_users_items():
    interactions_file = r"./dataset/Amazon_Electronics/Amazon_Electronics-filtered.inter"
    items_file = r"./dataset/Amazon_Electronics/Amazon_Electronics-filtered.item"
    
    # Load interactions and items data
    df_inter = pd.read_csv(interactions_file, sep='\t')
    df_item = pd.read_csv(items_file, sep='\t')

    # -------------------------------
    # 2. Iterative Filtering
    # -------------------------------
    min_interactions = 5

    # We'll iterate until the number of interactions/users doesn't change.
    prev_interactions_count = -1
    prev_users_count = -1

    while True:
        # --- Filter Items ---
        # Count interactions per item
        item_counts = df_inter['item_id:token'].value_counts()
        # Identify items with at least min_interactions
        valid_items = item_counts[item_counts >= min_interactions].index
        # Filter interactions DataFrame to keep only valid items
        df_inter = df_inter[df_inter['item_id:token'].isin(valid_items)]

        # --- Filter Users ---
        # Count interactions per user
        user_counts = df_inter['user_id:token'].value_counts()
        # Identify users with at least min_interactions
        valid_users = user_counts[user_counts >= min_interactions].index
        # Filter interactions DataFrame to keep only valid users
        df_inter = df_inter[df_inter['user_id:token'].isin(valid_users)]

        # Check for convergence
        current_interactions_count = df_inter.shape[0]
        current_users_count = df_inter['user_id:token'].nunique()

        if (current_interactions_count == prev_interactions_count) and (current_users_count == prev_users_count):
            break  # No further changes
        else:
            prev_interactions_count = current_interactions_count
            prev_users_count = current_users_count

    # -------------------------------
    # 3. Update the Items DataFrame
    # -------------------------------
    # Keep only items that still appear in the interactions DataFrame
    df_item = df_item[df_item['item_id:token'].isin(df_inter['item_id:token'].unique())]

    # -------------------------------
    # 4. (Optional) Save the Filtered Data
    # -------------------------------
    df_inter.to_csv(r'./dataset/Amazon_Electronics/Amazon_Electronics.inter', index=False, sep='\t')
    df_item.to_csv(r'./dataset/Amazon_Electronics/Amazon_Electronics.item', index=False, sep='\t')

    print("Filtering complete:")
    print(f" - Interactions: {df_inter.shape[0]} records")
    print(f" - Users: {df_inter['user_id:token'].nunique()} unique users")
    print(f" - Items: {df_item.shape[0]} records")
    
    

def plot_binned_bar_chart(csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    column_name = df.columns[0]  # Assuming there is only one column
    
    # Define bins
    bins = np.arange(-0.4, 0.45, 0.05)  # Bin edges from -0.4 to 0.4 with step 0.05
    
    # Bin the data
    counts, _ = np.histogram(df[column_name], bins=bins)
    
    # Define bin centers for plotting
    bin_centers = bins[:-1] + np.diff(bins) / 2
    
    # Create the bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(bin_centers, counts, width=0.04, edgecolor='black', align='center')
    plt.xlabel('Value Bins')
    plt.ylabel('Count')
    plt.title('Histogram with Bin Size 0.05')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def make_items_unpopular(item_seq_len, dataset):

    item_labels = pd.read_csv(rf"./dataset/{dataset}/item_popularity_labels_with_titles.csv")
    
    # Filter rows where popularity_label == -1
    filtered_items = item_labels[item_labels['popularity_label'] == -1]
    available_ids = filtered_items['item_id:token'].tolist()
    
    # Count how many items are in each row of the batch
    nonzero_counts = (item_seq_len != 0).sum(dim=1).tolist()
    selected_item_ids = []

    for count in nonzero_counts:
        sampled = pd.Series(available_ids).sample(n=count, replace=True).tolist()
        
        # Pad with 0s if needed to reach length 50
        if len(sampled) < 50:
            sampled += [0] * (50 - len(sampled))
        else:
            sampled = sampled[:50]  # In case count > 50 for any reason

        selected_item_ids.append(sampled)

    # Convert to tensor of shape (batch_size, 50)
    selected_tensor = torch.tensor(selected_item_ids)

    return selected_tensor



def make_items_popular(item_seq_len, dataset):
    item_labels = pd.read_csv(rf"./dataset/{dataset}/item_popularity_labels_with_titles.csv")
    
    # Filter rows where popularity_label == -1
    filtered_items = item_labels[item_labels['popularity_label'] == 1]
    available_ids = filtered_items['item_id:token'].tolist()
    
    # Count how many items are in each row of the batch
    nonzero_counts = (item_seq_len != 0).sum(dim=1).tolist()
    selected_item_ids = []

    for count in nonzero_counts:
        sampled = pd.Series(available_ids).sample(n=count, replace=True).tolist()
        
        # Pad with 0s if needed to reach length 50
        if len(sampled) < 50:
            sampled += [0] * (50 - len(sampled))
        else:
            sampled = sampled[:50]  # In case count > 50 for any reason

        selected_item_ids.append(sampled)

    # Convert to tensor of shape (batch_size, 50)
    selected_tensor = torch.tensor(selected_item_ids)

    return selected_tensor



def save_mean_SD(dataset, popular=None):
    # Load your .h5 file
    if popular == True:  
        file_path = rf"./dataset/{dataset}/neuron_activations_sasrecsae_final_pop.h5"
    elif popular == False:  
        file_path = rf"./dataset/{dataset}/neuron_activations_sasrecsae_final_unpop.h5"
    dataset_name = 'dataset'  # Replace with actual dataset name inside the h5 file

    # Load the real indices from the filtered CSV
    # index_csv = r"./dataset/ml-1m/nonzero_activations_sasrecsae_k48-32.csv"
    # real_indices = pd.read_csv(index_csv, index_col=0).index.tolist()

    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name][()]  # Reads full dataset into memory

    # Compute mean and standard deviation for each row
    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1)
    # Combine into a DataFrame with the correct index
    df = pd.DataFrame({
        'mean': means,
        'std': stds,
    })

    if popular == True:  
        output_csv_path = rf"./dataset/{dataset}/row_stats_popular.csv"
    if popular == False:  
        output_csv_path = rf"./dataset/{dataset}/row_stats_unpopular.csv"
    df.to_csv(output_csv_path)
    print(f"Row-wise mean and std saved to {output_csv_path}")
    
    



def save_cohens_d(dataset):
    df1 = pd.read_csv(rf"./dataset/{dataset}/row_stats_popular.csv", index_col=0)
    df2 = pd.read_csv(rf"./dataset/{dataset}/row_stats_unpopular.csv", index_col=0)

    # Compute pooled standard deviation
    s_pooled = np.sqrt((df1['std']**2 + df2['std']**2) / 2)

    # Compute Cohen's d
    cohen_d = (df1['mean'] - df2['mean']) / s_pooled

    # Create result DataFrame with same index
    df_result = pd.DataFrame({'cohen_d': cohen_d})

    # Save to CSV with index column
    df_result.to_csv(rf"./dataset/{dataset}/cohens_d.csv")

    print("Cohen's d values saved to cohens_d.csv")
    
    
    
from scipy.stats import pearsonr


def find_pair_cor():
    # === CONFIG ===
    # === CONFIG ===
    file_path = r"./dataset/ml-1m/neuron_activations_popular_sasrec.h5"
    dataset_name = "dataset"  # Replace with actual dataset key inside the HDF5
    output_csv = "correlation_pairs_popular.csv"

    # === LOAD DATA ===
    with h5py.File(file_path, "r") as f:
        data = f[dataset_name][...]  # shape: (64, X)

    assert data.shape[0] == 64, "Expected 64 rows (neurons) in the data."

    # === COMPUTE CORRELATIONS ===
    correlations = []

    for i in range(64):
        for j in range(i + 1, 64):  # Only upper triangle (i < j)
            r, _ = pearsonr(data[i], data[j])
            correlations.append((i, j, r))

    # === SAVE TO CSV ===
    df = pd.DataFrame(correlations, columns=["row_i", "row_j", "correlation"])
    df["abs_corr"] = df["correlation"].abs()
    df.sort_values("abs_corr", ascending=False, inplace=True)
    df.drop(columns="abs_corr", inplace=True)
    df.to_csv(output_csv, index=False)

    print(f"Saved correlation results to '{output_csv}'")
    


def find_diff():
    file1 = "correlation_pairs_popular.csv"  # Replace with your actual file name
    file2 = "correlation_pairs_unpopular.csv"
    output_file = "correlation_diff.csv"

    # === LOAD FILES ===
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # === MERGE ON (row_i, row_j) ===
    merged = pd.merge(df1, df2, on=["row_i", "row_j"], suffixes=("_1", "_2"))

    # === COMPUTE DIFFERENCE ===
    merged["correlation_diff"] = merged["correlation_1"] - merged["correlation_2"]
    merged["abs_diff"] = merged["correlation_diff"].abs()

    # === SORT AND SAVE ===
    merged.sort_values("abs_diff", ascending=False, inplace=True)
    result = merged[["row_i", "row_j", "correlation_diff"]]
    result.to_csv(output_file, index=False)

    print(f"Saved correlation differences to '{output_file}' (sorted by abs diff)")
    
    
    

def build_popularity_tensor(num_items=3706):
    """
    Returns a tensor of shape [num_items], where index i holds the
    popularity label for item ID (i + 1). Assumes item IDs ∈ [1, num_items].
    """
    csv_path = r"./dataset/ml-1m/item_popularity_labels_with_titles.csv"
    df = pd.read_csv(csv_path)

    # Rename for clarity (optional)
    df = df.rename(columns={'item_id:token': 'item_id'})

    # Clamp non-1 labels to 0
    df['popularity_label'] = (df['popularity_label'] == 1).astype(int)

    # Initialize tensor: index 0 = item ID 1
    popularity_tensor = torch.zeros(num_items, dtype=torch.float32)

    for _, row in df.iterrows():
        item_id = row['item_id']
        index = item_id - 1  # shift: ID 1 → index 0
        if 0 <= index < num_items:
            popularity_tensor[index] = row['popularity_label']
    return 

def get_extreme_correlations(file_name: str, unpopular_only: bool, dataset=None):
    """
    Retrieves all positive and all negative correlation indexes and their values.

    Parameters:
    file_name (str): CSV file name containing correlation values.
    unpopular_only (bool): If True, returns an empty positive list and the full negative list.

    Returns:
    tuple:
      - pos_list: list of (index, value) for all positives (empty if unpopular_only=True)
      - neg_list: list of (index, value) for all negatives
    """
    

    # 1) load
    df = pd.read_csv(rf"./dataset/{dataset}/{file_name}")
    # indices = pd.read_csv(r"./dataset/ml-1m/nonzero_activations_sasrecsae_k48-32.csv")["index"].tolist()
    # # 2) if they passed a subset of row positions, slice with .iloc
    # if indices is not None:
    #     df = df.iloc[indices]

    # 3) split out positives / negatives
    pos_series = df.loc[df["cohen_d"] > 0, "cohen_d"]
    neg_series = df.loc[df["cohen_d"] < 0, "cohen_d"]

    # 4) zip index-labels (which by default are 0,1,2… or the original row numbers)
    pos_list = list(pos_series.items())  # each item is (index_label, value)
    neg_list = list(neg_series.items())

    # 5) if they only want “unpopular” (negatives), empty the positives
    if unpopular_only:
        pos_list = []

    return pos_list, neg_list
    # file_path = "./dataset/ml-1m/" + file_name
    # df = pd.read_csv(file_path)  # 🟢 This is the fix


    # # get the rows where cohen_d > 0
    # pos_df = df.loc[df["cohen_d"] > 0, ["index", "cohen_d"]]
    # neg_df = df.loc[df["cohen_d"] < 0, ["index", "cohen_d"]]

    # # now zip the column values directly
    # pos_list = list(zip(pos_df["index"].tolist(), pos_df["cohen_d"].tolist()))
    # neg_list = list(zip(neg_df["index"].tolist(), neg_df["cohen_d"].tolist()))
    # # grab the first (aintnd only) column of correlation scores
    # vals = df["cohen_d"]
    
    # # if only unpopular, empty out the positives
    # if unpopular_only:
    #     pos_list = []

    # return pos_list, neg_list



import torch
import math
from collections import Counter



def calculate_average_popularity(item_ids: torch.Tensor, label) -> float:
    df = pd.read_csv(r"./dataset/ml-1m/item_popularity_labels_with_titles.csv")
    
    # Create a mapping from item_id to interaction_count
    interaction_dict = dict(zip(df['item_id:token'], df['interaction_count']))

    # Calculate total number of interactions
    total_interactions = df['interaction_count'].sum()

    # Compute the popularity scores list
    popularity_scores = [
        interaction_dict.get(item_id, 0) / total_interactions
        for item_id in item_ids.tolist()
    ]

    # Print the average popularity score
    average_popularity = sum(popularity_scores) / len(popularity_scores)
    print(label, " Average Popularity Score:", average_popularity)    


def calculate_IPS(item_ids, reverse=False):
    """
    Compute inverse propensity scores based on item popularity.

    Args:
        item_ids (Tensor): Tensor of item IDs (shape: N,)
        reverse (bool): If True, return inverse popularity (for SKEW). If False, return direct popularity.

    Returns:
        List[float]: Propensity scores per item.
    """
    csv_path = r"./dataset/steam/item_popularity_labels_with_titles.csv"
    df = pd.read_csv(csv_path)

    df['interaction_count'] = df['interaction_count'].astype(float)
    df['item_id'] = df['item_id:token'].astype(str)

    total_interactions = df['interaction_count'].sum()
    interaction_series = df.set_index('item_id')['interaction_count']

    fallback_value = 1.0  # fallback in case item ID is missing

    ips_list = []
    for item in item_ids:
        item_str = str(item.item())
        if item_str in interaction_series:
            count = interaction_series[item_str]
            ips = (total_interactions / count) if reverse else (count / total_interactions)
        else:
            ips = fallback_value
        ips_list.append(ips)

    return ips_list


def skew_sample(interaction, num_samples):
    """
    Select a subset of interactions based on SKEW sampling (inverse item popularity).

    Args:
        interaction (dict): Dictionary with keys 'item_id', 'item_id_list', 'item_length'.
        num_samples (int): Number of interactions to keep based on inverse popularity.

    Returns:
        Tensor: Indices of the sampled (retained) interactions (shape: num_samples,).
    """

    item_ids = interaction['item_id']             # (N,)
    item_id_list = interaction['item_id_list']    # (N, 50)
    item_length = interaction['item_length']      # (N,)
    print(torch.unique(item_ids).numel())
    # Log and compute entropy before sampling
    calculate_average_popularity(item_ids, 'Before sampling')
    # entropy_before = calculate_normalized_entropy(item_ids)
    # print(f"🟡 Normalized Entropy BEFORE: {entropy_before:.4f}")

    # Compute inverse popularity scores (low for popular, high for rare items)
    sample_probs = calculate_IPS(item_ids, reverse=True)
    sample_probs = list(map(lambda x: x**2, sample_probs))
    probs = torch.tensor(sample_probs, dtype=torch.float)
    probs = probs / probs.sum()

    if torch.isnan(probs).any() or probs.sum() == 0:
        raise ValueError("Invalid sampling probabilities. Check your popularity scores.")

    N = item_ids.size(0)
    all_indices = torch.arange(N)
    sampled_indices = torch.multinomial(probs, num_samples, replacement=False)

    # Keep only sampled interactions
    interaction['item_id'] = item_ids[sampled_indices]
    interaction['item_id_list'] = item_id_list[sampled_indices]
    interaction['item_length'] = item_length[sampled_indices]

    # Log and compute entropy after sampling
    calculate_average_popularity(interaction['item_id'], 'After sampling')
    # entropy_after = calculate_normalized_entropy(interaction['item_id'])
    # print(f"🟢 Normalized Entropy AFTER:  {entropy_after:.4f}")

    return sampled_indices


def get_popularity_label_indices(id_tensor, dataset=None):
    """
    Given a 1D tensor of item IDs, returns a 1D tensor of the same shape 
    that indicates the popularity label for each item.
    
    Args:
        id_tensor (torch.Tensor): 1D tensor of item IDs of shape (N,)
        
    Returns:
        torch.Tensor: 1D tensor of popularity labels corresponding to 
                      each item in id_tensor.
    """
    # Read the CSV that maps item IDs to popularity labels.
    df = pd.read_csv(rf"./dataset/{dataset}/item_popularity_labels_with_titles.csv", encoding='latin1')
    
    # Create a mapping from item ID to popularity label.
    id_to_label = dict(zip(df['item_id:token'], df['popularity_label']))
    
    # For each item in the tensor, retrieve the corresponding label.
    # If an item ID is not found, we assign a default label of -2.
    default_label = -2
    labels = [id_to_label.get(item_id, default_label) for item_id in id_tensor.tolist()]
    
    # Convert the list of labels to a torch tensor.
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return label_tensor
    
from scipy.stats import chi2_contingency


import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from kneed import KneeLocator



def rank_neurons_by_chi2(popular_csv, unpopular_csv, total_per_group, output_csv, plot_path=None):
    """
    Computes chi-square scores for neurons where unpopular > popular activations,
    ranks them, plots curve, finds elbow point, and saves output.

    Returns:
    - elbow_index: number of unpop-biased neurons (before flattening)
    """
    # Load data
    df_pop = pd.read_csv(popular_csv)
    df_unpop = pd.read_csv(unpopular_csv)
    df = pd.merge(df_pop, df_unpop, on='index', suffixes=('_pop', '_unpop'))

    # Filter: only where unpopular > popular
    df = df[df['count_unpop'] > df['count_pop']].reset_index(drop=True)

    chi2_scores = []
    for _, row in df.iterrows():
        pop_count = row['count_pop']
        unpop_count = row['count_unpop']
        table = [
            [pop_count, total_per_group - pop_count],
            [unpop_count, total_per_group - unpop_count]
        ]
        try:
            chi2_stat, _, _, _ = chi2_contingency(table)
        except ValueError:
            chi2_stat = 0
        chi2_scores.append(chi2_stat)

    df['chi2_score'] = chi2_scores
    df_sorted = df[['index', 'chi2_score']].sort_values(by='chi2_score', ascending=False)
    df_sorted.to_csv(output_csv, index=False)
    print(f"Saved filtered & ranked results to {output_csv}")

    # Elbow detection
    x = list(range(len(df_sorted)))
    y = df_sorted['chi2_score'].tolist()
    knee = KneeLocator(x, y, curve='convex', direction='decreasing')
    elbow_index = knee.knee if knee.knee is not None else 0

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, marker='o', linestyle='', alpha=0.6)
    plt.axhline(3.841, color='red', linestyle='--', label='p = 0.05 threshold (χ² = 3.841)')
    if elbow_index:
        plt.axvline(elbow_index, color='orange', linestyle='--', label=f'Elbow ≈ {elbow_index}')
    plt.xlabel('Neuron Rank (sorted by chi2)')
    plt.ylabel('Chi-square Statistic')
    plt.title('Unpop-Biased Neuron Chi-square Scores')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return elbow_index


def get_top_n_neuron_indexes(n):
    """
    Reads a CSV with ['index', 'chi2_score'], sorts by chi2 descending,
    and returns the top-n indexes as a PyTorch tensor.

    Parameters:
    - csv_path: path to the CSV file
    - n: number of top indexes to return

    Returns:
    - torch.Tensor of shape (n,)
    """
    csv_path = r"./dataset/ml-1m/ranked_neuron_bias_scores.csv"
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values(by='chi2_score', ascending=False).head(n)
    top_indexes = df_sorted['index'].values
    return torch.tensor(top_indexes, dtype=torch.long)



import h5py
import matplotlib.pyplot as plt

def plot_h5_columns(row_x=None, row_y=None, row_z=None, num_rows=100000):
    """
    Plots data from two HDF5 files (with dataset name 'dataset') using different modes
    depending on the provided row parameters.
    
    Modes:
      1. Scatter Plot:
         - If both row_x and row_y are provided, a scatter plot is produced.
         - If row_z is also provided, a 3D scatter plot is generated.
      
      2. Histogram for Single Row:
         - If only row_x is provided (row_y and row_z are None), a histogram (bar chart) is created.
           The histogram uses bins from -2.5 to 2.5 (with a bin width of 0.05) and plots frequency counts.
      
      3. Histograms for All Rows:
         - If row_x is not provided (i.e. row_x is None), then a histogram is generated for each row (all indices)
           from both files in a grid of subplots.
    
    Parameters:
        row_x (int or None): Index for the x-axis data or, if used alone, the row whose histogram is computed.
        row_y (int or None): Index for the y-axis data (required for scatter plot).
        row_z (int or None): Index for the z-axis data (optional, for 3D scatter plot).
        num_rows (int, optional): Number of columns to read from the dataset. Defaults to 100000.
    
    The function loads the dataset named 'dataset' from two files:
      - "./dataset/ml-1m/sasrec_unpop_activations.h5"
      - "./dataset/ml-1m/sasrec_pop_activations.h5"
    """
    # Define file paths.
    file1 = r"./dataset/gowalla/neuron_activations_sasrecsae_final_unpop.h5"
    file2 = r"./dataset/gowalla/neuron_activations_sasrecsae_final_pop.h5"
    
    # Load the first num_rows columns from the 'dataset' in both files.
    with h5py.File(file1, 'r') as f1:
        data1 = f1['dataset'][:, :num_rows]
    with h5py.File(file2, 'r') as f2:
        data2 = f2['dataset'][:, :num_rows]
    
    # Define histogram bins: from -2.5 to 2.5, bin width of 0.05.
    bins = np.arange(-2.5, 2.5 + 0.05, 0.05)
    
    # Case 1: Scatter plot if both row_x and row_y are provided.
    if row_x is not None and row_y is not None:
        # Extract the specified rows from each dataset.
        x1, y1 = data1[row_x, :], data1[row_y, :]
        x2, y2 = data2[row_x, :], data2[row_y, :]
        
        if row_z is not None:
            # 3D scatter plot.
            z1 = data1[row_z, :]
            z2 = data2[row_z, :]
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x1, y1, z1, color='blue', marker='o', label='Unpopular')
            ax.scatter(x2, y2, z2, color='red', marker='x', label='Popular')
            ax.set_xlabel(f'Row {row_x}')
            ax.set_ylabel(f'Row {row_y}')
            ax.set_zlabel(f'Row {row_z}')
            ax.set_title('3D Scatter Plot of Selected Rows from Two HDF5 Files')
            ax.legend()
        else:
            # 2D scatter plot.
            plt.figure(figsize=(8, 6))
            plt.scatter(x1, y1, color='blue', marker='o', label='Unpopular')
            plt.scatter(x2, y2, color='red', marker='x', label='Popular')
            plt.xlabel(f'Row {row_x}')
            plt.ylabel(f'Row {row_y}')
            plt.title('2D Scatter Plot of Selected Rows from Two HDF5 Files')
            plt.legend()
            plt.grid(True)
        plt.show()
    
    # Case 2: Histogram for a single row if only row_x is provided.
    elif row_x is not None and row_y is None and row_z is None:
        # Extract the data for the selected row from both datasets.
        data_row1 = data1[row_x, :]
        data_row2 = data2[row_x, :]
        
        # Compute histograms for each file.
        hist1, _ = np.histogram(data_row1, bins=bins)
        hist2, _ = np.histogram(data_row2, bins=bins)
        # Compute bin centers.
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        plt.figure(figsize=(8, 6))
        # Offset the bars slightly to avoid overlap.
        width = 0.025  # Half of the bin width.
        plt.bar(bin_centers - width, hist1, width=0.025, color='blue', alpha=0.7, label='File 1')
        plt.bar(bin_centers + width, hist2, width=0.025, color='red', alpha=0.7, label='File 2')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Histogram for Row {row_x} from -2.5 to 2.5 (bin width 0.05)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Case 3: No row parameters provided: histogram for all rows.
    elif row_x is None:
        n_rows_data = data1.shape[0]  # Total number of rows (activations) in the dataset.
        # Determine subplot grid size.
        ncols = int(math.ceil(math.sqrt(n_rows_data)))
        nrows_subplot = int(math.ceil(n_rows_data / ncols))
        
        fig, axs = plt.subplots(nrows_subplot, ncols, figsize=(4 * ncols, 3 * nrows_subplot))
        axs = axs.flatten()  # Flatten the array for easier indexing.
        
        for idx in range(n_rows_data):
            # Compute histograms for the current row for both files.
            d1 = data1[idx, :]
            d2 = data2[idx, :]
            h1, _ = np.histogram(d1, bins=bins)
            h2, _ = np.histogram(d2, bins=bins)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            ax = axs[idx]
            width = 0.025
            ax.bar(bin_centers - width, h1, width=0.025, color='blue', alpha=0.7, label='F1')
            ax.bar(bin_centers + width, h2, width=0.025, color='red', alpha=0.7, label='F2')
            ax.set_title(f'Row {idx}')
            ax.set_xlim([-2.5, 2.5])
            ax.grid(True)
            # Optionally, add legend only for the first subplot.
            if idx == 0:
                ax.legend()
        
        # Hide any unused subplots.
        for j in range(n_rows_data, len(axs)):
            axs[j].axis('off')
            
        plt.tight_layout()
        plt.show()
        
        
def compute_covariances(h5_file, row_index):
    """
    Compute the covariance between the specified row and all rows in the dataset.

    Parameters:
        h5_file (str): Path to the HDF5 file.
        row_index (int): Index of the row whose covariance with every row is computed.

    Returns:
        cov (np.ndarray): A 1D array of covariance values of shape (n_rows,).
                          cov[j] is the covariance between the row at row_index and row j.
    """
    # Open the HDF5 file and load the entire dataset.
    with h5py.File(h5_file, 'r') as f:
        data = f['dataset'][:]  # Assumed shape: (n_rows, n_columns)
    
    # Number of rows and columns.
    n_rows, n_cols = data.shape
    
    # Get the specified row and its mean.
    x = data[row_index, :]
    x_mean = np.mean(x)
    
    # Compute the mean for each row.
    row_means = np.mean(data, axis=1)  # shape: (n_rows,)
    
    # Compute the covariance between row_index and every row.
    # For each row j:
    #   cov(x, y_j) = sum((x - mean_x) * (y_j - mean_y_j)) / (n_cols - 1)
    # We compute this in a vectorized way.
    cov = np.sum((x - x_mean) * (data - row_means[:, None]), axis=1) / (n_cols - 1)
    
    return cov

import numpy as np
import h5py

def compute_covariances(h5_file, row_index):
    """
    Compute the covariance between the specified row and all rows in the dataset.

    Parameters:
        h5_file (str): Path to the HDF5 file.
        row_index (int): Index of the row whose covariance with every row is computed.

    Returns:
        cov (np.ndarray): A 1D array of covariance values of shape (n_rows,).
                          cov[j] is the covariance between the row at row_index and row j.
    """
    # Open the HDF5 file and load the entire dataset.
    with h5py.File(h5_file, 'r') as f:
        data = f['dataset'][:]  # Assumed shape: (n_rows, n_columns)
    
    # Number of rows and columns.
    n_rows, n_cols = data.shape
    
    # Get the specified row and compute its mean.
    x = data[row_index, :]
    x_mean = np.mean(x)
    
    # Compute the mean for each row.
    row_means = np.mean(data, axis=1)  # shape: (n_rows,)
    
    # Compute the covariance between the specified row and every row.
    cov = np.sum((x - x_mean) * (data - row_means[:, None]), axis=1) / (n_cols - 1)
    
    return cov



def compute_covariances(h5_file, row_index, num_rows=100000):
    """
    Compute the covariance between the specified row and all rows in the dataset.

    Parameters:
        h5_file (str): Path to the HDF5 file.
        row_index (int): Index of the row for which covariance with every row is computed.
        num_rows (int): Number of columns to read from the dataset (default is 100000).

    Returns:
        np.ndarray: An array of covariance values between the specified row and every row.
    """
    with h5py.File(h5_file, 'r') as f:
        data = f['dataset'][:, :num_rows]  # Assumed data shape: (n_rows, n_columns)
    
    n_rows, n_cols = data.shape
    x = data[row_index, :]
    x_mean = np.mean(x)
    # Compute the mean for every row.
    row_means = np.mean(data, axis=1)
    # Compute the covariance between the specified row and each row.
    cov = np.sum((x - x_mean) * (data - row_means[:, None]), axis=1) / (n_cols - 1)
    return cov


def compute_and_save_correlations(row1, row2, min_corr, num_rows=500000, output_prefix="correlation_results"):
    """
    For each of the two given rows, compute the Pearson correlation between that row 
    and all other rows in the HDF5 file's dataset, normalize using per-row standard deviations 
    obtained from a CSV file, filter out correlations below a given threshold, and save the results.

    The CSV files saved will each have two columns: 'row' and 'correlation'.

    Parameters:
        h5_file (str): Path to the HDF5 file.
        row1 (int): Index of the first row.
        row2 (int): Index of the second row.
        min_corr (float): The correlation threshold. Only correlation values greater than this value are saved.
        stats_csv_path (str): Path to the CSV file containing row statistics (with a "std" column).
        num_rows (int, optional): Number of columns to load from the dataset. Defaults to 100000.
        output_prefix (str, optional): Prefix for the output CSV filenames.
    """
    
    h5_file = r"./dataset/ml-1m/sasrec_unpop_activations.h5"
    stats_csv_path = r"./dataset/ml-1m/row_stats_unpopular.csv"
    
    # Load the per-row statistics (we assume the CSV file uses the row index as its index).
    stats_df = pd.read_csv(stats_csv_path, index_col=0)
    # Ensure the 'std' column is numeric.
    std_row1 = stats_df.iloc[row1]["std"]
    std_row2 = stats_df.iloc[row2]["std"]
    # Also get the standard deviations for all rows (assumed order corresponds to dataset rows).
    std_all = stats_df["std"].to_numpy()
    
    # Compute the covariance arrays for each specified row.
    cov1 = compute_covariances(h5_file, row1, num_rows)
    cov2 = compute_covariances(h5_file, row2, num_rows)
    
    # Compute Pearson correlation for each row:
    # correlation = covariance / (std(target_row) * std(other_row))
    corr1 = cov1 / (std_row1 * std_all)
    corr2 = cov2 / (std_row2 * std_all)
    
    # Filter to retain only correlation values greater than the provided threshold.
    indices = np.arange(len(corr1))
    mask1 = (corr1 > min_corr) | (corr1 < -1 * min_corr)
    mask2 = (corr2 > min_corr) | (corr2 < -1 * min_corr)
    filtered_indices1 = indices[mask1]
    filtered_corr1 = corr1[mask1]
    filtered_indices2 = indices[mask2]
    filtered_corr2 = corr2[mask2]
    
    # Stack the filtered row positions and correlation values.
    results_row1 = np.column_stack((filtered_indices1, filtered_corr1))
    results_row2 = np.column_stack((filtered_indices2, filtered_corr2))
    
    # Save the filtered results to CSV files.
    filename1 = f"{output_prefix}_row{row1}.csv"
    filename2 = f"{output_prefix}_row{row2}.csv"
    
    np.savetxt(filename1, results_row1, delimiter=",", header="row,correlation", comments="")
    np.savetxt(filename2, results_row2, delimiter=",", header="row,correlation", comments="")
    
    print(f"Saved correlation results for row {row1} (corr > {min_corr}) to '{filename1}'.")
    print(f"Saved correlation results for row {row2} (corr > {min_corr}) to '{filename2}'.")





def remove_sparse_users_items(n):
    # --- Step 1: Load the Data ---
    # The files use tab as the delimiter and have headers that include type annotations.
    items = pd.read_csv(r"./dataset/lastfm/lastfm.item", sep="\t", header=0)
    interactions = pd.read_csv(r"./dataset/lastfm/lastfm.inter", sep="\t", header=0)
    # --- Step 2: Iterative Filtering ---
    # We use a threshold of at least 5 interactions for both users and items.
    iteration = 0
    while True:
        iteration += 1
        current_shape = interactions.shape[0]
        
        # Remove users with fewer than 5 interactions:
        user_counts = interactions["user_id:token"].value_counts()
        valid_users = user_counts[user_counts >= n].index
        interactions = interactions[interactions["user_id:token"].isin(valid_users)]
                
        # Remove items with fewer than 5 interactions:
        item_counts = interactions["item_id:token"].value_counts()
        valid_items = item_counts[item_counts >= n].index
        interactions = interactions[interactions["item_id:token"].isin(valid_items)]
        
        new_shape = interactions.shape[0]
        print(f"Iteration {iteration}: {current_shape} -> {new_shape} interactions remain")
        
        if new_shape == current_shape:
            break
    # --- Step 3: Synchronize Items With Interactions ---
    # Keep only those items that still appear in the filtered interactions.
    items = items[items["item_id:token"].isin(interactions["item_id:token"])]

    # --- Step 4: Save the Filtered Files ---
    # Files are saved with the header intact (including the type annotations).
    items.to_csv(r"./dataset/lastfm/lastfm.item.filtered", sep="\t", index=False, header=True)
    interactions.to_csv(r"./dataset/lastfm/lastfm.inter.filtered", sep="\t", index=False, header=True)

    print("Filtering complete. Files saved as 'ml-1m.item.filtered', 'ml-1m.inter.filtered', and 'ml-1m.user.filtered'.")



def create_unbiased_set():
    # -------------------------------
    # Utility function for KL Divergence
    # -------------------------------
    def kl_divergence(P, Q, epsilon=1e-12):
        """
        Compute KL divergence KL(P || Q) = sum_i P[i] * log(P[i] / (Q[i] + epsilon)).
        P and Q should be valid probability distributions.
        """
        Q_safe = Q + epsilon
        return np.sum(P * np.log(P / Q_safe))
    
    # -------------------------------
    # Step 1: Load all three NPZ files and compute overall item counts.
    # -------------------------------
    train_file = r"./dataset/ml-1m/biased_eval_train.npz"
    val_file  = r"./dataset/ml-1m/biased_eval_test.npz"
    test_file   = r"./dataset/ml-1m/biased_eval_val.npz"
    
    train_data = np.load(train_file)
    test_data  = np.load(test_file)
    val_data   = np.load(val_file)
    
    train_labels = train_data["labels"].astype(int)
    test_labels  = test_data["labels"].astype(int)
    val_labels   = val_data["labels"].astype(int)
    
    # Total number of possible items (IDs 0 to 3416)
    n_items = 3417
    
    # Combine labels from all three files to compute overall frequencies.
    all_labels = np.concatenate((train_labels, test_labels, val_labels))
    counts_all = np.bincount(all_labels, minlength=n_items)
    
    # Compute the minimum nonzero count among items.
    nonzero_counts = counts_all[counts_all > 0]
    if len(nonzero_counts) == 0:
        min_count = 1  # Should not happen normally.
    else:
        min_count = np.min(nonzero_counts)
    print("Minimum (nonzero) count across all items:", min_count)
    
    # For each item, calculate the negative sampling acceptance probability.
    # Items with the least interactions get p_i = 1, others get p_i = (min_count)/(count_i)
    p_items = np.zeros(n_items)
    for i in range(n_items):
        if counts_all[i] > 0:
            p_items[i] = min_count / counts_all[i]
        else:
            p_items[i] = 0.0  # If an item is never seen.
    
    # -------------------------------
    # Step 2: Apply negative sampling on the test set.
    # -------------------------------
    # We use the test set to select interactions.
    test_features = test_data["features"]  # shape: (N_test, dim)
    test_labels   = test_data["labels"].astype(int)  # shape: (N_test,)
    N_test = len(test_labels)
    
    # For each test interaction with item label i, accept it with probability p_items[i].
    random_values = np.random.rand(N_test)
    accept_mask = random_values < p_items[test_labels]
    accepted_indices = np.where(accept_mask)[0]
    
    new_test_features = test_features[accepted_indices]
    new_test_labels = test_labels[accepted_indices]
    N_new = len(new_test_labels)
    
    print("Original number of test interactions:", N_test)
    print("Number of test interactions after negative sampling:", N_new)
    
    # -------------------------------
    # Step 3: Compute KL divergence before and after negative sampling.
    # -------------------------------
    counts_test = np.bincount(test_labels, minlength=n_items)
    p_test = counts_test / N_test

    if N_new > 0:
        counts_new = np.bincount(new_test_labels, minlength=n_items)
        p_new = counts_new / N_new
    else:
        p_new = np.zeros(n_items)
    
    # Dream (optimal) uniform distribution.
    p_optimal = np.ones(n_items) / n_items
    
    kl_before = kl_divergence(p_optimal, p_test)
    kl_after  = kl_divergence(p_optimal, p_new)
    
    print("KL divergence in test set before negative sampling: {:.6f}".format(kl_before))
    print("KL divergence in test set after negative sampling:  {:.6f}".format(kl_after))
    
    # -------------------------------
    # Step 4: Save the new unbiased test set.
    # -------------------------------
    output_file = r"./dataset/ml-1m/val_unbiased.npz"
    np.savez(output_file, features=new_test_features, labels=new_test_labels)
    print("Saved unbiased test set with {} interactions to '{}'".format(N_new, output_file))
    
    # -------------------------------
    # Step 5: Plot frequency counts before and after negative sampling,
    #         sorted by test set frequency (most popular items first).
    # -------------------------------
    # For plotting, sort the original test frequency counts in descending order.
    sorted_idx = np.argsort(counts_test)[::-1]
    
    counts_test_sorted = counts_test[sorted_idx]
    counts_new_sorted = (np.bincount(new_test_labels, minlength=n_items))[sorted_idx]
    
    plt.figure(figsize=(14, 6))
    
    # Plot before negative sampling.
    plt.subplot(1, 2, 1)
    plt.bar(range(n_items), counts_test_sorted, color='skyblue')
    plt.xlabel("Rank (sorted by test frequency)")
    plt.ylabel("Count")
    plt.title("Test Set Frequency Before Negative Sampling")
    
    # Plot after negative sampling.
    plt.subplot(1, 2, 2)
    plt.bar(range(n_items), counts_new_sorted, color='salmon')
    plt.xlabel("Rank (sorted by test frequency)")
    plt.ylabel("Count")
    plt.title("Test Set Frequency After Negative Sampling")
    
    plt.tight_layout()
    plt.show()


def create_item_popularity_csv(p):
    # -------------------------------
    # Step 1: Load the training NPZ file and compute item frequencies.
    # -------------------------------
    train_npz_path = r"./dataset/ml-1m/biased_eval_train.npz"
    data = np.load(train_npz_path)
    labels = data["labels"]  # assuming this array contains item IDs (item_id:token)
    total_interactions = len(labels)
    
    # Compute frequency counts for each unique item.
    unique_items, counts = np.unique(labels, return_counts=True)
    # Calculate popularity score: interaction_count divided by total_interactions.
    pop_scores = counts / total_interactions
    
    # Create a DataFrame from the computed counts.
    df_counts = pd.DataFrame({
        "item_id:token": unique_items,
        "interaction_count": counts,
        "pop_score": pop_scores
    })
    
    # -------------------------------
    # Step 2: Load the items_remapped CSV file.
    # -------------------------------
    items_csv_path = r"./dataset/ml-1m/items_remapped.csv"
    df_titles = pd.read_csv(items_csv_path)
    
    # -------------------------------
    # Step 3: Merge the NPZ counts with the items_remapped DataFrame.
    # -------------------------------
    df_merged = pd.merge(df_titles, df_counts, on="item_id:token", how="left")
    df_merged["interaction_count"] = df_merged["interaction_count"].fillna(0).astype(int)
    df_merged["pop_score"] = df_merged["pop_score"].fillna(0)

    # -------------------------------
    # Step 5: Compute popularity_label based on contribution to total interactions.
    # -------------------------------
    # To decide which items contribute to the top 20% (and bottom 20%) of total interactions,
    # sort by interaction_count and use the cumulative sum.
    
    # Make a copy for computing the top labels.
    df_top = df_merged.sort_values(by="interaction_count", ascending=False).reset_index(drop=True)
    total_sum = df_top["interaction_count"].sum()
    # Compute cumulative interaction count and its fraction.
    df_top["cum_interaction"] = df_top["interaction_count"].cumsum()
    df_top["cum_frac"] = df_top["cum_interaction"] / total_sum
    # Mark items in the top 20% cumulative.
    df_top["popularity_label_top"] = (df_top["cum_frac"] <= p).astype(int)
    
    # Similarly, compute for bottom labels.
    df_bottom = df_merged.sort_values(by="interaction_count", ascending=True).reset_index(drop=True)
    df_bottom["cum_interaction"] = df_bottom["interaction_count"].cumsum()
    df_bottom["cum_frac"] = df_bottom["cum_interaction"] / total_sum
    # Mark items in the bottom 20% cumulative.
    df_bottom["popularity_label_bottom"] = (df_bottom["cum_frac"] <= p).astype(int)
    
    # Create dictionaries mapping item_id:token to top and bottom labels.
    top_labels = df_top.set_index("item_id:token")["popularity_label_top"].to_dict()
    bottom_labels = df_bottom.set_index("item_id:token")["popularity_label_bottom"].to_dict()
    
    # Now assign overall popularity_label:
    #   If an item is marked as top (i.e. top_labels == 1) then label 1.
    #   Else if it is marked as bottom (i.e. bottom_labels == 1) then label -1.
    #   Else label 0.
    def assign_pop_label(item_id):
        if top_labels.get(item_id, 0) == 1:
            return 1
        elif bottom_labels.get(item_id, 0) == 1:
            return -1
        else:
            return 0
    
    df_merged["popularity_label"] = df_merged["item_id:token"].apply(assign_pop_label)
    
    # Drop the temporary columns if they exist in our working frames.
    df_merged = df_merged.drop(columns=[], errors="ignore")
    
    # -------------------------------
    # Step 6: Save the final DataFrame to a CSV file.
    # -------------------------------
    # Optionally, sort the final DataFrame by item_id for consistent ordering.
    df_final = df_merged.sort_values(by="interaction_count", ascending=False).reset_index(drop=True)
    output_csv =  r"./dataset/ml-1m/item_popularity_labels_with_titles.csv"
    df_final.to_csv(output_csv, index=False)
    print(f"CSV file '{output_csv}' created successfully.")
    

import requests
    
    
    
    
def search_movies(token, query, year=None, **kwargs):
    """
    Search for movies using the TMDB API with a Bearer Token.
    
    :param token: Your API Read Access Token.
    :param query: The search query (e.g., movie title).
    :param year: The release year of the movie (optional).
    :param kwargs: Optional parameters like language, page, etc.
    :return: JSON response from the API or an error message.
    """
    BASE_URL = "https://api.themoviedb.org/3"
    SEARCH_ENDPOINT = "/search/movie"
    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    params = {
        "query": query
    }
    if year is not None:
        params["year"] = year
    params.update(kwargs)
    
    response = requests.get(BASE_URL + SEARCH_ENDPOINT, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Request failed with status {response.status_code}", "details": response.text}



import json
import time
def process_and_save_movies(token, input_file=r"./dataset/ml-1m/items_remapped.csv", output_file=r"./dataset/ml-1m/processed_movies.csv", failed_file=r"./dataset/ml-1m/failed_movies.csv", num_rows=50):
    """
    Process a CSV file to fetch additional movie details from TMDB API and save results.
    
    :param token: Your API Read Access Token.
    :param input_file: Path to the input CSV file.
    :param output_file: Path to save the processed DataFrame.
    :param failed_file: Path to save the failed rows.
    :param num_rows: Number of rows to process (default is 50).
    """
    # Load genre mappings from JSON file
    with open(r"./dataset/ml-1m/genres.json", 'r') as f:
        genre_data = json.load(f)
        genre_map = {genre['id']: genre['name'] for genre in genre_data['genres']}

        # Load and preprocess the data
        df = pd.read_csv(input_file)
        df = df.rename(columns={
            'item_id:token': 'item_id',
            'movie_title:token_seq': 'movie_title',
            'release_year:token': 'release_year'
        })
        df['release_year'] = df['release_year'].astype(int)

        # Define new columns from the API response
        new_columns = [
            'adult', 'backdrop_path', 'genre_ids', 'id', 'original_language',
            'original_title', 'overview', 'popularity', 'poster_path',
            'release_date', 'title', 'video', 'vote_average', 'vote_count'
        ]
        for col in new_columns:
            df[col] = pd.NA

        # Process API requests with delay
        failed_indices = []
        request_count = 0
        for row in df.itertuples():
            query = row.movie_title
            year = row.release_year
            response = search_movies(token, query, year)
            request_count += 1

            # Add delay after every 50 requests
            if request_count % 50 == 0:
                print(f"Processed {request_count} requests, waiting for 1.1 seconds...")
                time.sleep(1.1)

            if "error" in response or len(response.get("results", [])) == 0:
                failed_indices.append(row.Index)
            else:
                result = response["results"][0]
                for col in new_columns:
                    value = result.get(col, pd.NA)
                    # Convert genre_ids to a comma-separated string of genre names
                    if col == 'genre_ids' and isinstance(value, list):
                        genre_names = [genre_map.get(genre_id, "Unknown") for genre_id in value]
                        value = ','.join(genre_names)
                    df.loc[row.Index, col] = value

    # Save results
    df.to_csv(output_file, index=False)
    if failed_indices:
        df_failed = df.loc[failed_indices]
        df_failed.to_csv(failed_file, index=False)


def get_movie_info(item_id):
    """
    Retrieve specific movie information for a given item_id from a CSV file.
    
    Args:
        item_id (int): The ID of the movie to look up.
        csv_file_path (str): Path to the CSV file containing movie data.
    
    Returns:
        list: A list containing [release_year, adult, genre_ids, original_language, overview].
              Returns None if item_id is not found.
    """
    
    csv_file_path = r"./dataset/ml-1m/processed_movies.csv"
    
    # Read only the required columns from the CSV
    df = pd.read_csv(csv_file_path, usecols=['item_id', 'release_year', 'adult', 'genre_ids', 'original_language', 'overview'])
    
    # Filter for the given item_id
    result = df[df['item_id'] == (item_id)]
    
    # If item_id is found, return the values as a list
    if not result.empty:
        return result[['release_year', 'adult', 'genre_ids', 'original_language', 'overview']].iloc[0].tolist()
    print("yoxduda blet ", item_id)
    # Return None if item_id is not found
    return None



def sample_users_interactions(
    X,
    inter_file=r"./dataset/mind_small_train/mind_small_train.inter",
    item_file=r"./dataset/mind_small_train/mind_small_train.item",
    sampled_inter_file=r"./dataset/mind_small_train/mind_small_train-sampled.inter",
    sampled_item_file=r"./dataset/mind_small_train/mind_small_train-sampled.item",
    sep='\t',
    random_state=None
):
    """
    Samples interactions of X random users from the interactions file and saves
    the sampled interactions and corresponding items to new files.

    Parameters
    ----------
    X : int
        Number of random users to sample.
    inter_file : str, optional
        Path to the original interactions file.
    item_file : str, optional
        Path to the original items file.
    sampled_inter_file : str, optional
        Path where the sampled interactions will be saved.
    sampled_item_file : str, optional
        Path where the sampled items will be saved.
    sep : str, optional
        Field separator used in the files (default: '\t').
    random_state : int or None, optional
        Random seed for reproducibility.
    """
    # Load interactions
    inter_df = pd.read_csv(inter_file, sep=sep)

    # Ensure there are enough users
    unique_users = inter_df['user_id:token'].unique()
    if X > len(unique_users):
        raise ValueError(
            f"X={X} is greater than the number of unique users ({len(unique_users)})"
        )

    # Sample users
    sampled_users = pd.Series(unique_users).sample(n=X, random_state=random_state).tolist()

    # Filter interactions
    sampled_inter_df = inter_df[inter_df['user_id:token'].isin(sampled_users)]

    # Save sampled interactions
    sampled_inter_df.to_csv(sampled_inter_file, sep=sep, index=False)

    # Load items
    item_df = pd.read_csv(item_file, sep=sep)

    # Filter items to those in sampled interactions
    sampled_item_ids = sampled_inter_df['item_id:token'].unique()
    sampled_item_df = item_df[item_df['item_id:token'].isin(sampled_item_ids)]

    # Save sampled items
    sampled_item_df.to_csv(sampled_item_file, sep=sep, index=False)

    print(f"Saved {len(sampled_inter_df)} interactions for {X} users to '{sampled_inter_file}'.")
    print(f"Saved {len(sampled_item_df)} items to '{sampled_item_file}'.")

def plot_interaction_distribution(file_path):
    df = pd.read_csv(file_path)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(df) + 1), df['interaction_count'], linewidth=1)
    plt.xlabel('Item Rank (Most Popular to Least Popular)')
    plt.ylabel('Interaction Count')
    plt.title('Item Popularity Distribution')
    plt.grid(True)
    plt.show()




def extract_sort_top_neurons(dataset_name):
    """
    Reads neuron activations and Cohen's d CSVs for a given dataset.
    Extracts all neurons whose activation 'count' exceeds 500, then splits them based on the sign of their 'cohen_d' values:
    1. Positive Cohen's d: sorted by descending absolute value and written to a CSV.
    2. Negative Cohen's d: sorted by descending absolute value and written to a separate CSV.
    Returns a tuple of the positive and negative output file paths.
    Raises KeyError if required columns are missing or indices are not found.
    """
    base_path = f"./dataset/{dataset_name}"
    activations_file = f"{base_path}/neuron_activations.csv"
    cohens_file = f"{base_path}/cohens_d.csv"
    pos_output = f"{base_path}/positive_cohens_d.csv"
    neg_output = f"{base_path}/negative_cohens_d.csv"

    # Load CSVs with index as first column
    df1 = pd.read_csv(activations_file, index_col=0)
    df2 = pd.read_csv(cohens_file, index_col=0)

    # Verify required columns
    if 'count' not in df1.columns:
        raise KeyError(f"'count' column not found in {activations_file}")
    if 'cohen_d' not in df2.columns:
        raise KeyError(f"'cohen_d' column not found in {cohens_file}")

    # Select all indices with activation count > 500
    selected = df1.loc[df1['count'] > 500].index

    # Retrieve Cohen's d values for selected indices
    try:
        cohen_d = df2.loc[selected, 'cohen_d']
    except KeyError as e:
        missing = list(set(selected) - set(df2.index))
        raise KeyError(f"Indices {missing} from activations not found in {cohens_file}") from e

    # Positive Cohen's d: sort by absolute value and save
    pos = cohen_d[cohen_d > 0].to_frame(name='cohen_d')
    pos['abs_cohen_d'] = pos['cohen_d'].abs()
    pos = pos.sort_values('abs_cohen_d', ascending=False).drop(columns='abs_cohen_d')
    pos.to_csv(pos_output)

    # Negative Cohen's d: sort by absolute value and save
    neg = cohen_d[cohen_d < 0].to_frame(name='cohen_d')
    neg['abs_cohen_d'] = neg['cohen_d'].abs()
    neg = neg.sort_values('abs_cohen_d', ascending=False).drop(columns='abs_cohen_d')
    neg.to_csv(neg_output)

    return pos_output, neg_output
