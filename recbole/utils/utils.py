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

import datetime
import importlib
import os
import random
import pandas as pd
import h5py

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


def early_stopping(value, best, cur_step, max_step, bigger=True):
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


import torch
import pandas as pd
from pathlib import Path
from typing import Union


# def compute_neuron_stats_by_row(
#         activations: torch.Tensor,
#         dataset: str
#     ) -> None:
    
#     labels_csv_path = rf"./dataset/{dataset}/item_popularity_labels.csv"
#     popular_out = rf"./dataset/{dataset}/neuron_stats_popular.csv"
#     unpopular_out = rf"./dataset/{dataset}/neuron_stats_unpopular.csv"
#     cohens_d_out = rf"./dataset/{dataset}/cohens_d.csv"
#     if activations.ndim != 2:
#         raise ValueError("`activations` must have shape (B, N)")
#     B, N = activations.shape

#     # ── 1. Load popularity labels ───────────────────────────────────────────────
#     label_ser = (
#         pd.read_csv(labels_csv_path, usecols=["item_id:token", "popularity_label"])
#         .rename(columns={"item_id:token": "item_id"})
#         .set_index("item_id")["popularity_label"]
#     )

#     # ── 2. Build per-row label tensor (1, −1, 0) ───────────────────────────────
#     labels = torch.zeros(B, dtype=torch.int8)
#     known_idx = label_ser.index.intersection(range(B))
#     labels[known_idx] = torch.tensor(label_ser.loc[known_idx].values, dtype=torch.int8)

#     pop_mask  = labels ==  1
#     unpop_mask = labels == -1

#     # Helper: stats for a boolean mask
#     def _stats(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
#         n = int(mask.sum().item())
#         if n:
#             subset = activations[mask]        # (n, N)
#             mean  = subset.mean(0)            # (N,)
#             sd    = subset.std(0, unbiased=False)
#         else:
#             mean = torch.zeros(N)
#             sd   = torch.zeros(N)
#         return mean, sd, n

#     # ── 3. Compute group stats ──────────────────────────────────────────────────
#     mean_pop,  sd_pop,  n_pop  = _stats(pop_mask)
#     mean_unp,  sd_unp,  n_unp  = _stats(unpop_mask)

#     # ── 4. Save per-group CSVs ──────────────────────────────────────────────────
#     def _to_csv(fname: str | Path, mean: torch.Tensor, sd: torch.Tensor):
#         pd.DataFrame({
#             "neuron": range(N),
#             "mean":   mean.tolist(),
#             "sd":     sd.tolist(),
#         }).to_csv(fname, index=False)

#     _to_csv(popular_out,   mean_pop, sd_pop)
#     _to_csv(unpopular_out, mean_unp, sd_unp)

#     # ── 5. Cohen’s d per neuron ────────────────────────────────────────────────
#     # pooled SD: sqrt( ((n1-1)*s1² + (n2-1)*s2²) / (n1+n2−2) )
#     # handle zero-row or zero-variance cases gracefully
#     denom = max(n_pop + n_unp - 2, 1)                      # scalar, ≥1
#     pooled_var = ((n_pop - 1) * sd_pop.pow(2) +
#                   (n_unp - 1) * sd_unp.pow(2)) / denom
#     pooled_sd = torch.sqrt(pooled_var)

#     valid = (pooled_sd != 0) & (n_pop > 0) & (n_unp > 0)
#     cohens_d = torch.full((N,), float('nan'))
#     cohens_d[valid] = (mean_pop[valid] - mean_unp[valid]) / pooled_sd[valid]

#     pd.DataFrame({
#         "neuron":   range(N),
#         "cohens_d": cohens_d.tolist(),
#     }).to_csv(cohens_d_out, index=False)


def compute_weighted_neuron_stats_by_row_item(
        activations: torch.Tensor,
        dataset: str,
        side: str
    ) -> None:
    labels_csv_path = rf"./dataset/{dataset}/{side}_popularity_labels.csv"
    popular_out = rf"./dataset/{dataset}/{side}/neuron_stats_pop.csv"
    unpopular_out = rf"./dataset/{dataset}/{side}/neuron_stats_unpop.csv"
    cohens_d_out = rf"./dataset/{dataset}/{side}/cohens_d.csv"
    if activations.ndim != 2:
        raise ValueError("`activations` must have shape (B, N)")
    B, N = activations.shape

    df = pd.read_csv(labels_csv_path, usecols=[rf"{side}_id:token", "popularity_label", "pop_score"])
    df['item_id'] = df[rf"{side}_id:token"].astype(int)  # Assuming general 'item_id' for index
    label_ser = df.set_index('item_id')["popularity_label"]
    pop_score_ser = df.set_index('item_id')["pop_score"]

    labels = torch.zeros(B, dtype=torch.int8)
    known_idx = label_ser.index.intersection(range(B))
    labels[known_idx] = torch.tensor(label_ser.loc[known_idx].values, dtype=torch.int8)

    pop_scores = torch.zeros(B, dtype=torch.float)
    pop_scores[known_idx] = torch.tensor(pop_score_ser.loc[known_idx].values, dtype=torch.float)
    # Normalize pop_scores to [0, 1]
    min_pop = pop_scores.min()
    max_pop = pop_scores.max()
    if max_pop > min_pop:
        pop_scores = (pop_scores - min_pop) / (max_pop - min_pop)

    pop_mask = labels == 1
    unpop_mask = labels == -1  # Assuming -1 for unpopular, adjust if necessary to ==0

    # Helper: weighted stats for a boolean mask
    def _stats(mask: torch.Tensor, is_pop: bool) -> tuple[torch.Tensor, torch.Tensor, float]:
        mask_idx = mask.nonzero(as_tuple=False).squeeze(-1)
        n_items = len(mask_idx)
        if n_items == 0:
            return torch.zeros(N), torch.zeros(N), 0.0
        subset = activations[mask_idx]  # (n, N)
        group_pop_scores = pop_scores[mask_idx]
        weights = group_pop_scores if is_pop else (1.0 - group_pop_scores)
        effective_n = weights.sum().item()
        if effective_n <= 0:
            return torch.zeros(N), torch.zeros(N), 0.0
        # Weighted mean
        mean = torch.sum(weights.unsqueeze(1) * subset, dim=0) / effective_n
        # Weighted variance (population style, matching original std unbiased=False)
        var = torch.sum(weights.unsqueeze(1) * (subset - mean.unsqueeze(0))**2, dim=0) / effective_n
        sd = torch.sqrt(var)
        return mean, sd, effective_n

    # Compute group stats
    mean_pop, sd_pop, effective_n_pop = _stats(pop_mask, is_pop=True)
    mean_unp, sd_unp, effective_n_unp = _stats(unpop_mask, is_pop=False)

    # Save per-group CSVs
    def _to_csv(fname: str, mean: torch.Tensor, sd: torch.Tensor):
        pd.DataFrame({
            "neuron": range(N),
            "mean":   mean.tolist(),
            "sd":     sd.tolist(),
        }).to_csv(fname, index=False)

    _to_csv(popular_out, mean_pop, sd_pop)
    _to_csv(unpopular_out, mean_unp, sd_unp)

    # Cohen’s d per neuron
    denom = max(effective_n_pop + effective_n_unp - 2, 1)
    pooled_var = ((effective_n_pop - 1) * sd_pop.pow(2) +
                  (effective_n_unp - 1) * sd_unp.pow(2)) / denom
    pooled_sd = torch.sqrt(pooled_var)

    valid = (pooled_sd != 0) & (effective_n_pop > 0) & (effective_n_unp > 0)
    cohens_d = torch.full((N,), float('nan'))
    cohens_d[valid] = (mean_pop[valid] - mean_unp[valid]) / pooled_sd[valid]

    pd.DataFrame({
        "neuron":   range(N),
        "cohens_d": cohens_d.tolist(),
    }).to_csv(cohens_d_out, index=False)



def compute_neuron_stats_by_row(
        activations: torch.Tensor,
        dataset: str,
        side: str
    ) -> None:
    labels_csv_path = rf"./dataset/{dataset}/{side}_popularity_labels.csv"
    popular_out = rf"./dataset/{dataset}/{side}/neuron_stats_pop.csv"
    unpopular_out = rf"./dataset/{dataset}/{side}/neuron_stats_unpop.csv"
    cohens_d_out = rf"./dataset/{dataset}/{side}/cohens_d.csv"
    if activations.ndim != 2:
        raise ValueError("`activations` must have shape (B, N)")
    B, N = activations.shape


    label_ser = (
        pd.read_csv(labels_csv_path, usecols=[rf"{side}_id:token", "popularity_label"])
        .set_index(rf"{side}_id:token")["popularity_label"]
    )

    labels = torch.zeros(B, dtype=torch.int8)
    known_idx = label_ser.index.intersection(range(B))
    labels[known_idx] = torch.tensor(label_ser.loc[known_idx].values, dtype=torch.int8)

    pop_mask  = labels ==  1
    unpop_mask = labels == 0

    # Helper: stats for a boolean mask
    def _stats(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        n = int(mask.sum().item())
        if n:
            subset = activations[mask]        # (n, N)
            mean  = subset.mean(0)            # (N,)
            sd    = subset.std(0, unbiased=False)
        else:
            mean = torch.zeros(N)
            sd   = torch.zeros(N)
        return mean, sd, n

    # ── 3. Compute group stats ──────────────────────────────────────────────────
    mean_pop,  sd_pop,  n_pop  = _stats(pop_mask)
    mean_unp,  sd_unp,  n_unp  = _stats(unpop_mask)

    # ── 4. Save per-group CSVs ──────────────────────────────────────────────────
    def _to_csv(fname: str | Path, mean: torch.Tensor, sd: torch.Tensor):
        pd.DataFrame({
            "neuron": range(N),
            "mean":   mean.tolist(),
            "sd":     sd.tolist(),
        }).to_csv(fname, index=False)

    _to_csv(popular_out,   mean_pop, sd_pop)
    _to_csv(unpopular_out, mean_unp, sd_unp)

    # ── 5. Cohen’s d per neuron ────────────────────────────────────────────────
    # pooled SD: sqrt( ((n1-1)*s1² + (n2-1)*s2²) / (n1+n2−2) )
    # handle zero-row or zero-variance cases gracefully
    denom = max(n_pop + n_unp - 2, 1)                      # scalar, ≥1
    pooled_var = ((n_pop - 1) * sd_pop.pow(2) +
                  (n_unp - 1) * sd_unp.pow(2)) / denom
    pooled_sd = torch.sqrt(pooled_var)

    valid = (pooled_sd != 0) & (n_pop > 0) & (n_unp > 0)
    cohens_d = torch.full((N,), float('nan'))
    cohens_d[valid] = (mean_pop[valid] - mean_unp[valid]) / pooled_sd[valid]

    pd.DataFrame({
        "neuron":   range(N),
        "cohens_d": cohens_d.tolist(),
    }).to_csv(cohens_d_out, index=False)






def get_extreme_correlations(file_name: str, dataset=None):
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
    pos_series = df.loc[df["cohens_d"] > 0, "cohens_d"]
    neg_series = df.loc[df["cohens_d"] < 0, "cohens_d"]

    # 4) zip index-labels (which by default are 0,1,2… or the original row numbers)
    pos_list = list(pos_series.items())  # each item is (index_label, value)
    neg_list = list(neg_series.items())


    return pos_list, neg_list


import matplotlib.pyplot as plt


def plot_tensor_sorted_by_popularity(tensor: torch.Tensor, dataset: str):
    """
    Sorts the given tensor (index 1 onwards) based on the pop_score from CSV,
    and plots the sorted tensor values.

    Parameters:
        tensor (torch.Tensor): 1D tensor of size N+1, where index 0 is unused (item ID 0 doesn't exist).
        csv_path (str): Path to the CSV file with 'item_id:token' and 'pop_score' columns.
    """
    # Load CSV
    df = pd.read_csv(rf"./dataset/{dataset}/item_popularity_labels.csv")

    # Use item_id:token as integer item ID
    df['item_id'] = df['item_id:token'].astype(int)

    # Sanity check
    assert tensor.shape[0] == df['item_id'].max() + 1, "Tensor size must match max item ID + 1"

    # Build pop_score tensor aligned to item ID
    pop_scores = torch.zeros_like(tensor)
    pop_scores[df['item_id'].values] = torch.tensor(df['pop_score'].values, dtype=torch.long)

    # Skip index 0 (no item with ID 0)
    tensor_valid = tensor[1:]
    pop_scores_valid = pop_scores[1:]

    # Sort tensor by popularity score
    sorted_indices = torch.argsort(pop_scores_valid)
    sorted_tensor = tensor_valid[sorted_indices]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(sorted_tensor)), sorted_tensor.numpy())
    plt.xlabel('Items sorted by pop_score')
    plt.ylabel('Tensor values')
    plt.title('Tensor values sorted by item popularity')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


import csv

import os
import csv
import matplotlib.pyplot as plt

import os, csv
import matplotlib.pyplot as plt
from math import isnan

import os, csv
import matplotlib.pyplot as plt
import numpy as np
import math


import os, csv, math, numpy as np
import matplotlib.pyplot as plt


import os
import csv
import math
from pathlib import Path
from typing import Dict, List, Any


import os
import csv
import math
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np


def plot_ndcg_vs_fairness(
    dataset: str,
    model: str = "LightGCN",
    alpha_n: int | None = None,
    alpha_i: int | None = None,
    alpha_u: int | None = None,
    show: bool = True,
    facet: bool = True,
):
    """Plot NDCG‑driven evaluation figures.

    This version fixes the *tail* calculation so that

        ``ndcgtail@10 = ndcg@10 − ndcghead@10``

    instead of relying on a potentially noisy ``ndcgtail@10`` column in the
    result CSV files.

    Besides the existing *item‑slice* plots (Head/Mid/Tail), this version also
    generates *user‑slice* plots (Passive/Mid/Active) so that you get **two**
    3‑panel figures:

    1. ``NDCG(head/mid/tail)   vs overall NDCG@10``  (items)
    2. ``NDCG(passive/mid/active) vs overall NDCG@10`` (users)

    The user‑slice figure is stored in ``figs["slice_user"]`` and is therefore
    handled exactly like the original slice figure (``figs["slice"]``).
    """

    if not dataset:
        raise ValueError("Please provide a dataset name, e.g. dataset='lastfm'.")

    # ------------------------------------------------------------------ paths
    files = {
        "PopSteer":        rf"dataset/{dataset}/results/{model}_user_{dataset}-results.csv",
        "Random-reranker": rf"dataset/{dataset}/results/{model}_random_{dataset}-resultss.csv",
        "IPR":             rf"dataset/{dataset}/results/{model}_ipr_{dataset}-results.csv",
        "FAIR":            rf"dataset/{dataset}/results/{model}_fair_{dataset}-results.csv",
        "PCT":             rf"dataset/{dataset}/results/{model}_pct_{dataset}-results.csv",
        "Min-reg":             rf"dataset/{dataset}/results/{model}_min_reg_{dataset}-results.csv",

    }

    # -------------------------------------------------------------- csv loader
    def load_rows(path: str) -> List[Dict[str, Any]]:
        if not os.path.isfile(path):
            return []
        rows: List[Dict[str, Any]] = []
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                numeric_row: Dict[str, Any] = {}
                for k, v in r.items():
                    try:
                        numeric_row[k] = float(v)
                    except (ValueError, TypeError):
                        numeric_row[k] = v
                rows.append(numeric_row)
        return rows

    data = {lbl: load_rows(p) for lbl, p in files.items()}

    # ────────────────────────────── filter PopSteer rows by α‑parameters
    if data.get("PopSteer"):
        first = data["PopSteer"][:1]
        rest: List[Dict[str, Any]] = []
        for r in data["PopSteer"][1:]:
            keep = True
            if alpha_n is not None:
                keep &= int(float(r.get("alpha_n", -1))) == alpha_n
            if alpha_i is not None:
                keep &= int(float(r.get("alpha_i", -1))) == alpha_i
            if alpha_u is not None:
                keep &= int(float(r.get("alpha_u", -1))) == alpha_u
            # Ensure α_i = α_u for paired settings
            if keep:
                rest.append(r)
        data["PopSteer"] = first + rest

    # ------------------ pull baseline from FAIR → SASRec
    if data.get("FAIR"):
        data["SASRec"] = [data["FAIR"][0]]
        data["FAIR"] = data["FAIR"][1:]
        if not data["FAIR"]:
            del data["FAIR"]

    # ───────────────────────────── compute *tail* on‑the‑fly
    # The core fix: overwrite / create ndcgtail@10 = ndcg − ndcghead@10
    for rows in data.values():
        for r in rows:
            if r.get("ndcg") is not None and r.get("ndcghead@10") is not None:
                r["ndcgtail@10"] = r["ndcg"] - r["ndcghead@10"]

    # ------------------ thresholds: 3 % & 5 % drops
    baseline_ndcg = data.get("SASRec", [{}])[0].get("ndcg")
    thr_03 = 0.97 * baseline_ndcg if baseline_ndcg is not None else None  # 3 % drop
    thr_05 = 0.95 * baseline_ndcg if baseline_ndcg is not None else None  # 5 % drop

    # ------------------------------------------- style maps
    colours = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    markers = ["o", "s", "^", "D", "P", "X", "v", "*"]
    labels_present = [lbl for lbl, rows in data.items() if rows]
    col_map = {lbl: colours[i % len(colours)] for i, lbl in enumerate(labels_present)}
    mrk_map = {lbl: markers[i % len(markers)] for i, lbl in enumerate(labels_present)}

    figs: Dict[str, plt.Figure] = {}

    # ========================================================================
    # 1) ITEM‑SLICE PLOT ───────────────────────── head / mid / tail
    # ========================================================================
    item_slice_keys = ["ndcghead@10", "ndcgtail@10"]
    item_slice_titles = {
        "ndcghead@10": "Head NDCG@10",
        "ndcgtail@10": "Tail NDCG@10 (computed)",
    }

    def _make_slice_figure(keys: List[str], titles: Dict[str, str], fig_key: str, super_title: str):
        """Internal helper to avoid copy‑pasting slice code."""
        if facet:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
            axes = np.atleast_1d(axes)
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            axes = [ax] * 3  # type: ignore[assignment]

        for ax_idx, (ax, sk) in enumerate(zip(axes, keys)):
            for lbl, rows in data.items():
                pts = [
                    (r.get("ndcg"), r.get(sk))
                    for r in rows
                    if r.get("ndcg") is not None and r.get(sk) is not None and not math.isnan(r[sk])
                ]
                if not pts:
                    continue
                xs, ys = zip(*sorted(pts, key=lambda t: t[0]))
                linestyle = "-" if len(xs) > 1 else "None"
                ax.plot(
                    xs,
                    ys,
                    marker=mrk_map[lbl],
                    linestyle=linestyle,
                    color=col_map[lbl],
                    linewidth=1,
                    markersize=6,
                    label=lbl if sk == keys[0] else "_nolegend_",
                )

            # vertical drop lines (one label per legend)
            if thr_03 is not None:
                ax.axvline(
                    thr_03,
                    linestyle="--",
                    linewidth=1,
                    color="grey",
                    label="3 % drop" if ax_idx == 0 else "_nolegend_",
                )
            if thr_05 is not None:
                ax.axvline(
                    thr_05,
                    linestyle=":",
                    linewidth=1,
                    color="grey",
                    label="5 % drop" if ax_idx == 0 else "_nolegend_",
                )

            ax.set_xlabel("NDCG@10 (overall)")
            ax.set_ylabel(titles[sk])
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            if facet:
                ax.set_title(titles[sk])

        axes[0].legend(title="File", fontsize=8, frameon=True)
        fig.suptitle(super_title, y=1.02 if facet else 1.03)
        fig.tight_layout()
        figs[fig_key] = fig

    # create original item‑slice figure
    _make_slice_figure(
        item_slice_keys,
        item_slice_titles,
        fig_key="slice",
        super_title=f"{dataset}: NDCG(head/mid/tail) vs overall NDCG@10",
    )

    # ========================================================================
    # 2) USER‑SLICE PLOT ───────────────────── passive / mid / active
    # ========================================================================
    user_slice_keys = ["ndcgtailuser@10", "ndcgmiduser@10", "ndcgheaduser@10"]
    user_slice_titles = {
        "ndcgtailuser@10": "Tail Users NDCG@10",
        "ndcgmiduser@10": "Neutral Users NDCG@10",
        "ndcgheaduser@10": "Head Users NDCG@10",
    }

    # _make_slice_figure(
    #     user_slice_keys,
    #     user_slice_titles,
    #     fig_key="slice_user",
    #     super_title=f"{dataset}: NDCG(passive/mid/active) vs overall NDCG@10",
    # )

    # ========================================================================
    # 3) FAIRNESS SCATTER PLOTS ───────────────────────────────────────────────
    # ========================================================================
    fairness_specs = [
        ("avgpop@10", "Average Popularity @10", "avgpop@10"),
        ("gini@10", "Gini Index @10", "gini@10"),
        ("covn@10", "Coverage‑5 @10", "covn@10"),
    ]

    for metric_key, metric_title, dict_key in fairness_specs:
        fig, ax = plt.subplots()
        for lbl, rows in data.items():
            xs = [r["ndcg"] for r in rows if "ndcg" in r and dict_key in r]
            ys = [r[dict_key] for r in rows if "ndcg" in r and dict_key in r]
            if not xs:
                continue
            ax.scatter(
                xs,
                ys,
                marker=mrk_map[lbl],
                label=lbl,
                edgecolors="none",
                alpha=0.85,
                color=col_map[lbl],
            )

        if thr_03 is not None:
            ax.axvline(thr_03, linestyle="--", linewidth=1, color="grey", label="3 % drop")
        if thr_05 is not None:
            ax.axvline(thr_05, linestyle=":", linewidth=1, color="grey", label="5 % drop")

        ax.set_xlabel("NDCG@10")
        ax.set_ylabel(metric_title)
        ax.set_title(f"{dataset}: NDCG@10 vs {metric_title}")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend()
        figs[metric_key] = fig

    # ------------------------------------------------------------ show / return
    if show:
        plt.show()

    return figs


# ---------------------------------------------------------------------------
# Helper: remove_sparse_users_items (unchanged, kept for completeness)
# ---------------------------------------------------------------------------
import pandas as pd
import shutil

def remove_sparse_users_items(n: int, dataset: str, base_dir: str = "./dataset") -> None:
    """Iteratively filter users/items with fewer than *n* interactions."""

    ds_dir = Path(base_dir) / dataset
    inter_path = ds_dir / f"{dataset}.inter"
    inter_bak  = ds_dir / f"{dataset}.inter.original"

    # --- Step 0: Backups (only once) ---
    if not inter_bak.exists():
        shutil.copy2(inter_path, inter_bak)

    # --- Step 1: Load ---
    interactions = pd.read_csv(inter_path, sep="\t", header=0)

    # --- Step 2: Iterative filtering ---
    iteration = 0
    while True:
        iteration += 1
        before = interactions.shape[0]

        valid_users = interactions["user_id:token"].value_counts()
        valid_users = valid_users[valid_users >= n].index
        interactions = interactions[interactions["user_id:token"].isin(valid_users)]

        valid_items = interactions["artists_id:token"].value_counts()
        valid_items = valid_items[valid_items >= n].index
        interactions = interactions[interactions["artists_id:token"].isin(valid_items)]

        after = interactions.shape[0]
        print(f"Iteration {iteration}: {before} -> {after} interactions remain")
        if after == before:
            break

    # --- Step 3: Overwrite originals (atomic‑ish) ---
    tmp_inter = inter_path.with_suffix(".inter.tmp")
    interactions.to_csv(tmp_inter, sep="\t", index=False)
    tmp_inter.replace(inter_path)

    print(
        f"Done. Wrote {interactions.shape[0]} interactions and "
        f"{len(interactions['artists_id:token'].unique())} items."
    )



import shutil

def remove_sparse_users_items(n: int, dataset: str, base_dir: str = "./dataset") -> None:
    ds_dir = Path(base_dir) / dataset
    inter_path = ds_dir / f"{dataset}.inter"
    # item_path  = ds_dir / f"{dataset}.item"
    inter_bak  = ds_dir / f"{dataset}.inter.original"
    # item_bak   = ds_dir / f"{dataset}.item.original"

    # --- Step 0: Backups (only once) ---
    if not inter_bak.exists():
        shutil.copy2(inter_path, inter_bak)
    # if not item_bak.exists():
        # shutil.copy2(item_path, item_bak)

    # --- Step 1: Load ---
    interactions = pd.read_csv(inter_path, sep="\t", header=0)
    # items        = pd.read_csv(item_path,  sep="\t", header=0)

    # --- Step 2: Iterative filtering ---
    iteration = 0
    while True:
        iteration += 1
        before = interactions.shape[0]

        valid_users = interactions["user_id:token"].value_counts()
        valid_users = valid_users[valid_users >= n].index
        interactions = interactions[interactions["user_id:token"].isin(valid_users)]

        valid_items = interactions["item_id:token"].value_counts()
        valid_items = valid_items[valid_items >= n].index
        interactions = interactions[interactions["item_id:token"].isin(valid_items)]

        after = interactions.shape[0]
        print(f"Iteration {iteration}: {before} -> {after} interactions remain")
        if after == before:
            break

    # --- Step 3: Sync items ---
    # items = items[items["item_id:token"].isin(interactions["item_id:token"])]

    # --- Step 4: Overwrite originals (atomic-ish) ---
    tmp_inter = inter_path.with_suffix(".inter.tmp")
    # tmp_item  = item_path.with_suffix(".item.tmp")

    interactions.to_csv(tmp_inter, sep="\t", index=False)
    # items.to_csv(tmp_item, sep="\t", index=False)

    tmp_inter.replace(inter_path)
    # tmp_item.replace(item_path)

    print(f"Done. Wrote {interactions.shape[0]} interactions and {len(interactions['item_id:token'].unique())} items.")



SECONDS_PER_DAY = 24 * 60 * 60  # 86,400

def retain_last_x_days(dataset: str,
                       days: int,
                       *,
                       sep: str = '\t') -> pd.DataFrame:
    """
    Retain rows whose `timestamp:float` lies within the last *days* days
    and print the total time span of the dataset, ignoring known bad timestamps.

    Parameters
    ----------
    dataset : str
        Dataset name (assumes "./dataset/{dataset}/{dataset}.inter").
    days : int
        Number of days to keep (≥ 1).
    sep : str, default '\\t'
        Field separator used in the .inter file.

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame containing only the last *days* of data.
    """
    if days < 1:
        raise ValueError("`days` must be at least 1.")

    # ── 1. Resolve paths ───────────────────────────────────────────
    csv_path = Path(f"./dataset/{dataset}/{dataset}.inter")
    out_path = Path(f"./dataset/{dataset}/{dataset}.inter.last{days}d")

    # ── 2. Load the interactions file ──────────────────────────────
    df = pd.read_csv(csv_path, sep=sep)

    # ── 3. Exclude known bad timestamps for span calculations ─────
    bad_timestamps = {1997728387, 1685138202, 1470888595, 1471057491}
    valid_ts = df[~df["timestamp:float"].isin(bad_timestamps)]["timestamp:float"]
    min_ts = valid_ts.min()
    max_ts = valid_ts.max()
    span_days = (max_ts - min_ts) / SECONDS_PER_DAY

    print(f"Dataset time span: {span_days:.2f} days "
          f"({min_ts:.0f} → {max_ts:.0f})")

    # ── 4. Compute the cutoff timestamp for the last *days* ────────
    cutoff = max_ts - days * SECONDS_PER_DAY

    # ── 5. Keep only rows at or after the cutoff ───────────────────
    filtered = df[df["timestamp:float"] >= cutoff].copy()

    # ── 6. Save the result ─────────────────────────────────────────
    filtered.to_csv(out_path, sep=sep, index=False)

    return filtered



def keep_random_users(
                      dataset: str,
                      x: int,
                      user_col: str = "session_id:token",
                      sep: str = "\t",
                      seed: int = 42,
                      chunksize: int = 1_000_000):
    """
    Keep only rows whose user_id is in a random sample of X users.

    Parameters
    ----------
    input_path : str
        Path to yoochose-clicks.inter (original file).
    output_path : str
        Path to yoochose-clicks-new.inter (filtered file).
    x : int
        Number of distinct users to keep.
    user_col : str
        Name of the user id column.
    sep : str
        Field separator (RecBole .inter files are usually tab-separated).
    seed : int
        RNG seed for reproducibility.
    chunksize : int
        Number of rows per chunk when scanning with pandas.
    """
    random.seed(seed)
    input_path = rf"./dataset/{dataset}/{dataset}.inter"
    output_path = rf"./dataset/{dataset}/{dataset}-new.inter"

    # ---------- Pass 1: collect all unique user ids ----------
    user_ids = set()
    header = pd.read_csv(input_path, nrows=0, sep=sep).columns.tolist()
    for chunk in pd.read_csv(input_path, sep=sep, chunksize=chunksize):
        user_ids.update(chunk[user_col].unique())

    if x > len(user_ids):
        raise ValueError(f"Requested {x} users but file only has {len(user_ids)}.")

    sampled_users = set(random.sample(list(user_ids), x))

    # ---------- Pass 2: write filtered rows ----------
    with open(output_path, "w", encoding="utf-8") as out_f:
        # write header
        out_f.write(sep.join(header) + "\n")

        for chunk in pd.read_csv(input_path, sep=sep, chunksize=chunksize):
            keep = chunk[chunk[user_col].isin(sampled_users)]
            keep.to_csv(out_f, sep=sep, index=False, header=False, mode="a")





def create_pop_unpop_mappings(dataset: str, embeddings: torch.Tensor) -> None:
    """
    Creates mapping CSV files for popular and unpopular item pairs based on embeddings and popularity labels.

    Args:
        embeddings (torch.Tensor): Tensor of item embeddings with shape (N, 64), where N is the number of items
                                   and the nth row corresponds to item ID n (0 to N-1).
        item_pop_csv (str): Path to the input CSV file containing 'item_id:token' and 'popularity_label' columns.
        pop_mapping_csv (str): Path to save the popular mapping CSV (columns: item_id, paired_id).
        unpop_mapping_csv (str): Path to save the unpopular mapping CSV (columns: item_id, paired_id).
    """
    
    dataset_path = Path(".", "dataset", dataset)
    item_pop_csv = dataset_path / "item_popularity_labels.csv"
    unpop_mapping_csv = dataset_path / "unpop_mapping.csv"
    pop_mapping_csv = dataset_path / "pop_mapping.csv"

    df_items = pd.read_csv(item_pop_csv, usecols=["item_id:token", "popularity_label"])
    df_items = df_items.dropna(subset=["popularity_label"])
    df_items = df_items.rename(columns={"item_id:token": "item_id"})
    df_items["item_id"] = df_items["item_id"].astype(int)

    # Get N from embeddings
    N = embeddings.shape[0]

    # Extract popular and unpopular item IDs
    popular_ids = df_items[df_items["popularity_label"] == 1]["item_id"].values
    unpopular_ids = df_items[df_items["popularity_label"] == -1]["item_id"].values

    # Create a dict for quick label lookup (default to 0 if missing)
    label_dict = df_items.set_index("item_id")["popularity_label"].to_dict()

    # Unpopular mapping
    if len(unpopular_ids) > 0:
        unpop_embeddings = embeddings[unpopular_ids]  # (num_unpop, 64)
        sim_unpop = embeddings @ unpop_embeddings.T  # (N, num_unpop)
    else:
        sim_unpop = torch.empty((N, 0))  # Handle edge case with no unpopular items

    pairs_unpop = []
    for i in range(N):
        label = label_dict.get(i, 0)
        if label == -1:
            pairs_unpop.append(i)
        else:
            if len(unpopular_ids) == 0:
                pairs_unpop.append(i)  # Fallback to self if no unpopular items
            else:
                closest_idx = sim_unpop[i].argmax().item()
                pairs_unpop.append(unpopular_ids[closest_idx])

    df_unpop = pd.DataFrame({"item_id": range(N), "paired_id": pairs_unpop})
    df_unpop.to_csv(unpop_mapping_csv, index=False)

    # Popular mapping
    if len(popular_ids) > 0:
        pop_embeddings = embeddings[popular_ids]  # (num_pop, 64)
        sim_pop = embeddings @ pop_embeddings.T  # (N, num_pop)
    else:
        sim_pop = torch.empty((N, 0))  # Handle edge case with no popular items

    pairs_pop = []
    for i in range(N):
        label = label_dict.get(i, 0)
        if label == 1:
            pairs_pop.append(i)
        else:
            if len(popular_ids) == 0:
                pairs_pop.append(i)  # Fallback to self if no popular items
            else:
                closest_idx = sim_pop[i].argmax().item()
                pairs_pop.append(popular_ids[closest_idx])

    df_pop = pd.DataFrame({"item_id": range(N), "paired_id": pairs_pop})
    df_pop.to_csv(pop_mapping_csv, index=False)


def create_pop_unpop_mappings(dataset: str, embeddings: torch.Tensor) -> None:
    """
    Creates mapping CSV files for popular and unpopular item pairs based on embeddings and popularity labels.

    Args:
        embeddings (torch.Tensor): Tensor of item embeddings with shape (N, 64), where N is the number of items
                                   and the nth row corresponds to item ID n (0 to N-1).
        item_pop_csv (str): Path to the input CSV file containing 'item_id:token' and 'popularity_label' columns.
        pop_mapping_csv (str): Path to save the popular mapping CSV (columns: item_id, paired_id).
        unpop_mapping_csv (str): Path to save the unpopular mapping CSV (columns: item_id, paired_id).
    """
    
    dataset_path = Path(".", "dataset", dataset)
    item_pop_csv = dataset_path / "item_popularity_labels.csv"
    unpop_mapping_csv = dataset_path / "unpop_mapping.csv"
    pop_mapping_csv = dataset_path / "pop_mapping.csv"

    df_items = pd.read_csv(item_pop_csv, usecols=["item_id:token", "popularity_label"])
    df_items = df_items.dropna(subset=["popularity_label"])
    df_items = df_items.rename(columns={"item_id:token": "item_id"})
    df_items["item_id"] = df_items["item_id"].astype(int)

    # Get N from embeddings
    N = embeddings.shape[0]

    # Extract popular and unpopular item IDs
    popular_ids = df_items[df_items["popularity_label"] == 1]["item_id"].values
    unpopular_ids = df_items[df_items["popularity_label"] == -1]["item_id"].values

    # Create a dict for quick label lookup (default to 0 if missing)
    label_dict = df_items.set_index("item_id")["popularity_label"].to_dict()

    # Unpopular mapping
    if len(unpopular_ids) > 0:
        unpop_embeddings = embeddings[unpopular_ids]  # (num_unpop, 64)
        sim_unpop = embeddings @ unpop_embeddings.T  # (N, num_unpop)
    else:
        sim_unpop = torch.empty((N, 0))  # Handle edge case with no unpopular items

    pairs_unpop = []
    for i in range(N):
        if i == 0:
            pairs_unpop.append(0)
            continue
        label = label_dict.get(i, 0)
        if label == -1:
            pairs_unpop.append(i)
        else:
            if len(unpopular_ids) == 0:
                pairs_unpop.append(i)  # Fallback to self if no unpopular items
            else:
                closest_idx = sim_unpop[i].argmax().item()
                pairs_unpop.append(unpopular_ids[closest_idx])

    df_unpop = pd.DataFrame({"item_id": range(N), "paired_id": pairs_unpop})
    df_unpop.to_csv(unpop_mapping_csv, index=False)

    # Popular mapping
    if len(popular_ids) > 0:
        pop_embeddings = embeddings[popular_ids]  # (num_pop, 64)
        sim_pop = embeddings @ pop_embeddings.T  # (N, num_pop)
    else:
        sim_pop = torch.empty((N, 0))  # Handle edge case with no popular items

    pairs_pop = []
    for i in range(N):
        if i == 0:
            pairs_pop.append(0)
            continue
        label = label_dict.get(i, 0)
        if label == 1:
            pairs_pop.append(i)
        else:
            if len(popular_ids) == 0:
                pairs_pop.append(i)  # Fallback to self if no popular items
            else:
                closest_idx = sim_pop[i].argmax().item()
                pairs_pop.append(popular_ids[closest_idx])

    df_pop = pd.DataFrame({"item_id": range(N), "paired_id": pairs_pop})
    df_pop.to_csv(pop_mapping_csv, index=False)



def replace_with_mappings(sequences: torch.Tensor, popular: bool, dataset: str) -> torch.Tensor:
    """
    Replaces item IDs in the input sequences tensor with their mapped paired IDs based on the popularity flag.

    Args:
        sequences (torch.Tensor): Input tensor of shape (B, M) containing item IDs (integers from 0 to N-1).
        popular (bool): If True, use popular mapping; if False, use unpopular mapping.
        pop_mapping_csv (str): Path to the popular mapping CSV (columns: item_id, paired_id).
        unpop_mapping_csv (str): Path to the unpopular mapping CSV (columns: item_id, paired_id).

    Returns:
        torch.Tensor: Output tensor of shape (B, M) with replaced item IDs.
    """
    dataset_path = Path(".", "dataset", dataset)
    unpop_mapping_csv = dataset_path / "unpop_mapping.csv"
    pop_mapping_csv = dataset_path / "pop_mapping.csv"

    mapping_csv = pop_mapping_csv if popular else unpop_mapping_csv
    df_map = pd.read_csv(mapping_csv)
    max_id = df_map['item_id'].max()
    map_list = [0] * (max_id + 1)
    for _, row in df_map.iterrows():
        map_list[int(row['item_id'])] = int(row['paired_id'])
    map_tensor = torch.tensor(map_list, dtype=torch.long, device=sequences.device)
    result = map_tensor[sequences]
    return result


def save_batch_activations(bulk_data, neuron_count, dataset, popular):
    print(popular, " blya")
    if popular == True:
        file_path = rf"./dataset/{dataset}/neuron_activations_sasrecsae_final_pop.h5"
    elif popular == False:
        file_path = rf"./dataset/{dataset}/neuron_activations_sasrecsae_final_unpop.h5"
    else:
        file_path = rf"./dataset/{dataset}/neuron_activations_sasrecsae_final.h5"
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
            

def save_mean_SD(dataset: str, *, popular: bool) -> int:
    """
    Compute row-wise mean and SD for the neuron-activation tensor in an .h5 file,
    save them to CSV, and RETURN the sample count (n) as an int.

    Parameters
    ----------
    dataset : str
        Name of the dataset subdirectory (e.g. "ml-1m").
    popular : bool
        True  → read the '_pop' file; False → read the '_unpop' file.

    Returns
    -------
    int
        Number of samples each mean/SD was computed from.
    """
    suffix   = "_pop" if popular else "_unpop"
    suffix = "" if popular is None else suffix
    h5_path  = Path(f"./dataset/{dataset}/neuron_activations_sasrecsae_final{suffix}.h5")
    csv_path = Path(f"./dataset/{dataset}/user/neuron_stats{suffix}.csv")
    dataset_name = "dataset"   # change if the key inside the HDF5 is different

    # --- Load tensor -------------------------------------------------------
    with h5py.File(h5_path, "r") as f:
        data = f[dataset_name][()]            # shape: (n_neurons, n_samples)

    n_samples = data.shape[1]                # <-- what you wanted

    # --- Compute stats -----------------------------------------------------
    means = np.nanmean(data, axis=1)
    stds  = np.nanstd(data, axis=1, ddof=0)

    pd.DataFrame({"mean": means, "sd": stds}).to_csv(csv_path)
    print(f"Row-wise mean & SD saved to {csv_path}")

    return int(n_samples)


def save_cohens_d(dataset: str, n1=None, n2=None) -> None:
    """
    Reads per-neuron summary stats for ‘popular’ and ‘unpopular’ users and
    saves Cohen’s d (with sample-size–weighted pooled SD) to
    dataset/<dataset>/user/cohens_d.csv.
    
    Expected columns in each CSV:
        mean   – group mean
        sd     – group standard deviation
        n      – number of samples in that group
    The index (first column) is treated as the neuron identifier.
    """
    base = Path(f"./dataset/{dataset}/user")
    df1 = pd.read_csv(base / "neuron_stats_pop.csv", index_col=0)
    df2 = pd.read_csv(base / "neuron_stats_unpop.csv", index_col=0)

    # Extract vectors for clarity
    m1, s1= df1["mean"], df1["sd"],
    m2, s2= df2["mean"], df2["sd"],

    # Pooled standard deviation accounting for sample size
    s_pooled = np.sqrt(((n1 - 1) * s1.pow(2) + (n2 - 1) * s2.pow(2)) / (n1 + n2 - 2))

    # Cohen's d
    d = (m1 - m2) / s_pooled

    # Assemble result DataFrame
    df_result = pd.DataFrame({"cohens_d": d})
    df_result.to_csv(base / "cohens_d.csv")

    print("Cohen's d values saved to cohens_d.csv")




def make_items_popular(item_seq_len, dataset, n):
    item_labels = pd.read_csv(rf"./dataset/{dataset}/item_popularity_labels.csv")
    
    # Filter rows where popularity_label == 1
    filtered_items = item_labels[item_labels['popularity_label'] == 1]
    available_ids = filtered_items['item_id:token'].tolist()
    
    # Get batch size
    batch_size = item_seq_len.shape[0]
    selected_item_ids = []

    for _ in range(batch_size):
        sampled = pd.Series(available_ids).sample(n=n, replace=True).tolist()
        selected_item_ids.append(sampled)

    # Convert to tensor of shape (batch_size, n)
    selected_tensor = torch.tensor(selected_item_ids)
    return selected_tensor

def make_items_unpopular(item_seq_len, dataset, n):
    item_labels = pd.read_csv(rf"./dataset/{dataset}/item_popularity_labels.csv")
    
    # Filter rows where popularity_label == 1
    filtered_items = item_labels[item_labels['popularity_label'] == -1]
    available_ids = filtered_items['item_id:token'].tolist()
    
    # Get batch size
    batch_size = item_seq_len.shape[0]
    selected_item_ids = []

    for _ in range(batch_size):
        sampled = pd.Series(available_ids).sample(n=n, replace=True).tolist()
        selected_item_ids.append(sampled)

    # Convert to tensor of shape (batch_size, n)
    selected_tensor = torch.tensor(selected_item_ids)
    return selected_tensor



import pandas as pd
import numpy as np

def make_labels(dataset=None,
                sep="	",
                alpha=0.9,
                holdout_k=2):
    """
    Create two labels:
      1. `label:int`     -> single user-level label (-1,0,1) from the user's last state (excluding last `holdout_k` interactions).
      2. `label_cur:int` -> per-interaction label (-1,0,1) computed for the *current* interaction using the same formula.

    Everything is saved to `./dataset/{dataset}/yoochoose-clicks.inter.new` with headers.
    """

    out_path = rf"./dataset/{dataset}/yoochoose-clicks.inter.new"
    in_path  = rf"./dataset/{dataset}/yoochoose-clicks.inter"

    # ---- Load ----------------------------------------------------------------
    df = pd.read_csv(
        in_path,
        sep=sep,
        header=0,
        low_memory=False,
        dtype={
            "user_id:token": "string",
            "item_id:token": "string",
            "timestamp:float": "float64"
        }
    )

    # Sort once for all subsequent ops
    df = df.sort_values(["user_id:token", "timestamp:float"]).reset_index(drop=True)

    # ---- Identify each user's last k interactions ----------------------------
    rdesc = df.groupby("user_id:token")["timestamp:float"].rank(method="first", ascending=False)
    holdout_mask = rdesc <= holdout_k

    # Data used to compute popularity & scores
    calc_df = df.loc[~holdout_mask].copy()

    # ---- Popularity labels p_i -----------------------------------------------
    item_counts = calc_df["item_id:token"].value_counts()
    n_items = len(item_counts)
    top_n = int(np.ceil(0.2 * n_items))
    bot_n = int(np.floor(0.2 * n_items))

    sorted_items = item_counts.sort_values()
    bottom_items = set(sorted_items.index[:bot_n])
    top_items    = set(sorted_items.index[-top_n:])

    pop_map = {iid: (-1 if iid in bottom_items else (1 if iid in top_items else 0))
               for iid in item_counts.index}

    calc_df["p_i"] = calc_df["item_id:token"].map(pop_map).fillna(0).astype(int)

    # ---- Recency n (0 = most recent after holdout) ---------------------------
    calc_df = calc_df.sort_values(["user_id:token", "timestamp:float"], ascending=[True, False])
    calc_df["n"] = calc_df.groupby("user_id:token").cumcount()

    # ---- Score and labels ----------------------------------------------------
    calc_df["num_user_interactions"] = calc_df.groupby("user_id:token")["item_id:token"].transform("size")
    calc_df["score"] = (alpha ** calc_df["n"]) * calc_df["p_i"] / calc_df["num_user_interactions"]
    calc_df["label_cur"] = np.sign(calc_df["score"]).astype(int)  # per-interaction label

    # ---- Single user label from last state ----------------------------------
    user_labels = (
        calc_df.loc[calc_df["n"] == 0, ["user_id:token", "label_cur"]]
               .rename(columns={"label_cur": "user_label"})
    )
    label_map = dict(zip(user_labels["user_id:token"], user_labels["user_label"]))

    # ---- Merge per-interaction labels back (including holdouts -> 0) ---------
    df = df.merge(
        calc_df[["user_id:token", "item_id:token", "tiuser_colmestamp:float", "label_cur"]],
        on=["user_id:token", "item_id:token", "timestamp:float"],
        how="left"
    )
    df["label_cur:token"] = df["label_cur"].fillna(0).astype("int8")
    df.drop(columns=["label_cur"], inplace=True)

    # ---- Assign single label per user (including holdouts) ------------------
    df["label:token"] = df["user_id:token"].map(label_map).fillna(0).astype("int8")

    # keep column order tidy
    df = df[["user_id:token", "item_id:token", "timestamp:float", "label:token", "label_cur:token"]]

    # ---- Save ----------------------------------------------------------------
    df.to_csv(out_path, sep=sep, index=False, header=True)

# Example usage:
# make_labels(dataset="yoochoose-clicks")
# make_labels("yoochoose.inter", "yoochoose.inter.new", sep="\t")
