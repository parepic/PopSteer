# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/10/3, 2020/10/1
# @Author : Zhen Tian, Yupeng Hou, Zihan Lin
# @Email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
from scipy.stats import pearsonr
import h5py
from itertools import combinations
from multiprocessing import Pool, cpu_count



import pandas as pd
from IPython.display import display
import itertools



from recbole.quick_start import run_recbole, load_data_and_model, run
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    label_popular_items,
    init_seed,
    set_color,
    get_flops,
    get_environment,
    remove_sparse_users_items,
    plot_binned_bar_chart,
    make_items_unpopular,
    save_mean_SD,
    save_cohens_d,
    plot_h5_columns,
    create_item_popularity_csv,
    plot_interaction_distribution, 
    extract_sort_top_neurons
)



def calculate_percentage_change(new_values, base_value):
    result = []
    for new in new_values:
        if base_value == 0:
            change = "inf" if new != 0 else "0.00%"
        else:
            change = f"{((new - base_value) / base_value) * 100:.2f}%"
        result.append(f"{new:.4f} ({change})")
    return result



def display_metrics_table(dampen_percs, ndcgs, hits, coverages, lt_coverages, deep_lt_coverages, ginis, arps, ndcg_heads, ndcg_mids, ndcg_tails):
    # Hardcoded first row for 'sasrec'
    dampen_labels = [f'{dp}' for dp in dampen_percs]
    
    data = {
        'beta': dampen_labels,
        'NDCG@10': calculate_percentage_change(ndcgs, ndcgs[0]),
        'NDCG-HEAD@10': calculate_percentage_change(ndcg_heads, ndcg_heads[0]),
        'NDCG-MID@10': calculate_percentage_change(ndcg_mids, ndcg_mids[0]),
        'NDCG-TAIL@10': calculate_percentage_change(ndcg_tails, ndcg_tails[0]),
        # 'Hit@10': calculate_percentage_change(hits, hits[0]),
        # 'Coverage@10': calculate_percentage_change(coverages, coverages[0]),
        # 'LT Coverage@10': calculate_percentage_change(lt_coverages, lt_coverages[0]),
        'Deep LT Coverage@10': calculate_percentage_change(deep_lt_coverages, deep_lt_coverages[0]),
        'Gini coefficient@10': calculate_percentage_change(ginis, ginis[0]),
        # 'IPS NDCG@10': calculate_percentage_change(ips_ndcgs, ips_ndcgs[0]),
        'ARP@10': calculate_percentage_change(arps, arps[0])
        
    }
    df = pd.DataFrame(data)
    
    # Display table
    print(df.to_string(index=False))  # Print the entire table without truncation



def ablate1():
    Ns = np.linspace(0, 229, 231)     
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.path,
        sae=(args.model == 'SASRec_SAE'),
        device=device
    )
    records = []
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)       
    for n in Ns:
        res = trainer.evaluate(
                    valid_data,
                    model_file=args.path,
                    show_progress=config["show_progress"],
                    N=n)
        records.append({
            'N': n,
            'ndcg': res['ndcg@10'], 'gini': res['Gini_coef@10'],
            'Deep long tail coverage': res.get('Deep_LT_coverage@10'),
            'arp': res.get('ARP@10'),
        })
    df = pd.DataFrame(records)
    out_csv = getattr(args, 'output_csv', 'interpretation.csv')
    df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")



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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )
    
    # ------------------------------------------------------------------
    # Paths & dataset selection
    # ------------------------------------------------------------------
    parser.add_argument('--path', '-p', type=str, default=None,
                        help="Model path, not required for training SASRec")

    parser.add_argument('--dataset', '-d', type=str, default='ml-1m',
                        choices=['ml-1m', 'lastfm', 'gowalla', 'amazon-book', 'custom'],
                        help="Name of the dataset to use (e.g. 'ml-1m', 'lastfm').")

    # ------------------------------------------------------------------
    # Hyper‑parameters (see paper §5.2)
    # ------------------------------------------------------------------
    parser.add_argument('--alpha', '-a', type=float, default=1.5,
                        help="Steering strength \u03B1 controlling neuron adjustment in Eq. 3 of the paper.")

    parser.add_argument('--num_neurons', '-N', type=int, default=4096,
                        help="Hidden dimension N of the Sparse Autoencoder (SAE).")

    parser.add_argument('--top_k', '-k', type=int, default=32,
                        help="Sparsity parameter K: keep only the top‑k activations per input in the SAE (Eq. 1).")

    parser.add_argument('--scale', '--scale_size', type=int, default=8,
                        dest='scale',
                        help="Scale factor s controlling the SAE hidden size relative to the input (s × d).")

    # ------------------------------------------------------------------
    # Training / analysis switches
    # ------------------------------------------------------------------
    parser.add_argument('--train_recommender', action='store_true',
                        help="Train or fine‑tune the SASRec backbone model only.")

    parser.add_argument('--train_popsteer', action='store_true',
                        help="Train the Sparse Autoencoder and apply PopSteer neuron steering.")

    parser.add_argument('--analyze_neurons', action='store_true',
                        help="Run neuron‑level analysis (compute Cohen's d, identify biased neurons, etc.).")
    
    parser.add_argument('--test_popsteer', action='store_true',
                    help="Test Popsteer.")
    
    parser.add_argument('--valid_set', action='store_true',
                    help="including this flag will test on valid data, otherwise test data")

    # Parse the arguments
    args = parser.parse_args()

    args, _ = parser.parse_known_args()
    
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        
    
    if(args.train_recommender):
        config_file_list = (
                args.config_files.strip().split(" ") if args.config_files else None
            )
        parameter_dict = {
            'train_neg_sample_args': None,
            'dataset':  args.dataset
        }   

        # config_file_list = [r'./recbole/recbole/properties/overall.yaml',
        #             r'./recbole/recbole/properties/model/SASRec.yaml',
        #             r'./recbole/recbole/properties/dataset/ml-1m.yaml'
        #             ]
        run(
            'SASRec',
            args.dataset,
            # config_file_list=config_file_list,
            config_dict=parameter_dict,
            nproc=args.nproc,
            world_size=args.world_size,
            ip=args.ip,
            port=args.port,
            group_offset=args.group_offset,
        )
    if args.train_popsteer:
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
            model_file=args.path, sae=True, device=device, scale_size=args.scale, k=args.top_k
        )  
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        trainer.fit_SAE(config, 
            args.path,
            train_data,
            dataset,
            valid_data=valid_data,
            show_progress=True,
            )
    
    if args.analyze_neurons:
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
            model_file=args.path, device=device,
        )  
        
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        trainer.analyze_neurons(train_data,  model_file=args.path, eval_data=False, sae=False)
        save_mean_SD(config["dataset"], popular=True)
        save_mean_SD(config["dataset"], popular=False)
        save_cohens_d(config["dataset"])
    
    if args.test_popsteer: 
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
            model_file=args.path, device=device,
        )  
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        # data = test_data if args.valid_set else test_data
        test_result = trainer.evaluate(
            test_data, model_file=args.path, show_progress=config["show_progress"], N=args.num_neurons, beta=args.alpha
        )      
        keys = [
            'recall@10',
            'ndcg@10',
            'hit@10',
            'ARP@10',
            'Deep_LT_coverage@10',
            'Gini_coef@10'
        ]

        max_key_len = max(len(k) for k in keys)

        # print header
        print(f"{'Metric':<{max_key_len}} | Value")
        print(f"{'-'*max_key_len}-|-------")

        # print each metric with its dynamic value
        for key in keys:
            value = test_result[key]             # get value from your OrderedDict
            print(f"{key:<{max_key_len}} | {value:>7.4f}")