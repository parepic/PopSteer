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
    count,
    calculate_pearson_correlation,
    compute_averages,
    remove_sparse_users_items,
    plot_binned_bar_chart,
    make_items_unpopular,
    save_mean_SD,
    save_cohens_d,
    find_diff, 
    rank_neurons_by_chi2,
    plot_h5_columns,
    compute_and_save_correlations,
    create_unbiased_set,
    create_item_popularity_csv,
    search_movies,
    process_and_save_movies,
    sample_users_interactions,
    plot_interaction_distribution, 
    extract_sort_top_neurons
)


def plot_graphs(ndcgs, hits, coverages, lt_coverages, dampen_percs):
    bar_width = 0.2  # Width of each bar
    # dampen_percs = [x + 1 for x in dampen_percs]
    index = np.arange(len(dampen_percs))  # X-axis positions for groups, adding 'sasrec'
    
    plt.figure(figsize=(10, 6))
    # Hardcoded first set of bars for 'sasrec'
    
    dampen_labels =  [f'{dp}' for dp in dampen_percs]
    
    plt.bar(index, ndcgs, bar_width, label='NDCG@10')
    plt.bar(index + bar_width, hits, bar_width, label='Hit@10')
    plt.bar(index + 2 * bar_width, coverages, bar_width, label='Coverage@10')
    plt.bar(index + 3 * bar_width, lt_coverages, bar_width, label='LT Coverage@10')
    
    # Add trend lines starting from the top of each respective bar
    plt.plot(index, ndcgs, marker='o', linestyle='--', color='blue', linewidth=2, label='NDCG@10 Trend')
    plt.plot(index + bar_width, hits, marker='o', linestyle='--', color='orange', linewidth=2, label='Hit@10 Trend')
    plt.plot(index + 2 * bar_width, coverages, marker='o', linestyle='--', color='green', linewidth=2, label='Coverage@10 Trend')
    plt.plot(index + 3 * bar_width, lt_coverages, marker='o', linestyle='--', color='red', linewidth=2, label='LT Coverage@10 Trend')
    
    plt.xlabel('Dampen Percentage')
    plt.ylabel('Values')
    plt.title('Performance Metrics by Dampen Percentage')
    
    plt.xticks(index + 1.5 * bar_width, dampen_labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


import pandas as pd
from IPython.display import display
import itertools



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

    
import time
    

def tune_hyperparam():
    # 1) load everything
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.path,
        sae=(args.model == 'SASRec_SAE'),
        device=device
    )
    
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    
    # 2) build your grid
    all_Ns   = list(np.linspace(512, 4096, 8))
    betas    = np.linspace(1, 4, 7)

    # 3) baseline & bookkeeping
    baseline_stats = {
            'ndcg@10':          0.1212,
            'Gini_coef@10':     0.7573,
            'Deep_LT_coverage@10': 0.4859,
            'ndcg-head@10':     0.1848,
            'ndcg-mid@10':      0.1234,
            'ndcg-tail@10':     0.0621,
            'arp':              3.0278,
            'time':             0.0
            }
    best_diff    = 0.0
    best_pair    = None
    it_num       = 0
    records      = []
    
    # include inference_time for the baseline row as 0 (or NA)
    records.append({
        'N': -1, 'beta': None,
        'ndcg': baseline_stats['ndcg@10'],
        'gini': baseline_stats['Gini_coef@10'],
        'gain': 0,
        'Deep long tail coverage': baseline_stats['Deep_LT_coverage@10'],
        'ndcg-head': baseline_stats['ndcg-head@10'],
        'ndcg-mid': baseline_stats['ndcg-mid@10'],
        'ndcg-tail': baseline_stats['ndcg-tail@10'],
        'arp': baseline_stats['arp'],
        'inference_time': baseline_stats['time']
    })
    
    # 4) single n=0 evaluation
    print("=== Running n=0 case ===")
    
    # start timing
    start_time = time.time()
    res0 = trainer.evaluate(
        valid_data,
        model_file=args.path,
        show_progress=config["show_progress"]
    )
    # stop timing
    inference_time0 = time.time() - start_time

    # compute and log n=0 metrics
    ndcg0 = res0['ndcg@10']
    gini0 = res0['Gini_coef@10']
    diff_ndcg0 = abs(ndcg0 - baseline_stats['ndcg@10']) / baseline_stats['ndcg@10']
    diff_gini0 = abs(gini0 - baseline_stats['Gini_coef@10']) / baseline_stats['Gini_coef@10']
    gain0 = diff_gini0 - diff_ndcg0

    print(f"[Iter {it_num:04d}] n=0 → ndcgΔ={diff_ndcg0:.3f}, giniΔ={diff_gini0:.3f}, time={inference_time0:.2f}s")
    if diff_ndcg0 <= 0.1 and gain0 > best_diff:
        best_diff = gain0
        best_pair = (0, None)
    
    records.append({
        'N': 0, 'beta': None,
        'ndcg': ndcg0, 'gini': gini0, 'gain': gain0,
        'Deep long tail coverage': res0.get('Deep_LT_coverage@10'),
        'ndcg-head': res0.get('ndcg-head@10'),
        'ndcg-mid': res0.get('ndcg-mid@10'),
        'ndcg-tail': res0.get('ndcg-tail@10'),
        'arp': res0.get('ARP@10'),
        'inference_time': inference_time0
    })
    
    it_num += 1

    # 5) full sweep for n>0
    for n, beta in itertools.product(all_Ns, betas):
        # start timing
        start_time = time.time()
        res = trainer.evaluate(
            valid_data,
            model_file=args.path,
            show_progress=config["show_progress"],
            N=n, beta=beta
        )
        # stop timing
        inference_time = time.time() - start_time

        # compute metrics
        ndcg       = res['ndcg@10']
        gini       = res['Gini_coef@10']
        diff_ndcg  = abs(ndcg - baseline_stats['ndcg@10']) / baseline_stats['ndcg@10']
        diff_gini  = abs(gini - baseline_stats['Gini_coef@10']) / baseline_stats['Gini_coef@10']
        gain       = diff_gini - diff_ndcg

        # update best
        if diff_ndcg <= 0.05 and gain > best_diff:
            best_diff = gain
            best_pair = (n, beta)

        # record + print
        records.append({
            'N': n, 'beta': beta, 
            'ndcg': ndcg, 'gini': gini, 'gain': gain,
            'Deep long tail coverage': res.get('Deep_LT_coverage@10'),
            'ndcg-head': res.get('ndcg-head@10'),
            'ndcg-mid': res.get('ndcg-mid@10'),
            'ndcg-tail': res.get('ndcg-tail@10'),
            'arp': res.get('ARP@10'),
            'inference_time': inference_time
        })
        print(f"[Iter {it_num:04d}] N={n:.0f}, β={beta:.2f} → "
              f"ndcgΔ={diff_ndcg:.3f}, giniΔ={diff_gini:.3f}, time={inference_time:.2f}s, best_gain={best_diff:.3f}")
        it_num += 1

    # 6) save results + final report
    df = pd.DataFrame(records)
    out_csv = getattr(args, 'output_csv', 'tuning_results_PopSteer.csv')
    df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")

    print("=== DONE ===")
    print(f"Best triplet: {best_pair}, best gain: {best_diff:.3f}")
    return best_pair



def tune_hyperparam_FAIRSTAR():
    # 1) load everything
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.path,
        sae=(args.model == 'SASRec_SAE'),
        device=device
    )
    
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    

    # 2) build your grid
    all_Ns   = [0.3, 0.5, 0.7, 0.9, 0.99]
    betas   = [0.01, 0.05, 0.1]

    # 3) baseline & bookkeeping (updated values)
    baseline_stats = {
            'ndcg@10':          0.1212,
            'Gini_coef@10':     0.7573,
            'Deep_LT_coverage@10': 0.4859,
            'ndcg-head@10':     0.1848,
            'ndcg-mid@10':      0.1234,
            'ndcg-tail@10':     0.0621,
            'arp':              3.0278,
            'time':             0.0
            }
    best_diff = 0.0
    best_pair = None
    it_num    = 0
    records   = []
    
    # baseline record with zero inference time
    records.append({
        'N': -1, 'beta': None,
        'ndcg': baseline_stats['ndcg@10'],
        'gini': baseline_stats['Gini_coef@10'],
        'gain': 0,
        'Deep long tail coverage': baseline_stats['Deep_LT_coverage@10'],
        'ndcg-head': baseline_stats['ndcg-head@10'],
        'ndcg-mid': baseline_stats['ndcg-mid@10'],
        'ndcg-tail': baseline_stats['ndcg-tail@10'],
        'arp': baseline_stats['arp'],
        'inference_time': 0.0
    })
    it_num += 1

    # 5) full sweep for n>0
    for n, beta in itertools.product(all_Ns, betas):
        # measure inference time
        start_time = time.time()
        res = trainer.evaluate(
            valid_data,
            model_file=args.path,
            show_progress=config["show_progress"],
            N=n, beta=beta
        )
        inference_time = time.time() - start_time

        # compute metrics
        ndcg      = res['ndcg@10']
        gini      = res['Gini_coef@10']
        diff_ndcg = abs(ndcg  - baseline_stats['ndcg@10'])      / baseline_stats['ndcg@10']
        diff_gini = abs(gini  - baseline_stats['Gini_coef@10']) / baseline_stats['Gini_coef@10']
        gain      = diff_gini - diff_ndcg

        # update best
        if diff_ndcg <= 0.05 and gain > best_diff:
            best_diff = gain
            best_pair = (n, beta)

        # record + print
        records.append({
            'N': n, 'beta': beta,
            'ndcg': ndcg, 'gini': gini, 'gain': gain,
            'Deep long tail coverage': res.get('Deep_LT_coverage@10'),
            'ndcg-head': res.get('ndcg-head@10'),
            'ndcg-mid': res.get('ndcg-mid@10'),
            'ndcg-tail': res.get('ndcg-tail@10'),
            'arp': res.get('ARP@10'),
            'inference_time': inference_time
        })
        print(
            f"[Iter {it_num:04d}] N={n:.2f}, β={beta:.2f} → "
            f"ndcgΔ={diff_ndcg:.3f}, giniΔ={diff_gini:.3f}, "
            f"time={inference_time:.2f}s, best_gain={best_diff:.3f}"
        )
        it_num += 1

    # 6) save results + final report
    df = pd.DataFrame(records)
    out_csv = getattr(args, 'output_csv', 'tuning_results_FAIR.csv')
    df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")
    
    print("=== DONE ===")
    print(f"Best triplet: {best_pair}, best gain: {best_diff:.3f}")
    return best_pair




def tune_hyperparam_pct():
    # 1) load everything
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.path,
        sae=(args.model == 'SASRec_SAE'),
        device=device
    )
    
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    

    # 2) build your grid
    all_Ns   = [0.3, 0.5, 0.7, 0.9, 0.99]
    betas   = [0.0, 0.3, 0.5, 0.7, 0.9]
        
    
    # 3) baseline & bookkeeping (updated values)
    baseline_stats = {
            'ndcg@10':          0.1212,
            'Gini_coef@10':     0.7573,
            'Deep_LT_coverage@10': 0.4859,
            'ndcg-head@10':     0.1848,
            'ndcg-mid@10':      0.1234,
            'ndcg-tail@10':     0.0621,
            'arp':              3.0278,
            'time':             0.0
            }
    best_diff = 0.0
    best_pair = None
    it_num    = 0
    records   = []
    
    # baseline record with zero inference time
    records.append({
        'N': -1, 'beta': None,
        'ndcg': baseline_stats['ndcg@10'],
        'gini': baseline_stats['Gini_coef@10'],
        'gain': 0,
        'Deep long tail coverage': baseline_stats['Deep_LT_coverage@10'],
        'ndcg-head': baseline_stats['ndcg-head@10'],
        'ndcg-mid': baseline_stats['ndcg-mid@10'],
        'ndcg-tail': baseline_stats['ndcg-tail@10'],
        'arp': baseline_stats['arp'],
        'inference_time': 0.0
    })
    it_num += 1

    # 5) full sweep for n>0
    for n, beta in itertools.product(all_Ns, betas):
        # measure inference time
        start_time = time.time()
        res = trainer.evaluate(
            valid_data,
            model_file=args.path,
            show_progress=config["show_progress"],
            N=n, beta=beta
        )
        inference_time = time.time() - start_time

        # compute metrics
        ndcg      = res['ndcg@10']
        gini      = res['Gini_coef@10']
        diff_ndcg = abs(ndcg  - baseline_stats['ndcg@10'])      / baseline_stats['ndcg@10']
        diff_gini = abs(gini  - baseline_stats['Gini_coef@10']) / baseline_stats['Gini_coef@10']
        gain      = diff_gini - diff_ndcg

        # update best
        if diff_ndcg <= 0.05 and gain > best_diff:
            best_diff = gain
            best_pair = (n, beta)

        # record + print
        records.append({
            'N': n, 'beta': beta,
            'ndcg': ndcg, 'gini': gini, 'gain': gain,
            'Deep long tail coverage': res.get('Deep_LT_coverage@10'),
            'ndcg-head': res.get('ndcg-head@10'),
            'ndcg-mid': res.get('ndcg-mid@10'),
            'ndcg-tail': res.get('ndcg-tail@10'),
            'arp': res.get('ARP@10'),
            'inference_time': inference_time
        })
        print(
            f"[Iter {it_num:04d}] N={n:.2f}, β={beta:.2f} → "
            f"ndcgΔ={diff_ndcg:.3f}, giniΔ={diff_gini:.3f}, "
            f"time={inference_time:.2f}s, best_gain={best_diff:.3f}"
        )
        it_num += 1

    # 6) save results + final report
    df = pd.DataFrame(records)
    out_csv = getattr(args, 'output_csv', 'tuning_results_PCT.csv')
    df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")
    
    print("=== DONE ===")
    print(f"Best triplet: {best_pair}, best gain: {best_diff:.3f}")
    return best_pair


def tune_hyperparam_pct():
    # 1) load everything
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.path,
        sae=(args.model == 'SASRec_SAE'),
        device=device
    )
    
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    

    # 2) build your grid
    all_Ns   = [0.3, 0.5, 0.7, 0.9, 0.99]
    betas   = [0.0, 0.3, 0.5, 0.7, 0.9]
        
    
    # 3) baseline & bookkeeping (updated values)
    baseline_stats = {
            'ndcg@10':          0.1212,
            'Gini_coef@10':     0.7573,
            'Deep_LT_coverage@10': 0.4859,
            'ndcg-head@10':     0.1848,
            'ndcg-mid@10':      0.1234,
            'ndcg-tail@10':     0.0621,
            'arp':              3.0278,
            'time':             0.0
            }
    best_diff = 0.0
    best_pair = None
    it_num    = 0
    records   = []
    
    # baseline record with zero inference time
    records.append({
        'N': -1, 'beta': None,
        'ndcg': baseline_stats['ndcg@10'],
        'gini': baseline_stats['Gini_coef@10'],
        'gain': 0,
        'Deep long tail coverage': baseline_stats['Deep_LT_coverage@10'],
        'ndcg-head': baseline_stats['ndcg-head@10'],
        'ndcg-mid': baseline_stats['ndcg-mid@10'],
        'ndcg-tail': baseline_stats['ndcg-tail@10'],
        'arp': baseline_stats['arp'],
        'inference_time': 0.0
    })
    it_num += 1

    # 5) full sweep for n>0
    for n, beta in itertools.product(all_Ns, betas):
        # measure inference time
        start_time = time.time()
        res = trainer.evaluate(
            valid_data,
            model_file=args.path,
            show_progress=config["show_progress"],
            N=n, beta=beta
        )
        inference_time = time.time() - start_time

        # compute metrics
        ndcg      = res['ndcg@10']
        gini      = res['Gini_coef@10']
        diff_ndcg = abs(ndcg  - baseline_stats['ndcg@10'])      / baseline_stats['ndcg@10']
        diff_gini = abs(gini  - baseline_stats['Gini_coef@10']) / baseline_stats['Gini_coef@10']
        gain      = diff_gini - diff_ndcg

        # update best
        if diff_ndcg <= 0.05 and gain > best_diff:
            best_diff = gain
            best_pair = (n, beta)

        # record + print
        records.append({
            'N': n, 'beta': beta,
            'ndcg': ndcg, 'gini': gini, 'gain': gain,
            'Deep long tail coverage': res.get('Deep_LT_coverage@10'),
            'ndcg-head': res.get('ndcg-head@10'),
            'ndcg-mid': res.get('ndcg-mid@10'),
            'ndcg-tail': res.get('ndcg-tail@10'),
            'arp': res.get('ARP@10'),
            'inference_time': inference_time
        })
        print(
            f"[Iter {it_num:04d}] N={n:.2f}, β={beta:.2f} → "
            f"ndcgΔ={diff_ndcg:.3f}, giniΔ={diff_gini:.3f}, "
            f"time={inference_time:.2f}s, best_gain={best_diff:.3f}"
        )
        it_num += 1

    # 6) save results + final report
    df = pd.DataFrame(records)
    out_csv = getattr(args, 'output_csv', 'tuning_results_PCT.csv')
    df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")
    
    print("=== DONE ===")
    print(f"Best triplet: {best_pair}, best gain: {best_diff:.3f}")
    return best_pair




def tune_hyperparam_random():
    # 1) load everything
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.path,
        sae=(args.model == 'SASRec_SAE'),
        device=device
    )
    
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    

    # 2) build your grid
    all_Ns   = [15, 30, 50, 75, 100]
    betas = [1]
    # 3) baseline & bookkeeping (updated values)
    baseline_stats = {
            'ndcg@10':          0.1212,
            'Gini_coef@10':     0.7573,
            'Deep_LT_coverage@10': 0.4859,
            'ndcg-head@10':     0.1848,
            'ndcg-mid@10':      0.1234,
            'ndcg-tail@10':     0.0621,
            'arp':              3.0278,
            'time':             0.0
            }
    best_diff = 0.0
    best_pair = None
    it_num    = 0
    records   = []
    
    # baseline record with zero inference time
    records.append({
        'N': -1, 'beta': None,
        'ndcg': baseline_stats['ndcg@10'],
        'gini': baseline_stats['Gini_coef@10'],
        'gain': 0,
        'Deep long tail coverage': baseline_stats['Deep_LT_coverage@10'],
        'ndcg-head': baseline_stats['ndcg-head@10'],
        'ndcg-mid': baseline_stats['ndcg-mid@10'],
        'ndcg-tail': baseline_stats['ndcg-tail@10'],
        'arp': baseline_stats['arp'],
        'inference_time': 0.0
    })
    it_num += 1

    # 5) full sweep for n>0
    for n, beta in itertools.product(all_Ns, betas):
        # measure inference time
        start_time = time.time()
        res = trainer.evaluate(
            valid_data,
            model_file=args.path,
            show_progress=config["show_progress"],
            N=n, beta=beta
        )
        inference_time = time.time() - start_time

        # compute metrics
        ndcg      = res['ndcg@10']
        gini      = res['Gini_coef@10']
        diff_ndcg = abs(ndcg  - baseline_stats['ndcg@10'])      / baseline_stats['ndcg@10']
        diff_gini = abs(gini  - baseline_stats['Gini_coef@10']) / baseline_stats['Gini_coef@10']
        gain      = diff_gini - diff_ndcg

        # update best
        if diff_ndcg <= 0.05 and gain > best_diff:
            best_diff = gain
            best_pair = (n, beta)

        # record + print
        records.append({
            'N': n, 'beta': beta,
            'ndcg': ndcg, 'gini': gini, 'gain': gain,
            'Deep long tail coverage': res.get('Deep_LT_coverage@10'),
            'ndcg-head': res.get('ndcg-head@10'),
            'ndcg-mid': res.get('ndcg-mid@10'),
            'ndcg-tail': res.get('ndcg-tail@10'),
            'arp': res.get('ARP@10'),
            'inference_time': inference_time
        })
        print(
            f"[Iter {it_num:04d}] N={n:.2f}, β={beta:.2f} → "
            f"ndcgΔ={diff_ndcg:.3f}, giniΔ={diff_gini:.3f}, "
            f"time={inference_time:.2f}s, best_gain={best_diff:.3f}"
        )
        it_num += 1

    # 6) save results + final report
    df = pd.DataFrame(records)
    out_csv = getattr(args, 'output_csv', 'tuning_results_random.csv')
    df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")
    
    print("=== DONE ===")
    print(f"Best triplet: {best_pair}, best gain: {best_diff:.3f}")
    return best_pair


def tune_hyperparam_ipr():
    # 1) load everything
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.path,
        sae=(args.model == 'SASRec_SAE'),
        device=device
    )
    
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    

    # 2) build your grid
    all_Ns   = [0.1, 0.3, 0.5, 0.5, 1.0]
    betas = [1]
    # 3) baseline & bookkeeping (updated values)
    baseline_stats = {
            'ndcg@10':          0.1212,
            'Gini_coef@10':     0.7573,
            'Deep_LT_coverage@10': 0.4859,
            'ndcg-head@10':     0.1848,
            'ndcg-mid@10':      0.1234,
            'ndcg-tail@10':     0.0621,
            'arp':              3.0278,
            'time':             0.0
            }
    best_diff = 0.0
    best_pair = None
    it_num    = 0
    records   = []
    
    # baseline record with zero inference time
    records.append({
        'N': -1, 'beta': None,
        'ndcg': baseline_stats['ndcg@10'],
        'gini': baseline_stats['Gini_coef@10'],
        'gain': 0,
        'Deep long tail coverage': baseline_stats['Deep_LT_coverage@10'],
        'ndcg-head': baseline_stats['ndcg-head@10'],
        'ndcg-mid': baseline_stats['ndcg-mid@10'],
        'ndcg-tail': baseline_stats['ndcg-tail@10'],
        'arp': baseline_stats['arp'],
        'inference_time': 0.0
    })
    it_num += 1

    # 5) full sweep for n>0
    for n, beta in itertools.product(all_Ns, betas):
        # measure inference time
        start_time = time.time()
        res = trainer.evaluate(
            valid_data,
            model_file=args.path,
            show_progress=config["show_progress"],
            N=n, beta=beta
        )
        inference_time = time.time() - start_time

        # compute metrics
        ndcg      = res['ndcg@10']
        gini      = res['Gini_coef@10']
        diff_ndcg = abs(ndcg  - baseline_stats['ndcg@10'])      / baseline_stats['ndcg@10']
        diff_gini = abs(gini  - baseline_stats['Gini_coef@10']) / baseline_stats['Gini_coef@10']
        gain      = diff_gini - diff_ndcg

        # update best
        if diff_ndcg <= 0.05 and gain > best_diff:
            best_diff = gain
            best_pair = (n, beta)

        # record + print
        records.append({
            'N': n, 'beta': beta,
            'ndcg': ndcg, 'gini': gini, 'gain': gain,
            'Deep long tail coverage': res.get('Deep_LT_coverage@10'),
            'ndcg-head': res.get('ndcg-head@10'),
            'ndcg-mid': res.get('ndcg-mid@10'),
            'ndcg-tail': res.get('ndcg-tail@10'),
            'arp': res.get('ARP@10'),
            'inference_time': inference_time
        })
        print(
            f"[Iter {it_num:04d}] N={n:.2f}, β={beta:.2f} → "
            f"ndcgΔ={diff_ndcg:.3f}, giniΔ={diff_gini:.3f}, "
            f"time={inference_time:.2f}s, best_gain={best_diff:.3f}"
        )
        it_num += 1

    # 6) save results + final report
    df = pd.DataFrame(records)
    out_csv = getattr(args, 'output_csv', 'tuning_results_ipr.csv')
    df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")
    
    print("=== DONE ===")
    print(f"Best triplet: {best_pair}, best gain: {best_diff:.3f}")
    return best_pair



def tune_hyperparam_pmmf():
    # 1) load everything
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.path,
        sae=(args.model == 'SASRec_SAE'),
        device=device
    )
    
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    

    # 2) build your grid
    all_Ns   = [0.001, 0.01, 0.1, 1, 10]
    betas   = [1e-4, 1e-3, 1e-2]

    # 3) baseline & bookkeeping (updated values)
    baseline_stats = {
            'ndcg@10':          0.1212,
            'Gini_coef@10':     0.7573,
            'Deep_LT_coverage@10': 0.4859,
            'ndcg-head@10':     0.1848,
            'ndcg-mid@10':      0.1234,
            'ndcg-tail@10':     0.0621,
            'arp':              3.0278,
            'time':             0.0
            }
    
    
    # baseline_stats = {
    #     'ndcg@10':               0.6273,
    #     'Gini_coef@10':          0.5849518873628745,
    #     'Deep_LT_coverage@10':   0.8716216216216216,
    #     'ndcg-head@10':          0.6589,
    #     'ndcg-mid@10':           0.5763,
    #     'ndcg-tail@10':          0.6798,
    #     'arp':                   0.00035533782557826705,
    #     'time':                  0.24
    # }
    
    
    
    best_diff = 0.0
    best_pair = None
    it_num    = 0
    records   = []
    
    # baseline record with zero inference time
    records.append({
        'N': -1, 'beta': None,
        'ndcg': baseline_stats['ndcg@10'],
        'gini': baseline_stats['Gini_coef@10'],
        'gain': 0,
        'Deep long tail coverage': baseline_stats['Deep_LT_coverage@10'],
        'ndcg-head': baseline_stats['ndcg-head@10'],
        'ndcg-mid': baseline_stats['ndcg-mid@10'],
        'ndcg-tail': baseline_stats['ndcg-tail@10'],
        'arp': baseline_stats['arp'],
        'inference_time': 0.0
    })
    
    it_num += 1

    # 5) full sweep for n>0
    for n, beta in itertools.product(all_Ns, betas):
        # measure inference time
        start_time = time.time()
        res = trainer.evaluate(
            valid_data,
            model_file=args.path,
            show_progress=config["show_progress"],
            N=n, beta=beta
        )
        inference_time = time.time() - start_time

        # compute metrics
        ndcg      = res['ndcg@10']
        gini      = res['Gini_coef@10']
        diff_ndcg = abs(ndcg  - baseline_stats['ndcg@10'])      / baseline_stats['ndcg@10']
        diff_gini = abs(gini  - baseline_stats['Gini_coef@10']) / baseline_stats['Gini_coef@10']
        gain      = diff_gini - diff_ndcg

        # update best
        if diff_ndcg <= 0.05 and gain > best_diff:
            best_diff = gain
            best_pair = (n, beta)

        # record + print
        records.append({
            'N': n, 'beta': beta,
            'ndcg': ndcg, 'gini': gini, 'gain': gain,
            'Deep long tail coverage': res.get('Deep_LT_coverage@10'),
            'ndcg-head': res.get('ndcg-head@10'),
            'ndcg-mid': res.get('ndcg-mid@10'),
            'ndcg-tail': res.get('ndcg-tail@10'),
            'arp': res.get('ARP@10'),
            'inference_time': inference_time
        })
        print(
            f"[Iter {it_num:04d}] N={n:.2f}, β={beta:.2f} → "
            f"ndcgΔ={diff_ndcg:.3f}, giniΔ={diff_gini:.3f}, "
            f"time={inference_time:.2f}s, best_gain={best_diff:.3f}"
        )
        it_num += 1

    # 6) save results + final report
    df = pd.DataFrame(records)
    out_csv = getattr(args, 'output_csv', 'tuning_results_pmmf.csv')
    df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")
    
    print("=== DONE ===")
    print(f"Best triplet: {best_pair}, best gain: {best_diff:.3f}")
    return best_pair



def create_visualizations_neurons():
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.path, sae=(args.model=='SASRec_SAE'), device=device
    )  
    
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    arps = [0.00035533782557826705]
    ndcgs = [0.6273]
    hits = [0.6725]
    coverages = [0.892436974789916]
    lt_coverages = [0.8908966928060007]
    deep_lt_coverages = [0.8716216216216216]
    dampen_percs = [0.0]  # still no value provided
    ginis = [0.5849518873628745]
    ndcg_heads = [0.6589]
    ndcg_mids = [0.5763]
    ndcg_tails = [0.6798]

    neuron_count = 0
    count = 0
    tochange = np.linspace(0, 4096, 9).tolist()
    # tochange = np.linspace(-5, 5, 1)
    # tochange = [[0.0, 1.0],  [0.0, 0.25], [0.5, 1.0], [0.0, 0.5], [0.5, 1.5], [0, 1.5], [0.5, 2.0], [1.0, 2.0], [1.0, 2.0], [1.5, 2.0]]
    toc = [[0.0, 1.0], [0.5, 1.0], [0.0, 0.5], [0.5, 1.5], [0, 1.5], [0.5, 2.0], [1.0, 2.0], [1.0, 2.0], [1.5, 2.0], [1.5, 2.5]]
    
    # tochange = np.linspace(0, 64, 17).tolist()
    for change in tochange:
        if count==0:
            test_result = trainer.evaluate(
                valid_data, model_file=args.path, show_progress=config["show_progress"]
            )
            print(test_result) 
        else:
            test_result = trainer.evaluate(
                valid_data, model_file=args.path, show_progress=config["show_progress"], N=change, beta=1.0
            )
        count += 1
        ndcgs.append(test_result['ndcg@10'])
        ndcg_heads.append(test_result['ndcg-head@10'])
        ndcg_mids.append(test_result['ndcg-mid@10'])
        ndcg_tails.append(test_result['ndcg-tail@10'])
        hits.append(test_result['hit@10'])
        coverages.append(test_result['coverage@10'])
        lt_coverages.append(test_result['LT_coverage@10'])
        deep_lt_coverages.append(test_result['Deep_LT_coverage@10'])
        ginis.append(test_result['Gini_coef@10'])
        dampen_percs.append(change)
        arps.append(test_result['ARP@10'])
        print(test_result['ndcg@10'])
        print(test_result['Deep_LT_coverage@10'])
        print(test_result['Gini_coef@10'])
        
        neuron_count += 2
    display_metrics_table(dampen_percs, ndcgs, hits, coverages, lt_coverages, deep_lt_coverages, ginis,
                         arps, ndcg_heads, ndcg_mids, ndcg_tails)




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



from scipy.stats import pearsonr
import h5py
from itertools import combinations
from multiprocessing import Pool, cpu_count
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



# file_path = r"./dataset/ml-1m/neuron_activations_popular_sasrec.h5"
# dataset_name = "dataset"  # Replace with actual dataset key inside the HDF5
# output_csv = "correlation_pairs_popular.csv"
# num_workers = 6  # You said 6 cores!
# with h5py.File(file_path, "r") as f:
#     data = f[dataset_name][...]  # shape: (64, X)

# assert data.shape[0] == 64, "Expected 64 rows (neurons)."

# === PREPARE PAIRS (i < j only) ===
# row_pairs = list(combinations(range(64), 2))  # Only unique pairs
# save_cohens_d()
# exit()

def compute_corr(pair):
    i, j = pair
    r, _ = pearsonr(data[i], data[j])
    return (i, j, r)



if __name__ == "__main__":
    

    # exit()
    # create_item_popularity_csv(0.2)
    # exit()
    # remove_sparse_users_items()
    # remove_sparse_users_items(10)
    # exit()
    # sample_users_interactions(15000)
    # exit()
    # save_cohens_d()
    # exit()
    # with open(r"./dataset/Amazon_Electronics/Amazon_Electronics.inter", 'r', encoding='utf-8') as f:
    #     for i, line in enumerate(f):
    #         if i >= 10000000:
    #             break
    #         # Strip newline and split
    #         parts = line.rstrip('\n').split('\t')
    #         if len(parts) != 4:
    #             raise ValueError(
    #                 f"Line {i+1} in hu has {len(parts)} columns (expected {4}): {parts!r}"
    #             )
    #     print(
    #         f"Validation passed: first {min(1000, i+1)} lines ofgg each have {4} columns."
    #     )


    # exit()
    # df = pd.read_csv(r"./dataset/yelp2018/yelp2018.inter", sep='\t', nrows=5)
    # print(df.columns.tolist())
    # sample_users_interactions(20000)
    # exit()

    # df = pd.read_csv(r"./dataset/mind_small_train/mind_small_train.inter", sep='\t')

    # # 2. Extract unique item IDs
    # unique_items = df['item_id:token'].drop_duplicates()

    # # 3. Write them out to the new file with the correct header
    # unique_items.to_frame().to_csv(
    #     r"./dataset/mind_small_train/mind_small_train.item",     # output filename
    #     sep='\t',                     # same separator
    #     index=False,                  # no index column
    #     header=['item_id:token']      # ensure the column name is exactly this
    # )

    # exit()
    # plot_interaction_distribution(r"./dataset/lastfm/item_popularity_labels_with_titles.csv")
    # exit()
    # save_cohens_d()
    # exit()
    # create_item_popularity_csv(0.2)
    # exit()
    # save_cohens_d()
    # exit()
    # save_cohens_d()
    # exit()
    # remove_sparse_users_items(5)
    # exit()
    # save_mean_SD()
    # exit()
    # create_item_popularity_csv(0.2)
    # exit()
    
    # save_mean_SD()
    # exit()
    # extract_sort_top_neurons("lastfm")
    # exit()
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

        # compute column width for alignment
        max_key_len = max(len(k) for k in keys)

        # print header
        print(f"{'Metric':<{max_key_len}} | Value")
        print(f"{'-'*max_key_len}-|-------")

        # print each metric with its dynamic value
        for key in keys:
            value = test_result[key]             # get value from your OrderedDict
            print(f"{key:<{max_key_len}} | {value:>7.4f}")

    else:
        # config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        #     model_file=args.path, sae=(args.model=='SASRec_SAE'), device=device
        # )  
        
        # trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        # trainer.save_neuron_activations3(model_file=args.path)
        # exit()
        # trainer.fit_gate( 
        #     train_data,
        #     valid_data=test_data,
        #     show_progress=True,
        #     device=config["device"],
        #     path=args.path
        #     )

        if(args.test):
            # if(args.corr_file):
            #     test_result = trainer.dampen_neurons(
            #         train_data, model_file=args.path, show_progress=config["show_progress"], eval_data=args.eval_data,
            #         corr_file=args.corr_file, neuron_count=args.neuron_count,
            #         damp_percent=args.damp_percent, unpopular_only = args.unpopular_only
            #     )            
            # tune_hyperparam_pmmf()
            # ablate1()
            create_visualizations_neurons()
            # test_result = trainer.evaluate(
            #     valid_data, model_file=args.path, show_progress=config["show_progress"]
            # )      
            # print(test_result)
        elif(args.model == "SASRec_SAE" and args.save_neurons):
            data = test_data if args.eval_data else train_data
            trainer.save_neuron_activations(data,  model_file=args.path, eval_data=args.eval_data, sae=True)
        elif(args.model == "SASRec" and args.save_neurons):
            data = test_data if args.eval_data else train_data
            trainer.save_neuron_activations(data,  model_file=args.path, eval_data=args.eval_data, sae=False)
        elif(args.model == "SASRec_SAE" and args.train):
            trainer.fit_SAE(config, 
                args.path,
                train_data,
                dataset,
                valid_data=valid_data,
                show_progress=True,
                # sasrec_sae_file=r"./recbole/saved/SASRec_SAE-Apr-26-k32-64-lastfm.pth"
                )
