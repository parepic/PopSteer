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
    get_trainer
)




def tune_hyperparam(args, device):
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



def tune_hyperparam_FAIRSTAR(args, device):
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




def tune_hyperparam_pct(args, device):
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


def tune_hyperparam_pct(args, device):
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




def tune_hyperparam_random(args, device):
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


def tune_hyperparam_ipr(args, device):
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



def tune_hyperparam_pmmf(args, device):
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
