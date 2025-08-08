# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/10/3, 2020/10/1
# @Author : Zhen Tian, Yupeng Hou, Zihan Lin
# @Email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

import argparse
import torch
import pandas as pd
from recbole.quick_start import run, run_recbole, load_data_and_model
from recbole.utils import (
    get_trainer,
    plot_ndcg_vs_fairness,
    remove_sparse_users_items,
    keep_random_users,
    make_labels,
    retain_last_x_days,
    create_atlas_visualizations,
    analyze_activation_popularity,
    top_neurons_by_effect_size
)
import csv
import os
from tune import tune

from recbole.data import create_item_popularity_csv


if __name__ == "__main__":
    # datasett = "ml-1mm"
    # if os.path.exists(rf"./dataset/{datasett}/activations.h5"):
    #     os.remove(rf"./dataset/{datasett}/activations.h5")

    # print(top_neurons_by_effect_size(dataset="ml-1mm", n=1000))
    # exit()
    # analyze_activation_popularity(dataset="ml-1mm", h5_filename="activations.h5")
    # exit()
    # create_atlas_visualizations(dataset="ml-1mm", subsample=10000)
    # exit()
    # retain_last_x_days(dataset="music", days=178)
    # exit()
    # keep_random_users(dataset="yoochoose-clicks", x=25000)
    # remove_sparse_users_items(5, "ml-1m")
    # exit()
    # parameter_dict = {
    # 'train_neg_samplze_args': None,
    # }
    # run_recbole(model='SASRec', dataset='ml-100k', config_dict=parameter_dict)
    # exit()
    # create_item_popularity_csv("ml-1m", 0.2)
    # plot_ndcg_vs_fairness(dataset="Amazon_Gift_Cards", alpha_i=0.0, model="SASRec")
    # exit()
    # make_labels(dataset="yoochoose-clicks")
    # exit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
    parser.add_argument("--train", action="store_true", help="Whether to train model")
    parser.add_argument("--test", action="store_true", help="Whether to test model")
    parser.add_argument("--plot", action="store_true", help="Whether to test model")
    parser.add_argument("--min_reg", action="store_true", help="Whether to use min-regularizer")

    parser.add_argument("--fair", action="store_true", help="Whether to use FAIR")
    parser.add_argument("--random", action="store_true", help="Whether to use random reranker")
    parser.add_argument("--ipr", action="store_true", help="Whether to use random reranker")
    parser.add_argument("--pct", action="store_true", help="Whether to use random reranker")

    parser.add_argument("--analyze", action="store_true", help="Whether to analyze neurons")
    parser.add_argument("--int", action="store_true", help="Whether to analyze for interpretation")

    parser.add_argument("--tune", action="store_true", help="Whether to train model")

    parser.add_argument('--config_json', type=str, default=None,
                    help="JSON string with config overrides")

    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-100k", help="name of datasets"
    )

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
        "--base_path", type=str, default='no path', help="base model path"
    )
    parser.add_argument(
        "--path", type=str, default='no path', help="model path"
    )

    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )

    args, _ = parser.parse_known_args()
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )
    if args.plot:
        plot_ndcg_vs_fairness(dataset="yelp2018", alpha_n=None, alpha_i=None, alpha_u=None, model="SASRec")
        exit()
    config_dict = dict()
    if args.config_json:
        import json, ast
        # Allow either strict JSON or python-literal (for lists)
        try:
            config_dict = json.loads(args.config_json)
        except json.JSONDecodeError:
            config_dict = ast.literal_eval(args.config_json)
    
    if args.tune == True:
        tune(args)
        exit()
    if args.train == True:
        if args.config_json is None:
            config_dict = {
                "base_path": "./saved/sasrec_beer.pth",
                # "load": "./saved/sasrec_beer-32-44.pth",
                "sae_scale_size": [64, 64],
                "sae_k": [32, 44],
                "learning_rate": 1e-3,
                "alpha": [1.0, 1.0],
                "steer": [0, 0],
                "steer_dir": [0, 0],
                "metrics": ["Recall","NDCG","Hit", "Deep_LT_Coverage", "GiniIndex", "AveragePopularity", "ItemCoverageN","ItemCoverage", 'Deep_LT_Coverage',
                             "NDCGTail", "NDCGHead", "NDCGMid", "NDCGPassive", "NDCGNeutral", "NDCGActive", "NDCGHeadUser", "NDCGMidUser", "NDCGTailUser"],
                # "train_neg_sample_args": None,
                "hidden_size": 64,
                "input_dim": 64
                }
        if args.model in ["LightGCN_SAE", "SASRec_SAE"]:
            config_dict["metrics"].extend(["SAE_Loss_i", "SAE_Loss_u", "SAE_Loss_total"])
            config_dict["valid_metric"] = "SAE_LOSS_u"
        run(
            args.model,
            args.dataset,
            config_file_list=config_file_list,
            config_dict=config_dict,
            nproc=args.nproc,
            world_size=args.world_size,
            ip=args.ip,
            port=args.port,
            group_offset=args.group_offset,
        )


    elif args.test == True:
        if args.config_json is None:
            config_dict = {
                "alpha": [0.1, 1],
                "steer": [0, 0],
                "steer_dir": [0, 0],
                "analyze": True,
                "tail_ratio": 0.2,
                "metrics": ["Recall","NDCG","Hit", "Deep_LT_Coverage", "GiniIndex", "AveragePopularity", "ItemCoverageN","ItemCoverage", 'Deep_LT_Coverage',
                             "NDCGTail", "NDCGHead", "NDCGMid", "NDCGPassive", "NDCGNeutral", "NDCGActive", "NDCGHeadUser", "NDCGMidUser", "NDCGTailUser", "EpochTime"],
                }

        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
            model_file=args.path, dict=config_dict
        )
        # model.sae_module_u.N=2048
        # model.sae_module_u.alpha=3
        # model.sae_module_u.beta=1

        if args.fair:
            model.fair = True
            model.a1 = 0.9
            model.a2 = 0.1
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        trainer.eval_collector.data_collect(train_data)
        if args.analyze:
            if not args.int:
                trainer.analyze_neurons(train_data, model_file=args.path, eval_data=False)
            elif args.int:
                trainer.analyze_neurons_int(train_data, model_file=args.path, eval_data=False)
            exit()
        test_result = trainer.evaluate(
            test_data, model_file=args.path, load_best_model = False, show_progress=config["show_progress"]
        )
        
        keys = [
            'recall@10',
            'ndcg@10',
            'hit@10',
            'giniindex@10',
            'averagepopularity@10',
            'itemcoverage@10',
            'itemcoveragen@10',
            'ndcgtail@10',
            'ndcghead@10',
            'ndcgmid@10',
            'ndcgtailuser@10',
            'ndcgheaduser@10',
            'ndcgmiduser@10',
            "epochtime"
            ]

        max_key_len = max(len(k) for k in keys)
        
        # print header
        print(f"{'Metric':<{max_key_len}} | Value")
        print(f"{'-'*max_key_len}-|-------")
        print(test_result)
        # print each metric with its dynamic value
        for key in keys:
            value = test_result[key]             # get value from your OrderedDict
            print(f"{key:<{max_key_len}} | {value:>7.4f}")


