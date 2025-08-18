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
    create_atlas_visualizations,
)

import matplotlib.pyplot as plt

from experiments import ablate_neurons
from tune import tune
from pathlib import Path


def fix_ndcg_columns(dataset: str) -> None:
    """
    Adjust all result CSVs in ./dataset/{dataset}/results-final:
    • ndcg@10tail   ←  ndcg - ndcg@head
    • drop column 'ndcg@mid'
    • save the amended files in a new sibling folder
      ./dataset/{dataset}/results-final/new  (same filenames)

    Parameters
    ----------
    dataset : str
        The dataset folder name (e.g. "msmarco", "trec23", …).
    """
    # Resolve paths
    base_dir = Path(f"./dataset/duor_baseline").resolve()
    new_dir  = base_dir / "new"
    new_dir.mkdir(parents=True, exist_ok=True)

    # Process every *.csv in the source directory (non-recursive)
    for csv_path in base_dir.glob("*.csv"):
        df = pd.read_csv(csv_path)

        # Re-compute ndcg@10tail and drop ndcg@mid if present
        if {"ndcg", "ndcghead@10"}.issubset(df.columns):
            df["ndcgtail@10"] = df["ndcg"] - df["ndcghead@10"]
        else:
            raise KeyError(
                f"Required columns missing in {csv_path.name}: "
                "expected 'ndcg' and 'ndcg@head'."
            )

        if "ndcgmid@10" in df.columns:
            df = df.drop(columns="ndcgmid@10")

        # Write out with identical filename into the /new folder
        df.to_csv(new_dir / csv_path.name, index=False)

    print(f"✔ All CSVs processed; output written to: {new_dir}")



if __name__ == "__main__":
    # dataset = "ml-1mm"  # replace with your dataset name

    # # Load CSV
    # df = pd.read_csv(f"./dataset/{dataset}/neuron_stats_test.csv")

    # # Filter out rows with activation_count < 100
    # df = df[df['activation_count'] >= 100]

    # # Calculate percentage change
    # df['percentage_change'] = (df['apr_org'] - df['apr_steered']) / df['apr_org'] * 100

    # # Calculate correlation (Pearson)
    # correlation = df['cohens_d'].corr(df['percentage_change'])
    # print(f"Pearson correlation between Cohen's d and Percentage Change: {correlation:.4f}")

    # # Plot with smaller and semi-transparent points
    # plt.scatter(df['cohens_d'], df['percentage_change'], s=10, alpha=0.6)
    # plt.xlabel("Cohen's d")
    # plt.ylabel("Percentage Change")
    # plt.title("Percentage Change vs Cohen's d")
    # plt.ylim(-5, 5)  # Cap y-axis
    # plt.grid(True)
    # plt.show()

    # exit()
    # fix_ndcg_columns(dataset="BeerAdvocate")
    # exit()
    # datasett = "ml-1mm"
    # if os.path.exists(rf"./dataset/{datasett}/activations.h5"):
    #     os.remove(rf"./dataset/{datasett}/activations.h5")
    # analyze_activation_popularity(dataset="ml-1mm", h5_filename="activations.h5", binsize=0.5, nomid=True)
    # exit()

    # a, b = top_neurons_by_effect_size(dataset="ml-1mm", n=500)
    # print(len(a), " ", len(b))
    # exit()
    # print(top_neurons_by_effect_size(dataset="ml-1mm", n=1000))
    # exit()
    # create_atlas_visualizations(dataset="ml-1mm", subsample=5000, hidden_dim=8192)
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
    parser.add_argument("--pct", action="store_true", help="Whether to use pct")
    parser.add_argument("--duor", action="store_true", help="Whether to use duor")

    parser.add_argument("--analyze", action="store_true", help="Whether to analyze neurons")
    parser.add_argument("--int", action="store_true", help="Whether to analyze for interpretation")
    parser.add_argument("--lightgcn", action="store_true", help="Whether to analyze for interpretation")

    parser.add_argument("--tune", action="store_true", help="Whether to train model")
    parser.add_argument("--ablate", action="store_true", help="Whether to ablate neurons")

    parser.add_argument('--config_json', type=str, default=None,
                    help="JSON string with config overrides")

    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-100k", help="name of datasets"
    )

    parser.add_argument("--steer", action="store_true", help="Whether to steer PopSteer or not when testing")
    parser.add_argument("--a_pop", default=None, type=float, help="alpha_pop hyperparameter value")
    parser.add_argument("--a_unpop",default=None,  type=float, help="alpha_unpop hyperparameter value")
    parser.add_argument("--D", default=None, type=float,  help="Cohens d hyperparameter")

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
        # plot_ablation_results(dataset="steam")
        plot_ndcg_vs_fairness(dataset="ml-1mm", alpha_n=None, alpha_i=None, alpha_u=None, model="SASRec")
        exit()
    if args.ablate:
        ablate_neurons(args)
        exit()
    config_dict = dict()

    if args.tune == True:
        tune(args)
        exit()
    if args.train == True:
        # if args.config_json is None:
            # config_dict = {
            #     "base_path": "./saved/sasrec_beer.pth",
            #     "load": "./saved/sasrec_beer-32-52.pth",
            #     "sae_scale_size": [32, 96],
            #     "sae_k": [32, 52],
            #     "learning_rate": 1e-3,
            #     "alpha": [1.0, 1.0],
            #     "steer": [0, 0],
            #     "metrics": ["Recall","NDCG","Hit", "Deep_LT_Coverage", "GiniIndex", "AveragePopularity", "ItemCoverageN","ItemCoverage", 'Deep_LT_Coverage',
            #                  "NDCGTail", "NDCGHead", "NDCGMid", "NDCGPassive", "NDCGNeutral", "NDCGActive", "NDCGHeadUser", "NDCGMidUser", "NDCGTailUser"],
            #     "train_neg_sample_args": None,
            #     "hidden_size": 64,
            #     "input_dim": 64
            #     }


            # config_dict = {
            #     "base_path": "./saved/lightgcn_beer.pth",
            #     # "load": "./saved/LightGCN-Aug-11-2025_03-45-54.pth",
            #     "sae_scale_size": [48, 48],
            #     "sae_k": [16, 16],
            #     "learning_rate": 1e-3,
            #     "alpha": [1.0, 1.0],
            #     "steer": [0, 0],
            #     "steer_dir": [0, 0],
            #     "metrics": ["Recall","NDCG","Hit", "Deep_LT_Coverage", "GiniIndex", "AveragePopularity", "ItemCoverageN","ItemCoverage", 'Deep_LT_Coverage',
            #                  "NDCGTail", "NDCGHead", "NDCGMid", "NDCGPassive", "NDCGNeutral", "NDCGActive", "NDCGHeadUser", "NDCGMidUser", "NDCGTailUser"],
            #     # "train_neg_sample_args": None,
            #     "hidden_size": 128,
            #     "input_dim": 128
            #     }
        # if args.model in ["LightGCN_SAE", "SASRec_SAE"]:
        #     config_dict["metrics"].extend(["SAE_Loss_i", "SAE_Loss_u", "SAE_Loss_total"])
        #     config_dict["valid_metric"] = "SAE_LOSS_u"
        config_dict = {"train_neg_sample_args": None}
        run(
            args.model,
            args.dataset,
            config_file_list=config_file_list,
            config_dict=config_dict,
            nproc=args.nproc,
            ip=args.ip,
            port=args.port,
            group_offset=args.group_offset,
        )


    elif args.test == True:
        if args.config_json is None:
            config_dict = {
                "alpha_pop": args.a_pop,
                "alpha_unpop": args.a_unpop,
                "D": args.D,
                "steer": args.steer
                }

        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
            model_file=args.path, dict=config_dict
        )
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        trainer.eval_collector.data_collect(train_data)
        if args.analyze:
            if args.lightgcn:
                trainer.synthetic_lightgcn(data=test_data, eval_data=True, model_file=args.path)
                exit()
            if not args.int:
                trainer.analyze_neurons(train_data, model_file=args.path, eval_data=False)
            elif args.int:
                trainer.analyze_neurons_int(test_data, model_file=args.path, eval_data=True)
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


