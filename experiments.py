from recbole.quick_start import load_data_and_model
from recbole.utils import (
    get_trainer,
    top_neurons_by_effect_size
)

import os
import pandas as pd


def ablate_neurons(args):
    """Run neuron ablation experiments and save evaluation results.

    The function now saves all collected rows in CSV format to
    ``./dataset/<dataset_name>/results/ablation_results.csv``.
    """
    if args.config_json is None:
        config_dict = {
            "alpha": [0, 0],
            "steer": [0, 0],
            "steer_dir": [0, 0],
            "analyze": True,
            "tail_ratio": 0.2,
            "sae_mode": "test",
            "metrics": [
                "Recall",
                "NDCG",
                "Hit",
                "Deep_LT_Coverage",
                "GiniIndex",
                "AveragePopularity",
                "ItemCoverageN",
                "ItemCoverage",
                "Deep_LT_Coverage",
                "NDCGTail",
                "NDCGHead",
                "NDCGMid",
                "NDCGPassive",
                "NDCGNeutral",
                "NDCGActive",
                "NDCGHeadUser",
                "NDCGMidUser",
                "NDCGTailUser",
                "epochtime",
            ],
        }

    # load data and model
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.path, dict=config_dict
    )

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.eval_collector.data_collect(train_data)

    pop_neurons, unpop_neurons = top_neurons_by_effect_size(dataset=config["dataset"], threshold=1, n=50000)
    print(len(pop_neurons), " pop neurons length ")
    print(len(unpop_neurons), " unpop neurons length ")

    rows_raw = []
    trainer.model.restore_item_e = None

    # --- Ablate unpopular neurons first
    for i in range(len(unpop_neurons)):
        trainer.model.sae_module_u.ablate_list = unpop_neurons[:i]
        trainer.model.sae_module_u.dampen_now = False

        test_result = trainer.evaluate(
            test_data,
            model_file=args.path,
            load_best_model=False,
            show_progress=config["show_progress"],
        )
        rows_raw.append(
            {
                "popular": False,
                "n": i,
                "giniindex@10": test_result["giniindex@10"],
                "covn@10": test_result["itemcoveragen@10"],
                "cov@10": test_result["itemcoverage@10"],
                "avgpop@10": test_result["averagepopularity@10"]
            }
        )
        # print(rows_raw[-1])  # print only the latest row to reduce clutter

    # --- Ablate popular neurons next
    # print(pop_neurons)
    for i in range(len(pop_neurons)):
        # print(i)
        trainer.model.sae_module_u.ablate_list = pop_neurons[:i]
        trainer.model.sae_module_u.dampen_now = True
        test_result = trainer.evaluate(
            test_data,
            model_file=args.path,
            load_best_model=False,
            show_progress=config["show_progress"],
        )
        # print(test_result)
        rows_raw.append(
            {
                "popular": True,
                "n": i,
                "giniindex@10": test_result["giniindex@10"],
                "covn@10": test_result["itemcoveragen@10"],
                "cov@10": test_result["itemcoverage@10"],
                "avgpop@10": test_result["averagepopularity@10"]
            }
        )
        # print(rows_raw[-1])

    # --- Save results
    results_dir = rf"./dataset/{config['dataset']}/results"
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "ablation_results.csv")
    pd.DataFrame(rows_raw).to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
