from recbole.quick_start import load_data_and_model
from recbole.utils import (
    get_trainer,
)
import csv
import torch

PCT_METRICS = {
    'ndcg@10',                
    'giniindex@10',            
    'averagepopularity@10',    
    'itemcoveragen@10',        
}


fieldnames = ["param1", "param2", "param3", "ndcg", "gini@10", "covn@10"]

metric_keys = [
    'ndcg@10',
    'giniindex@10',
    'itemcoveragen@10'
    ]

SHORT_NAMES = {
    'ndcg@10': 'NDCG@10',
    'giniindex@10': 'GINI@10',
    'itemcoveragen@10': 'COVN@10'
    }


def tune(args):
    if args.fair or args.random or args.ipr or args.pct or args.min_reg or args.duor:
        tune_baseline(args)
        exit()

    if args.config_json is None:
        config_dict = {
            "alpha": [0, 0],
            "steer": [0, 1],
            "steer_dir": [0, 0],
            # "analyze": True,
            "tail_ratio": 0.2,
            "sae_mode": "test",
            "metrics": ["Recall","NDCG","Hit", "Deep_LT_Coverage", "GiniIndex", "AveragePopularity", "ItemCoverageN","ItemCoverage", 'Deep_LT_Coverage',
                            "NDCGTail", "NDCGHead", "NDCGMid", "NDCGPassive", "NDCGNeutral", "NDCGActive", "NDCGHeadUser", "NDCGMidUser", "NDCGTailUser",'epochtime'],
            }
    
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.path, dict=config_dict
    )
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.eval_collector.data_collect(train_data)
    # trainer.model.N = 140
    # change1 = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    # change2 = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    change1 = [0, 0.25, 0.5, 0.75, 1.0]
    change2 = [0, 0.25, 0.5, 0.75, 1.0]

    change3 = [0, 0.5, 1.0, 1.5, 2.0]


    rows_raw = []

    trainer.model.sae_module_u.steer = False
    test_result = trainer.evaluate(
        test_data,
        model_file=args.path,
        load_best_model=False,
        show_progress=config["show_progress"]
    )
    trainer.model.restore_item_e = None
    rows_raw.append({
        'param1': 0,
        'param2': 0,
        'param3': 0,
        **{k: test_result[k] for k in metric_keys}
    })

    trainer.model.sae_module_u.steer = True

    for c3 in change3:
        for c1 in change1:
            for c2 in change2:
                trainer.model.sae_module_u.d_min = c3
                trainer.model.sae_module_u.steer_dir = 0
                trainer.model.sae_module_u.alpha_pop = c1
                trainer.model.sae_module_u.alpha_unpop = c2
                trainer.model.sae_module_u._steer_ready = False

                test_result = trainer.evaluate(
                    test_data,
                    model_file=args.path,
                    load_best_model=False,
                    show_progress=config["show_progress"]
                )
                trainer.model.restore_user_e = None
                rows_raw.append({
                    'param1': c2,
                    'param2': c1,
                    'param3': c3,
                    **{k: test_result[k] for k in metric_keys}
                })

    # Baseline: first (alpha_u, alpha_i) pair (assumes change lists start with 0.0)
    baseline = rows_raw[0]

    value_decimals = 4
    pct_decimals = 2
    show_zero_pct_on_baseline = False  # set True if you want (+0.00%)

    # Headers (rename alpha columns)
    header_labels = ['param1', 'param2', 'param3'] + [SHORT_NAMES[k] for k in metric_keys]

    # Build formatted rows
    formatted_rows = []
    for i, r in enumerate(rows_raw):
        is_baseline = (i == 0)
        formatted_row = {
            'param1': f"{r['param1']:.2f}",
            'param2': f"{r['param2']:.2f}",
            'param3': f"{r['param3']:.2f}"
        }
        for k in metric_keys:
            val  = r[k]
            base = baseline[k]

            wants_pct = k in PCT_METRICS and not is_baseline and base != 0

            if wants_pct:
                pct  = (val - base) / base * 100.0
                sign = '+' if pct >= 0 else ''
                formatted_row[SHORT_NAMES[k]] = (
                    f"{val:.{value_decimals}f} ({sign}{pct:.{pct_decimals}f}%)"
                )
            else:
                formatted_row[SHORT_NAMES[k]] = f"{val:.{value_decimals}f}"
        formatted_rows.append(formatted_row)

    # Compute column widths
    col_width = {}
    for h in header_labels:
        max_cell = max(len(row[h]) for row in formatted_rows)
        col_width[h] = max(len(h), max_cell)

    # Print table
    header_line = " | ".join(f"{h:<{col_width[h]}}" for h in header_labels)
    sep_line = "-+-".join("-" * col_width[h] for h in header_labels)
    print(header_line)
    print(sep_line)
    for fr in formatted_rows:
        line = " | ".join(f"{fr[h]:<{col_width[h]}}" for h in header_labels)
        print(line)

    # --- Write selected results to CSV (with separate alphas) --
    csv_path = rf'./dataset/{config["dataset"]}/results/LightGCN_popsteer_{config["dataset"]}-results.csv'

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_raw:
            writer.writerow({
                "param1": r["param1"],
                "param2": r["param2"],
                "param3": r["param3"],
                "ndcg": r["ndcg@10"],
                "gini@10": r["giniindex@10"],
                "covn@10": r["itemcoveragen@10"]
                })
    return rows_raw, formatted_rows


def tune_baseline(args):
    if args.config_json is None:
        config_dict = {
            "alpha": [0.5, 0.5],
            "metrics": ["Recall","NDCG","Hit", "Deep_LT_Coverage", "GiniIndex", "AveragePopularity", "ItemCoverageN","ItemCoverage", 'Deep_LT_Coverage',
                            "NDCGTail", "NDCGHead", "NDCGMid", "NDCGPassive", "NDCGNeutral", "NDCGActive", "NDCGHeadUser", "NDCGMidUser", "NDCGTailUser", "EpochTime"],
            }
    
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.path, dict=config_dict
    )

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.eval_collector.data_collect(train_data)


    test_result = trainer.evaluate(
        test_data,
        model_file=args.path,
        load_best_model=False,
        show_progress=config["show_progress"]
    )
    trainer.model.restore_item_e = None
    rows_raw   = []
    baseline = {
        'param1': 0,
        'param2': 0,
        **{k: test_result[k] for k in metric_keys}}
    rows_raw.append(baseline)
    formatted_cells = [
        f"{0.0}",
        f"{0.0}",
    ]
    for k in metric_keys:
        val  = baseline[k]
        formatted_cells.append(f"{val:.4f}")
    print(" | ".join(formatted_cells))
    

    
    if args.fair:
        model.fair = True
    elif args.random:
        model.random = True
    elif args.ipr:
        model.ipr = True
    elif args.pct:
        model.pct = True
    elif args.min_reg:
        model.min_reg = True
    elif args.duor:
        model.duor = True

    change1 = [0.0, 0.0]
    change2 = [0.0]

    if args.fair:
        change1 = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        change2 = [0.01, 0.05, 0.1]
    if args.random:
        change1 = [15, 30, 50, 75, 100]
        change2 = [0.0]
    if args.ipr:
        change1 = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        change2 = [0.0]
    if args.pct:
        change1 = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        change2 = [0.01, 0.05, 0.1]
    if args.min_reg:
        change1 = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        change2 = [0.0]
    if args.duor:
        change1 = [50, 100, 250, 500, 1000]
        change2 = [0.0]


    # --- prepare header printing ---
    header_labels = ['param1', 'param2'] + [SHORT_NAMES[k] for k in metric_keys]
    header_line = " | ".join(header_labels)
    sep_line    = "-+-".join("-" * len(h) for h in header_labels)
    print(header_line)
    print(sep_line)


    for a_u in change1:
        for a_i in change2:
            trainer.model.recommendation_count = torch.zeros(
                trainer.model.n_items, dtype=torch.long, device=trainer.device
            )
            trainer.model.param1 = a_u
            trainer.model.param2 = a_i

            test_result = trainer.evaluate(
                test_data,
                model_file=args.path,
                load_best_model=False,
                show_progress=config["show_progress"]
            )
            trainer.model.restore_item_e = None

            current = {
                'param1': a_u,
                'param2': a_i,
                **{k: test_result[k] for k in metric_keys}
            }
            rows_raw.append(current)
            # ----- format & print this row immediately -----
            formatted_cells = [
                f"{a_u:.2f}",
                f"{a_i:.2f}",
            ]
            for k in metric_keys:
                val  = current[k]
                base = baseline[k]

                wants_pct = (
                    k in PCT_METRICS           # only for the four chosen metrics
                    and base != 0
                )

                if wants_pct:
                    pct  = (val - base) / base * 100.0
                    sign = '+' if pct >= 0 else ''
                    formatted_cells.append(f"{val:.4f} ({sign}{pct:.2f}%)")
                else:
                    formatted_cells.append(f"{val:.4f}")
            print(" | ".join(formatted_cells))
    string = "org"
    if args.ipr:
        string = "ipr"
    if args.fair:
        string = "fair"
    if args.random:
        string = "random"
    if args.pct:
        string = "pct"
    if args.min_reg:
        string = "min_reg"
    if args.duor:
        string = "duor"

    # --- Write selected results to CSV (unchanged) ---
    csv_path = rf'./dataset/{config["dataset"]}/results/LightGCN_{string}_{config["dataset"]}-results.csv'

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_raw:
            writer.writerow({
                "param1": r["param1"],
                "param2": r["param2"],
                "ndcg": r["ndcg@10"],
                "gini@10": r["giniindex@10"],
                "covn@10": r["itemcoveragen@10"],
                })
    return rows_raw
