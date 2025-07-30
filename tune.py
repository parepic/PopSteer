from recbole.quick_start import load_data_and_model
from recbole.utils import (
    get_trainer,
)
import csv
import torch

PCT_METRICS = {
    'ndcg@10',                 # NDCG
    'giniindex@10',            # GINI
    'averagepopularity@10',    # AVGPOP
    'itemcoveragen@10',        # COVN
}



metric_keys = [
    'ndcg@10',
    'giniindex@10',
    'averagepopularity@10',
    'itemcoverage@10',
    'ndcgtail@10',
    'ndcghead@10',
    'ndcgmid@10',
    'itemcoveragen@10',
    'dltc@10'
    ]

SHORT_NAMES = {
    'ndcg@10': 'NDCG@10',
    'giniindex@10': 'GINI@10',
    'averagepopularity@10': 'AVGPOP@10',
    'itemcoverage@10': 'COV@10',
    'itemcoveragen@10': 'COVN@10',
    'ndcgtail@10':'NDCGTAIL@10',
    'ndcgmid@10':'NDCGMID@10',
    'ndcghead@10':'NDCGHEAD@10',
    'dltc@10':'DLTC@10'
}


def tune(args):
    if args.fair:
        tune_FAIR(args)
        exit()

    if args.config_json is None:
        config_dict = {
            "alpha": [0, 0],
            "steer": [0, 1],
            "steer_dir": [-1, -1],
            "analyze": True,
            "tail_ratio": 0.2,
            "sae_mode": "test",
            "metrics": ["Recall","NDCG","Hit", "Deep_LT_Coverage", "GiniIndex", "AveragePopularity", "ItemCoverageN", "Deep_LT_Coverage", "ItemCoverage",
                        "NDCGTail", "NDCGHead", "NDCGMid", "NDCGUserTail", "NDCGUserHead", "NDCGUserMid"]
            }
    
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.path, dict=config_dict
    )
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.eval_collector.data_collect(train_data)
    # change1 = [0.0, 0.5, 1, 1.5,  2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    # change2 = [0.0]

    change1= [0]
    change2 = [0.0, 0.5, 1, 1.5, 2.0, 2.5, 3.0, 4.0]


    rows_raw = []

    trainer.model.sae_module_u.alpha = 0.0
    test_result = trainer.evaluate(
        valid_data,
        model_file=args.path,
        load_best_model=False,
        show_progress=config["show_progress"]
    )
    trainer.model.restore_item_e = None
    rows_raw.append({
        'alpha_u': 0,
        'alpha_i': 0,
        **{k: test_result[k] for k in metric_keys}
    })

    for c1 in change1:
        for c2 in change2:
            trainer.model.sae_module_u.steer_dir = c1
            trainer.model.sae_module_u.alpha = c2
            trainer.model.sae_module_u._steer_ready = False

            test_result = trainer.evaluate(
                valid_data,
                model_file=args.path,
                load_best_model=False,
                show_progress=config["show_progress"]
            )
            trainer.model.restore_item_e = None
            rows_raw.append({
                'alpha_u': c2,
                'alpha_i': c1,
                **{k: test_result[k] for k in metric_keys}
            })

    # Baseline: first (alpha_u, alpha_i) pair (assumes change lists start with 0.0)
    baseline = rows_raw[0]

    value_decimals = 4
    pct_decimals = 2
    show_zero_pct_on_baseline = False  # set True if you want (+0.00%)

    # Headers (rename alpha columns)
    header_labels = ['alpha_u', 'alpha_i'] + [SHORT_NAMES[k] for k in metric_keys]

    # Build formatted rows
    formatted_rows = []
    for i, r in enumerate(rows_raw):
        is_baseline = (i == 0)
        formatted_row = {
            'alpha_u': f"{r['alpha_u']:.2f}",
            'alpha_i': f"{r['alpha_i']:.2f}",
        }
        for k in metric_keys:
            val  = r[k]
            base = baseline[k]

            # --- decide whether this metric should have a Δ % ---
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
    csv_path = rf'./dataset/{config["dataset"]}/results/SASRec_user_{config["dataset"]}-test.csv'
    fieldnames = ["alpha_u", "alpha_i", "ndcg",  "dltc@10", "avgpop@10", "gini@10", "cov@10", "covn@10", 'ndcgtail@10', 'ndcgmid@10', 'ndcghead@10'
                  ]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_raw:
            writer.writerow({
                "alpha_u": r["alpha_u"],
                "alpha_i": r["alpha_i"],
                "ndcg": r["ndcg@10"],
                "avgpop@10": r["averagepopularity@10"],
                "gini@10": r["giniindex@10"],
                "cov@10": r["itemcoverage@10"],
                "covn@10": r["itemcoveragen@10"],
                'ndcgtail@10': r["ndcgtail@10"],
                'ndcgmid@10': r["ndcgmid@10"],
                'ndcghead@10': r["ndcghead@10"],
                'dltc@10': r["dltc@10"]
                })

    return rows_raw, formatted_rows




def tune_FAIR(args):
    if args.config_json is None:
        config_dict = {
            "alpha": [0.5, 0.5],
            "metrics": ["Recall","NDCG","Hit", "Deep_LT_Coverage", "GiniIndex",
                        "AveragePopularity", "ItemCoverageN", "ItemCoverage",
                        "NDCGTail", "NDCGHead", "NDCGMid", 'Deep_LT_Coverage']
        }

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.path, dict=config_dict
    )

    model.fair = True
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.eval_collector.data_collect(train_data)

    change1 = [0.2, 0.4, 0.6, 0.8, 1.0]
    change2 = [0.01, 0.05, 0.1]

    # --- prepare header printing ---
    header_labels = ['alpha_u', 'alpha_i'] + [SHORT_NAMES[k] for k in metric_keys]
    header_line = " | ".join(header_labels)
    sep_line    = "-+-".join("-" * len(h) for h in header_labels)
    print(header_line)
    print(sep_line)

    rows_raw   = []
    baseline   = None

    for a_u in change1:
        for a_i in change2:
            trainer.model.recommendation_count = torch.zeros(
                trainer.model.n_items, dtype=torch.long, device=trainer.device
            )
            trainer.model.a1 = a_u
            trainer.model.a2 = a_i

            test_result = trainer.evaluate(
                valid_data,
                model_file=args.path,
                load_best_model=False,
                show_progress=config["show_progress"]
            )
            trainer.model.restore_item_e = None

            current = {
                'alpha_u': a_u,
                'alpha_i': a_i,
                **{k: test_result[k] for k in metric_keys}
            }
            rows_raw.append(current)

            # establish baseline on first iteration
            if baseline is None:
                baseline = current

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
                    and current is not baseline
                    and base != 0
                )

                if wants_pct:
                    pct  = (val - base) / base * 100.0
                    sign = '+' if pct >= 0 else ''
                    formatted_cells.append(f"{val:.4f} ({sign}{pct:.2f}%)")
                else:
                    formatted_cells.append(f"{val:.4f}")
            print(" | ".join(formatted_cells))

    # --- Write selected results to CSV (unchanged) ---
    csv_path = rf'./dataset/{config["dataset"]}/results/SASRec_fair_{config["dataset"]}-test.csv'
    fieldnames = ["alpha_u", "alpha_i", "ndcg",  "dltc@10", "avgpop@10", "gini@10", "cov@10", "covn@10", 'ndcgtail@10', 'ndcgmid@10', 'ndcghead@10']

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_raw:
            writer.writerow({
                "alpha_u": r["alpha_u"],
                "alpha_i": r["alpha_i"],
                "ndcg": r["ndcg@10"],
                "avgpop@10": r["averagepopularity@10"],
                "gini@10": r["giniindex@10"],
                "cov@10": r["itemcoverage@10"],
                "covn@10": r["itemcoveragen@10"],
                'ndcgtail@10': r["ndcgtail@10"],
                'ndcgmid@10': r["ndcgmid@10"],
                'ndcghead@10': r["ndcghead@10"],
                'dltc@10': r["dltc@10"]
                })
    return rows_raw
