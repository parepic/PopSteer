from recbole.quick_start import run_recbole, load_data_and_model, run
import numpy as np

from recbole.utils import (
    get_trainer
)
import pandas as pd


def ablate1(args, device):
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