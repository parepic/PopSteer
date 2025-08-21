# PopSteer

This repository contains the code used for the experiments in "From Insight to Intervention: Interpretable Neuron Steering for
Controlling Popularity Bias in Recommender Systems"

# Installation

```bash
git clone https://github.com/ANONYMOUS/PopSteer.git
cd PopSteer

# install Python dependencies
pip install -r requirements.txt
```

## Dataset preparation

PopSteer expects datasets to be provided in an [atomic](https://recbole.io/docs/user_guide/data/atomic_files.html) format and stored in the `./dataset` folder.  

The repository already includes the datasets used in the paper. You can easily extend it by adding additional datasets in the same format.  

### 1 · Train PopSteer

First, train a baseline recommender model that will act as the teacher for PopSteer. Later, train PopSteer pointing to the base recommender.   
All hyperparameters are controlled via the YAML configuration file.

    python run.py --model=SASRec --dataset=ml-1m --config_files=example_config.yaml --train

#### Flags

| Flag             | Description                                                                 | Default (used in paper) |
|------------------|-----------------------------------------------------------------------------|-------------------------|
| `--train`        | Runs the training pipeline (presence-based flag).                           | –                       |
| `--dataset`      | Dataset identifier. Options: `ml-1m`, `Steam`, `BeerAdvocate`, `Yelp`.      | All four datasets       |
| `--model`        | Model architecture to train.                                                | `SASRec` / `SASRec_SAE` |
| `--config_files` | YAML configuration file(s) with hyperparameters.                            | `example_config.yaml`   |

#### Notes
- Uuse **`SASRec`** to train the base recommender model.  
- Use **`SASRec_SAE`** to train PopSteer. In this case, add the `base_path` parameter in your YAML to point to the pretrained recommender file.


### 1 · Test PopSteer

Run evaluation with PopSteer steering enabled. The flags map to the paper’s main 3 hyperparameters:
`--a_pop` → α_pop (suppresses popularity-aligned neurons),
`--a_unpop` → α_unpop (amplifies long-tail neurons),
`--D` → β (Cohen’s-d threshold).

    python run.py --path=saved/sasrec_ml-1m-44.pth --a_pop=1.0 --a_unpop=1.0 --D=0 --steer --test

#### Flags

| Flag        | Description                                                                                           | Default / Example |
|-------------|-------------------------------------------------------------------------------------------------------|-------------------|
| `--path`    | Path to the trained checkpoint to load for testing/steering (e.g., SASRec(+SAE) run).                 | `saved/sasrec_ml-1m-44.pth` |
| `--a_pop`   | Steering strength for popularity-aligned neurons (**αPop**). Larger → stronger suppression.           | `1.0`             |
| `--a_unpop` | Steering strength for unpopularity-aligned neurons (**αUnpop**). Larger → stronger amplification.     | `1.0`             |
| `--D`       | Cohen’s-d threshold (**β**) selecting which neurons to steer (`0` steers all; larger steers fewer).   | `0`               |
| `--steer`   | Enable neuron steering at inference                                                                   | –                 |
| `--test`    | Flag to indicate testing                                                                              | –                 |

### 3 · Neuron analysis

Analyzes neurons through generating synthetic data and feeding it to model.

```
python run.py --path=saved/sasrec_ml-1m-44.pth  --analyze    
```

| Flag      | Description                                                                                  | Default / Example              |
|-----------|----------------------------------------------------------------------------------------------|--------------------------------|
| `--path`  | Path to the trained checkpoint to analyze (e.g., SASRec + SAE run).                          | `saved/sasrec_ml-1m-44.pth`    |
| `--analyze` | Runs neuron analysis: generates synthetic profiles, records activations, computes metrics. | Presence-based flag            |


### 4 · Tuning
We provide code for tuning PopSteer and the baselines. To tune PopSteer, use:

```
python run.py --tune --path=saved/model-name.pth 
```

To tune the baselines, use one of the flags `--fair`, `--ipr`, `--duor`, `--pct`, `--min_reg`. For instance:

```
python run.py --tune --path=saved/model-name.pth --fair
```


### 4 · LightGCN experiments
We also tested PopSteer when using LighGCN as a base recommender. For training PopSteer and LightGCN, use:

`
python run.py --model=LightGCN --dataset=ml-1m --config_files=example_config_lightgcn.yaml --train
`
Rest of the initial steps also apply to LightGCN version of PopSteer.









