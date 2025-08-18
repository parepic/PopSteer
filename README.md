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

### 1 · Train the Base Recommender

Train a baseline recommender model that will act as the **teacher** for PopSteer.  
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
- Use **`SASRec`** to train the plain teacher model.  
- Use **`SASRec_SAE`** to train PopSteer. In this case, add the `base_path` parameter in your YAML to point to the pretrained recommender checkpoint.



