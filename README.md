# PopSteer

A post‑hoc Sparse‑Autoencoder–based framework that **interprets and steers deep recommender models to mitigate popularity bias while keeping accuracy high**.

---

## Installation

# clone anonymous repo
git clone https://github.com/ANONYMOUS/PopSteer.git
cd PopSteer

# install Python dependencies
pip install -r requirements.txt
```

<!-- --- -->

<!-- ## Dataset preparation

PopSteer follows the common **5‑core sequential splits** used in the paper (MovieLens‑1M and Last.fm‑1K). If you place the raw files in `data/` with the usual RecBole naming convention, the helper scripts will preprocess them automatically the first time you run a command. -->

---

## Usage

### 1 · Train the base recommender

*This step fits a vanilla recommender model (in our experiments, SASRec was used) that later acts as the teacher for PopSteer.*

```bash
python run.py   --train_recommender   --dataset "ml-1m"        # MovieLens‑1M (default)
```

| Flag                  | Description                                                  | Default used in paper  |
| --------------------- | ------------------------------------------------------------ | ---------------------- |
| `--train_recommender` | Activates the recommender model training pipeline. Presence‑based flag. | –                      |
| `--dataset`           | Identifier of the dataset (`"ml-1m"` or `"lastfm"`).         | `"ml-1m"` / `"lastfm"` |

**Outputs**: checkpoint under `recbole/saved/SASRec-<DATE>-<DATASET>.pth`.

---

### 2 · Train PopSteer (Sparse Autoencoder)

*This fits a sparse autoencoder that replicates base model's output while exposing interpretable neurons.*

```bash
python run.py   --train_popsteer   -p "recbole/saved/SASRec-Apr-27-2025-ml-1m.pth"   --top_k 32    --scale 64
```

| Flag                      | Description                                                | Default used in paper |
| ------------------------- | ---------------------------------------------------------- | --------------------- |
| `--train_popsteer`        | Triggers SAE training.                                     | –                     |
| `-p`, `--pretrained_path` | Path to the **base recommender checkpoint**.                    | *(required)*          |
| `--top_k`                 | Sparsity *K*: number of hidden activations kept per input. | **32**                |
| `--scale`                 | Scale factor *s*: hidden‑dim = *s* × input‑dim.            | **64**                |


**Outputs**: checkpoint under `recbole/saved/SASRec_SAE-<DATE>-k<top_k>-<scale>-<DATASET>.pth`.

---

### 3 · Neuron analysis

*Runs the synthetic‑user probing routine to compute each neuron’s Cohen‑**d** popularity score.*

```bash
python run.py   --analyze_neurons   -p "recbole/saved/SASRec_SAE-Apr-27-k32-64-ml-1m.pth"
```

| Flag                | Description                                                     | Default used in paper       |
| ------------------- | --------------------------------------------------------------- | --------------------------- |
| `--analyze_neurons` | Executes Section 3 pipeline (synthetic profiles + effect size). | –                           |
| `-p`                | Path to the **trained SAE**.                                    | *(required)*                |
**Outputs**: three CSV reports are saved to `datasets/{dataset}/` after the run:

- `cohens_d.csv` – per‑neuron Cohen‑*d* values
- `row_stats_popular.csv` – mean and standard‑deviation activations for every **synthetic popular‑only** user profile.
- `row_stats_unpopular.csv` – the same statistics for **synthetic unpopular‑only** profiles.

---

### 4 · Test / steer PopSteer

*Applies neuron‑level steering (Eq. 4) at inference time and evaluates fairness & accuracy.*

```bash
python run.py   --test_popsteer   -p "recbole/saved/SASRec_SAE-Apr-27-k32-64-ml-1m.pth"   --num_neurons 4096   --alpha 1.5          # recommended for MovieLens; use 4.0 for Last.fm
```

| Flag              | Description                                                      | Default used in paper                   |
| ----------------- | ---------------------------------------------------------------- | --------------------------------------- |
| `--test_popsteer` | Switches run‑mode to evaluation with steering.                   | –                                       |
| `-p`              | Path to the trained SAE.                                         | *(required)*                            |
| `--num_neurons`   | How many neurons to steer, i.e. *N* in the paper. | **4096** |
| `--alpha`         | Scaling hyper‑parameter α (Eq. 3) controlling steering strength. | **1.5 (MovieLens)** / **4.0 (Last.fm)** |

**Metrics printed**: nDCG@k, precision@10, Long‑tail Coverage@k and Gini@k
