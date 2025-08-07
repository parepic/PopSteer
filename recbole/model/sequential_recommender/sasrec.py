# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import torch
from torch import nn
import pandas as pd
import numpy as np
import math
from scipy.optimize import linprog
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
from recbole.utils import create_pop_unpop_mappings, make_items_popular, make_items_unpopular,save_batch_activations,get_extreme_correlations
from typing import Literal, Union, Optional
Array = Union[np.ndarray, torch.Tensor]

class SASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASRec, self).__init__(config, dataset)
        self.alpha = config['alpha'][1]
        self.dtype = torch.float32

        # self.steer = config['steer'][1]
        # self.steer_dir = config['steer_dir'][1]
        # self._steer_ready = False
        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.N = self.hidden_size
        self.a1 = config["alpha"][0]
        self.a2 = config["alpha"][1]
        self.fair = False
        self.random = False
        self.ipr = False
        self.pct = False
        self.min_reg = False
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self._item2provider = None
        self._A = None
        self._rho = None
        self._iid2pid = None

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.dataset = config["dataset"]
        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        print("suka", self.device)
        self.dataset = config["dataset"]
        self.last_activations = None
        
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        # if self.steer == True and self.N != 0:
        #     output = self.dampen_neurons(output, dataset=self.dataset)
        self.last_activations = output
        return output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        # if self.val_fvu_i.item() != 0:

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction, popular=None):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        if popular is not None:
            if popular == True:
                item_seq = make_items_popular(item_seq, self.dataset, self.max_seq_length).to(self.device)
            elif popular == False:
                item_seq = make_items_unpopular(item_seq, self.dataset, self.max_seq_length).to(self.device)
            seq_output = self.forward(item_seq, item_seq_len)
            save_batch_activations(self.last_activations, self.hidden_size, self.dataset, popular) 
            return
        else:
            seq_output = self.forward(item_seq, item_seq_len)
            test_items_emb = self.item_embedding.weight
            scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
            if self.fair:
                scores = self.FAIR(scores, p=self.a1,alpha=self.a2).to(self.device)
            elif self.random:
                scores = self.random_reranker(scores=scores, top_k=self.a1)
            elif self.ipr:
                scores = self.ipr_baseline(scores=scores, dataset = self.dataset, alpha=self.a1)
            if self.pct:
                scores = self.pct_rerank(scores=scores, user_interest=item_seq, p=self.a1, lambda_= self.a2)
            if self.min_reg:
                scores = self.min_reg_algo(dataset=self.dataset, scores=scores, lambd=self.a1)
            return scores


    def create_synthetic_dataset(self):
        create_pop_unpop_mappings(dataset=self.dataset, embeddings=self.item_embedding.weight)



    def FAIR(self, scores, *, p: float = 0.9, alpha: float = 0.1,
            L: int = 250, K: int = 10):
        """
        Re-rank each batch row with FA*IR.
            p      – target minimum proportion of protected items
            alpha  – family-wise significance level for the binomial test
        Remaining arguments are kept for backward-compatibility.
        """
        scores = scores.detach().cpu()

        # ---- load popularity labels (unchanged) -----------------------
        df   = pd.read_csv(rf"./dataset/{self.dataset}/item_popularity_labels.csv")
        ids  = df["item_id:token"].astype(int).values
        labs = df["popularity_label"].astype(int).values
        max_id = ids.max()

        popularity_label = torch.zeros(max_id + 1, dtype=torch.bool)
        popularity_label[ids] = torch.from_numpy(labs != -1)  # True = popular
        # We treat *unpopular* as protected
        popularity_label = ~popularity_label

        # ---- take top-L candidates per row ----------------------------
        B, N          = scores.size()
        top_idx       = torch.argsort(scores, dim=1, descending=True)[:, :L]
        protected_top = popularity_label[top_idx]                  # (B,L) bool

        # ---- run FA*IR row-wise ---------------------------------------
        for b in range(B):
            row_scores    = scores[b, top_idx[b]]          # (L,)
            row_protected = protected_top[b]               # (L,)
            sel_in_top    = self.fair_topk(row_scores,
                                        row_protected,
                                        K, p, alpha)    # indices into 0..L-1

            # map back to original positions and overwrite scores
            orig_pos = top_idx[b, sel_in_top]
            base     = scores[b].max().item() + 1.0
            offsets  = torch.arange(K - 1, -1, -1, dtype=scores.dtype)
            scores[b, orig_pos] = base + offsets            # keep FA*IR order
        return scores


    def fair_topk(self,
                scores1d: torch.Tensor,
                protected1d: torch.Tensor,
                K: int,
                p: float,
                alpha: float = 0.10):
        """
        One-dimensional FA*IR (Algorithm 2) that *exactly* follows the
        binomial rule with Šidák-style multiple-test correction.
        """
        # --------------------------------------------------------------
        # helper: minimum #protected required at each prefix
        def _min_protected_per_prefix(k, p_, alpha_):
            alpha_c = 1.0 - (1.0 - alpha_) ** (1.0 / k)          # Šidák
            m = np.zeros(k, dtype=int)
            for t in range(1, k + 1):                            # prefix length
                cdf = 0.0
                for z in range(t + 1):                           # binomial CDF
                    cdf += math.comb(t, z) * (p_ ** z) * ((1.0 - p_) ** (t - z))
                    if cdf > alpha_c:
                        m[t - 1] = z
                        break
            return m

        m_needed = _min_protected_per_prefix(K, p, alpha)

        # --------------------------------------------------------------
        # build two quality-sorted lists
        idx_sorted   = np.argsort(-scores1d)                     # high→low
        prot_list    = [i for i in idx_sorted if protected1d[i]]
        nonprot_list = [i for i in idx_sorted if not protected1d[i]]

        sel  = []
        tp = tn = pp = np_ptr = 0

        for pos in range(K):                                     # positions 0..K-1
            need = m_needed[pos]                                 # min protected so far
            if tp < need:                                        # *must* take protected
                if pp < len(prot_list):  # NEW: Check if protected available
                    choose = prot_list[pp];  pp += 1;  tp += 1
                else:  # NEW: Fall back to non-protected if exhausted
                    choose = nonprot_list[np_ptr];  np_ptr += 1;  tn += 1
            else:                                                # free to take best
                next_p  = prot_list[pp]  if pp  < len(prot_list)     else None
                next_np = nonprot_list[np_ptr] if np_ptr < len(nonprot_list) else None

                if next_np is None or (next_p is not None and
                                    scores1d[next_p] >= scores1d[next_np]):
                    choose = next_p;   pp += 1;  tp += 1
                else:
                    choose = next_np;  np_ptr += 1;  tn += 1

            sel.append(choose)

        return np.array(sel, dtype=int)
    

    def _build_steering_vector(self, dataset):
        pop_neurons, unpop_neurons = get_extreme_correlations(
            rf"user/cohens_d.csv", dataset=dataset
        )
        
        if self.steer_dir == -1:
            combined = ([(i, d, "unpop")   for i, d in unpop_neurons])
        elif self.steer_dir == 1:
            combined = ([(i, d, "pop")   for i, d in pop_neurons])
        elif self.steer_dir == 0:
            combined = ([(i, d, "pop")   for i, d in pop_neurons] +
                        [(i, d, "unpop")   for i, d in unpop_neurons])

        combined_sorted = sorted(combined, key=lambda x: abs(x[1]), reverse=True)
        top_neurons = combined_sorted[: self.N]

        stats_unpop = pd.read_csv(rf"./dataset/{dataset}/user/neuron_stats_unpop.csv")
        stats_pop   = pd.read_csv(rf"./dataset/{dataset}/user/neuron_stats_pop.csv")

        abs_cohens = torch.tensor([abs(c) for _, c, _ in top_neurons],
                                device=self.device, dtype=self.dtype)

        def normalize_to_range(x, new_min, new_max):
            max_val = torch.max(x)
            if max_val == 0:
                return torch.full_like(x, (new_min + new_max) / 2)
            return (x / max_val) * (new_max - new_min) + new_min

        weights = normalize_to_range(abs_cohens, 0, self.alpha)

        steer = torch.zeros(self.hidden_size, device=self.device, dtype=self.dtype)

    
        for i, (neuron_idx, _, group) in enumerate(top_neurons):
            w = weights[i]
            if group == "unpop":
                unpop_sd = stats_unpop.iloc[neuron_idx]["sd"]
                steer[neuron_idx] += w * unpop_sd
            if group == "pop":
                pop_sd = stats_pop.iloc[neuron_idx]["sd"]
                steer[neuron_idx] -= w * pop_sd

        self.steer_vec = steer.to(self.device)
        self._steer_ready = True

    def dampen_neurons(self, pre_acts, dataset=None):
        if getattr(self, "N", None) in (None, 0):
            return pre_acts
        if not self._steer_ready:
            self._build_steering_vector(dataset)
        if self.steer_vec.device != pre_acts.device:
            self.steer_vec = self.steer_vec.to(pre_acts.device)

        return pre_acts + self.steer_vec


    def random_reranker(
        self,
        scores: torch.Tensor,
        top_k: int = 50,
        sample_k: int = 10,
        boost_margin: float = 1.0,
        seed: int = None
    ):
        """
        Args:
            scores:      Tensor of shape [B, N]
            top_k:       How many of the highest‐scoring indices to consider (default 50)
            sample_k:    How many to randomly sample from those top_k (default 10)
            boost_margin:Base increment unit for boosting (default 1.0)
            seed:        Optional random seed for reproducibility
        Returns:
            boosted_scores: Tensor of shape [B, N] with the selected indices boosted
            selected_idx:   LongTensor of shape [B, sample_k] giving the boosted indices per row
        """
        if seed is not None:
            torch.manual_seed(seed)

        B, N = scores.shape

        # 1) Get top_k indices per row
        topk_vals, topk_idx = torch.topk(scores, top_k, dim=1)  # shapes: [B, top_k]

        # 2) Randomly sample sample_k of those top_k **without** replacement
        #    This gives positions in the topk array (0..top_k-1), shape [B, sample_k]
        rand_vals = torch.ones(B, top_k)
        samp_pos = torch.multinomial(rand_vals, sample_k, replacement=True)

        # 3) Map back to the original indices in [0..N)
        batch_idx = torch.arange(B).unsqueeze(1).expand(-1, sample_k)  # [B, sample_k]
        selected_idx = topk_idx[batch_idx, samp_pos]                  # [B, sample_k]

        # 4) Compute per‐row max scores so we know where to boost from
        row_max, _ = torch.max(scores, dim=1, keepdim=True)           # [B, 1]

        # 5) Build boost values so that
        #      - the first sampled index gets row_max + sample_k*boost_margin
        #      - the next gets row_max + (sample_k-1)*boost_margin
        #      - … down to row_max + 1*boost_margin
        boost_steps = torch.arange(sample_k, 0, -1, device=scores.device).float()  # [sample_k]
        boost_vals = row_max + boost_steps.unsqueeze(0) * boost_margin            # [B, sample_k]

        # 6) Clone and scatter the boosts into a copy of the original scores
        boosted_scores = scores.clone()
        boosted_scores[batch_idx, selected_idx] = boost_vals

        return boosted_scores


    def ipr_baseline(self, scores: torch.Tensor, dataset: str, alpha: float, long_list_size: int = 250) -> torch.Tensor:
        """
        Implements the IPR baseline to adjust scores for popularity bias mitigation.
        Loads popularity scores from the specified CSV file based on the dataset.
        Assumes the nth column in scores corresponds to item_id n (0-based indexing).
        Optionally applies the adjustment only to a long list of top candidates per batch.

        Args:
            scores: Tensor of shape (B, N) containing relevance scores.
            dataset: The dataset name to construct the CSV file path.
            alpha: Hyperparameter controlling the degree of bias mitigation.
            long_list_size: Optional; if provided, select the top long_list_size items per batch based on original scores,
                            apply IPR only to them, and set other scores to -inf to exclude from ranking.

        Returns:
            Adjusted scores tensor of shape (B, N).
        """
        # Load the CSV file
        file_path = f"./dataset/{dataset}/item_popularity_labels.csv"
        df = pd.read_csv(file_path)
        
        # Assume columns are 'item_id' and 'pop_score'; map item_id to pop_score
        pop_dict = dict(zip(df['item_id:token'], df['pop_score']))
        
        # Derive item_ids as 0 to N-1
        N = scores.shape[1]
        item_ids = list(range(N))
        
        # Get pop values for the derived item_ids
        pop_list = [pop_dict.get(item_id, 0.0) for item_id in item_ids]
        pop = torch.tensor(pop_list, dtype=torch.float, device=scores.device)
        
        if pop.max() == 0:
            raise ValueError("Popularity values must include at least one positive value.")
        
        rho = pop / pop.max()
        boost_factor = 1 + alpha * (1 - rho)
        boost_factor = boost_factor.unsqueeze(0).expand(scores.shape[0], -1)
        
        adjusted_scores = scores.clone()
        
        if long_list_size is not None:
            # Set all to -inf initially
            adjusted_scores.fill_(-float('inf'))
            # For each batch, select top long_list_size indices and apply boost to those
            for b in range(scores.shape[0]):
                # Get top indices based on original scores
                _, top_indices = torch.topk(scores[b], min(long_list_size, N), sorted=False)
                # Apply boost to those positions
                adjusted_scores[b, top_indices] = scores[b, top_indices] * boost_factor[b, top_indices]
        
        else:
            # Apply to all
            adjusted_scores = scores * boost_factor
        
        return adjusted_scores
    
    def _solve_personal_targets(self, p_u: np.ndarray, q_hat: np.ndarray, chunk: int = 5000) -> np.ndarray:
        """Linear‑programming solver for personalised targets (2 groups)."""
        B = p_u.shape[0]                 # users
        gradient = p_u.mean(0) - q_hat   # len‑2
        if np.allclose(gradient, 0):
            return p_u.copy()
        g = gradient / np.linalg.norm(gradient)  # len‑2, g0 + g1 = 0

        tile_g = np.tile(g, (B, 1))      # (B,2) – per‑user grad direction
        # per‑user upper limits ensuring q_hat_u stays in [0,1]
        lim = np.where(tile_g > 0, p_u / (tile_g + 1e-10), (p_u - 1) / (tile_g + 1e-10)).min(1)

        # equality constraint  sum_u gamma_u * g0 = sum_u (p_u0 - q_hat0)
        A_eq_full = tile_g[:, 0].reshape(1, B)          # (1,B)
        b_eq_full = np.array([(p_u[:, 0] - q_hat[0]).sum()])  # shape (1,)

        gamma = np.empty(B)
        solved = 0
        while solved < B:
            end = min(solved + chunk, B)
            A_eq = A_eq_full[:, solved:end]
            # account for already‑solved part
            # subtract contribution of already‑solved users (only when solved>0)
            b_eq = b_eq_full - (A_eq_full[:, :solved] @ gamma[:solved]).ravel() if solved else b_eq_full.copy()
            bounds = [(0, lim[i]) for i in range(solved, end)]
            res = linprog(c=np.ones(end - solved), A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
            gamma[solved:end] = res.x
            solved = end

        return p_u - gamma[:, None] * g   # (B,2)


    def pct_rerank(
        self,
        scores: Array,
        *,
        list_size: Optional[int] = 250,
        top_k: int = 10,
        policy: Literal["Equal", "AvgEqual"] = "Equal",
        p: float = 0.5,
        personal: bool = True,
        user_interest: Optional[Array] = None,
        lambda_: float = 0.7,
    ) -> Array:
        """Post‑process *scores* so the Top‑k per user is PCT‑calibrated.

        `user_interest` options when *personal* is **True**:
        • 1‑D `(B,)` float → already the niche fraction per user.
        • 2‑D `(B,C)` int  → item‑id history, zero‑padded.  Non‑zeros are
            looked‑up in `niche_labels` to derive the fraction internally.
        """
        if list_size is not None and list_size < top_k:
            raise ValueError("list_size must be None or >= top_k")

        df = pd.read_csv(rf"./dataset/{self.dataset}/item_popularity_labels.csv")
        ids  = df["item_id:token"].astype(int).values      # e.g. [1, 2, 3, …, 3417]
        labs = df["popularity_label"].astype(int).values   # e.g. [1, 0, 1, …, 0]

        # 2) Build a 1D BoolTensor of size (max_id+1,) so we can index by ID directly
        max_id = ids.max()
        niche_labels = np.zeros(max_id+1, dtype=bool)

        # 3) Fill it: True where label == 1 (popular)
        #    If your “popular” is actually encoded as -1, just change (labs == 1) to (labs == -1)
        niche_labels[ids] = (labs == -1)

        # ---- Normalise inputs ---------------------------------------------------
        scores_np = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
        niche_np  = niche_labels.detach().cpu().numpy().astype(bool) if isinstance(niche_labels, torch.Tensor) else np.asarray(niche_labels, bool)
        B, N = scores_np.shape
        if niche_np.shape != (N,):
            raise ValueError("niche_labels must have shape (N,)")

        # ---- Exposure weights & system target -----------------------------------
        pos_weight = 1.0 / np.log2(np.arange(top_k) + 2)
        exp_budget = pos_weight.sum()
        if policy == "Equal":
            target_ratio = np.array([1-p, p])
        elif policy == "AvgEqual":
            target_ratio = np.array([1 - niche_np.mean(), niche_np.mean()])
        else:
            raise ValueError("policy must be 'Equal' or 'AvgEqual'")

        quality_sign = niche_np.astype(int)
        # ---- Personalised targets ----------------------------------------------
        if personal:
            if user_interest is None:
                raise ValueError("personal=True requires 'user_interest'")
            ui = user_interest.detach().cpu().numpy() if isinstance(user_interest, torch.Tensor) else np.asarray(user_interest)
            if ui.ndim == 2:  # (B,C) id history
                if ui.shape[0] != B:
                    raise ValueError("user_interest first dim must match batch size B")
                frac = np.zeros(B)
                for u in range(B):
                    ids = ui[u][ui[u] != 0]
                    # print(ids, " sikim?")
                    if ids.size == 0:
                        # print("suka blya")
                        frac[u] = target_ratio[1]  # fallback to global ratio
                    else:
                        valid = ids[ids < N]  # ignore out‑of‑range
                        # print(valid, "sikim 2")
                        frac[u] = niche_np[valid].mean() if valid.size else target_ratio[1]
                        # print(frac, "sikim 3")
                # print(frac, " sikim 4")
            elif ui.ndim == 1:
                if ui.shape != (B,):
                    raise ValueError("user_interest must be shape (B,) or (B,C)")
                frac = ui.astype(float)
            else:
                raise ValueError("user_interest must be 1‑D or 2‑D tensor/array")
            
            p_u = np.column_stack([1.0 - frac, frac])
            print(p_u.size, " sikim 5")

            q_hat_u = self._solve_personal_targets(p_u, target_ratio, chunk=B) * exp_budget
        else:
            q_hat_u = np.tile(target_ratio * exp_budget, (B, 1))

            # ---- Reranking core ------------------------------------------------------
        reranked = scores_np.copy()

        # sort once up-front
        order_idx_full = (-scores_np).argsort(1)    # (B, N) indices

        # if list_size is given, slice the candidate pool
        if list_size is not None:
            order_idx = order_idx_full[:, :list_size]      # (B, list_size)
        else:
            order_idx = order_idx_full                    # (B, N)

        for u in range(B):
            chosen   = np.full(top_k, -1, dtype=int)
            cur_exp  = np.zeros(2)
            sel      = set()
            target_exp = q_hat_u[u]

            # ------------ Pass-1  (keep highest items if safe) -------------
            for pos in range(top_k):
                for j in order_idx[u]:
                    if j in sel:
                        continue
                    g = quality_sign[j]
                    if cur_exp[g] + pos_weight[pos] <= target_exp[g]:
                        sel.add(j); chosen[pos] = j; cur_exp[g] += pos_weight[pos]
                        break

            # ------------ Pass-2  (MMR fill the gaps) ----------------------
            for pos in range(top_k):
                if chosen[pos] != -1:
                    continue

                best_s = -np.inf
                best_j = None
                for rnk, j in enumerate(order_idx[u]):
                    if j in sel:
                        continue
                    g = quality_sign[j]
                    assume = cur_exp.copy(); assume[g] += pos_weight[pos]
                    disp   = 0.5 * ((assume - target_exp) ** 2).sum()
                    mmr    = lambda_ * (1 / (rnk + 1)) - (1 - lambda_) * disp
                    if mmr > best_s:
                        best_s, best_j = mmr, j

                if best_j is None:            # <-- no candidates left
                    break                     #    leave the remaining slots -1
                sel.add(best_j)
                chosen[pos] = best_j
                cur_exp[quality_sign[best_j]] += pos_weight[pos]
            # -----------  bump scores so the chosen items surface ----------
            bump = scores_np[u].max() + 1
            for r, j in enumerate(chosen[::-1]):
                if j == -1:          # <-- nothing was chosen for this rank
                    continue
                reranked[u, j] = bump + r

        # return the same type the caller provided
        return (
            torch.as_tensor(reranked, dtype=scores.dtype, device=scores.device)
            if isinstance(scores, torch.Tensor) else reranked
        )
    


    def min_reg_algo(self, scores, dataset, M=250, lambd=0.0001, eta=0.001):
        """
        Function to perform min-regularizer re-ranking for fairness.
        Inputs:
        - scores: torch.Tensor of shape (B, N), user-item scores
        - dataset: str, dataset name for loading CSV
        - M: int, list size (top-K), default=250
        - lambd: float, fairness trade-off hyperparameter, default=0.1
        - eta: float, another hyperparameter (learning rate, though not used in this adaptation), default=0.001
        
        Outputs:
        - new_scores: torch.Tensor of shape (B, N), with selected items boosted
        """        
        B, N = scores.shape
        T = B  # Set horizon T to batch size B
        
        # Load provider data if not already loaded
        if self._item2provider is None:
            csv_path = f"./dataset/{dataset}/item_popularity_labels.csv"
            df = pd.read_csv(csv_path)
            # Map popularity_label (-1,0,1) to provider ids (0,1,2)
            self._item2provider = {row['item_id:token']: row['popularity_label'] + 1 for _, row in df.iterrows()}
            num_providers = 3  # Fixed to 3 as per user
            
            # Compute providerLen
            providerLen = np.zeros(num_providers)
            for label in self._item2provider.values():
                providerLen[int(label)] += 1
            
            # Compute rho
            self._rho = (1 + 1 / num_providers) * providerLen / np.sum(providerLen)
            
            # Build A (item-provider matrix)
            self._A = np.zeros((N, num_providers))
            self._iid2pid = [-1] * N  # Default -1 if not found
            for i in range(N):
                if i in self._item2provider:
                    pid = self._item2provider[i]
                    self._iid2pid[i] = pid
                    self._A[i, int(pid)] = 1
        
        # Convert scores to numpy
        batch_UI = scores.cpu().numpy()
        
        # Initialize remaining resources B_t
        B_t = T * M * self._rho
        
        result_x = []  # List to store selected item ids per user
        
        for t in range(T):
            # Compute penalty term
            min_B = np.min(B_t)
            gap_term = (-B_t + min_B) / (T * self._rho)
            penalty = np.matmul(self._A, gap_term)
            
            # Compute effective scores
            x_title = batch_UI[t, :] - lambd * penalty
            
            # Mask for depleted providers
            mask = np.matmul(self._A, (B_t > 0).astype(np.float64))
            mask = (1.0 - mask) * -10000.0
            
            # Sort to get top-M candidates
            x = np.argsort(x_title + mask, axis=-1)[::-1]
            x_allocation = x[:M]
            
            # Re-sort selected based on original scores (descending)
            re_allocation = np.argsort(batch_UI[t, x_allocation])[::-1]
            x_allocation = x_allocation[re_allocation]
            
            result_x.append(x_allocation)
            
            # Update B_t
            exposures = np.sum(self._A[x_allocation, :], axis=0)
            B_t = B_t - exposures
        
        # Create new_scores by boosting selected items
        new_scores = scores.clone()
        for b in range(B):
            selected = result_x[b]
            orig_sel = batch_UI[b, selected]  # Already in descending order
            
            # Find a boost value larger than current max
            orig_max = batch_UI[b].max()
            boost_base = orig_max + 10.0  # Arbitrary large boost; adjust if scores are very large
            eps = 1e-6
            
            for idx in range(M):
                item_id = selected[idx]
                new_scores[b, item_id] = float(boost_base - idx * eps)
        
        return new_scores