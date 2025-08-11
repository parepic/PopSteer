# -*- coding: utf-8 -*-
# @Time   : 2020/8/31
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

# UPDATE:
# @Time   : 2020/9/16, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com

r"""
LightGCN
################################################

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN
"""

import numpy as np
import scipy.sparse as sp
import torch
import os
import csv
import torch
from scipy.optimize import linprog
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
from recbole.utils import create_pop_unpop_mappings, make_items_popular, make_items_unpopular,save_batch_activations,get_extreme_correlations
from typing import Literal, Union, Optional
Array = Union[np.ndarray, torch.Tensor]

import math
import pandas as pd
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class LightGCN(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)
        self.a1 = config["alpha"][0]
        self.a2 = config["alpha"][1]
        self.fair = False
        self.random = False
        self.ipr = False
        self.pct = False
        self.min_reg = False
        self.duor = False
        self._item2provider = None
        self._A = None
        self._rho = None
        self._iid2pid = None


        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.dataset = config["dataset"]
        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size of lightGCN
        self.n_layers = config["n_layers"]  # int type:the layer num of lightGCN
        self.reg_weight = config[
            "reg_weight"
        ]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        for (row, col), value in data_dict.items():
            A[row, col] = value
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]
        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        scores[:, 0] =  float("-inf")
        if self.fair:
            scores = self.FAIR(scores, p=self.a1,alpha=self.a2).to(self.device)
        elif self.random:
            scores = self.random_reranker(scores=scores, top_k=self.a1)
        elif self.ipr:
            scores = self.ipr_baseline(scores=scores, dataset = self.dataset, alpha=self.a1)
        # if self.pct:
        #     scores = self.pct_rerank(scores=scores, user_interest=item_seq, p=self.a1, lambda_= self.a2)
        if self.min_reg:
            scores = self.min_reg_algo(dataset=self.dataset, scores=scores, lambd=self.a1)
        if self.duor:
            scores = self.duor_boost_scores_from_user_csv(scores=scores, user_id=self.USER_ID, candidate_size=self.a1)

        return scores.view(-1)

    def FAIR(self, scores, *, p: float = 0.9, alpha: float = 0.1,
            L: int = 10000, K: int = 10):
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
    
    def _load_user_popularity_scores(self, path: str) -> dict[str, float]:
        """
        Load one popularity score per user from user_popularity_labels.csv.
        Cache after first read.
        """
        if getattr(self, "_user_pop_score_cache", None) is not None:
            return self._user_pop_score_cache

        if not os.path.exists(path):
            raise FileNotFoundError(f"User popularity file not found: {path}")

        scores: dict[str, float] = {}
        with open(path, newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                uid = row.get("user_id:token")
                s = row.get("user_popularity_score")
                if not uid or s is None:
                    continue
                if uid in scores:
                    continue
                try:
                    val = float(s)
                except ValueError:
                    continue
                # clamp to [0,1]
                if val < 0.0: val = 0.0
                if val > 1.0: val = 1.0
                scores[uid] = val

        self._user_pop_score_cache = scores
        return scores


    def _load_popularity_labels(self, num_items: int,
                                csv_path: str,
                                device: torch.device) -> torch.Tensor:
        """
        Returns a length-N tensor with values {+1 (Head), -1 (Tail)}.
        Missing ids default to Tail (-1).
        """
        df = pd.read_csv(csv_path)
        # Columns are "item_id:token" and "popularity_label" (1 for Head, -1 for Tail)
        ids = df["item_id:token"].astype(int).to_numpy()
        labs = df["cum_popularity_label"].astype(int).to_numpy()
        labels = torch.full((num_items,), -1, dtype=torch.int8, device=device)
        valid = (ids >= 0) & (ids < num_items)
        if valid.any():
            labels[torch.as_tensor(ids[valid], device=device)] = torch.as_tensor(labs[valid], device=device, dtype=torch.int8)
        return labels  # shape: (N,), values in {1, -1}




    @torch.no_grad()
    def duor_boost_scores_from_user_csv(
        self,
        scores: torch.Tensor,             # (B, N)
        user_id: Union[int, str],         # same user for the whole batch
        candidate_size: int = 250,
        topk: int = 10,
    ) -> torch.Tensor:
        """
        Dynamic user-oriented re-ranking with score boosting for one user.
        All rows in `scores` belong to the same user.
        Returns a tensor of shape (B, N) with boosted scores for the selected top-k.
        """
        assert scores.dim() == 2, "scores must be (B, N)"
        B, N = scores.shape
        device = scores.device
        dtype = scores.dtype

        # Item popularity labels: +1 Head, -1 Tail, 0 Mid
        item_pop_csv = rf"./dataset/{self.dataset}/item_popularity_labels.csv"
        pop_labels = self._load_popularity_labels(N, item_pop_csv, device=device)  # (N,)

        # User popularity score in [0,1]
        user_pop_csv = rf"./dataset/{self.dataset}/user_popularity_labels.csv"
        user_score_map = self._load_user_popularity_scores(user_pop_csv)
        uid = str(int(user_id)) if isinstance(user_id, int) or (isinstance(user_id, str) and user_id.isdigit()) else str(user_id)
        pop_u = user_score_map.get(uid, 0.5)  # neutral fallback

        out = scores.clone()
        BIG = torch.tensor(1e6, device=device, dtype=dtype).item()

        M = min(candidate_size, N)

        for b in range(B):
            # Candidate pool: top-M by base scores
            cand_vals, cand_idx = torch.topk(scores[b], k=M, largest=True, sorted=True)  # (M,)
            cand_labels = pop_labels[cand_idx]  # (M,)

            available = torch.ones(M, device=device, dtype=torch.bool)
            chosen_item_ids: list[int] = []

            for k in range(topk):
                if not available.any():
                    break

                if k == 0:
                    masked = cand_vals.clone()
                    masked[~available] = -torch.inf
                    pick_j = torch.argmax(masked).item()
                else:
                    if len(chosen_item_ids) == 0:
                        pop_rec = 0.0
                    else:
                        chosen_is_head = (pop_labels[torch.as_tensor(chosen_item_ids, device=device)] == 1).sum().item()
                        pop_rec = chosen_is_head / float(len(chosen_item_ids))

                    if pop_rec < pop_u:
                        allow_mask = (cand_labels == 1) & available   # prefer Head
                    elif pop_rec > pop_u:
                        allow_mask = (cand_labels == -1) & available  # prefer Tail
                    else:
                        allow_mask = available

                    if not allow_mask.any():
                        allow_mask = available

                    masked = cand_vals.clone()
                    masked[~allow_mask] = -torch.inf
                    pick_j = torch.argmax(masked).item()

                chosen_item_ids.append(int(cand_idx[pick_j].item()))
                available[pick_j] = False

            if len(chosen_item_ids) == 0:
                continue

            chosen_item_ids_tensor = torch.as_tensor(chosen_item_ids, device=device)

            # Boost chosen items to dominate the top-k while keeping their mutual order
            base = out[b].max().item() + BIG
            out[b, chosen_item_ids_tensor] = base + scores[b, chosen_item_ids_tensor]

        return out
