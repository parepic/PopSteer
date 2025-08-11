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
        return scores.view(-1)



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