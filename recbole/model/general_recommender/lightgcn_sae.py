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

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType, compute_neuron_stats_by_row, compute_weighted_neuron_stats_by_row_item
from recbole.model.general_recommender.lightgcn import LightGCN




class LightGCN_SAE(LightGCN):
	r"""LightGCN is a GCN-based recommender model.

	LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
	collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
	propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
	learned at all layers as the final embedding.

	We implement the model following the original author with a pairwise training mode.
	"""

	input_type = InputType.PAIRWISE

	def __init__(self, config, dataset):
		super().__init__(config, dataset)
		model_path = config["base_path"]
		checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
		self.load_state_dict(checkpoint['state_dict'])
		self.sae_module_i = SAE(config, side="item")
		self.sae_module_u = SAE(config, side="user")
		self.restore_item_e = None
		self.restore_user_e = None
		self.val_fvu_i = torch.tensor(0.0, device=self.device)
		self.val_fvu_u = torch.tensor(0.0, device=self.device)
		self.dataset = config["dataset"]
		self.base_i = None
		self.base_u = None
		self.mode = config["sae_mode"]

		for param in self.parameters():
			param.requires_grad = False

		for param in self.sae_module_i.parameters():
			param.requires_grad = True  
		for param in self.sae_module_u.parameters():
			param.requires_grad = True  

	def forward(self, train_mode=None):
		u_emb, i_emb = self.base_u, self.base_i
		if self.base_i is None or self.base_u is None:
			self.base_u, self.base_i = super().forward()
			u_emb, i_emb = self.base_u, self.base_i
		if self.mode == "test":
			if self.sae_module_i.steer:
				i_emb = self.sae_module_i(self.base_i, train_mode=train_mode)
			if self.sae_module_u.steer:
				u_emb = self.sae_module_u(self.base_u, train_mode=train_mode)
		else:
			i_emb = self.sae_module_i(self.base_i, train_mode=train_mode)
			u_emb = self.sae_module_u(self.base_u, train_mode=train_mode)
		return u_emb, i_emb
	
	def calculate_loss(self, interaction):
		if self.restore_user_e is not None or self.restore_item_e is not None:
			self.restore_user_e, self.restore_item_e = None, None
		
		user_all_embeddings, item_all_embeddings = self.forward(train_mode=True)
		sae_loss_i = self.sae_module_i.fvu + self.sae_module_i.auxk_loss / 2
		sae_loss_u = self.sae_module_u.fvu + self.sae_module_u.auxk_loss / 2
		
		return sae_loss_i + sae_loss_u

	def full_sort_predict(self, interaction):
		user = interaction[self.USER_ID]
		# df = pd.read_csv(rf"./dataset/{self.dataset}/user_popularity_labels.csv")
		# print(user, " ffff")
		# row = df[df['user_id:token'] == user]
		# self.sae_module_i.dampen = (row.iloc[0]['popularity_label'] != 1)
		if self.restore_user_e is None or self.restore_item_e is None:
			self.restore_user_e, self.restore_item_e = self.forward(train_mode=False)
		u_embeddings = self.restore_user_e[user]

		scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
		scores[:, 0] =  float("-inf")
		self.val_fvu_i += (self.sae_module_i.fvu)
		self.val_fvu_u += (self.sae_module_u.fvu)
		return scores.view(-1)
	
	def synthetic_inference(self, interaction, popular=None):
		user = interaction[self.USER_ID]
		if self.restore_user_e is None or self.restore_item_e is None:
			self.restore_user_e, self.restore_item_e = self.forward(train_mode=False)
		u_embeddings = self.restore_user_e[user]
		scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
		scores[:, 0] =  float("-inf")
		self.val_fvu += (self.sae_module_i.fvu + self.sae_module_u.fvu)
		return scores.view(-1)


	def set_sae_mode(self, train_mode=True):
		self.sae_module_i.train_mode=train_mode
		self.sae_module_u.train_mode=train_mode


import torch
import numpy as np
import json
import torch
import torch.nn as nn
from recbole.utils import utils
import pandas as pd
import random


class SAE(nn.Module):
    
    def __init__(self,config, side="item"):
        super(SAE, self).__init__()
        self.side=side
        self.dataset = config["dataset"]
        self.index = 0 if side == "item" else 1
        self.k = config["sae_k"][self.index]
        self.scale_size = config["sae_scale_size"][self.index]
        self.alpha = config['alpha'][self.index]
        self.beta = None
        self.steer = config['steer'][self.index]
        self.steer_dir = config['steer_dir'][self.index]
        self.analyze = config['analyze']
        self.fvu = torch.tensor(0.0)
        self.dampen=False
        self.neuron_count = None
        self.unpopular_only = None
        self.corr_file = None
        self.device = config["device"]
        self.dtype = torch.float32
        self.to(self.device)
        self.d_in = config['input_dim']
        self.hidden_dim = self.d_in * self.scale_size
        self.N = self.hidden_dim
        self.steered_activations=None
        self.d_min = None
        self.activated_features = torch.zeros(config["train_batch_size"], self.hidden_dim)
        self.analyze_neurons = False
        self.activation_count = torch.zeros(self.hidden_dim, device=config["device"])
        self.encoder = nn.Linear(self.d_in, self.hidden_dim, device=self.device,dtype = self.dtype)
        self.encoder.bias.data.zero_()
        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        self.set_decoder_norm_to_unit_norm()
        self.b_dec = nn.Parameter(torch.zeros(self.d_in, dtype = self.dtype, device=self.device))
        self.activate_latents = set()
        self.previous_activate_latents = None
        self.epoch_idx=0
        self.new_epoch = False
        self.item_activations = np.zeros(self.hidden_dim)
        self.highest_activations = {
            j: {
                "values": torch.empty(0, dtype=torch.float32, device=self.device),
                "low_values": torch.empty(0, dtype=torch.float32, device=self.device),
                "items": torch.empty(0, dtype=torch.long, device=self.device),
                "low_items": torch.empty(0, dtype=torch.long, device=self.device),
                "recommendations": torch.empty((0, 10), dtype=torch.long, device=self.device)
            }
            for j in range(self.hidden_dim)
        }
        self.steer_vec = None        # cached steering vector
        self._steer_ready = False    # flag

        return  
  
    def get_dead_latent_ratio(self, need_update=0):
        # Calculate the dead latent ratio
        ans = 1 - len(self.activate_latents) / self.hidden_dim
        # Calculate the current number of dead latents
        current_dead = self.hidden_dim - len(self.activate_latents)
        print(f" Side: {self.side}, Dead percentage:  {ans}")
        print(f" Side: {self.side}, FVU: {self.fvu}, AUXK Loss: {self.auxk_loss}, AUXK Loss / 2: {self.auxk_loss / 2} SAE Total Loss: {(self.auxk_loss / 2) + self.fvu}")
        if need_update:
            # Convert current active latents to a tensor
            current_active = torch.tensor(list(self.activate_latents), device=self.device)
            
            # Compute revived latents if there’s a previous state
            if self.previous_activate_latents is not None:
                # Find latents in current_active that were not in previous_activate_latents
                revived_mask = ~torch.isin(current_active, self.previous_activate_latents)
                num_revived = revived_mask.sum().item()
                # Print the requested information
                print(f"Number of revived latents: {num_revived}, Current dead latents: {current_dead}")
            
            # Update previous_activate_latents to the current active latents
            self.previous_activate_latents = current_active
        
            # Reset activate_latents for the next period
            self.activate_latents = set()
        return ans


    def set_decoder_norm_to_unit_norm(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."
        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps


    def topk_activation(self, x, sequences, save_result, k=0):
        """
        Performs top-k activation on tensor x.
        If k is not None, reads the first k indices from the previously saved indices file
        and sets their activations in x to -10 before computing top-k.
        Returns a sparse tensor with only the top-k activations.
        """

        topk_values, topk_indices = torch.topk(x, self.k, dim=1)
        flat_indices = topk_indices.view(-1)

        # Count occurrences of each index
        counts = torch.bincount(flat_indices, minlength=self.hidden_dim)

        # Update activation count
        self.activation_count += counts.to(self.activation_count.device)
        self.activate_latents.update(topk_indices.cpu().numpy().flatten())

        # Save epoch activations if needed
        if save_result:
            values_np = topk_values.detach().cpu().numpy()
            inds_np = topk_indices.detach().cpu().numpy()
        # Build sparse output
        sparse_x = torch.zeros_like(x)
        sparse_x.scatter_(1, topk_indices, topk_values.to(self.dtype))
        return sparse_x

        

    def update_topk_recommendations(self, predictions, current_sequences, k=10):
        """
        Update top-k recommendations for sequences in highest_activations.

        Parameters:
        - predictions: Tensor of shape [B, N], where B is batch size and N is the number of items.
        - current_sequences: List of sequences (IDs) in the current batch.
        - k: Number of top recommendations to save.
        """
        # Convert current_sequences to a list of lists for easy comparison
        current_sequences_list = [seq.tolist() for seq in current_sequences]
  
        for neuron_idx, data in self.highest_activations.items():
            for idx, stored_sequence in enumerate(data["sequences"]):
                # Check if the stored sequence is in the current batch
                if stored_sequence in current_sequences_list:
                    # Find the index of the stored sequence in the current batch
                    batch_idx = current_sequences_list.index(stored_sequence)
                    
                    # Get predictions for this sequence
                    pred_scores = predictions[batch_idx].cpu().numpy()  # Convert to numpy for sorting
                    
                    # Find indices of the top-k scores
                    topk_indices = np.argsort(pred_scores, axis=1)[:, -k:][:, ::-1]  # Add 1 to match item IDs
     
                    # Update the recommendations for this sequence
                    data["recommendations"].append(topk_indices.tolist())
    

    def _build_steering_vector(self, dataset):
        """Build the steering vector according to the specified direction.

        If ``self.d_min`` is **not** ``None`` we do **not** rely on
        ``self.N``.  Instead, we include *all* neurons whose absolute
        Cohen's *d* is **greater than or equal to** ``self.d_min``.  When
        ``self.d_min`` *is* ``None``, we fall back to using the first
        ``self.N`` neurons ranked by |*d*|.
        """

        # ── 1. Load neurons ranked by effect‑size  ───────────────────────
        pop_neurons, unpop_neurons = utils.get_extreme_correlations(
            rf"{self.side}/cohens_d.csv", dataset=dataset
        )

        if self.steer_dir == -1:
            combined = [(i, d, "unpop") for i, d in unpop_neurons]
        elif self.steer_dir == 1:
            combined = [(i, d, "pop") for i, d in pop_neurons]
        else:  # self.steer_dir == 0  → both groups
            combined = (
                [(i, d, "pop") for i, d in pop_neurons]
                + [(i, d, "unpop") for i, d in unpop_neurons]
            )

        # Sort by |d|, descending
        combined_sorted = sorted(combined, key=lambda x: abs(x[1]), reverse=True)

        # ── 2. Select neurons either by threshold (d_min) or by a fixed N ──
        if getattr(self, "d_min", None) is not None:
            # Keep all neurons with |d| ≥ d_min
            top_neurons = [triplet for triplet in combined_sorted if abs(triplet[1]) >= self.d_min]
            # Update N for downstream code that might rely on its size
            self.N = len(top_neurons)
        else:
            top_neurons = combined_sorted[: self.N]

        if((self.N) == 0):
            self.steer_vec = torch.zeros(self.hidden_dim, device=self.device, dtype=self.dtype)
            self._steer_ready = True
            return
        # ── 3. Load per‑neuron statistics  ───────────────────────────────
        stats_unpop = pd.read_csv(rf"./dataset/{dataset}/{self.side}/neuron_stats_unpop.csv")
        stats_pop = pd.read_csv(rf"./dataset/{dataset}/{self.side}/neuron_stats_pop.csv")
        stats = pd.read_csv(rf"./dataset/{dataset}/{self.side}/neuron_stats.csv")

        abs_cohens = torch.tensor(
            [abs(c) for _, c, _ in top_neurons], device=self.device, dtype=self.dtype
        )

        # Placeholder for normalised weights
        weights = torch.empty_like(abs_cohens)

        def normalise(x, pop: bool | None = None):
            """Linearly map *x* → [0, α] or [0, β] (pop vs. unpop)."""
            thres = self.beta if pop else self.alpha
            xmax = torch.max(x)
            return torch.full_like(x, thres / 2) if xmax == 0 else (x / xmax) * thres

        # ── 4. Compute group‑specific weights  ───────────────────────────
        if self.steer_dir == 0:  # both groups
            pop_mask = torch.tensor([g == "pop" for *_, g in top_neurons], device=self.device)
            unpop_mask = ~pop_mask

            if pop_mask.any():
                weights[pop_mask] = normalise(abs_cohens[pop_mask], pop=True)
            if unpop_mask.any():
                weights[unpop_mask] = normalise(abs_cohens[unpop_mask])
        else:
            # steer_dir is ±1 → single group
            weights = normalise(abs_cohens)

        # ── 5. Assemble the steering vector  ─────────────────────────────
        steer = torch.zeros(self.hidden_dim, device=self.device, dtype=self.dtype)

        for i, (neuron_idx, _, group) in enumerate(top_neurons):
            tot_sd = stats.iloc[neuron_idx]["sd"]
            w = weights[i]
            if group == "unpop":
                unpop_sd = stats_unpop.iloc[neuron_idx]["sd"]
                steer[neuron_idx] += w * tot_sd
            else:  # group == "pop"
                pop_sd = stats_pop.iloc[neuron_idx]["sd"]
                steer[neuron_idx] -= w * tot_sd

        # Save and mark ready
        self.steer_vec = steer.to(self.device)
        self._steer_ready = True



    def dampen_neurons(self, pre_acts, dataset=None):
        if getattr(self, "N", None) in (None, 0):
            return pre_acts
        if not self._steer_ready:
            self._build_steering_vector(dataset=self.dataset)
        if self.steer_vec.device != pre_acts.device:
            self.steer_vec = self.steer_vec.to(pre_acts.device)
        return pre_acts + self.steer_vec
    

    def forward(self, x, sequences=None, train_mode=False, save_result=False, epoch=None, dataset=None, pop_scores=None):
            sae_in = x - self.b_dec
            pre_acts1 = self.encoder(sae_in)
            self.last_activations = pre_acts1
            # if self.analyze == True:
            #     if self.side == "item":
            #         compute_weighted_neuron_stats_by_row_item(activations=pre_acts1, dataset=self.dataset, side=self.side)
            if self.steer == True and self.N != 0:
                pre_acts1 = self.dampen_neurons(pre_acts1, dataset=self.dataset)
            self.steered_activations = pre_acts1
            # pre_acts = self.add_noise(pre_acts, std=self.beta)
            pre_acts = nn.functional.relu(pre_acts1)   
            z = self.topk_activation(pre_acts, sequences, save_result=False)
            x_reconstructed = z @ self.W_dec + self.b_dec
            e = x_reconstructed - x
            total_variance = (x - x.mean(0)).pow(2).sum()
            self.fvu = e.pow(2).sum() / total_variance
            if train_mode:
                if self.new_epoch == True:
                    self.new_epoch = False
                    dead = self.get_dead_latent_ratio(need_update=1)
                    print("Dead percentage ", dead)					
                # First epoch, do not have dead latent info
                if self.previous_activate_latents is None:
                    self.auxk_loss = 0.0
                    return x_reconstructed
                num_dead = self.hidden_dim - len(self.previous_activate_latents)
                k_aux = int(x.shape[-1]) * 2
                if num_dead == 0:
                    self.auxk_loss = 0.0
                    return x_reconstructed
                scale = min(num_dead / k_aux, 1.0)
                k_aux = min(k_aux, num_dead)
                dead_mask = torch.isin(
                    torch.arange(pre_acts.shape[-1]).to(self.device),
                    self.previous_activate_latents,
                    invert=True
                )
                auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
                auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
                # print("these are aux values, ", auxk_indices[0])
                # print("these are aux indices, ", auxk_acts[0])

                e_hat = torch.zeros_like(auxk_latents)
                e_hat.scatter_(1, auxk_indices, auxk_acts.to(self.dtype))
                e_hat = e_hat @ self.W_dec + self.b_dec

                auxk_loss = (e_hat - e).pow(2).sum()
                self.auxk_loss = scale * auxk_loss / total_variance

            return x_reconstructed



# class SAE(nn.Module):
	
# 	def __init__(self,config, side="item"):
# 		super(SAE, self).__init__()
# 		self.side=side
# 		self.dataset = config["dataset"]
# 		self.index = 0 if side == "item" else 1
# 		self.k = config["sae_k"][self.index]
# 		self.scale_size = config["sae_scale_size"][self.index]
# 		self.alpha = config['alpha'][self.index]
# 		self.steer = config['steer'][self.index]
# 		self.analyze = config['analyze']
# 		self.fvu = torch.tensor(0.0)
# 		self.dampen=False
# 		self.neuron_count = None
# 		self.unpopular_only = None
# 		self.corr_file = None
# 		self.device = config["device"]
# 		self.dtype = torch.float32
# 		self.to(self.device)
# 		self.d_in = config['input_dim']
# 		self.hidden_dim = self.d_in * self.scale_size
# 		self.N = self.hidden_dim
# 		self.activation_count = torch.zeros(self.hidden_dim, device=config["device"])
# 		self.encoder = nn.Linear(self.d_in, self.hidden_dim, device=self.device,dtype = self.dtype)
# 		self.encoder.bias.data.zero_()
# 		self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
# 		self.set_decoder_norm_to_unit_norm()
# 		self.b_dec = nn.Parameter(torch.zeros(self.d_in, dtype = self.dtype, device=self.device))
# 		self.activate_latents = set()
# 		self.previous_activate_latents = None
# 		self.epoch_idx=0
# 		self.new_epoch = False
# 		self.item_activations = np.zeros(self.hidden_dim)
# 		self.highest_activations = {
# 			j: {
# 				"values": torch.empty(0, dtype=torch.float32, device=self.device),
# 				"low_values": torch.empty(0, dtype=torch.float32, device=self.device),
# 				"items": torch.empty(0, dtype=torch.long, device=self.device),
# 				"low_items": torch.empty(0, dtype=torch.long, device=self.device),
# 				"recommendations": torch.empty((0, 10), dtype=torch.long, device=self.device)
# 			}
# 			for j in range(self.hidden_dim)
# 		}
# 		return  
  
# 	def get_dead_latent_ratio(self, need_update=0):
# 		# Calculate the dead latent ratio
# 		ans = 1 - len(self.activate_latents) / self.hidden_dim
# 		# Calculate the current number of dead latents
# 		current_dead = self.hidden_dim - len(self.activate_latents)
# 		print(f" Side: {self.side}, Dead percentage:  {ans}")
# 		print(f" Side: {self.side}, FVU: {self.fvu}, AUXK Loss: {self.auxk_loss}, AUXK Loss / 2: {self.auxk_loss / 2} SAE Total Loss: {(self.auxk_loss / 2) + self.fvu}")
# 		if need_update:
# 			# Convert current active latents to a tensor
# 			current_active = torch.tensor(list(self.activate_latents), device=self.device)
			
# 			# Compute revived latents if there’s a previous state
# 			if self.previous_activate_latents is not None:
# 				# Find latents in current_active that were not in previous_activate_latents
# 				revived_mask = ~torch.isin(current_active, self.previous_activate_latents)
# 				num_revived = revived_mask.sum().item()
# 				# Print the requested information
# 				print(f"Number of revived latents: {num_revived}, Current dead latents: {current_dead}")
			
# 			# Update previous_activate_latents to the current active latents
# 			self.previous_activate_latents = current_active
		
# 			# Reset activate_latents for the next period
# 			self.activate_latents = set()
# 		return ans


# 	def set_decoder_norm_to_unit_norm(self):
# 		assert self.W_dec is not None, "Decoder weight was not initialized."
# 		eps = torch.finfo(self.W_dec.dtype).eps
# 		norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
# 		self.W_dec.data /= norm + eps


# 	def topk_activation(self, x, sequences, save_result, k=0):
# 		"""
# 		Performs top-k activation on tensor x.
# 		If k is not None, reads the first k indices from the previously saved indices file
# 		and sets their activations in x to -10 before computing top-k.
# 		Returns a sparse tensor with only the top-k activations.
# 		"""

# 		topk_values, topk_indices = torch.topk(x, self.k, dim=1)
# 		flat_indices = topk_indices.view(-1)

# 		# Count occurrences of each index
# 		counts = torch.bincount(flat_indices, minlength=self.hidden_dim)

# 		# Update activation count
# 		self.activation_count += counts.to(self.activation_count.device)
# 		self.activate_latents.update(topk_indices.cpu().numpy().flatten())

# 		# Save epoch activations if needed
# 		if save_result:
# 			values_np = topk_values.detach().cpu().numpy()
# 			inds_np = topk_indices.detach().cpu().numpy()
# 		# Build sparse output
# 		sparse_x = torch.zeros_like(x)
# 		sparse_x.scatter_(1, topk_indices, topk_values.to(self.dtype))
# 		return sparse_x

		

# 	def update_topk_recommendations(self, predictions, current_sequences, k=10):
# 		"""
# 		Update top-k recommendations for sequences in highest_activations.

# 		Parameters:
# 		- predictions: Tensor of shape [B, N], where B is batch size and N is the number of items.
# 		- current_sequences: List of sequences (IDs) in the current batch.
# 		- k: Number of top recommendations to save.
# 		"""
# 		# Convert current_sequences to a list of lists for easy comparison
# 		current_sequences_list = [seq.tolist() for seq in current_sequences]
  
# 		for neuron_idx, data in self.highest_activations.items():
# 			for idx, stored_sequence in enumerate(data["sequences"]):
# 				# Check if the stored sequence is in the current batch
# 				if stored_sequence in current_sequences_list:
# 					# Find the index of the stored sequence in the current batch
# 					batch_idx = current_sequences_list.index(stored_sequence)
					
# 					# Get predictions for this sequence
# 					pred_scores = predictions[batch_idx].cpu().numpy()  # Convert to numpy for sorting
					
# 					# Find indices of the top-k scores
# 					topk_indices = np.argsort(pred_scores, axis=1)[:, -k:][:, ::-1]  # Add 1 to match item IDs
	 
# 					# Update the recommendations for this sequence
# 					data["recommendations"].append(topk_indices.tolist())
	 
# 	def dampen_neurons(self, pre_acts, dataset=None):
# 		# Early exit
# 		self.N = 512
# 		if getattr(self, "N", None) in (None, 0):
# 			return pre_acts

# 		pop_neurons, unpop_neurons = utils.get_extreme_correlations(
# 			rf"{self.side}/cohens_d.csv", dataset=dataset
# 		)

# 		combined_neurons = (
# 			[(idx, d, "unpop") for idx, d in unpop_neurons] +
# 			[(idx, d, "pop")   for idx, d in pop_neurons]
# 		)
# 		combined_sorted = sorted(combined_neurons, key=lambda x: abs(x[1]), reverse=True)
# 		top_neurons = combined_sorted[: self.N]

# 		stats_unpop = pd.read_csv(rf"./dataset/{dataset}/{self.side}/neuron_stats_unpopular.csv")
# 		stats_pop   = pd.read_csv(rf"./dataset/{dataset}/{self.side}/neuron_stats_popular.csv")

# 		abs_cohens = torch.tensor([abs(c) for _, c, _ in top_neurons],
# 								device=pre_acts.device, dtype=pre_acts.dtype)

# 		def normalize_to_range(x, new_min, new_max):
# 			max_val = torch.max(x)
# 			if max_val == 0:
# 				return torch.full_like(x, (new_min + new_max) / 2)
# 			return (x / max_val) * (new_max - new_min) + new_min

# 		weights = normalize_to_range(abs_cohens, new_min=0, new_max=self.alpha)
# 		user_labels = pd.read_csv(rf"./dataset/{self.dataset}/user_popularity_labels.csv")
# 		label_dict = dict(zip(user_labels['user_id:token'], user_labels['popularity_label']))
# 		user_mask = torch.tensor([label_dict.get(i, 0) == -1 for i in range(pre_acts.shape[0])],
#                              device=pre_acts.device, dtype=torch.bool)
# 		for i, (neuron_idx, cohen, group) in enumerate(top_neurons):
# 			w = weights[i]

# 			vals = pre_acts[:, neuron_idx]

# 			if group == "unpop":
# 				# only modify when activation > pop_mean
# 				unpop_mean = stats_unpop.iloc[neuron_idx]["mean"]
# 				pop_sd = stats_pop.iloc[neuron_idx]["mean"]

# 				unpop_sd = stats_unpop.iloc[neuron_idx]["sd"]

# 				mask = vals > unpop_mean + 2 * unpop_sd
# 				pre_acts[mask, neuron_idx] += w * unpop_sd

# 			else:  # group == "pop"
# 				# only modify when activation < unpop_mean
# 				unpop_mean = stats_unpop.iloc[neuron_idx]["mean"]
# 				unpop_sd = stats_unpop.iloc[neuron_idx]["sd"]

# 				pop_sd     = stats_pop.iloc[neuron_idx]["sd"]

# 				mask = vals < unpop_mean - 2 * unpop_sd
# 				pre_acts[mask, neuron_idx] -= w * pop_sd	 

# 		return pre_acts
# 	def add_noise(self, pre_acts, std):
# 		pre_actss = pre_acts.detach().cpu()
# 		if self.N is None:
# 			return pre_acts

# 		# pick N unique neurons
# 		top_neurons = random.sample(range(self.hidden_dim), int(self.N))

# 		# add Gaussian noise to each selected neuron
# 		# pre_acts shape: (batch_size, hidden_dim)
# 		batch_size = pre_actss.shape[0]
# 		for idx in top_neurons:
# 			# draw a vector of Gaussian noise
# 			noise = np.random.normal(
# 				loc=0.0,
# 				scale=std,
# 				size=(batch_size,)
# 			)
# 			pre_actss[:, idx] += noise

# 		return pre_actss.to(self.device)
	 
	 
# 	def forward(self, x, sequences=None, train_mode=False, save_result=False, epoch=None, dataset=None, pop_scores=None):
# 			sae_in = x - self.b_dec
# 			pre_acts1 = self.encoder(sae_in)
# 			self.last_activations = pre_acts1
# 			if self.analyze == True:
# 				if self.side == "user":
# 					compute_weighted_neuron_stats_by_row_item(activations=pre_acts1, dataset=self.dataset, side=self.side)
# 				else:
# 					compute_weighted_neuron_stats_by_row_item(activations=pre_acts1, dataset=self.dataset, side=self.side)
# 			if self.steer == True and self.N != 0:
# 				pre_acts1 = self.dampen_neurons(pre_acts1, dataset=self.dataset)
# 				# pre_acts = self.add_noise(pre_acts, std=self.beta)
# 			pre_acts = nn.functional.relu(pre_acts1)   
# 			z = self.topk_activation(pre_acts, sequences, save_result=False)

# 			x_reconstructed = z @ self.W_dec + self.b_dec
# 			e = x_reconstructed - x
# 			total_variance = (x - x.mean(0)).pow(2).sum()
# 			self.fvu = e.pow(2).sum() / total_variance
# 			if train_mode:
# 				if self.new_epoch == True:
# 					self.new_epoch = False
# 					dead = self.get_dead_latent_ratio(need_update=1)
# 					print("Dead percentage ", dead)					
# 				# First epoch, do not have dead latent info
# 				if self.previous_activate_latents is None:
# 					self.auxk_loss = 0.0
# 					return x_reconstructed
# 				num_dead = self.hidden_dim - len(self.previous_activate_latents)
# 				k_aux = int(x.shape[-1]) * 2
# 				if num_dead == 0:
# 					self.auxk_loss = 0.0
# 					return x_reconstructed
# 				scale = min(num_dead / k_aux, 1.0)
# 				k_aux = min(k_aux, num_dead)
# 				dead_mask = torch.isin(
# 					torch.arange(pre_acts.shape[-1]).to(self.device),
# 					self.previous_activate_latents,
# 					invert=True
# 				)
# 				auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
# 				auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
# 				# print("these are aux values, ", auxk_indices[0])
# 				# print("these are aux indices, ", auxk_acts[0])

# 				e_hat = torch.zeros_like(auxk_latents)
# 				e_hat.scatter_(1, auxk_indices, auxk_acts.to(self.dtype))
# 				e_hat = e_hat @ self.W_dec + self.b_dec

# 				auxk_loss = (e_hat - e).pow(2).sum()
# 				self.auxk_loss = scale * auxk_loss / total_variance

# 			return x_reconstructed


