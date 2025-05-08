import torch
import numpy as np
import json
import torch
import torch.nn as nn
import os
from time import time
from tqdm import tqdm
import logging
from recbole.utils import utils
import pandas as pd
import random

class SAE(nn.Module):
	# @staticmethod
	# def parse_model_args(parser):
	# 	parser.add_argument('--sae_k', type=int, default=32,
	# 						help='top k activation')
	# 	parser.add_argument('--sae_scale_size', type=int, default=32,
	# 						help='scale size')
	# 	parser.add_argument('--recsae_model_path', type=str, default='',
	# 						help='Model save path.')
	# 	return parser
	
	def __init__(self,config,d_in):
		super(SAE, self).__init__()
		self.k = config["sae_k"]
		self.scale_size = config["sae_scale_size"]
		self.neuron_count = None
		self.damp_percent = None
		self.unpopular_only = None
		self.corr_file = None
		self.user_pop_scores = []
		self.device = config["device"]
		self.dtype = torch.float32
		self.to(self.device)
		self.d_in = d_in
		self.death_patience = 0
		self.hidden_dim = d_in * self.scale_size
		self.activation_count = torch.zeros(self.hidden_dim, device=config["device"])
		self.encoder = nn.Linear(self.d_in, self.hidden_dim, device=self.device,dtype = self.dtype)
		self.encoder.bias.data.zero_()
		self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
		self.set_decoder_norm_to_unit_norm()
		self.b_dec = nn.Parameter(torch.zeros(self.d_in, dtype = self.dtype, device=self.device))
		self.activate_latents = set()
		self.previous_activate_latents = None
		self.epoch_activations = {"indices": None, "values": None} 
		self.last_activations = torch.empty(0, dtype=torch.float32, device=self.device)
  
		self.epoch_idx=0
		self.item_activations = np.zeros(self.hidden_dim)

        # 2) highest_activations dict, each neuron j -> dict of GPU tensors
        #    (values, sequences, recommendations)
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
		return


	def set_dampen_hyperparam(self, corr_file=None, N=None, beta=None, unpopular_only=False):
		self.corr_file = corr_file
		self.N = N
		self.beta = beta
		self.unpopular_only = unpopular_only	
  
  
	def get_dead_latent_ratio(self, need_update=0):
		# Calculate the dead latent ratio
		ans = 1 - len(self.activate_latents) / self.hidden_dim
		# Calculate the current number of dead latents
		current_dead = self.hidden_dim - len(self.activate_latents)
		print("Dead percentage: ", ans)
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
		# If specified, mask out the first k indices from file by setting to -10
		# dataset_name="ml-1m"	
		# if k > 0:
		# 	idx_file = f"./dataset/{dataset_name}/negative_cohens_d.csv"
		# 	try:
		# 		df_idx = pd.read_csv(idx_file, index_col=0)
		# 	except FileNotFoundError:
		# 		raise FileNotFoundError(f"Index file not found: {idx_file}")
		# 	all_indices = df_idx.index.astype(int).tolist()
		# 	mask_indices = all_indices[:int(k)]
		# 	x = x.clone()
		# 	x[:, mask_indices] = 0

		# Compute top-k as before
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
			if self.epoch_activations["indices"] is None:
				self.epoch_activations["indices"] = inds_np
				self.epoch_activations["values"] = values_np
			else:
				self.epoch_activations["indices"] = np.concatenate((
					self.epoch_activations["indices"], inds_np), axis=0)
				self.epoch_activations["values"] = np.concatenate((
					self.epoch_activations["values"], values_np), axis=0)

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
     


	def dampen_neurons(self, pre_acts, dataset=None):
		if self.N is None or self.N==0:
			return pre_acts
		pop_neurons, unpop_neurons = utils.get_extreme_correlations(self.corr_file, self.unpopular_only, dataset=dataset)
  
		# Combine both groups into one list while labeling the group type.
		# 'unpop' neurons are those with higher activations for unpopular inputs (to be reinforced),
		# while 'pop' neurons are those with lower activations (to be dampened).
		combined_neurons = [(idx, cohen, 'unpop') for idx, cohen in unpop_neurons] + \
                        [(idx, cohen, 'pop') for idx, cohen in pop_neurons]
                        
		# Now sort by the absolute Cohen's d value (in descending order) and pick the overall top N neurons.
		combined_sorted = sorted(combined_neurons, key=lambda x: abs(x[1]), reverse=True)
		top_neurons = combined_sorted[:int(self.N)]
		# Load the corresponding statistics files.
		stats_unpop = pd.read_csv(rf"./dataset/{dataset}/row_stats_popular.csv")
		stats_pop = pd.read_csv(rf"./dataset/{dataset}/row_stats_unpopular.csv")
  
		# Create tensors of the absolute Cohen's d values for the selected neurons.
		abs_cohens = torch.tensor([abs(c) for _, c, _ in top_neurons], device=pre_acts.device)
		min_cohen = min(abs(c) for _, c, _ in combined_neurons)
		# Define a helper normalization function.
		def normalize_to_range(x, new_min, new_max):
			min_val = min_cohen
			max_val = torch.max(x)
			if max_val == min_val:
				return torch.full_like(x, (new_min + new_max) / 2)
			return (x - min_val) / (max_val - min_val) * (new_max - new_min) + new_min

		# Normalize the Cohen's d values to [0, 2.5]
		weights = normalize_to_range(abs_cohens, new_min=0, new_max=self.beta)

		# Now update the neuron activations based on group.
		for i, (neuron_idx, cohen, group) in enumerate(top_neurons):
			weight = weights[i]		
			if group == 'unpop':
				# For neurons to be reinforced, fetch stats from the unpopular file.
				row = stats_unpop.iloc[neuron_idx]
				mean_val = row["mean"]
				std_val = row["std"]
    
				# Identify positions where the neuron's activation is above its mean.
				vals = pre_acts[:, neuron_idx]
				# Increase activations by an amount proportional to the standard deviation and effective weight.
				pre_acts[:, neuron_idx] += weight * std_val
			else:  # group == 'pop'
				# For neurons to be dampened, use the popular statistics for impact.
				pop_mean = stats_pop.iloc[neuron_idx]["mean"]
				pop_sd = stats_pop.iloc[neuron_idx]["std"]

				# Still fetch the comparison stats from the unpopular stats file
				# (this is from your original logic; adjust if needed).
				row = stats_unpop.iloc[neuron_idx]
				mean_val = row["mean"]
				std_val = row["std"]

				# Identify positions where the neuron's activation is below its mean.
				vals = pre_acts[:, neuron_idx]
				# Decrease activations proportionally.
				pre_acts[:, neuron_idx] -= weight * pop_sd
    
		return pre_acts
     
     
    
	def add_noise(self, pre_acts, std):
		pre_actss = pre_acts.detach().cpu()
		if self.N is None:
			return pre_acts

		# pick N unique neurons
		top_neurons = random.sample(range(self.hidden_dim), int(self.N))

		# add Gaussian noise to each selected neuron
		# pre_acts shape: (batch_size, hidden_dim)
		batch_size = pre_actss.shape[0]
		for idx in top_neurons:
			# draw a vector of Gaussian noise
			noise = np.random.normal(
				loc=0.0,
				scale=std,
				size=(batch_size,)
			)
			pre_actss[:, idx] += noise

		return pre_actss.to(self.device)
     
     
     
     
	def forward(self, x, sequences=None, train_mode=False, save_result=False, epoch=None, dataset=None):
		sae_in = x - self.b_dec
		pre_acts = self.encoder(sae_in)
		self.last_activations = pre_acts
		if self.corr_file:
			pre_acts = self.dampen_neurons(pre_acts, dataset=dataset)
			# pre_acts = self.add_noise(pre_acts, std=self.beta)
		pre_acts = nn.functional.relu(pre_acts)   
		z = self.topk_activation(pre_acts, sequences, save_result=False)

		x_reconstructed = z @ self.W_dec + self.b_dec
		e = x_reconstructed - x
		total_variance = (x - x.mean(0)).pow(2).sum()
		self.fvu = e.pow(2).sum() / total_variance

		if train_mode:
			if self.epoch_idx != epoch:
				self.epoch_idx = epoch
				dead = self.get_dead_latent_ratio(need_update=1)
				# Resampling dead latents if any exist
				# if dead > 0.02:  # Threshold can be adjusted, e.g., dead > 0.1
				# 	# Compute mean residual over the batch
				# 	mean_e = (x - x_reconstructed).mean(dim=0)  # Shape: (d,)
				# 	norm_e = torch.norm(mean_e)
				# 	if norm_e > 1e-6:  # Avoid division by zero or insignificant updates
				# 		mean_e = mean_e / norm_e  # Normalize to unit vector
				# 		# Identify dead latents
				# 		all_latents = torch.arange(self.hidden_dim, device=self.device)
				# 		dead_mask = ~torch.isin(all_latents, self.previous_activate_latents)
				# 		dead_latents = all_latents[dead_mask]
				# 		# Update weights for each dead latent
				# 		for i in dead_latents:
				# 			self.W_dec.data[i, :] = mean_e
				# 			self.encoder.weight.data[i, :] = mean_e
				# 	else:
				# 		print("Norm of mean residual is too small, skipping resampling")
				self.death_patience = 0

			self.death_patience += pre_acts.shape[0]
			# First epoch, do not have dead latent info
			if self.previous_activate_latents is None:
				self.auxk_loss = 0.0
				return x_reconstructed
			num_dead = self.hidden_dim - len(self.previous_activate_latents)
			# print("num dead ", num_dead)
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


	def update_highest_activations2(self, item_ids, recommendations):

			"""
			1) Find top-10 and bottom-10 activations per neuron for this new batch on GPU.
			2) Merge them with the existing top-10 and bottom-10 stored in self.highest_activations.
			3) Keep only the top-10 and bottom-10 overall (no .cpu() or .tolist() here).
			
			:param item_ids:       [batch_size, seq_len] GPU tensor (sequences)
			:param recommendations: [batch_size, num_items] GPU tensor
									We'll extract top-10 recommended items per sample.
			"""
			# Save batch activations (optional)
			# utils.save_batch_activations(self.last_activations, self.hidden_dim) 

			# ------------------------
			# A) Get top-10 and bottom-10 per neuron (column)
			# ------------------------
			# self.last_activations has shape [batch_size, hidden_dim]
			batch_top_vals, batch_top_idxs = torch.topk(self.last_activations, k=10, dim=0)
			# rand_k   = 10                                     # how many random rows you want
			# num_rows = self.last_activations.size(0)          # rows = neurons / tokens / etc.
			# num_cols = self.last_activations.size(1)          # usually = batch size

			# batch_rand_idxs = []

			# for col in range(num_cols):
			# 	all_rows = torch.arange(num_rows, device=self.last_activations.device)
			# 	# remove the forbidden indices for this column
			# 	valid    = all_rows[~torch.isin(all_rows, batch_top_idxs[:, col])]
			# 	# shuffle and grab the first `rand_k`
			# 	rand_sel = valid[torch.randperm(valid.size(0))[:rand_k]]
			# 	batch_rand_idxs.append(rand_sel)

			# batch_rand_idxs = torch.stack(batch_rand_idxs, dim=1)   # shape: (rand_k, num_cols)

			# # optional: grab the actual activation values for those random rows
			# batch_rand_vals = self.last_activations_relu.gather(0, batch_rand_idxs)			# shapes: [10, hidden_dim] for both top and bottom
			batch_rand_vals, batch_rand_idxs = torch.topk(self.last_activations, k=10, dim=0, largest=False)

			# ------------------------
			# C) Merge each neuron's top-10 and bottom-10
			# ------------------------
			for j in range(self.hidden_dim):
				# --- Handle top-10 ---
				# 1) Gather new data for neuron j (top-10)
				new_top_indices = batch_top_idxs[:, j]         # [10] - indices in this batch
				new_top_vals = batch_top_vals[:, j]           # [10]
				new_top_seqs = item_ids[new_top_indices]      # [10, seq_len]
				new_top_recs = recommendations[new_top_indices] # [10, num_items]
        
				# 2) Retrieve old top-10 from self.highest_activations[j]
				old_top_vals = self.highest_activations[j].get("values", torch.tensor([]).to(self.last_activations.device))
				old_top_seqs = self.highest_activations[j].get("items", torch.tensor([], dtype=torch.long).to(self.last_activations.device))
				old_top_recs = self.highest_activations[j].get("recommendations", torch.tensor([], dtype=torch.long).to(self.last_activations.device))

				# 3) Concatenate old + new (-> up to 20)
				all_top_vals = torch.cat([old_top_vals, new_top_vals], dim=0)    # [<=20]
				all_top_seqs = torch.cat([old_top_seqs, new_top_seqs], dim=0)    # [<=20, seq_len]
				all_top_recs = torch.cat([old_top_recs, new_top_recs], dim=0)    # [<=20, num_items]

				# 4) Take top-10 again
				if all_top_vals.numel() > 0:
					topvals, topidxs = torch.topk(all_top_vals, k=min(15, all_top_vals.size(0)), dim=0)
					# 5) Update with new top-10 data
					self.highest_activations[j]["values"] = topvals
					self.highest_activations[j]["items"] = new_top_indices  # Store sequences, not indices
					self.highest_activations[j]["recommendations"] = all_top_recs[topidxs]
				
				# --- Handle bottom-10 ---
				# 1) Gather new data for neuron j (bottom-10)
				new_bottom_indices = batch_rand_idxs[:, j]         # [10] - indices in this batch
				new_bottom_vals = batch_rand_vals[:, j]           # [10]
				new_bottom_seqs = new_top_indices     # [10, seq_len]
				new_bottom_recs = recommendations[new_bottom_indices] # [10, num_items]

				# 2) Retrieve old bottom-10 from self.highest_activations[j]
				old_bottom_vals = self.highest_activations[j].get("low_values", torch.tensor([]).to(self.last_activations.device))
				old_bottom_seqs = self.highest_activations[j].get("low_items", torch.tensor([], dtype=torch.long).to(self.last_activations.device))

				# 3) Concatenate old + new (-> up to 20)
				all_bottom_vals = torch.cat([old_bottom_vals, new_bottom_vals], dim=0)    # [<=20]
				all_bottom_seqs = torch.cat([old_bottom_seqs, new_bottom_seqs], dim=0)    # [<=20, seq_len]

				# 4) Take bottom-10 again
				if all_bottom_vals.numel() > 0:
					# 5) Update with new bottom-10 data
					self.highest_activations[j]["low_values"] = new_bottom_vals
					self.highest_activations[j]["low_items"] = new_bottom_indices  # Store sequences, not indices


	def update_highest_activations(self, sequences, recommendations, user_ids):
		"""
		1) Find top-10 activations per neuron for this new batch on GPU.
		2) Merge them with the existing top-10 stored in self.highest_activations.
		3) Keep only the top-10 overall (no .cpu() or .tolist() here).
		
		:param sequences:       [batch_size, seq_len] GPU tensor
		:param recommendations: [batch_size, num_items] GPU tensor
								We'll extract top-10 recommended items per sample.
		"""
		# utils.save_user_popularity_score(0.9, user_ids, sequences)
		# total_pop_scores, total_unpop_scores = utils.fetch_user_popularity_score(user_ids,sequences)
		# utils.save_batch_user_popularities(total_pop_scores, total_unpop_scores)
		utils.save_batch_activations(self.last_activations, self.hidden_dim) 

		# ------------------------
		# A) Get top-10 per neuron (column)
		# ------------------------
		# self.last_activations has shape [batch_size, hidden_dim]
		batch_top_vals, batch_top_idxs = torch.topk(self.last_activations, k=10, dim=0)
		# shapes: [10, hidden_dim]

		# ------------------------
		# C) Merge each neuron's top-10
		# ------------------------
		for j in range(self.hidden_dim):
			# 1) Gather new data for neuron j
			new_indices = batch_top_idxs[:, j]         # [10] - indices in this batch
			new_vals = batch_top_vals[:, j]           # [10]
			new_seqs = sequences[new_indices]         # [10, seq_len]
			new_recs = recommendations[new_indices]    # [10, 10]

			# 2) Retrieve old top-k from self.highest_activations[j]
			old_vals = self.highest_activations[j]["values"]           # [<=10]
			old_seqs = self.highest_activations[j]["sequences"]        # [<=10, seq_len]
			old_recs = self.highest_activations[j]["recommendations"]  # [<=10, 10]

			# 3) Concatenate old + new (-> up to 20)
			all_vals = torch.cat([old_vals, new_vals], dim=0)    # [<=20]
			all_seqs = torch.cat([old_seqs, new_seqs], dim=0)    # [<=20, seq_len]
			all_recs = torch.cat([old_recs, new_recs], dim=0)    # [<=20, 10]

			# 4) Take top-10 again
			if all_vals.numel() > 0:
				topvals, topidxs = torch.topk(all_vals, k=min(10, all_vals.size(0)), dim=0)
			else:
				# no old or new data
				continue

			# 5) Index sequences & recs
			new_values = topvals
			new_sequences = all_seqs[topidxs]
			new_recommendations = all_recs[topidxs]

			# 6) Store back as GPU tensors
			self.highest_activations[j]["values"] = new_values
			self.highest_activations[j]["sequences"] = new_sequences
			self.highest_activations[j]["recommendations"] = new_recommendations

    
    
	def save_highest_activations(self, filename=r"./dataset/ml-1m/neuron_activations.csv"):		
		"""
		Save the top 5 highest activations and their corresponding sequences to a file.
		"""
		df = pd.DataFrame({
			'index': np.arange(len(self.activation_count)),
			'count': self.activation_count.cpu().numpy()
		})

		# Save to CSV
		df.to_csv(filename, index=False)
  


	def save_highest_activations2(self, filename=r"./dataset/lastfm/neuron_activations_unpopular.csv"):		
			"""
			Save the top 5 highest activations and their corresponding sequences to a file.
			"""
			df = pd.DataFrame({
				'index': np.arange(len(self.activation_count)),
				'count': self.activation_count.cpu().numpy()
			})
   
			# Save to CSV
			df.to_csv(filename, index=False)
	
	
			# # corr_pop = utils.calculate_pearson_correlation(r"./dataset/ml-1m/user_scores_pop.h5", r"./dataset/ml-1m/correlations_pop.csv")
			# # corr_unpop = utils.calculate_pearson_correlation(r"./dataset/ml-1m/user_scores_unpop.h5", r"./dataset/ml-1m/correlations_unpop.csv")
			# file_path = r'./dataset/ml-1m/item_popularity_labels_with_titles.csv'
			# data_item = pd.read_csv(file_path)  # Try 'latin1', change to 'cp1252' if needed
			# with open(filename, "w") as f:
			# 	for neuron, data in self.highest_activations.items():
			# 		f.write(f"Neuron {neuron}:\n")
			# 		# f.write(f"Popularity Correlation:{corr_pop[neuron]}\n")
			# 		# f.write(f"Unpopularity Correlation:{corr_unpop[neuron]}\n")
			# 		for value, sequence_ids, sequence, recommendations_ids, recommendations in zip(data["values"],  data["sequences"], utils.get_item_titles(data["sequences"], data_item), data["recommendations"], utils.get_item_titles(data["recommendations"], data_item)):
			# 			f.write(f"  Activation: {value}\n")
			# 			f.write(f"  Last 10 Sequence titles: {sequence[-10:]}\n")
			# 			f.write(f"  Sequence ids: {sequence_ids}\n")
			# 			f.write(f"  top recommendation ids: {recommendations_ids}\n")
			# 			f.write(f"  top recommendations: {recommendations}\n")
			# 		f.write("\n")

