"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:_
    https://github.com/kang205/SASRec

"""

import torch
from torch import nn
from recbole.model.sequential_recommender.sasrec import SASRec
import torch
import numpy as np
import json
import torch
import torch.nn as nn
from recbole.utils import utils
import pandas as pd
import random



class SASRec_SAE(SASRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        model_path = config["base_path"]
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.load_state_dict(checkpoint['state_dict'])
        self.sae_module_i = SAE(config, side="item")
        self.sae_module_u = SAE(config, side="user")
        self.a1 = 0.9
        self.a2 = 0.1
        for param in self.parameters():
            param.requires_grad = False
        for param in self.sae_module_i.parameters():
            param.requires_grad = True  
        for param in self.sae_module_u.parameters():
            param.requires_grad = True         
        self.val_fvu_i = torch.tensor(0.0, device=self.device)
        self.val_fvu_u = torch.tensor(0.0, device=self.device)

    def forward(self, item_seq, item_seq_len, train_mode=None):
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
        output_sae = self.sae_module_u(output, train_mode=train_mode)
        return output_sae
    

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len, train_mode=True)
        sae_loss_u = self.sae_module_u.fvu + self.sae_module_u.auxk_loss / 2
        return sae_loss_u

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores



    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len, train_mode=False)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        # self.val_fvu_i += (self.sae_module_i.fvu)
        self.val_fvu_u += (self.sae_module_u.fvu)
        return scores

        

class SAE(nn.Module):
    
    def __init__(self,config, side="item"):
        super(SAE, self).__init__()
        self.side=side
        self.dataset = config["dataset"]
        self.k = config["sae_k"]
        self.scale_size = config["sae_scale_size"]
        self.alpha_pop = config['alpha_pop']
        self.alpha_unpop = config['alpha_unpop']
        self.steer = config['steer']
        self.analyze = config['analyze']
        self.fvu = torch.tensor(0.0)
        self.neuron_count = None
        self.unpopular_only = None
        self.device = config["device"]
        self.dtype = torch.float32
        self.to(self.device)
        self.d_in = config['input_dim']
        self.hidden_dim = self.d_in * self.scale_size
        self.N = self.hidden_dim
        self.d_min =  config['D']
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

        counts = torch.bincount(flat_indices, minlength=self.hidden_dim)

        self.activation_count += counts.to(self.activation_count.device)
        self.activate_latents.update(topk_indices.cpu().numpy().flatten())

        if save_result:
            values_np = topk_values.detach().cpu().numpy()
            inds_np = topk_indices.detach().cpu().numpy()

        sparse_x = torch.zeros_like(x)
        sparse_x.scatter_(1, topk_indices, topk_values.to(self.dtype))
        return sparse_x

        
    def _build_steering_vector(self, dataset):
        """Build the steering vector according to the specified direction.

        If ``self.d_min`` is **not** ``None`` we do **not** rely on
        ``self.N``.  Instead, we include *all* neurons whose absolute
        Cohen's *d* is **greater than or equal to** ``self.d_min``.  When
        ``self.d_min`` *is* ``None``, we fall back to using the first
        ``self.N`` neurons ranked by |*d*|.
        """

        pop_neurons, unpop_neurons = utils.get_extreme_correlations(
            rf"{self.side}/cohens_d.csv", dataset=dataset
        )
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
            thres = self.alpha_pop if pop else self.alpha_unpop
            xmax = torch.max(x)
            return torch.full_like(x, thres / 2) if xmax == 0 else (x / xmax) * thres

        pop_mask = torch.tensor([g == "pop" for *_, g in top_neurons], device=self.device)
        unpop_mask = ~pop_mask

        if pop_mask.any():
            weights[pop_mask] = normalise(abs_cohens[pop_mask], pop=True)
        if unpop_mask.any():
            weights[unpop_mask] = normalise(abs_cohens[unpop_mask])

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
            # if self.analyze == True:
            #     if self.side == "item":
            #         compute_weighted_neuron_stats_by_row_item(activations=pre_acts1, dataset=self.dataset, side=self.side)
            self.last_activations = pre_acts1
            if self.steer == True and self.N != 0:
                pre_acts1 = self.dampen_neurons(pre_acts1, dataset=self.dataset)
            pre_acts = nn.functional.relu(pre_acts1)
            # self.last_activations = torch.where(pre_acts == 0, torch.tensor(-0.1, dtype=pre_acts.dtype, device=pre_acts.device), pre_acts)
            # self.last_activations = pre_acts1 - 1
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




