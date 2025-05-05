import torch
import torch.nn as nn
from .sasrec import SASRec
from .sae import SAE
from recbole.utils import (
    make_items_unpopular,
    make_items_popular,
    save_batch_activations
)

import numpy as np
import pandas as pd

class SASRec_SAE(SASRec):
    def __init__(self, config, dataset, sasrec_model_path=None, mode="train"):
        super(SASRec_SAE, self).__init__(config, dataset)
        self.device = config["device"]
        # Load SASRec model parameters if path is provided
        if sasrec_model_path is not None:
            self.load_sasrec(sasrec_model_path)
        self.sae_module = SAE(config, self.hidden_size)  # SAE initialization
        # Mode can be 'train', 'test', or 'inference'
        self.mode = mode
        self.total_loss = torch.tensor(0.0)
        self.to(config["device"])
        for param in self.parameters():
            param.requires_grad = False  # Freeze all parameters

        for param in self.sae_module.parameters():
            param.requires_grad = True  # Unfreeze SAE parameters
            
    def set_dampen_hyperparam(self, corr_file=None, N=None, beta=None, unpopular_only=False):
        self.sae_module.set_dampen_hyperparam(corr_file=corr_file, N=N, beta=beta, unpopular_only=unpopular_only)

    def set_sae_mode(self, mode):
        if mode in ['train', 'test', 'dampened']:
            self.mode = mode
        else:
            raise ValueError(f"Invalid mode {mode} set for SASRec_SAE")

    def forward(self, item_seq, item_seq_len, mode=None, scores=None):
        # Use SASRec to process the sequence
        sasrec_output = super().forward(item_seq, item_seq_len)  # Final hidden states from SASRec
        
        sae_output = self.sae_module(sasrec_output, train_mode=(mode=='train'), epoch=scores)


        return sae_output




    def save_item_activations(self):
        output = self.sae_module(self.item_embedding.weight, train_mode=False, epoch=None)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(output, test_items_emb.transpose(0, 1))  # [B n_items]
        top_recs = torch.argsort(scores, dim=1, descending=True)[:, :10]
        # if(self.mode == "test"):
        #     # user_ids = interaction['user_id']
        #     save_batch_activations(self.sae_module.last_activations, 4096) 
        self.sae_module.update_highest_activations2(self.item_embedding.weight, top_recs)
        # for key in top_recs.flatten():
        #     self.recommendation_count[key.item()] += 1
        return scores




    def calculate_loss(self, interaction, scores=None, show_res=False):
        # Compute SASRec loss first
        # sasrec_loss = super().calculate_loss(interaction)

        # Compute SAE loss
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        sasrec_output = self.forward(item_seq, item_seq_len, mode='train', scores=scores)
        if self.mode == 'train':
            sae_loss = self.sae_module.fvu + self.sae_module.auxk_loss / 2
            if show_res:
                print(f"FVU: {self.sae_module.fvu}, AUXK Loss: {self.sae_module.auxk_loss}, AUXK Loss / 32: {self.sae_module.auxk_loss / 32} SAE Total Loss: {sae_loss}")
        else:
            sae_loss = self.sae_module.fvu

        total_loss = sae_loss

        return total_loss



    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        # item_seq = make_items_unpopular(item_seq).to(self.device)
        seq_output = self.forward(item_seq, item_seq_len, mode='eval')
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        top_recs = torch.argsort(scores, dim=1, descending=True)[:, :10]
        scores[:, 0] =  float("-inf")
        # if(self.mode == "test"):
        #     # user_ids = interaction['user_id']
        # nonzero_idxs = pd.read_csv(r"./dataset/ml-1m/nonzero_activations_sasrecsae_k48-32.csv")["index"].tolist()
        # save_batch_activations(self.sae_module.last_activations, self.sae_module.hidden_dim) 
        # self.sae_module.update_highest_activations(item_seq, top_recs, None)

        if hasattr(self.sae_module, 'auxk_loss'):
            self.total_loss += (self.sae_module.fvu + self.sae_module.auxk_loss / 32)
        
        for key in top_recs.flatten():
            self.recommendation_count[key.item()] += 1
        return scores



    def save_sae(self, path):
        torch.save(self.sae_module.state_dict(), path)

    def load_sae(self, path):
        self.sae_module.load_state_dict(torch.load(path))

    def load_sasrec(self, path):
        """
        Load the saved SASRec model parameters into the current model.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        a = 5
        # self.load_other_parameter(checkpoint.get("other_parameter"))

