from torch import nn
import torch
import numpy as np
class SASRecWithGating(nn.Module):    
    def __init__(self, sasrec_model, gate_indices, device='cpu', popularity_labels=None):
        self.popularity_labels = popularity_labels
        self.popularity_labels = self.popularity_labels.to(device)
        super().__init__()
        self.to(device)        
        self.sasrec = sasrec_model
        self.gating = AdaptiveGating(hidden_dim=sasrec_model.hidden_size,
                                     gate_indices=gate_indices)
        self.gating = self.gating.to(device)
        self.recommendation_count = np.zeros(self.sasrec.n_items)
        self.device = device
        self.loss_fct = nn.CrossEntropyLoss()

    
    def forward(self, input_seq, input_seq_len):
        # for param in self.sasrec.parameters():
        #     print('required? ', param.requires_grad)
        hidden = self.sasrec.forward(input_seq, input_seq_len)  # [batch_size, seq_len, hidden_dim]
        gated_hidden, gate_values = self.gating(hidden)
        return gated_hidden
    


    def calculate_loss(self, interaction, lambda_reg):
        item_seq = interaction['item_id_list']
        item_seq_len = interaction['item_length']
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction['item_id']
        test_item_emb = self.sasrec.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        scores = torch.softmax(logits, dim=1)
        loss_main = self.loss_fct(logits, pos_items)
        scores = scores.to('cuda')
        self.popularity_labels.to('cuda')
        penalty = lambda_reg * torch.sum(scores[:, 1:] * self.popularity_labels)
        loss = loss_main + penalty
        print(f"Main Loss: {loss_main.item():.4f} | Penalty: {penalty.item():.4f} | λ: {lambda_reg}")
        top_recs = torch.argsort(logits, dim=1, descending=True)[:, :10]
        # print(top_recs[0])
        
        return loss      
    
    def full_sort_predict(self, interaction):
        item_seq = interaction[self.sasrec.ITEM_SEQ]
        # item_seq = make_items_unpopular(item_seq)
        item_seq_len = interaction[self.sasrec.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        # if self.corr_file:
        #     seq_output = self.dampen_neurons_sasrec(seq_output)
        # save_batch_activations(seq_output, 64)
        test_items_emb = self.sasrec.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        top_recs = torch.argsort(scores, dim=1, descending=True)[:, :10]
        for key in top_recs.flatten():
            self.recommendation_count[key.item()] += 1
        return scores  

    def load_sasrec(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.sasrec.load_state_dict(checkpoint['state_dict'])

class AdaptiveGating(nn.Module):
    def __init__(self, hidden_dim, gate_indices):
        super().__init__()
        self.gate_indices = gate_indices  # e.g., [5, 20, 38]
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, len(gate_indices))  # No sigmoid!
        )

    def forward(self, hidden):
        # hidden: [batch_size, hidden_dim]
        gate_values = self.gate_mlp(hidden)  # [batch_size, 3]
        gated_hidden = hidden.clone()
        for i, idx in enumerate(self.gate_indices):
            gated_hidden[:, idx] += gate_values[:, i]
        # print(f"Gate values: {gate_values[0]} | Gate hidden: {gated_hidden[0]} | Indices: {self.gate_indices[0]}")
                # print(f"Gate values: {gate_values[0]} | Gate hidden: {gated_hidden[0]} | Indices: {self.gate_indices[0]}")
        print('gate values ', gate_values[0])
        
        return gated_hidden, gate_values