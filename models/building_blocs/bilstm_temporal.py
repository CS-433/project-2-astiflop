import torch
import torch.nn as nn

class BiLSTMTemporal(nn.Module):
    def __init__(self, embed_dim, bilstm_layers):
        super().__init__()
        self.bilstm_layers = bilstm_layers
        self.bilstms = nn.ModuleList([
            nn.LSTM(
                input_size=embed_dim, 
                hidden_size=embed_dim//2, 
                num_layers=1, 
                batch_first=True, 
                bidirectional=True
            ) for _ in range(bilstm_layers)
        ])
        self.lstm_aggregation = nn.Linear(embed_dim * bilstm_layers, embed_dim)

    def compute_orthogonality_loss(self):
        if self.bilstm_layers <= 1:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        weights = []
        for lstm in self.bilstms:
            w = torch.cat([p.flatten() for n, p in lstm.named_parameters() if 'weight' in n])
            w_norm = w / (torch.norm(w) + 1e-8) 
            weights.append(w_norm.unsqueeze(1))
            
        W = torch.cat(weights, dim=1)
        corr = torch.matmul(W.t(), W) 
        
        eye = torch.eye(self.bilstm_layers, device=W.device)
        off_diag_corr = corr - eye
        ortho_loss = torch.sum(off_diag_corr ** 2)        
        return ortho_loss

    def forward(self, seg_emb, mask=None):
        B, T, _ = seg_emb.shape
        lstm_outputs = []
        for lstm in self.bilstms:
            if mask is not None:
                lengths = mask.sum(dim=1).cpu().to(torch.int64)
                packed_emb = torch.nn.utils.rnn.pack_padded_sequence(
                    seg_emb, lengths, batch_first=True, enforce_sorted=False
                )
                lstm_out_packed, _ = lstm(packed_emb)
                lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                    lstm_out_packed, batch_first=True, total_length=T
                )
            else:
                lstm_out, _ = lstm(seg_emb)
            lstm_outputs.append(lstm_out)
            
        lstm_cat = torch.cat(lstm_outputs, dim=-1)
        lstm_agg = self.lstm_aggregation(lstm_cat)
        
        ortho_loss = self.compute_orthogonality_loss()
        return lstm_agg, ortho_loss