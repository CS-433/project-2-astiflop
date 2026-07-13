import torch
import torch.nn as nn


class VanillaRNNTemporal(nn.Module):
    """
    Simplest recurrent temporal module: a single-layer vanilla RNN.
    """

    def __init__(self, embed_dim, num_layers=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, seg_emb, mask=None):
        B, T, _ = seg_emb.shape

        if mask is not None:
            lengths = mask.sum(dim=1).cpu().to(torch.int64)
            packed_emb = nn.utils.rnn.pack_padded_sequence(
                seg_emb, lengths, batch_first=True, enforce_sorted=False
            )
            rnn_out, _ = self.rnn(packed_emb)
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(
                rnn_out, batch_first=True, total_length=T
            )
        else:
            rnn_out, _ = self.rnn(seg_emb)

        aux_loss = torch.tensor(0.0, device=seg_emb.device)
        return rnn_out, aux_loss
