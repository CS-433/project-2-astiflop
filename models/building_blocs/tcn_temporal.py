import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class Chomp1d(nn.Module):
    """
    Removes the 'future' elements from the 1D convolution output
    to ensure the network is strictly causal.
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, : -self.chomp_size]
        return x.contiguous()


class TemporalBlock(nn.Module):
    """
    A single TCN block with two causal convolutions, residual connection, and optional dropout.
    Ensures strict causality through careful padding management.
    """

    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.3,
        dropout_1d=False,
    ):
        super(TemporalBlock, self).__init__()

        # Causal Conv 1
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        if dropout_1d:
            self.dropout1 = nn.Dropout1d(dropout)
        else:
            self.dropout1 = nn.Dropout(dropout)

        # Causal Conv 2
        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        if dropout_1d:
            self.dropout2 = nn.Dropout1d(dropout)
        else:
            self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )

        # Residual connection if input and output dimensions differ
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNTemporal(nn.Module):
    """
    Temporal Convolutional Network (TCN) for sequence modeling.
    Takes segment embeddings and outputs temporal representations.
    """

    def __init__(
        self,
        embed_dim,
        kernel_size=3,
        num_levels=6,
        dropout=0.3,
        dropout_1d=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.dropout = dropout

        # TCN Sequence Modeling with constant channels across levels
        # 6 levels provides receptive field to cover ~150 segments (T_max)
        num_channels = [embed_dim] * num_levels
        layers = []

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = embed_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            # Padding formula ensures strict causality
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                    dropout_1d=dropout_1d,
                )
            )

        self.tcn = nn.Sequential(*layers)

    def forward(self, seg_emb, mask=None):
        """
        Args:
            seg_emb: (B, T, embed_dim) - segment embeddings from feature extraction
            mask: (B, T) - optional mask for padded sequences

        Returns:
            tcn_out: (B, T, embed_dim) - temporal representations
            aux_loss: scalar - auxiliary loss (0 for TCN, no regularization)
        """
        B, T, _ = seg_emb.shape

        # TCN expects input shape: (Batch, Channels, Time_Sequence)
        tcn_input = seg_emb.transpose(1, 2)  # (B, embed_dim, T)

        tcn_out = self.tcn(tcn_input)  # (B, embed_dim, T)

        # Revert shape to (Batch, Time_Sequence, Channels)
        tcn_out = tcn_out.transpose(1, 2)  # (B, T, embed_dim)

        # TCN has no auxiliary loss (unlike BiLSTM orthogonality loss)
        aux_loss = torch.tensor(0.0, device=seg_emb.device)

        return tcn_out, aux_loss
