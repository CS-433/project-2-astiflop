import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeatureExtractor(nn.Module):
    """
    Encodes raw time series signals (segments) into a feature vector.
    Input: (Batch, 1, Length) -> Output: (Batch, Embedding_Dim)
    """

    def __init__(self, input_len, embedding_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(64, embedding_dim, kernel_size=3, stride=2, padding=1)
        self.global_pool = nn.AdaptiveMaxPool1d(1)  # Pooling to get fixed size vector
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        # x shape: (Batch * T * V, 1, Length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x).squeeze(-1)  # (Batch*T*V, Emb_Dim)
        x = F.relu(self.fc(x))
        return x


class GatedAttention(nn.Module):
    """
    Computes attention weights for a set of instances.
    Gated Attention (Ilse et al. 2018) is standard for MIL.
    """

    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (Batch, Num_Instances, Dim)
        # V-Attention: learn non-linearity
        # U-Attention: learn gating (like LSTM gate)
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)

        # Element-wise multiplication (Conjunctive-like mechanism)
        A = self.attention_weights(A_V * A_U)  # (Batch, Num_Instances, 1)

        # Softmax over instances to get probability distribution
        weights = F.softmax(A, dim=1)
        return weights


class TAIL_MIL(nn.Module):
    def __init__(self, segment_len, num_vars=3, embed_dim=512):
        super().__init__()
        self.num_vars = num_vars
        self.embed_dim = embed_dim

        # 1. Feature Extractor (Shared across all variables and segments)
        self.feature_extractor = FeatureExtractor(
            input_len=segment_len, embedding_dim=embed_dim
        )

        # 2. V-Attention (Variate Level)
        self.v_attention = GatedAttention(embed_dim)

        # 3. Time-Awareness (Positional Encoding)
        self.pos_encoder = nn.Parameter(
            torch.randn(1, 200, embed_dim)
        )  # Max 200 segments

        # 4. T-Attention (Time Level)
        self.t_attention = GatedAttention(embed_dim)

        # 5. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        """
        x shape: (Batch, T, V, Length)
           - Batch: 100 samples
           - T: ~75 segments
           - V: 3 variables (X, Y, Speed)
           - Length: ~866 raw signal points
        """
        B, T, V, L = x.shape

        # --- Step 1: Feature Extraction ---
        x_flat = x.view(B * T * V, 1, L)
        embeddings = self.feature_extractor(x_flat)  # (B*T*V, Emb_Dim)
        embeddings = embeddings.view(B, T, V, self.embed_dim)  # (B, T, V, Emb_Dim)

        # --- Step 2: V-Attention (Aggregation over Variables) ---
        # We want to aggregate the 3 vars into 1 segment representation
        # Flatten Batch and Time to treat them as independent "groups" for now
        curr_embeddings = embeddings.view(B * T, V, self.embed_dim)
        v_weights = self.v_attention(curr_embeddings)  # (B*T, V, 1)
        segment_embeddings = torch.sum(
            curr_embeddings * v_weights, dim=1
        )  # (B*T, Emb_Dim)
        segment_embeddings = segment_embeddings.view(
            B, T, self.embed_dim
        )  # (B, T, Emb_Dim)

        # --- Step 3: Time-Awareness ---
        # Add positional encoding to capture sequence order
        # Slice pos_encoder to matching time length T
        segment_embeddings = segment_embeddings + self.pos_encoder[:, :T, :]

        # --- Step 4: T-Attention (Aggregation over Time) ---
        t_weights = self.t_attention(segment_embeddings)  # (B, T, 1)
        bag_embedding = torch.sum(segment_embeddings * t_weights, dim=1)  # (B, Emb_Dim)

        # --- Step 5: Classification ---
        y_prob = self.classifier(bag_embedding)

        # Return probability and weights for interpretability
        return y_prob, t_weights, v_weights.view(B, T, V)


if __name__ == "__main__":
    # --- Dummy Data Example ---
    # Batch=2, Segments=75, Variables=3, Length=866
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TAIL_MIL(segment_len=866).to(device)
    dummy_input = torch.randn(2, 75, 3, 866).to(device)

    output, t_importance, v_importance = model(dummy_input)

    print(f"Device: {device}")
    print(f"Anomaly Score: {output.item() if output.numel()==1 else output[0].item()}")
    print(f"Time Attention Shape: {t_importance.shape}")  # Should be (2, 75, 1)
    print(f"Variable Attention Shape: {v_importance.shape}")  # Should be (2, 75, 3)


def TailMilModel(
    dataset,
    train_indices,
    test_indices,
    batch_size=8,
    lr=1e-4,
    embed_dim=64,
    epochs=100,
    patience=10,
    threshold=0.5,
    device="cpu",
):
    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices), batch_size=batch_size, shuffle=False
    )

    model = TAIL_MIL(
        segment_len=dataset.segment_len, num_vars=3, embed_dim=embed_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    best_val_acc = -float("inf")
    best_val_f1 = 0
    best_val_auc = 0
    best_val_prec = 0
    best_val_rec = 0
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).float()
            preds, _, _ = model(X)
            preds = preds.squeeze()
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                preds, _, _ = model(X)
                preds = preds.squeeze()
                val_labels.extend(y.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())

        val_preds_binary = (np.array(val_preds) > threshold).astype(int)
        val_acc = accuracy_score(val_labels, val_preds_binary)
        val_f1 = f1_score(val_labels, val_preds_binary)
        val_prec = precision_score(val_labels, val_preds_binary, zero_division=0)
        val_rec = recall_score(val_labels, val_preds_binary, zero_division=0)
        try:
            val_auc = roc_auc_score(val_labels, val_preds)
        except:
            val_auc = 0.5

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_val_prec = val_prec
            best_val_rec = val_rec
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), "tail_mil_worm_best.pth")
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break

    print("\n=== Train/Test Summary ===")
    print(f"Val Acc: {best_val_acc:.4f}")
    print(f"Val Prec: {best_val_prec:.4f}")
    print(f"Val Rec: {best_val_rec:.4f}")
    print(f"Val F1: {best_val_f1:.4f}")
    print(f"Val AUC: {best_val_auc:.4f}")

    return best_val_acc, best_val_prec, best_val_rec, best_val_f1
