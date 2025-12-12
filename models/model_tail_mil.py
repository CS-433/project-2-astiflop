import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
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

from tqdm import tqdm


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

    def forward(self, x, mask=None):
        # x: (Batch, Num_Instances, Dim)
        # V-Attention: learn non-linearity
        # U-Attention: learn gating (like LSTM gate)
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)

        # Element-wise multiplication (Conjunctive-like mechanism)
        A = self.attention_weights(A_V * A_U)  # (Batch, Num_Instances, 1)

        if mask is not None:
            # mask shape: (Batch, Num_Instances)
            # We want to mask out the padded instances (where mask is False/0)
            # A shape: (Batch, Num_Instances, 1)
            mask = mask.unsqueeze(-1)  # (Batch, Num_Instances, 1)
            A = A.masked_fill(mask == 0, -1e9)

        # Softmax over instances to get probability distribution
        weights = F.softmax(A, dim=1)
        return weights


class TAIL_MIL(nn.Module):
    def __init__(self, segment_len, embed_dim=512):
        super().__init__()
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

        if torch.isnan(x).any():
            print("NaN detected in the input")
            exit(0)

        # --- Step 1: Feature Extraction ---
        x_flat = x.view(B * T * V, 1, L)
        embeddings = self.feature_extractor(x_flat)  # (B*T*V, Emb_Dim)
        embeddings = embeddings.view(B, T, V, self.embed_dim)  # (B, T, V, Emb_Dim)

        # Check for NaN values in embeddings
        if torch.isnan(embeddings).any():
            print("NaN detected in embeddings")
            exit(0)

        # --- Step 2: V-Attention (Aggregation over Variables) ---
        curr_embeddings = embeddings.view(B * T, V, self.embed_dim)
        v_weights = self.v_attention(curr_embeddings)  # (B*T, V, 1)
        segment_embeddings = torch.sum(
            curr_embeddings * v_weights, dim=1
        )  # (B*T, Emb_Dim)
        segment_embeddings = segment_embeddings.view(
            B, T, self.embed_dim
        )  # (B, T, Emb_Dim)

        # Check for NaN values in segment_embeddings
        if torch.isnan(segment_embeddings).any():
            print("NaN detected in segment_embeddings")
            exit(0)

        # --- Step 3: Time-Awareness ---
        segment_embeddings = segment_embeddings + self.pos_encoder[:, :T, :]

        # --- Step 4: T-Attention (Aggregation over Time) ---
        mask = (x.view(B, T, -1).abs().sum(dim=-1) > 1e-6).float()  # (B, T)

        t_weights = self.t_attention(segment_embeddings, mask=mask)  # (B, T, 1)
        bag_embedding = torch.sum(segment_embeddings * t_weights, dim=1)  # (B, Emb_Dim)

        # Check for NaN values in bag_embedding
        if torch.isnan(bag_embedding).any():
            print("NaN detected in bag_embedding")
            exit(0)

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


class ScaledDataset(Dataset):
    def __init__(self, dataset, mean, std):
        self.dataset = dataset
        self.mean = mean.view(1, -1, 1)  # (1, F, 1)
        self.std = std.view(1, -1, 1)    # (1, F, 1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        # x shape: (max_segments, features, segment_len)
        
        # Identify non-padded segments (assuming padding is zero and valid data is not all zero)
        # x shape: (S, F, L) -> flatten F and L to check if segment is all zeros
        mask = (x.view(x.size(0), -1).abs().sum(dim=-1) > 1e-6).float().view(-1, 1, 1) # (max_segments, 1, 1)
        
        # Apply scaling
        x_scaled = (x - self.mean) / (self.std + 1e-8)
        
        # Re-apply mask to ensure padding remains zero
        x_scaled = x_scaled * mask
        
        # DEBUG: print stats about the scaled features
        # print(f"feature 1 after scaling: min={x_scaled[:,:,0].min().item():.4f}, max={x_scaled[:,:,0].max().item():.4f}, mean={x_scaled[:,:,0].mean().item():.4f}, std={x_scaled[:,:,0].std().item():.4f}")
        # print(f"feature 2 after scaling: min={x_scaled[:,:,1].min().item():.4f}, max={x_scaled[:,:,1].max().item():.4f}, mean={x_scaled[:,:,1].mean().item():.4f}, std={x_scaled[:,:,1].std().item():.4f}")
        # print(f"feature 3 after scaling: min={x_scaled[:,:,2].min().item():.4f}, max={x_scaled[:,:,2].max().item():.4f}, mean={x_scaled[:,:,2].mean().item():.4f}, std={x_scaled[:,:,2].std().item():.4f}")
        # exit(0)
        return x_scaled, y

def compute_stats(dataset, indices, batch_size=32):
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    
    n_samples = 0
    sum_x = None
    sum_sq_x = None
    
    for x, _ in loader:
        # x: (B, S, F, L)
        B, S, F, L = x.shape
        
        if sum_x is None:
            sum_x = torch.zeros(F)
            sum_sq_x = torch.zeros(F)
            
        # Mask for valid segments
        mask = (x.view(B, S, -1).abs().sum(dim=-1) > 1e-6)
        
        # x[mask] -> (N_valid, F, L)
        valid_x = x[mask]
        
        if valid_x.numel() == 0:
            continue
            
        # Reshape to (N_valid * L, F)
        valid_x_flat = valid_x.transpose(1, 2).reshape(-1, F)
        
        sum_x += valid_x_flat.sum(dim=0)
        sum_sq_x += (valid_x_flat ** 2).sum(dim=0)
        n_samples += valid_x_flat.shape[0]
        
    mean = sum_x / n_samples
    std = torch.sqrt(sum_sq_x / n_samples - mean ** 2)
    
    return mean, std


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
    use_scaler=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    if use_scaler:
        mean, std = compute_stats(dataset, train_indices, batch_size=batch_size)
        
        train_subset = ScaledDataset(Subset(dataset, train_indices), mean, std)
        test_subset = ScaledDataset(Subset(dataset, test_indices), mean, std)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(
            Subset(dataset, train_indices), batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            Subset(dataset, test_indices), batch_size=batch_size, shuffle=False
        )

    model = TAIL_MIL(
        segment_len=dataset.segment_len, embed_dim=embed_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    best_val_acc = -float("inf")
    best_val_f1 = 0
    best_val_auc = 0
    best_val_prec = 0
    best_val_rec = 0
    epochs_no_improve = 0
    best_model_state = None

    for epoch in tqdm(range(epochs), desc="Training TAIL-MIL"):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).float()
            preds, _, _ = model(X)
            preds = preds.squeeze()
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
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
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
        
        # Summary of epoch:
        tqdm.write(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}. Patience: {epochs_no_improve}/{patience} {"<- Best" if epochs_no_improve==0 else ""}")
        
        # Early stopping
        if epochs_no_improve >= patience:
            break

    # print("\n=== Train/Test Summary ===")
    # print(f"Val Acc: {best_val_acc:.4f}")
    # print(f"Val Prec: {best_val_prec:.4f}")
    # print(f"Val Rec: {best_val_rec:.4f}")
    # print(f"Val F1: {best_val_f1:.4f}")
    # print(f"Val AUC: {best_val_auc:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return best_val_acc, best_val_prec, best_val_rec, best_val_f1, model
