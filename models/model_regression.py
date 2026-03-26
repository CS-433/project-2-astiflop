import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

from .base import BaseModel
from .utils.cnn_features_extractor import CNNFeatureExtractor
from .utils.gated_attention import GatedAttention

import random

class CNNBiLSTMMLPRegressor(nn.Module):
    def __init__(self, segment_len, embed_dim=512, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.feature_extractor = CNNFeatureExtractor(input_len=segment_len, embedding_dim=embed_dim)
        
        self.variate_attention = GatedAttention(dim=embed_dim, hidden_dim=embed_dim//4)
        
        self.bilstm = nn.LSTM(
            input_size=embed_dim, 
            hidden_size=embed_dim//2, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.segment_attention = GatedAttention(dim=embed_dim, hidden_dim=embed_dim//4)
        
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim//2, 1)
        )

    def forward(self, x, mask=None):
        # x shape: (B, T, V, L)
        B, T, V, L = x.shape

        # == Feature Extraction ==
        x_reshaped = x.view(B * T * V, 1, L)
        features = self.feature_extractor(x_reshaped)  # (B*T*V, embed_dim)
        features = features.view(B * T, V, self.embed_dim)   # (B*T, V, embed_dim)

        # == Variate Attention (Fusion X, Y, Speed) ==
        v_weights = self.variate_attention(features, mask=None)  # (B*T, V, 1)
        seg_emb = torch.sum(features * v_weights, dim=1) # (B*T, embed_dim)
        seg_emb = seg_emb.view(B, T, self.embed_dim) # (B, T, embed_dim)
        
        # == BiLSTM ==
        if mask is not None:
            lengths = mask.sum(dim=1).cpu().to(torch.int64)
            packed_emb = torch.nn.utils.rnn.pack_padded_sequence(
                seg_emb, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out_packed, _ = self.bilstm(packed_emb)
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                lstm_out_packed, batch_first=True, total_length=T
            )
        else:
            lstm_out, _ = self.bilstm(seg_emb)  # (B, T, embed_dim)
        
        # == Segment Attention ==
        s_weights = self.segment_attention(lstm_out, mask=mask)
        context_vector = torch.sum(lstm_out * s_weights, dim=1) # (B, embed_dim)
          
        # == Final Regression ==
        output = self.regressor(context_vector).squeeze(-1) # (B,)
        
        return output, s_weights, v_weights



if __name__ == "__main__":
    # --- Dummy Data Example ---
    B, T, V, L = 4, 10, 3, 900
    dummy_input = torch.randn(B, T, V, L)
    model = CNNBiLSTMMLPRegressor(segment_len=L, embed_dim=128)
    output, s_weights, v_weights = model(dummy_input)
    print(f"Output shape: {output.shape} (expected: ({B},)) ")
    print(f"Segment Attention Weights shape: {s_weights.shape} (expected: ({B}, {T}, 1))")
    print(f"Variate Attention Weights shape: {v_weights.shape} (expected: ({B*T}, {V}, 1))")
    print(f"Nan detected in the output: {torch.isnan(output).any()}")
    



class RegressorWrapper(BaseModel): 
    def _forward_pass(self, model, batch_data, total_lengths, criterion, device, max_segment_number, is_training=True):
        B, T_max, V, L = batch_data.shape
        batch_data = batch_data.cpu()

        X_staircase = []
        Y_staircase = []
        
        # Sampling parameters
        num_samples_train = 4
        val_stride = 10         # Striding in validation for reproducibility
        
        for i in range(B):
            T_actual = int(total_lengths[i].item())
            full_trajectory = batch_data[i] # (T_max, V, L) sur CPU
            
            if is_training:
                if T_actual <= num_samples_train:
                    indices = list(range(1, T_actual + 1))
                else:
                    indices = random.sample(range(1, T_actual + 1), num_samples_train)
            else:
                indices = list(range(1, T_actual + 1, val_stride))
                if indices[-1] != T_actual: indices.append(T_actual)

            for t in indices:
                y = min(T_actual - t, max_segment_number//3) # Reduce difficulty of the task
                y = 3*float(y)/max_segment_number # Normalized between 0 and 1 for easier gradients computations
                X_staircase.append(full_trajectory[:t]) 
                Y_staircase.append(y) 

        X_padded = pad_sequence(X_staircase, batch_first=True).to(device)
        targets = torch.tensor(Y_staircase, device=device).float()
        
        # Attention mask
        indices = torch.arange(X_padded.size(1), device=device).expand(len(X_staircase), -1)
        lengths_tensor = torch.tensor([len(x) for x in X_staircase], device=device).unsqueeze(1)
        mask = (indices < lengths_tensor).float()

        # Forward pass
        preds, _, _ = model(X_padded, mask=mask)
        loss = criterion(preds, targets)

        return loss


    def train_on_fold(self, training_loader, validation_loader):
        lr = self.params.get("lr", 1e-4)
        embed_dim = self.params.get("embed_dim", 64)
        epochs = self.params.get("epochs", 100)
        patience = self.params.get("patience", 10)
        segment_len = self.params.get("segment_len", 900)
        loss = self.params.get("loss", "mse")
        device = self.params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        max_segment_number = 150 # Set in the dataset

        model = CNNBiLSTMMLPRegressor(segment_len=segment_len, embed_dim=embed_dim).to(device) 
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if loss == "mse":
            criterion = nn.MSELoss()
        elif loss == "mae":
            criterion = nn.L1Loss()
        elif loss == "huber":
            criterion = nn.SmoothL1Loss()

        best_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in tqdm(range(epochs), desc=f"Training {model.__class__.__name__}"):
            model.train()
            train_loss = 0.0

            for batch_data, _, total_lengths in training_loader:
                loss = self._forward_pass(model, batch_data, total_lengths, criterion, device, max_segment_number=max_segment_number, is_training=True)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(training_loader)
            
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X, _, total_segment_len in validation_loader:
                    loss = self._forward_pass(model, X, total_segment_len, criterion, device, max_segment_number=max_segment_number, is_training=False)
                    val_loss += loss.item()
            val_loss /= len(validation_loader)


            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
            
            # Summary of epoch:
            if epoch % 10 == 0:  # Print every 10 epochs
                tqdm.write(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}. Patience: {epochs_no_improve}/{patience} {'<- Best' if epochs_no_improve==0 else ''}")
            
            # Early stopping
            if epochs_no_improve >= patience:
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            torch.save(model.state_dict(), f"best_regressor_model.pth")
            print(f"Best model saved with validation loss: {best_loss:.4f}")

        return {"best_loss": best_loss}, model