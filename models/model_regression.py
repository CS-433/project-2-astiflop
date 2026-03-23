import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

from base import BaseModel
from utils.cnn_features_extractor import CNNFeatureExtractor
from utils.gated_attention import GatedAttention

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
    def _forward_pass(self, model, batch_data, total_lengths, criterion, device):
        batch_data = batch_data.cpu()
        B, T_max, V, L = batch_data.shape
                
        # Sliding window training: Create staircase of segments and corresponding RUL targets
        X_staircase = []
        Y_staircase = []
        for i in range(B):
            full_trajectory = batch_data[i] # (T_max, V, L)
            T_actual = int(total_lengths[i].item())
            
            for t in range(1, T_actual + 1):
                segment_sequence = full_trajectory[:t] 
                rul_target = T_actual - t
                X_staircase.append(segment_sequence)
                Y_staircase.append(rul_target)

        X_padded = pad_sequence(X_staircase, batch_first=True).to(device) 
        targets = torch.tensor(Y_staircase, device=device).float()
        
        # attention mask
        indices = torch.arange(X_padded.size(1), device=device).expand(len(X_staircase), -1)
        lengths_tensor = torch.tensor([len(x) for x in X_staircase], device=device).unsqueeze(1)
        mask = (indices < lengths_tensor).float()

        # Forward pass
        preds, _, _ = model(X_padded, mask=mask) # Passer le masque à l'attention
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
                loss = self._forward_pass(model, batch_data, total_lengths, criterion, device)
                
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
                    loss = self._forward_pass(model, X, total_segment_len, criterion, device)
                    val_loss += loss.item()
            val_loss /= len(validation_loader)


            if val_loss < best_loss:
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
            
            # Summary of epoch:
            tqdm.write(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}. Patience: {epochs_no_improve}/{patience} {'<- Best' if epochs_no_improve==0 else ''}")
            
            # Early stopping
            if epochs_no_improve >= patience:
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return {"best_loss": best_loss}, model