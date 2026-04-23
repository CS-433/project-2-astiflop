import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm
import random
import time
import os

from .wrappers import TrainingWrapper, BenchmarkWrapper
from .utils.cnn_features_extractor import CNNFeatureExtractor
from .utils.gated_attention import GatedAttention

class RotaryTimeEmbedding(nn.Module):
    def __init__(self, embed_dim, max_time=1000000.0):
        super().__init__()
        self.embed_dim = embed_dim
        # Frequencies scaled to handle time up to max_time
        inv_freq = 1.0 / (max_time ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t):
        # t: (B, T, 1)
        sinusoid_inp = t * self.inv_freq # (B, T, embed_dim / 2)
        emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        return emb

# ---------------------------------------------------------
# MODEL 1: The Original BiLSTM
# ---------------------------------------------------------
class CNNBiLSTMMLPRegressor(nn.Module):
    def __init__(self, segment_len, embed_dim=512, dropout=0.3, feature_extractor_layers=1, bilstm_layers=1, use_time_encoding=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_extractor_layers = feature_extractor_layers
        self.bilstm_layers = bilstm_layers
        
        self.feature_extractors = nn.ModuleList([
            CNNFeatureExtractor(input_len=segment_len, embedding_dim=embed_dim)
            for _ in range(feature_extractor_layers)
        ])
        
        self.cnn_aggregation = nn.Linear(embed_dim * feature_extractor_layers, embed_dim)
        
        self.variate_attention = GatedAttention(dim=embed_dim, hidden_dim=embed_dim//4)
        
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
        
        # Time projection for Lifetime feature
        self.use_time_encoding = use_time_encoding
        if self.use_time_encoding:
            self.time_projection = RotaryTimeEmbedding(embed_dim, max_time=1500000.0)
            print(f"[WARNING] Model: Time encoding enabled. Ensure presence of Lifetime feature and that max_time is set appropriately for the scale of Lifetime values.")
        
        self.segment_attention = GatedAttention(dim=embed_dim, hidden_dim=embed_dim//4)
        
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim//2, 1)
        )

    def compute_orthogonality_loss(self):
        if self.bilstm_layers <= 1:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        weights = []
        for lstm in self.bilstms:
            w = torch.cat([p.flatten() for n, p in lstm.named_parameters() if 'weight' in n])
            w_norm = w / (torch.norm(w) + 1e-8) 
            weights.append(w_norm.unsqueeze(1))
            
        W = torch.cat(weights, dim=1) # (num_params, bilstm_layers)        
        corr = torch.matmul(W.t(), W) 
        
        eye = torch.eye(self.bilstm_layers, device=W.device)
        off_diag_corr = corr - eye
        ortho_loss = torch.sum(off_diag_corr ** 2)        
        return ortho_loss

    def forward(self, x, mask=None):
        # x shape: (B, T, V, L) where V includes Lifetime
        B, T, V, L = x.shape
        
        if self.use_time_encoding:
            # Separation: X, Y, Speed are features; Lifetime is the last channel
            x_features = x[:, :, :-1, :]  # (B, T, V-1, L)
            x_lifetime = x[:, :, -1, :]  # (B, T, L)
            V_feat = V - 1
        else:
            x_features = x
            V_feat = V

        # == Feature Extraction ==
        x_reshaped = x_features.reshape(B * T * V_feat, 1, L)
        
        # Extract features from each branch and concatenate
        extracted_features = []
        for fe in self.feature_extractors:
            extracted_features.append(fe(x_reshaped)) # (B*T*V_feat, embed_dim)
            
        features_cat = torch.cat(extracted_features, dim=-1) # (B*T*V_feat, embed_dim * feature_extractor_layers)
        features_agg = self.cnn_aggregation(features_cat) # (B*T*V_feat, embed_dim)
        
        features = features_agg.view(B * T, V_feat, self.embed_dim)   # (B*T, V_feat, embed_dim)

        # == Variate Attention (Fusion X, Y, Speed, etc) ==
        v_weights = self.variate_attention(features, mask=None)  # (B*T, V_feat, 1)
        seg_emb = torch.sum(features * v_weights, dim=1) # (B*T, embed_dim)
        seg_emb = seg_emb.view(B, T, self.embed_dim) # (B, T, embed_dim)
        
        if self.use_time_encoding:
            # == Time Encoding (from Lifetime) ==
            # Calculate a single scalar timestamp for each segment (average of Lifetime)
            time_scalar = x_lifetime.mean(dim=-1).unsqueeze(-1)  # (B, T, 1)
            time_emb = self.time_projection(time_scalar)  # (B, T, embed_dim)
            seg_emb = seg_emb + time_emb  # Inject chronological awareness
        
        # == BiLSTM ==
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
                lstm_out, _ = lstm(seg_emb)  # (B, T, embed_dim)
            lstm_outputs.append(lstm_out)
            
        lstm_cat = torch.cat(lstm_outputs, dim=-1) # (B, T, embed_dim * bilstm_layers)
        lstm_agg = self.lstm_aggregation(lstm_cat) # (B, T, embed_dim)
        
        # == Segment Attention ==
        s_weights = self.segment_attention(lstm_agg, mask=mask)
        context_vector = torch.sum(lstm_agg * s_weights, dim=1) # (B, embed_dim)
          
        # == Final Regression ==
        output = self.regressor(context_vector).squeeze(-1) # (B,)
        
        # == Orthogonality Loss ==
        ortho_loss = self.compute_orthogonality_loss()
        
        return output, s_weights, v_weights, ortho_loss

# ---------------------------------------------------------
# MODEL 2: The New PyTorch-Struct HMM
# ---------------------------------------------------------
class CNNHMMMLPRegressor(nn.Module):
    def __init__(self, segment_len, embed_dim=512, dropout=0.3, feature_extractor_layers=1, num_states=16, use_time_encoding=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_extractor_layers = feature_extractor_layers
        self.num_states = num_states
        
        # 1. Feature Extraction (Identical to BiLSTM)
        self.feature_extractors = nn.ModuleList([
            CNNFeatureExtractor(input_len=segment_len, embedding_dim=embed_dim)
            for _ in range(feature_extractor_layers)
        ])
        
        self.cnn_aggregation = nn.Linear(embed_dim * feature_extractor_layers, embed_dim)
        self.variate_attention = GatedAttention(dim=embed_dim, hidden_dim=embed_dim//4)
        
        self.use_time_encoding = use_time_encoding
        if self.use_time_encoding:
            self.time_projection = RotaryTimeEmbedding(embed_dim, max_time=1500000.0)
            
        # 2. HMM Components 
        # Transition matrix: (num_states, num_states)
        self.transition_logits = nn.Parameter(torch.randn(num_states, num_states))
        
        self.init_logits = nn.Parameter(torch.randn(num_states))
        # Maps CNN features to emission log-probabilities for each state
        self.emission_network = nn.Linear(embed_dim, num_states)
        
        # 3. Attention over HMM State Marginals
        self.segment_attention = GatedAttention(dim=num_states, hidden_dim=num_states//2)
        
        # 4. Final Regression Head
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_states, num_states//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_states//2, 1)
        )

    def forward(self, x, mask=None):
        B, T, V, L = x.shape
        
        if self.use_time_encoding:
            x_features = x[:, :, :-1, :]  
            x_lifetime = x[:, :, -1, :]  
            V_feat = V - 1
        else:
            x_features = x
            V_feat = V

        # --- CNN Feature Extraction ---
        x_reshaped = x_features.reshape(B * T * V_feat, 1, L)
        extracted_features = [fe(x_reshaped) for fe in self.feature_extractors]
            
        features_cat = torch.cat(extracted_features, dim=-1) 
        features_agg = self.cnn_aggregation(features_cat) 
        features = features_agg.view(B * T, V_feat, self.embed_dim)  

        v_weights = self.variate_attention(features, mask=None)  
        seg_emb = torch.sum(features * v_weights, dim=1).view(B, T, self.embed_dim) 
        
        if self.use_time_encoding:
            time_scalar = x_lifetime.mean(dim=-1).unsqueeze(-1)  
            seg_emb = seg_emb + self.time_projection(time_scalar)  
            
        # Get emission log-probabilities for the sequence: (B, T, num_states)
        emission_logits = self.emission_network(seg_emb)
        
        # Normalize transitions and inits to log-probabilities
        trans_log_probs = torch.log_softmax(self.transition_logits, dim=-1)
        init_log_probs = torch.log_softmax(self.init_logits, dim=-1)
        
        # Determine sequence lengths for torch_struct
        lengths = mask.sum(dim=1).long() if mask is not None else torch.full((B,), T, device=x.device, dtype=torch.long)
        
        # Initialize alpha values (log probabilities)
        alpha = torch.zeros(B, T, self.num_states, device=x.device)
        alpha[:, 0, :] = init_log_probs.unsqueeze(0) + emission_logits[:, 0, :]
        
        # Differentiable Forward Pass
        for t in range(1, T):
            prev_alpha = alpha[:, t-1, :].unsqueeze(2)  # (B, num_states, 1)
            trans = trans_log_probs.unsqueeze(0)        # (1, num_states, num_states)
            
            # logsumexp computes log( sum( exp(prev_alpha + trans) ) ) safely
            log_prob_prior = torch.logsumexp(prev_alpha + trans, dim=1) 
            alpha[:, t, :] = log_prob_prior + emission_logits[:, t, :]
            
        # 1. Negative Log-Likelihood (NLL) Loss for training
        batch_indices = torch.arange(B, device=x.device)
        seq_log_likelihood = torch.logsumexp(alpha[batch_indices, lengths - 1, :], dim=-1)
        nll_loss = -seq_log_likelihood.mean()
        
        # 2. State Marginals P(z_t | x_{1:t})
        # Softmax over alpha gives exact causal filtering probabilities for attention
        marginals = torch.softmax(alpha, dim=-1)
        
        # --- Final Regression using HMM States ---
        # Instead of LSTM hidden states, we apply attention over the HMM state probabilities
        s_weights = self.segment_attention(marginals, mask=mask)
        context_vector = torch.sum(marginals * s_weights, dim=1) # (B, num_states)
          
        output = self.regressor(context_vector).squeeze(-1) 
        
        return output, s_weights, v_weights, nll_loss


if __name__ == "__main__":
    # --- Dummy Data Example ---
    B, T, V, L = 4, 10, 4, 900
    dummy_input = torch.randn(B, T, V, L)
    
    print("\nTESTING CNN-BiLSTM REGRESSOR")
    model_lstm = CNNBiLSTMMLPRegressor(segment_len=L, embed_dim=128, feature_extractor_layers=3, bilstm_layers=2)
    output_lstm, s_weights_lstm, v_weights_lstm, ortho_loss = model_lstm(dummy_input)
    print(f"Output shape: {output_lstm.shape} (expected: ({B},)) ")
    print(f"Segment Attention Weights shape: {s_weights_lstm.shape} (expected: ({B}, {T}, 1))")
    print(f"Variate Attention Weights shape: {v_weights_lstm.shape} (expected: ({B*T}, {V-1}, 1))")
    print(f"Ortho Loss shape: {ortho_loss.shape} - Value: {ortho_loss.item()}")
    print(f"NaN detected in the output: {torch.isnan(output_lstm).any()}\n")


    print("\nTESTING CNN-HMM REGRESSOR")
    # Using num_states=16 instead of bilstm_layers
    model_hmm = CNNHMMMLPRegressor(segment_len=L, embed_dim=128, feature_extractor_layers=3, num_states=16)
    output_hmm, s_weights_hmm, v_weights_hmm, nll_loss = model_hmm(dummy_input)
    print(f"Output shape: {output_hmm.shape} (expected: ({B},)) ")
    print(f"Segment Attention Weights shape: {s_weights_hmm.shape} (expected: ({B}, {T}, 1))")
    print(f"Variate Attention Weights shape: {v_weights_hmm.shape} (expected: ({B*T}, {V-1}, 1))")
    print(f"NLL (Partition) Loss shape: {nll_loss.shape} - Value: {nll_loss.item()}")
    print(f"NaN detected in the output: {torch.isnan(output_hmm).any()}")

# ---------------------------------------------------------
# UNIFIED WRAPPERS
# ---------------------------------------------------------
class RegressorBenchmarkWrapper(BenchmarkWrapper):
    def load(self, path):
        model_type = self.params.get("model_type", "bilstm")
        embed_dim = self.params.get("embed_dim", 64)
        segment_len = self.params.get("segment_len", 900)
        feature_extractor_layers = self.params.get("feature_extractor_layers", 1)
        use_time_encoding = self.params.get("use_time_encoding", True)
        device = self.params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        if model_type == "hmm":
            num_states = self.params.get("num_states", 16)
            self.model = CNNHMMMLPRegressor(segment_len=segment_len, embed_dim=embed_dim, feature_extractor_layers=feature_extractor_layers, num_states=num_states, use_time_encoding=use_time_encoding).to(device)
        else:
            bilstm_layers = self.params.get("bilstm_layers", 1)
            self.model = CNNBiLSTMMLPRegressor(segment_len=segment_len, embed_dim=embed_dim, feature_extractor_layers=feature_extractor_layers, bilstm_layers=bilstm_layers, use_time_encoding=use_time_encoding).to(device)
            
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.eval()

    def benchmark(self, test_loader):
        device = self.params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        max_segment_number = 150 # Set in the dataset
        
        all_trajectory_preds = []
        all_trajectory_vars = []
        
        with torch.no_grad():
            for X, _, total_segment_len in test_loader:
                B, T_max, V, L = X.shape
                X = X.cpu()
                
                for i in range(B):
                    T_actual = int(total_segment_len[i].item())
                    full_trajectory = X[i]
                    
                    X_staircase = []
                    for t in range(1, T_actual + 1):
                        X_staircase.append(full_trajectory[:t])
                        
                    # Batch prediction for this single trajectory
                    X_padded = pad_sequence(X_staircase, batch_first=True).to(device)
                    indices = torch.arange(X_padded.size(1), device=device).expand(len(X_staircase), -1)
                    lengths_tensor = torch.tensor([len(x) for x in X_staircase], device=device).unsqueeze(1)
                    mask = (indices < lengths_tensor).float()
                    
                    trajectory_preds, _, _, _ = self.model(X_padded, mask=mask)
                    trajectory_preds = trajectory_preds.cpu().numpy()

                    # Denormalize predictions to true RUL scale (number of segments)
                    trajectory_preds = trajectory_preds * (max_segment_number / 3.0)
                    
                    # Variance estimation with heuristic
                    # during the estimated healthy phase (predicted RUL > 45), we can assume high uncertainty. 
                    # during the critical phase (predicted RUL <= 45), we can assume lower uncertainty. 
                    trajectory_vars = [10.0 if pred > 45 else 2.0 for pred in trajectory_preds]

                    all_trajectory_preds.append(trajectory_preds)
                    all_trajectory_vars.append(trajectory_vars)
                    
        return {
            "predictions": all_trajectory_preds,
            "variances": all_trajectory_vars,
            "interpretability_score": 7.5
        }


class RegressorTrainingWrapper(TrainingWrapper): 
    def _forward_pass(self, model, batch_data, total_lengths, criterion, comparison_criterion, device, max_segment_number, is_training=True):
        B, T_max, V, L = batch_data.shape
        batch_data = batch_data.cpu()
        
        # This acts as Ortho Beta for BiLSTM, or NLL Beta for the HMM
        aux_beta = self.params.get("aux_beta", self.params.get("ortho_beta", 0.01))

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
        preds, _, _, aux_loss = model(X_padded, mask=mask)
        loss = criterion(preds, targets) + aux_beta * aux_loss
        
        if not is_training and comparison_criterion is not None:
            comparison_loss = comparison_criterion(preds, targets)
            return loss, comparison_loss

        return loss

    def train_on_fold(self, training_loader, validation_loader):
        model_type = self.params.get("model_type", "bilstm")
        name = self.params.get("name", f"{model_type}_regressor")
        lr = self.params.get("lr", 1e-4)
        embed_dim = self.params.get("embed_dim", 64)
        epochs = self.params.get("epochs", 100)
        patience = self.params.get("patience", 10)
        segment_len = self.params.get("segment_len", 900)
        loss_type = self.params.get("loss", "mse")
        feature_extractor_layers = self.params.get("feature_extractor_layers", 1)
        use_time_encoding = self.params.get("use_time_encoding", True)
        device = self.params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        max_segment_number = 150 

        # Unified Instantiation
        if model_type == "hmm":
            num_states = self.params.get("num_states", 16)
            model = CNNHMMMLPRegressor(segment_len=segment_len, embed_dim=embed_dim, feature_extractor_layers=feature_extractor_layers, num_states=num_states, use_time_encoding=use_time_encoding).to(device)
        else:
            bilstm_layers = self.params.get("bilstm_layers", 1)
            model = CNNBiLSTMMLPRegressor(segment_len=segment_len, embed_dim=embed_dim, feature_extractor_layers=feature_extractor_layers, bilstm_layers=bilstm_layers, use_time_encoding=use_time_encoding).to(device) 
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        if loss_type == "mse":
            criterion = nn.MSELoss()
        elif loss_type == "mae":
            criterion = nn.L1Loss()
        elif loss_type == "huber":
            criterion = nn.SmoothL1Loss()

        best_loss = float('inf')
        comparison_criterion = nn.MSELoss()
        best_comparison_loss = float('inf')

        epochs_no_improve = 0
        best_model_state = None

        for epoch in tqdm(range(epochs), desc=f"Training {model.__class__.__name__}"):
            model.train()
            train_loss = 0.0

            for batch_data, _, total_lengths in training_loader:
                loss = self._forward_pass(model, batch_data, total_lengths, criterion, None, device, max_segment_number=max_segment_number, is_training=True)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(training_loader)
            
            
            # Validation
            model.eval()
            val_loss = 0.0
            comparison_loss = 0.0
            with torch.no_grad():
                for X, _, total_segment_len in validation_loader:
                    loss, comparison = self._forward_pass(model, X, total_segment_len, criterion, comparison_criterion, device, max_segment_number=max_segment_number, is_training=False)
                    val_loss += loss.item()
                    comparison_loss += comparison.item()
            val_loss /= len(validation_loader)
            comparison_loss /= len(validation_loader)


            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1

            if comparison_loss < best_comparison_loss:
                best_comparison_loss = comparison_loss
            
            # Summary of epoch:
            if epoch % 10 == 0:  # Print every 10 epochs
                tqdm.write(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Comparison Loss: {comparison_loss:.4f}. Patience: {epochs_no_improve}/{patience} {'<- Best' if epochs_no_improve==0 else ''}")
            
            # Early stopping
            if epochs_no_improve >= patience:
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            datetime_str = time.strftime("%H-%M")
            os.makedirs("ckpts", exist_ok=True)
            torch.save(model.state_dict(), f"ckpts/best_{name}_{datetime_str}.pth")
            print(f"Best model saved with comparison loss: {best_comparison_loss:.4f} at time {datetime_str}")

        return {"best_loss": best_loss, "comparison_loss": best_comparison_loss}, model