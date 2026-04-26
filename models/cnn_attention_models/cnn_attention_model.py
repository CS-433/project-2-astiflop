import torch
import torch.nn as nn
from models.building_blocs.cnn_features_extractor import CNNFeatureExtractor
from models.building_blocs.gated_attention import GatedAttention
from models.building_blocs.time_embedding import SinusoidalTimeEmbedding
from models.building_blocs.bilstm_temporal import BiLSTMTemporal
from models.building_blocs.hmm_temporal import HMMTemporal


class CNNAttentionRegressor(nn.Module):
    def __init__(self, segment_len, embed_dim=512, dropout=0.3, feature_extractor_layers=1, temporal_type="bilstm", temporal_params=None, use_time_encoding=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_extractor_layers = feature_extractor_layers
        self.temporal_type = temporal_type
        
        temporal_params = temporal_params or {}
        
        # 1. Feature Extraction
        self.feature_extractors = nn.ModuleList([
            CNNFeatureExtractor(input_len=segment_len, embedding_dim=embed_dim)
            for _ in range(feature_extractor_layers)
        ])
        
        self.cnn_aggregation = nn.Linear(embed_dim * feature_extractor_layers, embed_dim)
        
        # 2. Variate Attention
        self.variate_attention = GatedAttention(dim=embed_dim, hidden_dim=embed_dim//4)
        
        # Time Encoding
        self.use_time_encoding = use_time_encoding
        if self.use_time_encoding:
            self.time_projection = SinusoidalTimeEmbedding(embed_dim, max_time=1500000.0)
            
        # 3. Temporal Model
        if self.temporal_type == "bilstm":
            bilstm_layers = temporal_params.get("bilstm_layers")
            self.temporal_model = BiLSTMTemporal(embed_dim, bilstm_layers)
            temporal_out_dim = embed_dim
        elif self.temporal_type == "hmm":
            num_states = temporal_params.get("num_states")
            self.temporal_model = HMMTemporal(embed_dim, num_states)
            temporal_out_dim = num_states
        else:
            raise ValueError(f"Unknown temporal type: {self.temporal_type}")
        
        # 4. Segment Attention
        self.segment_attention = GatedAttention(dim=temporal_out_dim, hidden_dim=temporal_out_dim//4 if self.temporal_type == "bilstm" else temporal_out_dim//2)
        
        # 5. Final Regression Head
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(temporal_out_dim, temporal_out_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_out_dim//2, 1)
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

        # --- Feature Extraction ---
        x_reshaped = x_features.reshape(B * T * V_feat, 1, L)
        extracted_features = [fe(x_reshaped) for fe in self.feature_extractors]
            
        features_cat = torch.cat(extracted_features, dim=-1) 
        features_agg = self.cnn_aggregation(features_cat) 
        features = features_agg.view(B * T, V_feat, self.embed_dim)  

        # --- Variate Attention ---
        v_weights = self.variate_attention(features, mask=None)  
        seg_emb = torch.sum(features * v_weights, dim=1).view(B, T, self.embed_dim) 
        
        if self.use_time_encoding:
            time_scalar = x_lifetime.mean(dim=-1).unsqueeze(-1)  
            seg_emb = seg_emb + self.time_projection(time_scalar)  
            
        # --- Temporal Model ---
        temporal_out, aux_loss = self.temporal_model(seg_emb, mask=mask)
        
        # --- Segment Attention ---
        s_weights = self.segment_attention(temporal_out, mask=mask)
        context_vector = torch.sum(temporal_out * s_weights, dim=1) 
          
        # --- Final Regression ---
        output = self.regressor(context_vector).squeeze(-1) 
        
        return output, s_weights, v_weights, aux_loss