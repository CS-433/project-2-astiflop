import torch
import torch.nn as nn

class HMMTemporal(nn.Module):
    def __init__(self, embed_dim, num_states):
        super().__init__()
        self.num_states = num_states
        
        # Transition matrix: (num_states, num_states)
        self.transition_logits = nn.Parameter(torch.randn(num_states, num_states))
        
        self.init_logits = nn.Parameter(torch.randn(num_states))
        # Maps CNN features to emission log-probabilities for each state
        self.emission_network = nn.Linear(embed_dim, num_states)

    def forward(self, seg_emb, mask=None):
        B, T, _ = seg_emb.shape
        
        # Get emission log-probabilities for the sequence: (B, T, num_states)
        emission_logits = self.emission_network(seg_emb)
        
        # Normalize transitions and inits to log-probabilities
        trans_log_probs = torch.log_softmax(self.transition_logits, dim=-1)
        init_log_probs = torch.log_softmax(self.init_logits, dim=-1)
        
        # Determine sequence lengths
        lengths = mask.sum(dim=1).long() if mask is not None else torch.full((B,), T, device=seg_emb.device, dtype=torch.long)
        
        # Initialize alpha values (log probabilities)
        alpha = torch.zeros(B, T, self.num_states, device=seg_emb.device)
        alpha[:, 0, :] = init_log_probs.unsqueeze(0) + emission_logits[:, 0, :]
        
        # Differentiable Forward Pass
        for t in range(1, T):
            prev_alpha = alpha[:, t-1, :].unsqueeze(2)  # (B, num_states, 1)
            trans = trans_log_probs.unsqueeze(0)        # (1, num_states, num_states)
            
            # logsumexp computes log( sum( exp(prev_alpha + trans) ) ) safely
            log_prob_prior = torch.logsumexp(prev_alpha + trans, dim=1) 
            alpha[:, t, :] = log_prob_prior + emission_logits[:, t, :]
            
        # Negative Log-Likelihood (NLL) Loss
        batch_indices = torch.arange(B, device=seg_emb.device)
        seq_log_likelihood = torch.logsumexp(alpha[batch_indices, lengths - 1, :], dim=-1)
        nll_loss = -seq_log_likelihood.mean()
        
        # State Marginals P(z_t | x_{1:t})
        marginals = torch.softmax(alpha, dim=-1)
        
        return marginals, nll_loss