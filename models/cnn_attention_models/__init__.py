"""
Those models are based on a common architecture:
- CNN-based feature extraction of several layers aggregated using a linear layer
- A variate-level attention mechanism that produces a single embedding per segment
- A temporal model that takes treats the sequence of segment embeddings and produces a new one
- A segment-level attention mechanism that produces a single embedding for the entire sample from 
    the sequence of segment embeddings
- A final MLP regression head that produces the output
"""