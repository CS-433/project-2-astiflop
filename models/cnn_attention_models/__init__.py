"""
CNN-attention regression architecture:
- CNN-based feature extraction of several layers aggregated using a linear layer
- A variate-level attention mechanism that produces a single embedding per segment
- A temporal model that treats the sequence of segment embeddings and produces a new one
- A segment-level attention mechanism that produces a single embedding for the entire sample
- A final MLP regression head that produces the output

Model construction and pipeline utilities live in utils/train_utils/.
"""
