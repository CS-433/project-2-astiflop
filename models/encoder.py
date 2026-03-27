# This model is a little bit special and is not meant to run in the same way as the others.
# This model is meant to be trained as an encoder to make Time Series Representation Learning. 
# It is trained on the whole dataset without validation.

from .utils.cnn_features_extractor import CNNFeatureExtractor

