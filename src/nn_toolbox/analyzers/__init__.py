"""Architecture-specific analyzers for specialized model families."""

from nn_toolbox.analyzers.cnn import CNNAnalyzer
from nn_toolbox.analyzers.transformer import TransformerAnalyzer

__all__ = [
    "CNNAnalyzer",
    "TransformerAnalyzer",
]
