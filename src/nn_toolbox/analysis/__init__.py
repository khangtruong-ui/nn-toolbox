"""Analysis functions and statistical engines."""

from nn_toolbox.analysis.distributions import (
    analyze_saturation,
    compute_approximate_histogram,
    detect_dead_features,
)
from nn_toolbox.analysis.sensitivity import compute_empirical_sensitivity
from nn_toolbox.analysis.similarity import (
    analyze_representation_collapse,
    compute_effective_rank,
    compute_gradient_cosine_similarity,
    compute_pairwise_cosine_similarity,
    compute_vector_cosine_similarity,
)
from nn_toolbox.analysis.statistics import (
    WelfordAccumulator,
    compute_tensor_stats,
)

__all__ = [
    "compute_tensor_stats",
    "WelfordAccumulator",
    "compute_approximate_histogram",
    "analyze_saturation",
    "detect_dead_features",
    "compute_vector_cosine_similarity",
    "compute_gradient_cosine_similarity",
    "compute_pairwise_cosine_similarity",
    "compute_effective_rank",
    "analyze_representation_collapse",
    "compute_empirical_sensitivity",
]
