"""
Example 6: "My Transformer representation seems collapsed."

Scenario:
In an embedding, self-supervised, or transformer model, distinct input tokens or samples
all map to nearly the same vectors (dimensional collapse or representation collapse).
nn-toolbox computes:
- Pairwise cosine similarity matrix across samples
- Singular values and effective rank (Roy & Vetterli)
- Attention entropy distributions
"""

import torch
import torch.nn as nn
from nn_toolbox.analysis.similarity import analyze_representation_collapse
from nn_toolbox.analyzers.transformer import TransformerAnalyzer


def main():
    print("--- Example 6: Diagnosing Representation Collapse & Attention Collapse ---")

    # 1. Simulating collapsed representations: all vectors pointing along the same axis
    batch_size = 8
    feature_dim = 32
    base_direction = torch.randn(1, feature_dim)
    base_direction = base_direction / base_direction.norm()
    # Tiny noise added -> all samples share 99% similarity
    collapsed_features = base_direction.repeat(batch_size, 1) + torch.randn(batch_size, feature_dim) * 0.01

    print("\n1. Analyzing Representation Similarity & Effective Rank:")
    collapse_info = analyze_representation_collapse(collapsed_features)
    print(f"  Is Collapsed: {collapse_info['is_collapsed']}")
    print(f"  Mean Pairwise Cosine Similarity: {collapse_info['mean_pairwise_similarity']:.4f}")
    print(f"  Effective Rank: {collapse_info['effective_rank']:.2f} (max possible: {collapse_info['sample_count']})")
    print(f"  Rank Ratio: {collapse_info['rank_ratio']:.4f}")

    # 2. Simulating collapsed attention weights
    print("\n2. Analyzing Attention Entropy:")
    dummy_model = nn.Linear(10, 10)
    tf_analyzer = TransformerAnalyzer(dummy_model)
    # One-hot attention matrix (always attending to token 0)
    collapsed_attn = torch.zeros(2, 4, 16, 16)
    collapsed_attn[:, :, :, 0] = 1.0  # Token 0 absorbs all attention
    attn_info = tf_analyzer.analyze_attention_weights(collapsed_attn)
    print(f"  Mean Entropy: {attn_info['mean_entropy']:.4f} (max possible: {attn_info['max_possible_entropy']:.4f})")
    print(f"  Entropy Ratio: {attn_info['entropy_ratio']:.4f}")
    for f in attn_info["findings"]:
        print(f"  [{f.severity.upper()}] {f.observation}")
        print(f"    Hypotheses: {f.hypotheses}")


if __name__ == "__main__":
    main()
