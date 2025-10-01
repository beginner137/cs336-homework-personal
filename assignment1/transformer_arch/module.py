import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        sigma = math.sqrt(2/(in_features + out_features))
        weight = torch.empty(in_features, out_features)
        bias = torch.empty(out_features)
        self.weight = nn.Parameter(nn.init.trunc_normal_(
            weight, mean=0, std=sigma, a=-3*sigma, b=3*sigma))
        self.bias = nn.Parameter(nn.init.trunc_normal_(
            bias, mean=0, std=sigma, a=-3*sigma, b=3*sigma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight) + self.bias


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        embedding = torch.empty(num_embeddings, embedding_dim)
        sigma = 1
        self.embedding = nn.Parameter(nn.init.trunc_normal_(
            embedding, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        d_model = self.embedding.shape[1]
        output = torch.zeros(batch_size, seq_len, d_model)

        # for b in range(batch_size):
        #     for s in range(seq_len):
        #         token_id = token_ids[b, s]
        #         output[b, s] = self.embedding[token_id]
        output = self.embedding[token_ids]
        return output
