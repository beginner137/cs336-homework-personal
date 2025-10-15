import torch
import torch.nn as nn
import math



class Linear(nn.Module):
    def __init__(self, out_features, in_features, device=None, dtype=None):
        super().__init__()

        sigma = math.sqrt(2/(in_features + out_features))
        weight = torch.empty(in_features, out_features)
        self.bias = torch.zeros(out_features)
        self.weight = nn.Parameter(nn.init.trunc_normal_(
            weight, mean=0, std=sigma, a=-3*sigma, b=3*sigma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.T) + self.bias


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

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms * self.weights).to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w3 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def forward(self, x):
        silu_w1x = self.w1(x)*torch.sigmoid(self.w1(x))
        w3x = self.w3(x)
        gated = silu_w1x * w3x
        return self.w2(gated)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device

        base_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))

        positions = torch.arange(
            max_seq_len, dtype=torch.float).unsqueeze(1)
        angles = positions * base_freq
        self.register_buffer('sin_cache', torch.sin(angles))
        self.register_buffer('cos_cache', torch.cos(angles))


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        sin = self.sin_cache[token_positions]
        cos = self.cos_cache[token_positions]
        # x1, x2 = x[..., 0::2], x[..., 1::2]
        # rotated_x1 = x1 * cos - x2 * sin
        # rotated_x2 = x1 * sin + x2 * cos

        # stacked_results = torch.stack((rotated_x1, rotated_x2), dim=-1)
        # return stacked_results.flatten(-2)

        # use complex number multiplication
        x_complex = torch.view_as_complex(
            x.float().reshape(*x.shape[:-1], -1, 2))
        rotations_complex = torch.complex(cos, sin)

        x_rotated_complex = x_complex * rotations_complex
        x_rotated = torch.view_as_real(x_rotated_complex)

        return x_rotated.flatten(-2).type_as(x)


def softmax(in_features: torch.Tensor, dim: int):
    max_val = torch.max(in_features, dim=dim, keepdim=True).values
    shifted_features = in_features - max_val
    exps = torch.exp(shifted_features)
    sum_exps = torch.sum(exps, dim=dim, keepdim=True)
    return exps / sum_exps


def scaled_dot_product_attention(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor = None):
    assert queries.shape[-1] == keys.shape[-1]
    d_k = queries.shape[-1]
    squared_d_k = torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    scores = queries @ keys.transpose(-2, -1)
    scores = scores / squared_d_k
    if mask is not None:
        scores = scores.masked_fill(mask == False, -1e9)
    attention_weights = softmax(scores, dim=-1)
    output = attention_weights @ values
    return output




class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 max_seq_len: int = 2048,
                 theta: float = 10000.0,
                 use_rope: bool = False,
                 device=None,
                 dtype=None):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.o_proj = Linear(num_heads * self.d_v, d_model,
                             device=device, dtype=dtype)

        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len)
        else:
            self.rope = None

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len,
                       dtype=torch.bool), diagonal=1),
            persistent=False
        )

    def forward(self,
                x: torch.Tensor,
                token_positions: torch.Tensor = None) -> torch.Tensor:
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]

        # Input shape: (batch_shape ..., seq_len, d_model)
        # Output shape: (batch_shape ..., seq_len, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # (batch shape ..., seq_len, d_model) -> (batch_shape..., seq_len, num_heads, d_k)
        Q = Q.view(*batch_shape, seq_len, self.num_heads,
                   self.d_k).transpose(-3, -2)
        K = K.view(*batch_shape, seq_len, self.num_heads,
                   self.d_k).transpose(-3, -2)
        V = V.view(*batch_shape, seq_len, self.num_heads,
                   self.d_k).transpose(-3, -2)

        if self.use_rope:
            if token_positions is not None:
                token_positions = torch.arange(seq_len)

            # if not token_positions:
            Q = self.rope(Q, token_positions=token_positions)
            K = self.rope(K, token_positions=token_positions)

        mask = self.causal_mask[:seq_len, :seq_len]
        mask = ~mask

        # attention_output shape: (batch_shape..., num_heads, seq_len, d_k)
        attention_output = scaled_dot_product_attention(
            Q, K, V, mask)

        # Transpose back to: (batch_shape..., seq_len, num_heads, d_v)
        attention_output = attention_output.transpose(-3, -2)

        concatenated_output = attention_output.contiguous().view(
            *batch_shape, seq_len, self.d_model)

        final_output = self.o_proj(concatenated_output)
        return final_output
