import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (self.head_dim * num_heads == embed_size), "Embedding size must be divisible by heads"

        self.value = nn.Linear(embed_size, embed_size, bias=False)
        self.key = nn.Linear(embed_size, embed_size, bias=False)
        self.query = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x, context):
        N = x.shape[0]
        value_len, key_len, query_len = context.shape[1], context.shape[1], x.shape[1]

        value = self.value(context)
        key = self.key(context)
        query = self.query(x)

        # Split into heads
        value = value.view(N, value_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(N, key_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        query = query.view(N, query_len, self.heads, self.head_dim).permute(0, 2, 1, 3)

        # Calculate attention scores
        energy = torch.einsum('nqhd,nkhd->nhqk', [query, key])
        attention = torch.softmax(energy, dim=3)

        # Weighted sum of values
        out = torch.einsum('nhql,nlhd->nqhd', [attention, value]).reshape(N, query_len, self.heads * self.head_dim)

        return self.fc_out(out)