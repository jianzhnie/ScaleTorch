import torch
from torch import nn as nn

class MutiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MutiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Projection matrices for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_state, attention_mask=None):
        batch_size = hidden_state.size()[0]

        query = self.q_proj(hidden_state)
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)
        
        query = self.split_head(query)
        key = self.split_head(key)
        value = self.split_head(value)
        
        # Compute scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))
        
        if attention_mask != None:
            attention_scores += attention_mask * -1e-9
        
        attention_probs = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, value)
        
        # Reshape and apply output projection
        output = output.transpose(-1, -2).contiguous().view(batch_size, -1, self.head_dim * self.num_heads)
        output = self.o_proj(output)
        
        return output

        
    def split_head(self, x):
        batch_size = x.size()[0]
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)