import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv, GCNConv, GINConv, RGCNConv
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from torch.nn import Parameter as Param
import numpy as np
import math
class DiagLayer(torch.nn.Module):
    def __init__(self, in_dim, num_et=1):
        super(DiagLayer, self).__init__()
        self.num_et = num_et
        self.in_dim = in_dim
        self.weight = Param(torch.Tensor(num_et, in_dim))

        self.reset_parameters()

    def forward(self, x):
        # print(self.weight)
        value = x * self.weight
        return value

    def reset_parameters(self):
        self.weight.data.normal_(std=1/np.sqrt(self.in_dim))
        # self.weight.data.fill_(1)
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(0.1)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        input_tensor=input_tensor.unsqueeze(0)
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = hidden_states.squeeze(0)
        return hidden_states
class GAT3(torch.nn.Module):
    def __init__(self, input_dim=109, input_dim_e=243, output_dim=200, output_dim_e=64, dropout=0.2, heads=10):
        super(GAT3, self).__init__()

        #self.att1 = SelfAttention(num_attention_heads=1,input_size=109,hidden_size=109,hidden_dropout_prob=0.1)
        #self.att2 = SelfAttention(num_attention_heads=8, input_size=2000, hidden_size=2000, hidden_dropout_prob=0.1)
        self.att5 = SelfAttention(num_attention_heads=10, input_size=200, hidden_size=200, hidden_dropout_prob=0.1)
        # graph layers : drug
        self.gcn1 = GATConv(input_dim, 200,heads=heads,dropout=dropout)
        self.gcn2 = GATConv(200 * heads, output_dim, heads=heads,dropout=dropout)
        self.gcn5 = GATConv(output_dim * heads, output_dim, dropout=dropout)

        self.fc_g1 = nn.Linear(output_dim, output_dim)
        self.fc_g2 = nn.Linear(output_dim, output_dim)

        #self.att3 = SelfAttention(num_attention_heads=9, input_size=243, hidden_size=243, hidden_dropout_prob=0.1)
        #self.att4 = SelfAttention(num_attention_heads=10, input_size=2000, hidden_size=2000, hidden_dropout_prob=0.1)
        self.att6 = SelfAttention(num_attention_heads=8, input_size=200, hidden_size=200, hidden_dropout_prob=0.1)
        # # graph layers : sideEffect
        self.gcn3 = GATConv(input_dim_e, 200, heads=heads, dropout=dropout)
        self.gcn4 = GATConv(200 * heads, output_dim,heads=heads, dropout=dropout)
        self.gcn6 = GATConv(output_dim * heads, output_dim, dropout=dropout)
        self.fc_g3 = nn.Linear(output_dim, output_dim)
        self.fc_g4 = nn.Linear(output_dim, output_dim)

        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        # activation and regularization
        self.relu = nn.ReLU()
        self.diag = DiagLayer(in_dim=output_dim)

    def forward(self, x, edge_index,batch,data_e, DF=False, not_FC=True):
        # graph input feed-forward

        x_e, edge_index_e = data_e.x, data_e.edge_index

        # 药物
        #x = self.att1(x)
        x = self.leaky_relu(self.gcn1(x, edge_index))
        #x = self.att5(x)
        x = self.leaky_relu(self.gcn2(x,edge_index))
        #x = self.att1(x)
        #x = self.leaky_relu(self.gcn2(x,edge_index))
        #x = self.att2(x)
        x = self.leaky_relu(self.gcn5(x, edge_index))
        x = self.att5(x)
        x = global_max_pool(x, batch)  # global max pooling

        # 副作用
        #x_e = self.att3(x_e)
        x_e = self.leaky_relu(self.gcn3(x_e, edge_index_e))
        #x_e = self.att4(x_e)
        x_e = self.leaky_relu(self.gcn4(x_e,edge_index_e))
        #x_e = self.att6(x_e)
        x_e = self.leaky_relu(self.gcn6(x_e, edge_index_e))
        x_e = self.att6(x_e)
        if not not_FC:
            x = self.relu(self.fc_g1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc_g2(x)
            x_e = self.relu(self.fc_g3(x_e))
            x_e = F.dropout(x_e, p=0.5, training=self.training)
            x_e = self.fc_g4(x_e)

        # 结合
        x_ = self.diag(x) if DF else x

        xc = torch.matmul(x_, x_e.T)
        return xc, x, x_e



