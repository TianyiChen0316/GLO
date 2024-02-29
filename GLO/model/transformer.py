from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from lib.torch import Sequence, BatchedSequence
from model.nn import ZScore

from .base.attention import TransformerEncoder
from .base.norm import SqrtZScore
from model.nn._models import TransposeBatchNorm1d


class Prediction(nn.Module):
    def __init__(self, in_feature=69, hid_units=256, out_dim=1, contract=1, mid_layers=True, res_con=True):
        super(Prediction, self).__init__()
        self.mid_layers = mid_layers
        self.res_con = res_con

        self.out_mlp1 = nn.Linear(in_feature, hid_units)

        self.mid_mlp1 = nn.Linear(hid_units, hid_units // contract)
        self.mid_mlp2 = nn.Linear(hid_units // contract, hid_units)

        self.out_mlp2 = nn.Linear(hid_units, out_dim)

    def forward(self, features):

        hid = F.relu(self.out_mlp1(features))
        if self.mid_layers:
            mid = F.relu(self.mid_mlp1(hid))
            mid = F.relu(self.mid_mlp2(mid))
            if self.res_con:
                hid = hid + mid
            else:
                hid = mid
        out = self.out_mlp2(hid)

        return out


class PlanTransformer(nn.Module):
    def __init__(self, emb_size=128, hidden_size=96, head_size=8,
                 dropout=0.1, attention_dropout_rate=0., n_layers=8,
                 out_dim=1, norm_pos='pre', use_node_attrs=True,
                 node_attr_norm_type='batch',
                 ):

        super().__init__()
        self.emb_size = emb_size
        self.hidden_dim = hidden_size
        self.head_size = head_size

        self.use_node_attrs = use_node_attrs

        ffn_dim = hidden_size * head_size

        self.rel_pos_encoder = nn.Embedding(64, head_size, padding_idx=0)

        self.height_encoder = nn.Embedding(64, self.hidden_dim, padding_idx=0)

        self.emb_to_hidden = nn.Sequential(
            nn.Linear(self.emb_size, self.hidden_dim),
        )

        self.input_dropout = nn.Dropout(dropout)
        encoders = [TransformerEncoder(self.hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size, norm_pos=norm_pos)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)

        self.final_ln = nn.LayerNorm(self.hidden_dim)

        self.node_type_embedding = nn.Embedding(64, self.hidden_dim, padding_idx=0)

        self.node_attr_norm_type = node_attr_norm_type
        if node_attr_norm_type == 'layer':
            norm = torch.nn.LayerNorm(self.hidden_dim)
        elif node_attr_norm_type == 'batch':
            norm = TransposeBatchNorm1d(self.hidden_dim)
        else:
            # node_attr_norm_type == 'none'
            norm = torch.nn.Sequential()

        self.node_attr_zscore = ZScore(num_features=3)
        self.node_attr_preprocess = torch.nn.Sequential(
                torch.nn.Linear(10, self.hidden_dim, bias=True),
                norm,
                torch.nn.LeakyReLU(),
                torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            )

        self.out_dim = out_dim
        self.tail = Prediction(self.hidden_dim, self.hidden_dim, out_dim)

    @property
    def node_attr_norm_training(self):
        if self.node_attr_norm_type == 'batch':
            return self.node_attr_preprocess[1].bn_training
        return None

    @node_attr_norm_training.setter
    def node_attr_norm_training(self, value):
        if self.node_attr_norm_type == 'batch':
            self.node_attr_preprocess[1].bn_training = value

    def forward(self, batch_plan : BatchedSequence, dropout=True, use_tail=True):
        not_batched = isinstance(batch_plan, Sequence)
        if not_batched:
            batch_plan = BatchedSequence([batch_plan])
        elif not isinstance(batch_plan, BatchedSequence):
            if isinstance(batch_plan, Iterable):
                # try to batch the sequences
                batch_plan = BatchedSequence(batch_plan)
            else:
                raise TypeError(f"'{batch_plan.__class__.__name__}' object is not batched plans")

        x = batch_plan['embedding']
        node_type_enc = batch_plan['node_type']
        heights = batch_plan['height']

        attn_bias = batch_plan.attention['adjacency_matrix'].transpose(1, 2)
        rel_pos = batch_plan.attention['distance'].transpose(1, 2)

        node_attr = batch_plan['node_attr']
        if self.use_node_attrs:
            zscore_node_attr = self.node_attr_zscore(node_attr[..., :3].transpose(-1, -2)).transpose(-1, -2)
            node_attr = torch.cat([zscore_node_attr, node_attr[..., 3:]], dim=-1)
        else:
            node_attr = torch.zeros_like(node_attr, dtype=node_attr.dtype, device=node_attr.device)
        node_attr_mask = batch_plan['node_attr_mask']
        node_attr = self.node_attr_preprocess(node_attr) * node_attr_mask.unsqueeze(-1)

        # num_batches * head_size * sequence_length * sequence_length
        tree_attn_bias = attn_bias.log().unsqueeze(1).repeat(1, self.head_size, 1, 1)

        # num_batches * head_size * sequence_length * sequence_length
        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2)
        tree_attn_bias = tree_attn_bias + rel_pos_bias

        node_feature = self.emb_to_hidden(x)
        node_feature = node_feature + node_attr + self.height_encoder(heights) + self.node_type_embedding(node_type_enc)

        # transformer encoder
        output = self.input_dropout(node_feature)
        for index, enc_layer in enumerate(self.layers):
            if index == 0:
                output = enc_layer(output, tree_attn_bias, dropout=dropout)
            else:
                output = enc_layer(output, tree_attn_bias, dropout=dropout)
        output = self.final_ln(output)
        res = output[:, 0, :]

        if not_batched:
            res = res.squeeze(0)

        if use_tail == 'detail':
            return res, self.tail(res)

        if use_tail:
            res = self.tail(res)

        return res

