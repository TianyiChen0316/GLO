import torch
import dgl.nn.pytorch as dglnn
from .egatconv import EGATConv

class AggrGATConv(torch.nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, *args, aggr=None, **kwargs):
        super().__init__()
        self.gatconv = dglnn.GATConv(in_feats, out_feats, num_heads, *args, **kwargs)
        self.aggr = aggr
        if self.aggr == 'fc':
            self.fc = torch.nn.Linear(out_feats * num_heads, out_feats)
        else:
            self.fc = torch.nn.Identity()

    def forward(self, graph, feat, get_attention=False):
        if self.aggr == 'mean':
            aggr = lambda x: x.mean(dim=-2)
        elif self.aggr == 'max':
            aggr = lambda x: x.max(dim=-2).values
        elif self.aggr == 'sum':
            aggr = lambda x: x.sum(dim=-2)
        elif self.aggr == 'fc':
            aggr = lambda x: self.fc(x.view(*x.shape[:-2], -1))
        else:
            aggr = lambda x: x

        res = self.gatconv(graph, feat, get_attention=get_attention)
        if get_attention:
            return aggr(res[0]), res[1]
        return aggr(res)

class AggrEGATConv(torch.nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, out_node_feats, out_edge_feats, num_heads, *args, aggr=None, **kwargs):
        super().__init__()
        self.gatconv = EGATConv(in_node_feats, in_edge_feats, out_node_feats, out_edge_feats, num_heads, *args, **kwargs)
        self.aggr = aggr
        if self.aggr == 'fc':
            self.fc_n = torch.nn.Linear(out_node_feats * num_heads, out_node_feats)
            self.fc_e = torch.nn.Linear(out_edge_feats * num_heads, out_edge_feats)
        else:
            self.fc_n = torch.nn.Identity()
            self.fc_e = torch.nn.Identity()

    def forward(self, graph, nfeats, efeats, get_attention=False):
        if self.aggr == 'mean':
            aggr_n = lambda x: x.mean(dim=-2)
            aggr_e = lambda x: x.mean(dim=-2)
        elif self.aggr == 'max':
            aggr_n = lambda x: x.max(dim=-2).values
            aggr_e = lambda x: x.max(dim=-2).values
        elif self.aggr == 'sum':
            aggr_n = lambda x: x.sum(dim=-2)
            aggr_e = lambda x: x.sum(dim=-2)
        elif self.aggr == 'fc':
            aggr_n = lambda x: self.fc_n(x.view(*x.shape[:-2], -1))
            aggr_e = lambda x: self.fc_e(x.view(*x.shape[:-2], -1))
        else:
            aggr_n = lambda x: x
            aggr_e = lambda x: x

        res_n, res_e, att = self.gatconv(graph, nfeats, efeats, get_attention=True)
        res_n, res_e = aggr_n(res_n), aggr_e(res_e)

        if get_attention:
            return res_n, res_e, att
        return res_n, res_e

