import math

import torch
import dgl
import torch.nn.functional as F

from model.dgl import AggrGATConv
from .base.norm import SqrtZScore, SqrtMinMaxScaler

from ..core.sql_featurizer import database


def u_matmul_v(lhs_field, rhs_field, out):
    def u_matmul_v(edges):
        return {out: torch.einsum('...ij,...jk->...ik', edges.src[lhs_field], edges.dst[rhs_field])}

    return u_matmul_v


class TableTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._hidden_size = database.config.feature_size
        _table_to_column_hidden_size = 32
        self._table_to_column_aggr_heads = 64
        self._predicate_num_heads = 16
        _table_cluster_onehot_size = database.schema.TABLE_CLUSTERS + 1
        _table_cluster_dense_embedding_size = database.schema.TABLE_DENSE_EMBEDDING_SIZE
        _column_statistics_size = database.schema.COLUMN_FEATURE_SIZE
        self._column_predicate_size = len(self._predicate_features)
        _table_feature_size = len(self._table_features)

        self.zscore_table_features = SqrtZScore(_table_feature_size)
        self.minmax_predicate_features = SqrtMinMaxScaler(self._column_predicate_size)
        self.fc_table_features = torch.nn.Sequential(
            # torch.nn.Linear(_table_feature_size, self._hidden_size, bias=False),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(self._hidden_size, self._hidden_size),
        )
        self.fc_table_cluster_onehot = torch.nn.Sequential(
            # torch.nn.Linear(_table_cluster_onehot_size, self._hidden_size, bias=False),
        )
        self.fc_table_cluster_dense_embedding = torch.nn.Sequential(
            # torch.nn.Linear(_table_cluster_dense_embedding_size, self._hidden_size, bias=False)
        )
        self.fc_table_all = torch.nn.Sequential(
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(3 * self._hidden_size, self._hidden_size, bias=False),
            torch.nn.Linear(_table_feature_size + _table_cluster_onehot_size + _table_cluster_dense_embedding_size,
                            self._hidden_size, bias=False)
        )
        a = math.sqrt(3) * 2 / math.sqrt(10 + 7)
        torch.nn.init.uniform_(self.fc_table_all[0].weight, -a, a)

        self.fc_column_statistics = torch.nn.Sequential(
            torch.nn.Linear(_column_statistics_size, _table_to_column_hidden_size, bias=False),
            torch.nn.Dropout(0.4),
        )
        self.fc_table_to_column_heads = torch.nn.Sequential(
            torch.nn.Linear(self._hidden_size, _table_to_column_hidden_size * self._table_to_column_aggr_heads)
        )
        self.fc_predicate_heads = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self._table_to_column_aggr_heads, self._predicate_num_heads * self._column_predicate_size,
                            bias=True)
        )
        self.group_fc_predicate_embedding = torch.nn.ModuleList((
            torch.nn.Linear(self._predicate_num_heads, self._hidden_size, bias=False)
            for _ in range(self._column_predicate_size)
        ))
        self.fc_table_final_embedding = torch.nn.Sequential(
            torch.nn.Linear(self._hidden_size * self._column_predicate_size + _table_feature_size, self._hidden_size,
                            bias=False),
        )

        self.gnn_column_to_table = AggrGATConv(
            (_column_statistics_size, self._hidden_size),
            self._hidden_size,
            num_heads=8,
            residual=True,
            aggr='mean',
        )

    _table_features = (
        'table_row_count',
        'table_log_selectivity', 'table_log_relative_rows',
        # 'table_out_degree', 'table_in_degree',
        # 'table_column_count',
        # 'table_cost', 'table_selected_rows', 'table_width',
    )

    @classmethod
    def _get_table_features(cls, g: dgl.DGLGraph):
        table_features = []
        for feature_name in cls._table_features:
            _table_feature = g.nodes['table'].data[f'features.{feature_name}']
            if _table_feature.ndim == 1:
                _table_feature = _table_feature.unsqueeze(-1)
            table_features.append(_table_feature)
        table_features = torch.cat(table_features, dim=-1)
        return table_features

    _predicate_features = (
        'predicate_filter', 'predicate_join',
    )

    @classmethod
    def _get_predicate_features(cls, g: dgl.DGLGraph):
        predicate_features = []
        predicate_masks = []
        for feature_name in cls._predicate_features:
            _predicate_feature = g.nodes['column'].data[f'{feature_name}.value']
            _predicate_mask = g.nodes['column'].data[f'{feature_name}.mask']
            if _predicate_feature.ndim == 1:
                _predicate_feature = _predicate_feature.unsqueeze(-1)
            if _predicate_mask.ndim == 1:
                _predicate_mask = _predicate_mask.unsqueeze(-1)
            predicate_features.append(_predicate_feature)
            predicate_masks.append(_predicate_mask)
        predicate_features = torch.cat(predicate_features, dim=-1)
        predicate_masks = torch.cat(predicate_masks, dim=-1)
        return predicate_features, predicate_masks

    def fit_norm(self, g: dgl.DGLGraph):
        table_features = self._get_table_features(g)
        #mean = table_features.mean(dim=0, keepdim=False)
        #std = table_features.std(dim=0, keepdim=False)
        #self.zscore_table_features.set(mean, std)
        predicate_features, predicate_masks = self._get_predicate_features(g)
        self.zscore_table_features.fit(table_features)
        self.minmax_predicate_features.fit(predicate_features)

    def forward(self, g: dgl.DGLGraph):
        origin_table_features = self._get_table_features(g)
        if database.config.extractor_use_normalization:
            origin_table_features = self.zscore_table_features(origin_table_features)
        origin_table_cluster_onehot = g.nodes['table'].data['onehot'].to(torch.float)
        origin_table_cluster_dense_embedding = g.nodes['table'].data['embedding']

        column_statistics = g.nodes['column'].data['statistic'].to(torch.float)
        column_to_table_subgraph = g['column', 'column_to_table', 'table']

        table_features = self.fc_table_features(origin_table_features)
        table_cluster_onehot = self.fc_table_cluster_onehot(origin_table_cluster_onehot)
        table_cluster_dense_embedding = self.fc_table_cluster_dense_embedding(origin_table_cluster_dense_embedding)
        table_embedding = self.fc_table_all(
            torch.cat([table_features, table_cluster_onehot, table_cluster_dense_embedding], dim=-1))
        if database.config.extractor_use_no_gat:
            column_to_table_embedding = table_embedding
        else:
            column_to_table_embedding = self.gnn_column_to_table(
                column_to_table_subgraph,
                (column_statistics, table_embedding),
            )
        table_new_embedding = F.dropout(column_to_table_embedding, 0.4, training=self.training)

        column_statistics_embedding = self.fc_column_statistics(column_statistics)
        column_statistics_embedding = column_statistics_embedding.view(*column_statistics_embedding.shape[:-1], 1,
                                                                       column_statistics_embedding.shape[-1])
        table_to_column_embedding = self.fc_table_to_column_heads(table_new_embedding)
        table_to_column_embedding = table_to_column_embedding.view(*table_to_column_embedding.shape[:-1], -1,
                                                                   self._table_to_column_aggr_heads)

        column_to_table_subgraph.srcdata['cs'] = column_statistics_embedding
        column_to_table_subgraph.dstdata['tc'] = table_to_column_embedding
        column_to_table_subgraph.apply_edges(u_matmul_v('cs', 'tc', 'out'))
        predicate_heads = self.fc_predicate_heads(column_to_table_subgraph.edata['out'])
        predicate_heads = predicate_heads.view(*predicate_heads.shape[:-2], -1, self._predicate_num_heads)
        column_to_table_subgraph.edata['ph'] = predicate_heads

        predicate_features, predicate_masks = self._get_predicate_features(g)
        if database.config.extractor_use_normalization:
            predicate_features = self.minmax_predicate_features(predicate_features)
        g.nodes['column'].data['pf'] = predicate_features
        g.nodes['column'].data['pm'] = predicate_masks

        def _predicate_func(edges):
            p = edges.src['pf'].unsqueeze(-1)
            p_mask = edges.src['pm'].unsqueeze(-1)
            res = p * edges.data['ph']
            #res = res + p_mask.log()
            res = res - (1 - p_mask) * 65536.
            return {'out_ct': res}

        g.multi_update_all({
            'column_to_table': (
                _predicate_func,
                dgl.function.max('out_ct', 'pe'),
            ),
        }, 'sum')
        predicate_embeddings = g.nodes['table'].data['pe']
        predicate_embeddings = predicate_embeddings * (predicate_embeddings > -32768.)
        #predicate_embeddings = torch.nan_to_num(predicate_embeddings, 0., 0., 0.)
        new_predicate_embeddings = []
        for i in range(self._column_predicate_size):
            new_predicate_embeddings.append(self.group_fc_predicate_embedding[i](predicate_embeddings[..., i, :]))
        predicate_embeddings = torch.cat(new_predicate_embeddings, dim=-1)

        # table_final_representation = torch.cat([predicate_embeddings, table_embedding], dim=-1)
        table_final_representation = torch.cat([predicate_embeddings, origin_table_features], dim=-1)
        table_final_representation = self.fc_table_final_embedding(table_final_representation)
        g.nodes['table'].data['_emb'] = table_final_representation
        return g


class Extractor(torch.nn.Module):
    def __init__(self, num_table_layers=3):
        super().__init__()

        self._feature_size = database.config.feature_size
        self._table_feature_size = database.schema.max_columns * database.config.feature_length + database.config.feature_extra_length

        self.table_transform = TableTransform()

        self.num_table_layers = num_table_layers
        self.table_to_table = torch.nn.ModuleList((
            AggrGATConv(
                self._feature_size,
                self._feature_size,
                num_heads=8,
                feat_drop=0.1,
                residual=True,
                # bias=False,
                aggr='fc',
            ) for i in range(self.num_table_layers)
        ))

    def fit_norm(self, g):
        if not isinstance(g, dgl.DGLGraph):
            g = dgl.batch(g)
        return self.table_transform.fit_norm(g)

    def forward(self, g):
        g = self.table_transform(g)
        table_x = g.nodes['table'].data['_emb']

        table_res = table_x
        if not database.config.extractor_use_no_gat:
            for i in range(self.num_table_layers):
                table_res = self.table_to_table[i](
                    (dgl.edge_type_subgraph(g, [('table', 'to', 'table')])),
                    table_res,
                )

        g.nodes['table'].data['res'] = table_res

        return g
