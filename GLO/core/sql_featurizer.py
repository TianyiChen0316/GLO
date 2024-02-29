import sys
import math
import re

from tqdm import tqdm
import torch
import dgl

from lib.torch import torch_summary, safe_save
from sql.database import Postgres, Schema, postgres_type
from sql import sql_parser
from sql.sql_parser import parse_select, PlanParser

from .config import Config

database = Postgres(
    retry_limit=3,
    auto_save_interval=400,
)

class GLOSchema(Schema):
    COLUMN_FEATURE_SIZE = 13
    TABLE_CLUSTERS = 6
    TABLE_DENSE_EMBEDDING_SIZE = 9

    def __init__(self, database : Postgres, cache_file = None):
        if cache_file is None:
            cache_file = f'schema_cache/{database.dbname}.schema.pkl'
        if safe_save.file_exists(cache_file):
            super().__init__(None)
            state_dict = safe_save.load_pickle(cache_file)
            self.load_state_dict(state_dict)
        else:
            super().__init__(database)

            table_cluster_file = f'data_driven/table_autoencoder.pkl'
            if safe_save.file_exists(table_cluster_file):
                state_dict = safe_save.load_torch(table_cluster_file, map_location='cpu')

                self.table_clusters = state_dict['cluster']
                table_dense_embeddings = state_dict['embeddings']
                self.table_dense_embeddings = {k : (v.detach().numpy() if isinstance(v, torch.Tensor) else v)
                                               for k, v in table_dense_embeddings.items()}
            else:
                self.table_clusters = {}
                self.table_dense_embeddings = {}

            print('Calculating column features', file=sys.stderr)

            self._column_features = []
            self.column_features_size = 0

            for index, tup in tqdm(enumerate(self.columns), total=len(self.columns)):
                sname, tname, cname = tup
                table_obj = self.name_to_table[tname]

                column_type = postgres_type(table_obj.column_types[cname])
                column_type_onehot = [0 for i in range(3)]
                column_type_onehot[column_type] = 1

                pg_stats_query = f'select null_frac, n_distinct from pg_stats ' \
                                 f'where schemaname = \'{"public" if sname is None else sname}\' ' \
                                 f'and tablename = \'{tname}\' ' \
                                 f'and attname = \'{cname}\';'
                res = database.execute(pg_stats_query)
                if not res:
                    # not enough data
                    null_frac = 0.0
                    n_distinct = -1.0
                else:
                    null_frac, n_distinct = res[0]
                table_row_count = table_obj.row_count

                is_increasing = n_distinct < 0
                if is_increasing:
                    real_n_distinct = -n_distinct
                else:
                    real_n_distinct = n_distinct / table_row_count

                if real_n_distinct < 0.001:
                    # type-like column
                    row_type = 0
                elif real_n_distinct < 0.999:
                    # common type
                    row_type = 1
                else:
                    row_type = 2

                n_distinct_onehot = [*(0 for i in range(3)), 1 if is_increasing else 0, 0 if is_increasing else 1]
                n_distinct_onehot[row_type] = 1

                column_indexes = self.column_db_indexes[tup]
                column_index_onehot = [1 if column_indexes else 0, 0 if column_indexes else 1]

                null_frac_onehot = [1 if null_frac == 0 else 0, 1 if 0 < null_frac < 0.5 else 0, 1 if null_frac >= 0.5 else 0]

                self._column_features.append([
                    *column_type_onehot, # 3
                    *n_distinct_onehot, # 5
                    *null_frac_onehot, # 3
                    *column_index_onehot, # 2
                ])
            self.column_features_size = self.total_columns + 2

            safe_save.save_pickle(self.state_dict(), cache_file)

    def state_dict(self):
        res = super().state_dict()
        for name in (
            'table_clusters', 'table_dense_embeddings', '_column_features', 'column_features_size',
        ):
            res[name] = getattr(self, name)
        return res

    def load_state_dict(self, state_dict):
        for name in (
            'table_clusters', 'table_dense_embeddings', '_column_features', 'column_features_size',
        ):
            setattr(self, name, state_dict[name])
        return super().load_state_dict(state_dict)

    def column_features(self, table_name, column_name, schema_name='public', *,
                        dtype=torch.float, device=torch.device('cpu')):
        column_index = self.column_index(table_name, column_name, schema_name)
        if dtype is not None:
            return torch.tensor(self._column_features[column_index], dtype=dtype, device=device)
        return list(self._column_features[column_index])

def _database_schema_init(database : Postgres):
    database.schema = GLOSchema(database, cache_file=None)
    database.config = Config()
    database.set_settings('max_parallel_workers', 1)
    database.set_settings('max_parallel_workers_per_gather', 1)

database.add_hook('after_setup', 'GLO_schema_hook', _database_schema_init)


class Sql(sql_parser.SqlBase):
    def __init__(self, statement, filename=None, device=torch.device('cpu')):
        self.statement = statement
        self.filename = filename

        parser_env = sql_parser.ParserEnvironment()
        parser_env.table_columns = database.schema.table_columns
        super().__init__(parse_select(statement, parser_env))

        self.__device = device
        self.parse()
        self.to(device)

    def to(self, device):
        self.__device = device
        self.table_features = torch_summary.dict_map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x,
                                                     self.table_features)
        self.column_features = torch_summary.dict_map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x,
                                                      self.column_features)
        self.edge_features = torch_summary.dict_map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x,
                                                    self.edge_features)
        self.graph = self.graph.to(device) if self.graph else self.graph
        return self

    @property
    def device(self):
        return self.__device

    def parse(self):
        self.table_features = {}
        self.column_features = {}
        self.edge_features = {}

        self.parse_features()
        self.graph, self.graph_node_indices = self._to_graph()

        self.baseline = PlanParser(database.plan(self.statement, geqo=False)[0][0][0]['Plan'])

    def get_table(self, alias):
        table_name = self.alias_to_table[alias].fullname
        table_index, table = database.schema.table(table_name)
        return table_index, table_name, table

    def table_selectivity_features(self, alias, explain=True):
        table_filters = self.table_filter_predicates.get(alias, {})
        table_filters_list = []
        for column, filters in table_filters.items():
            table_filters_list.extend(filters)

        _, _, table = self.get_table(alias)

        total_rows = table.row_count

        if table_filters_list:
            str_conditions = ' and '.join(map(str, table_filters_list))

            sql = f'explain select * from {table.name} {alias} ' \
                  f'where {str_conditions}'
            res = tuple(database.execute(sql, True))
            res = res[0][0]
            select_rows = int(re.search(r'rows=([0-9]+)', res).group(1))
            cost = float(re.search(r'cost=[0-9]+(?:\.[0-9]+)?\.\.([0-9]+(?:\.[0-9]+)?)', res).group(1))
            width = int(re.search(r'width=([0-9]+)', res).group(1))
            if not explain:
                sql = f'select count(*) from {table.name} {alias} ' \
                      f'where {str_conditions}'
                res = tuple(database.execute(sql))
                select_rows = res[0][0]
        else:
            select_rows = total_rows
            sql = f'explain select * from {table.name} {alias} '
            res = tuple(database.execute(sql, True))
            res = res[0][0]
            cost = float(re.search(r'cost=[0-9]+(?:\.[0-9]+)?\.\.([0-9]+(?:\.[0-9]+)?)', res).group(1))
            width = int(re.search(r'width=([0-9]+)', res).group(1))

        sel = select_rows / total_rows

        return {
            'selectivity': sel,
            'selected_rows': select_rows,
            'total_rows': total_rows,
            'cost': cost,
            'width': width,
        }

    def parse_features(self):
        table_global_extra_features = {}
        total_sel_rows = 0
        for alias in self.aliases:
            dic = self.table_selectivity_features(alias, explain=True)
            total_sel_rows += dic['selected_rows']
            table_global_extra_features[alias] = dic

        # this feature is used in plan featurization
        self.table_costs = {}

        for alias in self.aliases:
            dic = table_global_extra_features[alias]
            sel = dic['selectivity']
            rows = dic['selected_rows']
            cost = dic['cost']
            width = dic['width']
            self.table_costs[alias] = (cost, rows, width)
            log_sel = -math.log(max(sel, 1e-9))
            log_relative_rows = -math.log(max(rows / total_sel_rows, 1e-9))
            table_global_extra_features[alias] = (log_sel, log_relative_rows)

        self.table_features = {}
        self.column_features = {}
        for alias in self.aliases:
            table_index, table_name, table = self.get_table(alias)

            table_out_degree = len(database.schema.foreign_keys.get(table.name, ()))
            table_in_degree = len(database.schema.foreign_keys_from.get(table.name, ()))
            table_row_count = math.log10(table.row_count + 1)
            table_column_count = len(table.columns)

            _table_costs = self.table_costs[alias]
            _table_global_extra_features = table_global_extra_features[alias]

            cluster_id = database.schema.table_clusters.get(table.name, -1)
            cluster_onehot = [0 for i in range(GLOSchema.TABLE_CLUSTERS + 1)]
            cluster_onehot[cluster_id] = 1

            dense_embedding = database.schema.table_dense_embeddings.get(table.name, None)
            if dense_embedding is None:
                dense_embedding = torch.zeros(GLOSchema.TABLE_DENSE_EMBEDDING_SIZE, device=self.device)
            else:
                dense_embedding = torch.tensor(dense_embedding, device=self.device)

            for index, column_name in table.columns.items():
                features = database.schema.column_features(table.name, column_name, dtype=None)

                table_column = f'{alias}.{column_name}'
                # is_joined, sel, 1-sel
                #column_features = [0., 0., 0.]
                column_feature = {
                    'statistic': features,
                    'predicate_join': {
                        'value': 0.,
                        'mask': 0.,
                    },
                    'predicate_filter': {
                        'value': 0.,
                        'mask': 0.,
                    },
                    'related': False,
                }
                self.column_features[table_column] = column_feature

            table_feature = {
                'features': {
                    'table_out_degree': table_out_degree,
                    'table_in_degree': table_in_degree,
                    'table_row_count': table_row_count,
                    'table_column_count': table_column_count,
                    'table_cost': math.log10(max(_table_costs[0], 1)),
                    'table_selected_rows': -math.log(max(_table_costs[1], 1e-9)),
                    'table_width': _table_costs[2],
                    'table_log_selectivity': _table_global_extra_features[0],
                    'table_log_relative_rows': _table_global_extra_features[1],
                },
                'onehot': cluster_onehot,
                'embedding': dense_embedding,
            }
            self.table_features[alias] = table_feature

        # join predicates
        self.join_candidates = set()
        for join_predicate in self.eqjoin_predicates + self.neqjoin_predicates:
            left_concerned, right_concerned = list(sql_parser.concerned_columns(join_predicate.lexpr))[0], list(sql_parser.concerned_columns(join_predicate.rexpr))[0]
            left_alias, left_column = left_concerned
            right_alias, right_column = right_concerned

            self.join_candidates.add(tuple(sorted((left_alias, right_alias))))

            left_is_table = isinstance(self.alias_to_table[left_alias], sql_parser.FromTable)
            right_is_table = isinstance(self.alias_to_table[right_alias], sql_parser.FromTable)

            if left_is_table and right_is_table:
                left_table_column = f'{left_alias}.{left_column}'
                right_table_column = f'{right_alias}.{right_column}'

                column_features = self.column_features[left_table_column]
                #column_features['predicate'][0] = 1.
                #column_features['predicate_mask'][0] = 1
                column_features['predicate_join']['value'] = 1.
                column_features['predicate_join']['mask'] = 1.
                column_features['related'] = True

                column_features = self.column_features[right_table_column]
                #column_features['predicate'][0] = 1.
                #column_features['predicate_mask'][0] = 1
                column_features['predicate_join']['value'] = 1.
                column_features['predicate_join']['mask'] = 1.
                column_features['related'] = True

        # filters
        for alias, filters in self.table_filter_predicates.items():
            is_table = isinstance(self.alias_to_table[alias], sql_parser.FromTable)
            if is_table:
                table_index, table_name, table = self.get_table(alias)
                for column, column_filters in filters.items():
                    table_column = f'{alias}.{column}'
                    column_features = self.column_features[table_column]
                    for cmp in column_filters:
                        selectivity, row_count, total_row_count = database.selectivity(f'{table_name} {alias}', str(cmp), explain=True, detail=True)
                        _log_selectivity = -math.log(max(selectivity, 1e-9))
                        #_log_1_minus_selectivity = -math.log(max(1 - selectivity, 1e-9))
                        #column_features['predicate'][1] = max(column_features['predicate'][1], _log_selectivity)
                        #column_features['predicate'][2] = max(column_features['predicate'][2], _log_1_minus_selectivity)
                        #column_features['predicate_mask'][1] = 1
                        #column_features['predicate_mask'][2] = 1
                        column_features['predicate_filter']['value'] = max(column_features['predicate_filter']['value'], _log_selectivity)
                        column_features['predicate_filter']['mask'] = 1.
                        column_features['related'] = True

        self.edge_features = {}
        for left_alias, right_alias in self.join_candidates:
            self.edge_features[left_alias, right_alias] = {}
            self.edge_features[right_alias, left_alias] = {}

        self.table_features = torch_summary.torch_convert(self.table_features, convert_scalar=False, device=self.device)
        self.column_features = torch_summary.torch_convert(self.column_features, convert_scalar=False, device=self.device)
        self.edge_features = torch_summary.torch_convert(self.edge_features, convert_scalar=False, device=self.device)

    def _to_graph(self):
        node_indices = {}
        table_nodes_temp = set()
        table_to_table_temp = []

        for left_alias, right_alias in self.join_candidates:
            table_nodes_temp.add(left_alias)
            table_nodes_temp.add(right_alias)
            table_to_table_temp.append((left_alias, right_alias))
            table_to_table_temp.append((right_alias, left_alias))

        table_features = {}
        column_features = {}
        column_index = 0

        edge_dict = {
            ('table', 'to', 'table'): [],
            ('column', 'column_to_table', 'table'): [],
            ('table', 'table_to_column', 'column'): [],
        }

        for index, alias in enumerate(table_nodes_temp):
            node_indices['~' + alias] = index
            table_index, table_name, table = self.get_table(alias)
            for column in table.columns.values():
                table_column = f'{alias}.{column}'
                _column_features = self.column_features[table_column]
                related = _column_features['related']
                # ignore non-related columns
                if not related:
                    continue
                node_indices[table_column] = column_index

                edge_dict['column', 'column_to_table', 'table'].append((column_index, index))
                edge_dict['table', 'table_to_column', 'column'].append((index, column_index))

                _column_features : dict = _column_features.copy()
                _column_features.pop('related')
                _column_features = torch_summary.torch_flat(_column_features, convert_scalar=True, device=self.device)

                for k, v in _column_features.items():
                    if not isinstance(v, torch.Tensor):
                        print(f'Warning: \'{k}\' is not a tensor: \'{v}\'', file=sys.stderr)
                        continue
                    _column_feature_item = column_features.setdefault(k, [])
                    _column_feature_item.append(v)
                column_index += 1

            _table_features = self.table_features[alias]
            _table_features = torch_summary.torch_flat(_table_features, convert_scalar=True, device=self.device)
            for k, v in _table_features.items():
                if not isinstance(v, torch.Tensor):
                    print(f'Warning: \'{k}\' is not a tensor: \'{v}\'', file=sys.stderr)
                    continue
                _table_feature_item = table_features.setdefault(k, [])
                _table_feature_item.append(v)

        edge_features = {}
        for left_alias, right_alias in table_to_table_temp:
            edge_dict['table', 'to', 'table'].append([node_indices['~' + left_alias], node_indices['~' + right_alias]])
            _edge_features = self.edge_features[left_alias, right_alias]
            _edge_features = torch_summary.torch_flat(_edge_features, convert_scalar=True, device=self.device)
            for k, v in _edge_features.items():
                if not isinstance(v, torch.Tensor):
                    print(f'Warning: \'{k}\' is not a tensor: \'{v}\'', file=sys.stderr)
                    continue
                _edge_feature_item = edge_features.setdefault(k, [])
                _edge_feature_item.append(v)

        for edge_name, edge_list in edge_dict.items():
            if len(edge_list) == 0:
                tensor_edge = torch.zeros(0, 2, device=self.device, dtype=torch.long)
            else:
                tensor_edge = torch.tensor(edge_list, device=self.device, dtype=torch.long)
            edge_dict[edge_name] = tuple(tensor_edge.t())

        g = dgl.heterograph(edge_dict, device=self.device)
        for feature_type, feature_list in table_features.items():
            feature_list = torch.stack(feature_list, dim=0).to(self.device, dtype=torch.float32)
            g.nodes['table'].data[feature_type] = feature_list
        for feature_type, feature_list in column_features.items():
            feature_list = torch.stack(feature_list, dim=0).to(self.device, dtype=torch.float32)
            g.nodes['column'].data[feature_type] = feature_list
        for feature_type, feature_list in edge_features.items():
            feature_list = torch.stack(feature_list, dim=0).to(self.device, dtype=torch.float32)
            g.edges['to'].data[feature_type] = feature_list

        return g, node_indices
