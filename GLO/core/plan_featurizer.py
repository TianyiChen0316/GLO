import typing

import math
from copy import copy, deepcopy

import torch

from lib.torch import Sequence
from lib.syntax import view
from sql.plan_base import PlanBase, BranchNode
from sql.sql_parser import PlanParser, Operators

from .sql_featurizer import Sql, database

class Plan(PlanBase):
    NODE_ATTRIBUTE_SIZE = 10

    def __init__(self, sql : Sql, device=None):
        if not isinstance(sql, Sql):
            raise TypeError(f"'{sql.__class__.__name__}' object is not a '{Sql.__name__}' object")
        if device is None:
            device = sql.device
        self.__device = device

        self.sql : Sql = ...
        super().__init__(sql)

    def clone(self, detach=True):
        res = self._clone(self.__class__, self.sql, self.__device)
        res._node_attributes = deepcopy(self._node_attributes)
        res._alias_indices = self._alias_indices.copy()
        res._cpu_features = self._cpu_features.clone(detach=True)
        res._grad_features = self._grad_features.clone(detach=detach)
        return res

    def reset(self):
        super().reset()
        self._node_attributes = {}

        self._alias_indices = {alias : index for index, alias in enumerate(sorted(self.sql.aliases))}
        total_leaf_nodes = len(self.sql.aliases)

        # super node: 1, leaf nodes: N, branch nodes: N - 1
        self._max_sequence_length = len(self._alias_indices) * 2
        self._grad_features = Sequence(self._max_sequence_length, device=self.__device)
        self._grad_features.register('embedding', database.config.feature_size)
        if database.config.use_lstm:
            self._grad_features.register('cell', database.config.feature_size)

        if database.config.plan_features_on_cuda:
            self._feature_device = self.__device
        else:
            self._feature_device = torch.device('cpu')

        self._cpu_features = Sequence(self._max_sequence_length, device=self._feature_device)
        self._cpu_features.register('node_attr', self.NODE_ATTRIBUTE_SIZE)
        self._cpu_features.register('node_attr_mask', dtype=torch.int)
        self._cpu_features.register('operator_type', dtype=torch.int)

        self._cpu_features.register('sequence_mask', dtype=torch.int)
        self._cpu_features['sequence_mask'][:total_leaf_nodes + 1] = 1

        self._cpu_features.register('height', dtype=torch.int)
        self._cpu_features['height'][1 : total_leaf_nodes + 1] = 1

        self._cpu_features.register('node_type', dtype=torch.int)
        self._cpu_features['node_type'][0] = 3
        self._cpu_features['node_type'][1 : total_leaf_nodes + 1] = 1

        # due to the attention mechanism, the adjacency matrix should contain padding elements
        self._cpu_features.attention['distance'] = torch.eye(database.config.max_sequence_length, dtype=torch.int).log().abs()
        self._cpu_features.attention['distance'][0, 0] = -1
        self._cpu_features.attention['distance'][1 : total_leaf_nodes + 1, 0] = -2
        self._cpu_features.attention['adjacency_matrix'] = torch.eye(database.config.max_sequence_length, dtype=torch.int)

    def join(self, left, right, join_method=None):
        res = super().join(left, right, join_method=None)
        left_index, right_index = self._to_feature_index(left), self._to_feature_index(right)
        new_index = self._to_feature_index(res)
        left_index, right_index, new_index = 1 + left_index, 1 + right_index, 1 + new_index
        self._cpu_features['sequence_mask'][new_index] = 1
        self._cpu_features['node_type'][new_index] = 2
        new_height = 1 + torch.maximum(
            self._cpu_features['height'][left_index],
            self._cpu_features['height'][right_index],
        )
        self._cpu_features['height'][new_index] = new_height
        self._cpu_features['operator_type'][new_index] = 1 + res.join_method.value
        self._cpu_features.attention['adjacency_matrix'][left_index, new_index] = 1
        self._cpu_features.attention['adjacency_matrix'][right_index, new_index] = 1
        # the distance from other nodes to the new node is the minimal distance from other nodes to the children plus one
        self._cpu_features.attention['distance'][1:, new_index] = 1 + torch.minimum(
            self._cpu_features.attention['distance'][1:, left_index],
            self._cpu_features.attention['distance'][1:, right_index],
        )
        self._cpu_features.attention['distance'][new_index, new_index] = 0
        self._cpu_features.attention['distance'][new_index, 0] = -1 - new_height
        #if database.config.plan_aggr_node_link_to_root_nodes_only:
        #    self._cpu_features.attention['distance'][left_index, 0] = torch.inf
        #    self._cpu_features.attention['distance'][right_index, 0] = torch.inf
        return res

    def revert(self):
        res = super().revert()
        left_child, right_child, new_node = res.left_child, res.right_child, res.index
        left_index, right_index, new_index = self._to_feature_index(left_child), self._to_feature_index(right_child), self._to_feature_index(new_node)
        left_index, right_index, new_index = 1 + left_index, 1 + right_index, 1 + new_index
        self._cpu_features['sequence_mask'][new_index] = 0
        self._cpu_features['node_type'][new_index] = 0
        self._cpu_features['height'][new_index] = 0
        self._cpu_features['operator_type'][new_index] = 0
        self._cpu_features.attention['adjacency_matrix'][left_index, new_index] = 0
        self._cpu_features.attention['adjacency_matrix'][right_index, new_index] = 0
        self._cpu_features.attention['distance'][1:, new_index] = torch.inf
        self._cpu_features.attention['distance'][new_index, new_index] = 0
        self._cpu_features.attention['distance'][new_index, 0] = torch.inf
        #if database.config.plan_aggr_node_link_to_root_nodes_only:
        #    self._cpu_features.attention['distance'][left_index, 0] = -1
        #    self._cpu_features.attention['distance'][right_index, 0] = -1
        return res

    def to(self, device):
        self.__device = device
        self._cpu_features.to(device)

    @view.getter_view
    def features(self, item):
        feature_index = self._to_feature_index(item)
        return self._grad_features[feature_index]

    @features.setitem
    def features(self, item, value):
        feature_index = self._to_feature_index(item)
        self._grad_features[feature_index] = value

    def _to_feature_index(self, node):
        if isinstance(node, str):
            res = self._alias_indices.get(node, None)
            if res is None:
                raise IndexError(f"alias '{node}' does not exist")
            return res
        if isinstance(node, int):
            # allows pre-settings for branch nodes that are not yet created
            return len(self.sql.aliases) + node
        if isinstance(node, BranchNode):
            return len(self.sql.aliases) + node.index
        raise TypeError(f"invalid index type: '{node.__class__.__name__}'")

    @property
    def node_attrs(self):
        return self._node_attributes

    @node_attrs.setter
    def node_attrs(self, value : dict):
        if not isinstance(value, dict):
            raise TypeError(f"'{value.__class__.__name__}' object is not a dict")
        self._node_attributes = {node : {k : v.get(k, None) for k in self._node_attrs_used} for node, v in value.items()}

    def update_node_attrs(self, value : dict):
        if not isinstance(value, dict):
            raise TypeError(f"'{value.__class__.__name__}' object is not a dict")
        self._node_attributes.update({node : {k : v.get(k, None) for k in self._node_attrs_used} for node, v in value.items()})

    def clear_embeddings(self):
        res = self._grad_features['embedding']
        del self._grad_features['embedding']
        self._grad_features.register('embedding', database.config.feature_size)
        if database.config.use_lstm:
            del self._grad_features['cell']
            self._grad_features.register('cell', database.config.feature_size)
        return res

    def set_leaf_embeddings(self, tensors, node_indices):
        indices = []
        ori_indices = []
        for alias in self.sql.aliases:
            original_index = node_indices[f'~{alias}']
            this_index = self._to_feature_index(alias)
            indices.append(1 + this_index)
            ori_indices.append(original_index)
        indices = torch.tensor(indices, dtype=torch.long, device=self.__device)
        ori_indices = torch.tensor(ori_indices, dtype=torch.long, device=self.__device)
        tensors = tensors[ori_indices]
        self._index_update('embedding', indices, tensors)

    def _index_update(self, target_field, indices, values):
        emb = self._grad_features[target_field]
        if emb.requires_grad:
            new_emb = torch.zeros_like(emb, dtype=values.dtype, device=self.__device)
            new_emb[indices] = values
            new_emb_mask = torch.zeros_like(emb, device=self.__device)
            new_emb_mask[indices] = 1
            self._grad_features[target_field] = emb * (1 - new_emb_mask) + new_emb * new_emb_mask
        else:
            emb[indices] = values
            self._grad_features[target_field] = emb

    def set_node_attrs_from_parsed_plan(self, parsed : PlanParser):
        if isinstance(parsed, dict):
            parsed = PlanParser(parsed)
        self.update_node_attrs(parsed.plan_attributes(self))
        if database.config.plan_use_node_attrs:
            node_attr_list = []
            if database.config.plan_use_leaf_node_attrs:
                node_attr_list.extend(self._alias_indices.keys())
            if database.config.plan_use_branch_node_attrs:
                node_attr_list.extend(range(self.total_branch_nodes))
            for node in node_attr_list:
                node_index = 1 + self._to_feature_index(node)
                if isinstance(node, str) and database.config.plan_use_leaf_node_attrs_with_sql_info:
                    attrs = self._node_attr_from_sql(node)
                else:
                    attrs = self._node_attr_from_parsed_plan(node)
                self._cpu_features['node_attr'][node_index] = attrs
                self._cpu_features['node_attr_mask'][node_index] = 1

    _node_attrs_used = (
        'Total Cost', 'Plan Rows', 'Plan Width', 'Node Type',
    )
    def _node_attr_from_parsed_plan(self, target_node):
        if isinstance(target_node, dict):
            node_attrs = target_node
        else:
            node_attrs = self._node_attributes.get(target_node, None)
        if node_attrs is None:
            raise RuntimeError(f"attributes of node '{target_node}' does not exist")
        # size: 10
        attrs = [
            math.log(max(node_attrs['Total Cost'], 1) / 1000),
            math.log(max(node_attrs['Plan Rows'], 1) / 1000),
            math.log(max(node_attrs['Plan Width'], 1)),
            *_extra_input_nodetype_onehot(node_attrs)
        ]
        res = torch.tensor(attrs, dtype=torch.float, device=self._feature_device)
        return res

    def _node_attr_from_sql(self, target_node : str):
        if target_node not in self.sql.aliases:
            raise RuntimeError(f"invalid node: '{target_node}'")
        cost, rows, width = self.sql.table_costs[target_node]
        attrs = [
            math.log(max(cost, 1) / 1000),
            math.log(max(rows, 1) / 1000),
            math.log(max(width, 1)),
            *_extra_input_nodetype_onehot(None),
        ]
        res = torch.tensor(attrs, dtype=torch.float, device=self._feature_device)
        return res

    def to_sequence(self):
        features = self._cpu_features.clone(deep=False).to(self.__device)
        features.update(self._grad_features)
        if database.config.plan_use_height:
            features['height'] = features['height'].clip(None, database.config.plan_height_maximum)
        else:
            ori_height = features['height']
            features['height'] = torch.zeros_like(ori_height, dtype=ori_height.dtype, device=self.__device)
        attention_distance = features.attention['distance']
        attention_distance[attention_distance == torch.inf] = 0
        attention_distance = attention_distance.clip(-database.config.plan_relative_dist_maximum - 1, database.config.plan_relative_dist_maximum)
        dist_lt0 = attention_distance < 0
        attention_distance[dist_lt0] = database.config.max_sequence_length + attention_distance[dist_lt0]
        features.attention['distance'] = attention_distance.to(torch.long)
        features.resize(database.config.max_sequence_length, lazy=False)
        features.attention['adjacency_matrix'] = (
            (features.attention['distance'] != 0) |
            torch.eye(database.config.max_sequence_length, dtype=torch.bool, device=self.__device)
        ).to(torch.long)
        return features

    def layer_actions(self):
        if not self._branch_nodes:
            # not yet joined
            return []
        traversed = [0 for i in range(len(self._branch_nodes))]
        actions = []
        current_layer = self.sql.aliases
        while current_layer:
            next_layer = []
            for node in current_layer:
                parent_node = self.parent[node.index if isinstance(node, BranchNode) else node]
                if parent_node is not None:
                    parent_node = self._branch_nodes[parent_node]
                    assert isinstance(parent_node, BranchNode), 'invalid node type'
                    assert traversed[parent_node.index] < 2, f'node {parent_node.index} is traversed more than twice'
                    traversed[parent_node.index] += 1
                    if traversed[parent_node.index] >= 2:
                        # both the left child and the right child are reached
                        next_layer.append(parent_node)
            if next_layer:
                actions.append(next_layer)
            current_layer = next_layer
        return actions

    def get_last_action_embeddings(self):
        if not self._branch_nodes:
            raise RuntimeError('the plan has no joined nodes')
        return self.get_layer_embeddings([self._branch_nodes[-1]])

    LSTM_EXTRA_INPUT_SIZE = 3 + NODE_ATTRIBUTE_SIZE * 3
    def get_layer_embeddings(self, layer_nodes: typing.Iterable[BranchNode]):
        if not database.config.use_lstm:
            raise RuntimeError('cannot get cell embeddings')

        layer_nodes = tuple(layer_nodes)
        left_indices, right_indices, node_indices, operators = [], [], [], []
        for node in layer_nodes:
            left_index, right_index = 1 + self._to_feature_index(node.left_child), 1 + self._to_feature_index(node.right_child)
            index = 1 + self._to_feature_index(node.index)
            left_indices.append(left_index)
            right_indices.append(right_index)
            node_indices.append(index)
            operators.append(node.join_method)
        left_indices = torch.tensor(left_indices, dtype=torch.long, device=self.__device)
        right_indices = torch.tensor(right_indices, dtype=torch.long, device=self.__device)
        node_indices = torch.tensor(node_indices, dtype=torch.long, device=self.__device)
        left_embs = self._grad_features['embedding'][left_indices]
        right_embs = self._grad_features['embedding'][right_indices]
        left_cells = self._grad_features['cell'][left_indices]
        right_cells = self._grad_features['cell'][right_indices]

        operators = torch.tensor([Operators.to_binary(op) for op in operators], dtype=torch.long,
                                   device=self.__device)
        node_attrs_left = self._cpu_features['node_attr'][left_indices]
        node_attrs_right = self._cpu_features['node_attr'][right_indices]
        node_attrs_parent = self._cpu_features['node_attr'][node_indices]
        cpu_features = torch.cat([
            node_attrs_left, node_attrs_right, node_attrs_parent
        ], dim=-1).to(self.__device)
        extra_inputs = torch.cat([operators, cpu_features], dim=-1)

        seq = Sequence(len(layer_nodes), device=self.__device)
        seq['left_index'] = left_indices
        seq['left_hidden'] = left_embs
        seq['left_cell'] = left_cells
        seq['right_index'] = right_indices
        seq['right_hidden'] = right_embs
        seq['right_cell'] = right_cells
        seq['node_index'] = node_indices
        seq['extra_input'] = extra_inputs
        return seq

    def set_layer_embeddings(
        self,
        layer_nodes: typing.Union[torch.Tensor, Sequence, typing.Iterable[BranchNode]],
        embeddings: typing.Union[Sequence, dict, torch.Tensor],
        cells: typing.Optional[torch.Tensor] = None,
        source_indices = None,
    ):
        if not database.config.use_lstm:
            raise RuntimeError('cannot set cell embeddings')

        if isinstance(source_indices, int):
            source_indices = slice(source_indices, source_indices + 1)

        if isinstance(embeddings, (Sequence, dict)):
            if cells is None:
                cells = embeddings['node_cell']
            embeddings = embeddings['node_hidden']
        if source_indices is not None:
            embeddings, cells = embeddings[source_indices], cells[source_indices]

        if isinstance(layer_nodes, torch.Tensor):
            node_indices = layer_nodes
            if source_indices is not None:
                node_indices = node_indices[source_indices]
        elif isinstance(layer_nodes, Sequence):
            node_indices = layer_nodes['node_index']
            if source_indices is not None:
                node_indices = node_indices[source_indices]
        else:
            node_indices = []
            if source_indices is not None:
                layer_nodes = tuple(layer_nodes)[source_indices]
            for node in layer_nodes:
                node_indices.append(1 + self._to_feature_index(node.index))
            node_indices = torch.tensor(node_indices, dtype=torch.long, device=self.__device)

        self._index_update('embedding', node_indices, embeddings)
        self._index_update('cell', node_indices, cells)

    def get_root_embeddings(self):
        root_nodes = filter(
            lambda x: (self.parent[x] is None or self.parent[x] == x),
            (*self.sql.aliases, *range(self.total_branch_nodes)),
        )
        root_node_indices = list(map(lambda x: 1 + self._to_feature_index(x), root_nodes))
        embeddings = self._grad_features['embedding'][root_node_indices]
        return embeddings


def _extra_input_nodetype_onehot(dic):
    res = [0 for i in range(7)]
    if not isinstance(dic, dict):
        return res
    node_type = dic['Node Type'].lower()
    id_ = 0
    if node_type == 'nested loop' or 'join' in node_type:
        if node_type == 'nested loop':
            id_ = 1
        elif 'hash' in node_type:
            id_ = 2
        elif 'merge' in node_type:
            id_ = 3
    elif 'scan' in node_type:
        if 'seq' in node_type:
            id_ = 4
        elif 'index' in node_type:
            if 'only' in node_type:
                id_ = 6
            else:
                id_ = 5
    res[id_] = 1
    return res
