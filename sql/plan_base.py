from copy import deepcopy

from lib.syntax import view
from lib.iter.disjoint_set import DisjointSet
from . import sql_parser
from .sql_parser import BranchNode, Operators


class PlanBase:
    def __init__(self, sql : sql_parser.SqlBase):
        self.sql = sql
        self.reset()

    def reset(self):
        self._branch_nodes = []
        self._parent = {}
        self._roots = DisjointSet()
        self._bushy = 0

    def _clone(self, cls, *args, **kwargs):
        new_plan = cls(*args, **kwargs)
        new_plan._branch_nodes = deepcopy(self._branch_nodes)
        new_plan._parent = self._parent.copy()
        new_plan._roots = self._roots.copy()
        new_plan._bushy = self._bushy
        return new_plan

    def clone(self):
        return self._clone(PlanBase, self.sql)

    @property
    def total_branch_nodes(self):
        return len(self._branch_nodes)

    @property
    def total_nodes(self):
        return len(self._branch_nodes) + len(self.sql.aliases)

    @property
    def completed(self):
        return len(self._branch_nodes) == len(self.sql.aliases) - 1

    def _alias_to_node(self, index, throw=True):
        if isinstance(index, str):
            if index not in self.sql.aliases:
                if throw:
                    raise IndexError(f'alias {index} does not exist')
                return None
            return index
        elif isinstance(index, int):
            if len(self._branch_nodes) <= index:
                if throw:
                    raise IndexError(f'branch node {index} does not exist')
                return None
            return self._branch_nodes[index]
        if throw:
            raise TypeError(f'invalid type \'{index.__class__.__name__}\' for table alias')
        return None

    def _node_is_leaf(self, node):
        return not isinstance(node, BranchNode) and not isinstance(node, int)

    def _node_representation(self, node):
        return node.index if isinstance(node, BranchNode) else node

    def join(self, left, right, join_method=None):
        if join_method is None:
            join_method = Operators.default
        if self.completed:
            raise ValueError(f'the plan is already completed: {str(self)}')
        if self._roots.root(left, default=left) == self._roots.root(right, default=right):
            raise ValueError(f'the left table and the right table are already joined')

        left_child = self._alias_to_node(self._roots.root(left, left))
        right_child = self._alias_to_node(self._roots.root(right, right))

        is_bushy = self._node_is_leaf(left_child) and self._node_is_leaf(right_child)
        if is_bushy:
            self._bushy += 1

        left_representation = self._node_representation(left_child)
        right_representation = self._node_representation(right_child)

        new_node = BranchNode(len(self._branch_nodes), left_representation, right_representation, join_method)
        self._branch_nodes.append(new_node)
        parent_representation = new_node.index

        self._parent[left_representation] = new_node.index
        self._parent[right_representation] = new_node.index

        self._roots.set(left_representation, parent_representation)
        self._roots.set(right_representation, parent_representation)

        return new_node

    def revert(self):
        if self.total_branch_nodes == 0:
            raise RuntimeError('no join action to be reverted')
        last_node = self._branch_nodes.pop()
        left_child, right_child = last_node.left_child, last_node.right_child

        is_bushy = self._node_is_leaf(left_child) and self._node_is_leaf(right_child)
        if is_bushy:
            self._bushy -= 1

        del self._parent[left_child]
        del self._parent[right_child]
        self._reconstruct_roots()
        return last_node

    def _reconstruct_roots(self):
        self._roots.clear()
        for branch in self._branch_nodes:
            left_alias, right_alias = branch.left_child, branch.right_child
            parent_alias = branch.index
            self._roots.set(left_alias, parent_alias)
            self._roots.set(right_alias, parent_alias)

    @property
    def bushy(self):
        return self._bushy > 1

    @view.getter_view
    def parent(self, alias):
        node = self._alias_to_node(alias)
        return self._parent.get(self._node_representation(node), None)

    @property
    def actions(self):
        res = []
        for branch_node in self._branch_nodes:
            res.append((branch_node.left_child, branch_node.right_child, branch_node.join_method))
        return tuple(res)

    def candidates(self, bushy=False, table_level=False, unordered=True):
        """
        If table_level is True, this method returns joins of leaf nodes (tables).
        If unordered is True, this method takes no consideration of inner or outer tables.
        """
        res = set()
        for left, right in self.sql.join_candidates:
            left_root, right_root = self._roots.root(left, left), self._roots.root(right, right)
            if left_root == right_root:
                # already joined
                continue
            if not bushy and self._branch_nodes and not (self._node_is_leaf(left_root) ^ self._node_is_leaf(right_root)):
                # when branch nodes is not empty, only a branch node and a leaf node can be joined,
                #  and so the two roots must have exactly one leaf
                continue
            if not table_level:
                left, right = left_root, right_root
            if unordered:
                left_is_leaf, right_is_leaf = self._node_is_leaf(left), self._node_is_leaf(right)
                if (left_is_leaf and not right_is_leaf) or not (left_is_leaf ^ right_is_leaf) and right < left:
                    left, right = right, left
            res.add((left, right))
        return res

    def _children_leafs(self, node):
        """
        use 'leafs' but not 'leaves' to avoid misleading
        """
        if isinstance(node, int):
            node = self._branch_nodes[node]
        if not isinstance(node, BranchNode):
            if node in self.sql.aliases:
                return (node, )
            raise TypeError(f"'{node.__class__.__name__}' object is not a branch node")
        res = []
        for child in (node.left_child, node.right_child):
            if self._node_is_leaf(child):
                res.append(child)
            else:
                res.extend(self._children_leafs(child))
        return tuple(res)

    def _to_sql_node(self, node, parenthesis=True, oracle=False):
        if isinstance(node, int):
            node = self._branch_nodes[node]
        if isinstance(node, BranchNode):
            left_is_leaf, right_is_leaf = self._node_is_leaf(node.left_child), self._node_is_leaf(node.right_child)
            # the left tree can omit parentheses
            left_sql, left_conditions = self._to_sql_node(node.left_child, parenthesis=left_is_leaf, oracle=oracle)
            # the right tree must use parentheses
            right_sql, right_conditions = self._to_sql_node(node.right_child, parenthesis=True, oracle=oracle)

            # the predicates can be divided into:
            # - sql.table_filter_predicates : simple filter predicates that contains only one column
            # - sql.table_complicated_predicates : complicated single-table filter predicates that contains more than one column
            # - sql.table_eqjoin_predicates : eqjoin predicates of tables; each predicate has two entries for both the left table and the right table
            # - sql.table_neqjoin_predicates : non-eqjoin predicates of tables; two entries
            # - sql.other_complicated_predicates : complicated predicates that concerns no less than 2 tables and more than 2 columns

            # add filters on the leaf nodes
            filters = []
            if left_is_leaf:
                condition_dict = self.sql.table_filter_predicates.get(node.left_child, {})
                for column, conditions in condition_dict.items():
                    filters.extend(conditions)
                complicated_filters = self.sql.table_complicated_predicates.get(node.left_child, ())
                filters.extend(complicated_filters)
            if right_is_leaf:
                condition_dict = self.sql.table_filter_predicates.get(node.right_child, {})
                for column, conditions in condition_dict.items():
                    filters.extend(conditions)
                complicated_filters = self.sql.table_complicated_predicates.get(node.right_child, ())
                filters.extend(complicated_filters)

            join_predicates = []
            left_children, right_children = self._children_leafs(node.left_child), self._children_leafs(node.right_child)
            left_children, right_children = set(left_children), set(right_children)
            for left_child in left_children:
                for other_table, predicate in self.sql.table_eqjoin_predicates.get(left_child, []) + self.sql.table_neqjoin_predicates.get(left_child, []):
                    lexpr_concerned_aliases, rexpr_concerned_aliases = sql_parser.concerned_aliases(predicate.lexpr), sql_parser.concerned_aliases(predicate.rexpr)
                    assert len(lexpr_concerned_aliases) == 1 and len(rexpr_concerned_aliases), f'join predicate {predicate} contains more than 2 tables'
                    lexpr_alias, rexpr_alias = tuple(lexpr_concerned_aliases)[0], tuple(rexpr_concerned_aliases)[0]

                    right_child = rexpr_alias if lexpr_alias == left_child else lexpr_alias
                    if right_child in right_children:
                        # found an edge between the two trees
                        join_predicates.append(predicate)

            children = left_children | right_children
            other_complicated_predicates = []
            for concerned_aliases, complicated in self.sql.other_complicated_predicates:
                if not concerned_aliases.issubset(left_children) and not concerned_aliases.issubset(right_children) \
                    and concerned_aliases.issubset(children):
                    # complicated predicates that contains both left tables and right tables
                    other_complicated_predicates.append(complicated)

            if oracle:
                predicate_to_str = lambda x: x.oracle()
            else:
                predicate_to_str = str
            all_predicates = (*join_predicates, *filters, *other_complicated_predicates)
            if all_predicates:
                # theoretically, the 'inner join' statement only requires an 'on' statement with any condition,
                #  so 'inner join' can be applied even when no join predicate exists
                join_string = f'{left_sql} inner join {right_sql} on {" AND ".join(map(predicate_to_str, all_predicates))}'
            else:
                join_string = f'{left_sql} cross join {right_sql}'
            if parenthesis:
                join_string = f'({join_string})'

            return join_string, (*left_conditions, *right_conditions, *all_predicates)
        else:
            if oracle:
                return self.sql.alias_to_table[node].oracle(), ()
            return str(self.sql.alias_to_table[node]), ()

    def _unjoined_conditions(self):
        conditions = []

        # filter predicates
        for alias in self.sql.aliases:
            if self._parent.get(alias, None) is None:
                # all filter predicates of this table are not joined
                filters = self.sql.table_filter_predicates.get(alias, {})
                for column, _conditions in filters.items():
                    conditions.extend(_conditions)
                complicated_filters = self.sql.table_complicated_predicates.get(alias, {})
                conditions.extend(complicated_filters)

        aliases = set(self.sql.aliases)
        # join predicates
        while aliases:
            # ensure that each edge <table_i, table_j> appears only once
            alias = aliases.pop()
            alias_root = self._roots.root(alias, default=alias)
            join_predicates = self.sql.table_eqjoin_predicates.get(alias, []) + self.sql.table_neqjoin_predicates.get(alias, [])
            for other_alias, condition in join_predicates:
                if alias == other_alias:
                    # self join should always be considered
                    conditions.append(condition)
                if other_alias not in aliases:
                    # ensure that only one between <alias, other_alias> and <other_alias, alias> is traversed
                    continue
                other_alias_root = self._roots.root(other_alias, default=other_alias)
                if alias_root != other_alias_root:
                    # not joined yet
                    conditions.append(condition)

        # other complicated predicates
        for concerned_aliases, complicated_condition in self.sql.other_complicated_predicates:
            concerned_aliases = sql_parser.concerned_aliases(complicated_condition)
            concerned_roots = set(map(lambda x: self._roots.root(x, default=x), concerned_aliases))
            if len(concerned_roots) > 1:
                # the condition is originated from more than one subtree
                conditions.append(complicated_condition)

        return conditions

    def __to_sql_manually_join(self, oracle=False):
        tables = []
        for index, branch_node in enumerate(self._branch_nodes):
            if self._parent.get(index, None) is not None:
                # not root node
                continue
            joined_table, _joined_conditions = self._to_sql_node(branch_node, parenthesis=True, oracle=oracle)
            tables.append(joined_table)
        for alias in self.sql.aliases:
            if self._roots.root(alias, alias) != alias:
                # already joined
                continue
            if oracle:
                table_representation = self.sql.alias_to_table[alias].oracle()
            else:
                table_representation = str(self.sql.alias_to_table[alias])
            tables.append(table_representation)

        if oracle:
            to_str = lambda x: f'{x.oracle()}'
        else:
            to_str = lambda x: f'{str(x)}'

        unjoined_conditions = self._unjoined_conditions()
        if unjoined_conditions:
            unjoined_conditions_str = ' where ' + ' and '.join(map(to_str, unjoined_conditions))
        else:
            unjoined_conditions_str = ''

        return ''.join((
            'select ' if not oracle else '',
            ', '.join(map(to_str, self.sql.element.target_tables)),
            ' from ',
            ', '.join(tables),
            unjoined_conditions_str,
            self.sql.element.tail_clauses(oracle=oracle),
        ))

    def __node_hint(self, node, hints : list, children: list = None, oracle=False):
        """
        Recursively obtains pg_hint_plan style hint for a node.
        Join method hints are stored into 'hints', and leading hints are returned.
        The 'children' argument recursively obtains all children of a node.
        """
        if children is None:
            children = []

        if self._node_is_leaf(node):
            # directly returns the node name
            children.append(node)
            return node
        if isinstance(node, int):
            node = self._branch_nodes[node]

        left_is_leaf, right_is_leaf = self._node_is_leaf(node.left_child), self._node_is_leaf(node.right_child)
        if oracle:
            if not left_is_leaf and not right_is_leaf:
                raise ValueError('oracle does not support bushy plans')

        left_leading_hint = self.__node_hint(node.left_child, hints, children, oracle=oracle)
        right_leading_hint = self.__node_hint(node.right_child, hints, children, oracle=oracle)
        join_method_hint = Operators.hint_name(node.join_method, oracle=oracle)
        if join_method_hint is not None:
            if oracle:
                if left_is_leaf:
                    hints.append(f'{join_method_hint}({node.left_child})')
                if right_is_leaf:
                    hints.append(f'{join_method_hint}({node.right_child})')
            else:
                hints.append(f'{join_method_hint}({" ".join(children)})')

        if oracle:
            if not right_is_leaf:
                return f'{right_leading_hint} {left_leading_hint}'
            return f'{left_leading_hint} {right_leading_hint}'
        return f'({left_leading_hint} {right_leading_hint})'

    def _to_str(self, oracle=False):
        hint_str = f'/*+ {self.hints(leading=oracle, oracle=oracle)} */'

        sql = self.__to_sql_manually_join(oracle=oracle)
        if oracle:
            sql = f'select {hint_str} {sql}'
            if self.sql.element.limit_clause is not None:
                sql = f'select * from ({sql}) where {self.sql.element.limit_clause.oracle()}'
        else:
            sql = f'{hint_str} {sql}'

        return sql

    def hints(self, leading=True, operators=True, oracle=False):
        leading_hints = []
        hints = []
        for index, branch_node in enumerate(self._branch_nodes):
            if self._parent.get(index, None) is not None:
                # not root node
                continue
            node_leading_hint = self.__node_hint(branch_node, hints, oracle=oracle)
            leading_hints.append(f'{Operators.hint_name(Operators.leading, oracle=oracle)}({node_leading_hint})')
        res = []
        if leading:
            res.extend(leading_hints)
        if operators:
            res.extend(hints)
        return " ".join(res)

    def oracle(self):
        return self._to_str(oracle=True)

    def __str__(self):
        return self._to_str(oracle=False)

    def to_any_complete_plan_str(self, oracle=False):
        new_plan = self.clone()
        while not new_plan.completed:
            new_plan.join(*(tuple(new_plan.candidates(bushy=True))[0]))
        return new_plan._to_str(oracle=oracle)

    def to_sql_node(self, node, oracle=False):
        node = self._node_representation(node)
        joined_table, _joined_conditions = self._to_sql_node(node, parenthesis=True, oracle=oracle)
        res = f'select * from {joined_table}'
        return res
