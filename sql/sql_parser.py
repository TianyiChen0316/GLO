import re as _re
import copy
import sys
import typing
from collections.abc import Iterable as _Iterable
from enum import IntEnum

import psqlparse as _psqlparse

from lib.iter.disjoint_set import DisjointSet

class ParserEnvironment:
    def __init__(self):
        self.alias_to_table = {}
        self.table_columns = {}

    def clone(self):
        res = self.__class__()
        res.alias_to_table = self.alias_to_table.copy()
        res.table_columns = self.table_columns.copy()
        return res

def sql_expr_analyze(arg, env : ParserEnvironment):
    """
    Recursive analysis of select statements.
    """
    if 'ColumnRef' in arg:
        return ColumnRef(arg['ColumnRef'], env)
    elif 'FuncCall' in arg:
        arg['FuncCall']['_FuncName'] = None
        return FuncCall(arg['FuncCall'], env)
    elif 'GroupingFunc' in arg:
        arg['GroupingFunc']['_FuncName'] = 'grouping'
        return FuncCall(arg['GroupingFunc'], env)
    elif 'CoalesceExpr' in arg:
        arg['CoalesceExpr']['_FuncName'] = 'coalesce'
        return FuncCall(arg['CoalesceExpr'], env)
    elif 'A_Expr' in arg:
        return MathExpr(arg['A_Expr'], env)
    elif 'A_Const' in arg:
        return Const(arg['A_Const'], env)
    elif 'TypeCast' in arg:
        return TypeCast(arg['TypeCast'], env)
    elif 'BoolExpr' in arg:
        return BoolExpr(arg['BoolExpr'], env)
    elif 'NullTest' in arg:
        return NullTest(arg['NullTest'], env)
    elif 'SubLink' in arg:
        return SubLink(arg['SubLink'], env)
    elif 'WindowDef' in arg:
        return WindowDef(arg['WindowDef'], env)
    elif 'CaseExpr' in arg:
        return CaseExpr(arg['CaseExpr'], env)
    elif 'GroupingSet' in arg:
        return GroupingSet(arg['GroupingSet'], env)
    else:
        raise Exception(f'Unknown expression: {arg}')


class Element:
    """
    Parent class of all tree elements.
    """
    def oracle(self):
        return str(self)

    @property
    def concerned_columns(self):
        return set()

    @property
    def concerned_aliases(self):
        return set()

    def __init__(self, args=None, env : ParserEnvironment = None):
        if args is not None:
            self.setup(args, env)
        else:
            self.default_setup(env)

    def setup(self, args : dict, env : ParserEnvironment = None):
        pass

    def default_setup(self, env : ParserEnvironment = None):
        pass

    def clone(self):
        res = self.__class__()
        for field in filter(lambda x: not callable(getattr(self, x)) and not x.startswith('__'), dir(self)):
            # including public, protected and private fields
            value = getattr(self, field)
            if isinstance(value, Element):
                value = value.clone()
            else:
                value = copy.deepcopy(value)
            setattr(res, field, value)
        return res

def concerned_columns(element : typing.Union[Element, typing.Iterable]):
    if isinstance(element, dict):
        element = element.values()
    if isinstance(element, _Iterable):
        res = set()
        for e in element:
            res.update(concerned_columns(e))
        return res
    return element.concerned_columns

def concerned_aliases(element : typing.Union[Element, typing.Iterable]):
    if isinstance(element, dict):
        element = element.values()
    if isinstance(element, _Iterable):
        res = set()
        for e in element:
            res.update(concerned_aliases(e))
        return res
    return element.concerned_aliases

class CaseWhen(Element):
    def default_setup(self, env=None):
        self.expr = None
        self.result = None

    def setup(self, args : dict, env=None):
        self.expr = sql_expr_analyze(args['expr'], env)
        self.result = sql_expr_analyze(args['result'], env)

    def to_str(self, oracle=False):
        if oracle:
            return f'when {self.expr.oracle()} then {self.result.oracle()}'
        # TODO: do expr and result need parenthesises?
        return f'when {self.expr} then {self.result}'

    def __str__(self):
        return self.to_str(False)

    def oracle(self):
        return self.to_str(True)

    @property
    def concerned_columns(self):
        return self.expr.concerned_columns | self.result.concerned_columns

    @property
    def concerned_aliases(self):
        return self.expr.concerned_aliases | self.result.concerned_aliases

class CaseExpr(Element):
    def default_setup(self, env=None):
        self.default = None
        self.cases = []

    def setup(self, args : dict, env=None):
        if 'defresult' in args:
            self.default = sql_expr_analyze(args['defresult'], env)
        else:
            self.default = None
        cases = args.get('args', [])
        self.cases = []
        for case in cases:
            self.cases.append(CaseWhen(case['CaseWhen'], env))

    @property
    def concerned_aliases(self):
        res = set()
        if self.default:
            res |= self.default.concerned_aliases
        for case in self.cases:
            res |= case.concerned_aliases
        return res

    @property
    def concerned_columns(self):
        res = set()
        if self.default:
            res |= self.default.concerned_columns
        for case in self.cases:
            res |= case.concerned_columns
        return res

    def to_str(self, oracle=False):
        if oracle:
            _str = lambda x: x.oracle()
        else:
            _str = str

        if self.default is None:
            default = ''
        else:
            default = f' else {self.default}'
        cases = ' '.join(map(_str, self.cases))
        return f'case {cases}{default} end'

    def __str__(self):
        return self.to_str(False)

    def oracle(self):
        return self.to_str(True)

class WindowDef(Element):
    def default_setup(self, env=None):
        self.partition = []
        self.order = []

    def setup(self, args : dict, env=None):
        partition = args.get('partitionClause', [])
        self.partition = []
        for element in partition:
            self.partition.append(sql_expr_analyze(element, env))
        order = args.get('orderClause', [])
        self.order = []
        for element in order:
            self.order.append(OrderClause(element, env))

    @property
    def concerned_aliases(self):
        res = set()
        for element in self.partition:
            res |= element.concerned_aliases
        for element in self.order:
            res |= element.concerned_aliases
        return res

    @property
    def concerned_columns(self):
        res = set()
        for element in self.partition:
            res |= element.concerned_columns
        for element in self.order:
            res |= element.concerned_columns
        return res

    def to_str(self, oracle=False):
        if oracle:
            _str = lambda x: x.oracle()
        else:
            _str = str

        if self.partition:
            partition = f'partition by {", ".join(map(_str, self.partition))}'
        else:
            partition = ''
        if self.order:
            order = f'order by {", ".join(map(_str, self.order))}'
        else:
            order = ''
        return f'{partition} {order}'

    def __str__(self):
        return self.to_str(False)

    def oracle(self):
        return self.to_str(True)

class NullTest(Element):
    def default_setup(self, env=None):
        self.type = None
        self.element = None

    def setup(self, args : dict, env=None):
        arg = args['arg']
        self.type = args['nulltesttype']
        self.element = sql_expr_analyze(arg, env)

    @property
    def concerned_aliases(self):
        return self.element.concerned_aliases

    @property
    def concerned_columns(self):
        return self.element.concerned_columns

    def __str__(self):
        if self.type == 0:
            type_str = 'IS NULL'
        elif self.type == 1:
            type_str = 'IS NOT NULL'
        else:
            raise Exception(f'Unknown null test type: {self.type}')
        return f'{self.element} {type_str}'

class ColumnRef(Element):
    @classmethod
    def from_name(cls, column_name=None, alias=None, star=False):
        res = cls()
        res.column_name = column_name
        res.alias = alias
        res.star = star
        return res

    def default_setup(self, env=None):
        self.alias = None
        self.column_name = None
        self.star = False

    def setup(self, args : dict, env=None):
        fields = args['fields']
        if len(fields) == 1:
            # only contains column name
            self.star = 'A_Star' in fields[0]
            if not self.star:
                self.column_name = fields[0]['String']['str']
                self.alias = None
                if env and env.alias_to_table and env.table_columns:
                    for alias, table in env.alias_to_table.items():
                        table_columns = env.table_columns.get(table, None)
                        if table_columns and self.column_name in table_columns:
                            self.alias = alias
                            break
            else:
                self.column_name = None
                self.alias = None
        else:
            self.alias = fields[0]['String']['str']
            self.star = 'A_Star' in fields[1]
            if not self.star:
                self.column_name = fields[1]['String']['str']
            else:
                self.column_name = None

    @property
    def concerned_aliases(self):
        return {self.alias} if self.alias else set()

    @property
    def concerned_columns(self):
        return {(self.alias, self.column_name)} if self.alias and self.column_name else set()

    def __str__(self):
        if self.alias:
            if self.star:
                return f'{self.alias}.*'
            return f'{self.alias}.{self.column_name}'
        if self.star:
            return '*'
        return self.column_name

class Const(Element):
    @classmethod
    def from_value(cls, value):
        res = cls()
        if isinstance(value, (int, float)):
            res.type = type(value)
            res.value = value
        elif value is None:
            res.type = None
            res.value = None
        else:
            res.type = str
            res.value = str(value)
        return res

    def default_setup(self, env=None):
        self.type = None
        self.value = None

    def setup(self, args : dict, env=None):
        self.type = None
        value = args["val"]
        if "String" in value:
            self.type = str
            self.value = value["String"]["str"]
        elif "Integer" in value:
            self.type = int
            self.value = value["Integer"]["ival"]
        elif "Float" in value:
            self.type = float
            self.value = float(value["Float"]["str"])
        elif "Null" in value:
            self.type = None
            self.value = None
        else:
            raise Exception("unknown Value in Expr")

    @property
    def concerned_aliases(self):
        return set()

    @property
    def concerned_columns(self):
        return set()

    def __str__(self):
        if self.type == str:
            return f"'{self.value}'"
        if self.value is None:
            return 'null'
        return str(self.value)

class BoolExpr(Element):
    AND = 0
    OR = 1
    NOT = 2

    def default_setup(self, env=None):
        self.op = None
        self.args = []

    def setup(self, args : dict, env=None):
        self.op = args['boolop']
        args = args['args']
        self.args = []
        for arg in args:
            self.args.append(sql_expr_analyze(arg, env))

    @property
    def concerned_aliases(self):
        res = set()
        for arg in self.args:
            res |= arg.concerned_aliases
        return res

    @property
    def concerned_columns(self):
        res = set()
        for arg in self.args:
            res |= arg.concerned_columns
        return res

    def to_str(self, oracle=False):
        if oracle:
            _str = lambda x: x.oracle()
        else:
            _str = str

        if self.op == self.AND:
            return f"({' AND '.join(map(_str, self.args))})"
        elif self.op == self.OR:
            return f"({' OR '.join(map(_str, self.args))})"
        elif self.op == self.NOT:
            return f"({'NOT ' + _str(self.args[0])})"
        else:
            raise Exception('Unknown bool expression')

    def __str__(self):
        return self.to_str(False)

    def oracle(self):
        return self.to_str(True)

class TypeCast(Element):
    def default_setup(self, env=None):
        self.arg = None
        self.type_class = None
        self.type_args = []
        self.name = None

    def setup(self, args : dict, env = None):
        # TODO: is this a bug? env is None?
        self.arg = sql_expr_analyze(args['arg'], env)
        type_name = args['typeName']['TypeName']
        if len(type_name['names']) == 1:
            self.type_class = None
            self.name = type_name['names'][0]['String']['str']
        else:
            self.type_class = type_name['names'][0]['String']['str']
            self.name = type_name['names'][1]['String']['str']
        self.type_args = []
        if 'typmods' in type_name:
            for dic in type_name['typmods']:
                self.type_args.append(sql_expr_analyze(dic, env))

    @property
    def concerned_aliases(self):
        res = self.arg.concerned_aliases
        for arg in self.type_args:
            res |= arg.concerned_aliases
        return res

    @property
    def concerned_columns(self):
        res = self.arg.concerned_columns
        for arg in self.type_args:
            res |= arg.concerned_columns
        return res

    def __str__(self):
        type_name = f'{self.type_class}.{self.name}' if self.type_class and self.type_class != 'pg_catalog' else self.name
        type_args = f'({", ".join(map(str, self.type_args))})' if self.type_args else ''
        return str(self.arg) + '::' + type_name + type_args

    def oracle(self):
        type_name = f'{self.type_class}.{self.name}' if self.type_class and self.type_class != 'pg_catalog' else self.name
        if type_name == 'date':
            return f'to_date({str(self.arg)}, \'YYYY-MM-DD\')'
        return str(self.arg)

class MathExpr(Element):
    ARITHMETIC = 0
    IN = 6
    LIKE = 7
    ILIKE = 8
    BETWEEN = 10
    NOT_BETWEEN = 11

    def default_setup(self, env=None):
        self.kind = None
        self.name = None
        self.lexpr = None
        self.rexpr = None

    def setup(self, args : dict, env=None):
        self.kind = args['kind']
        self.name = args['name'][0]['String']['str']
        if 'lexpr' in args:
            lexpr = args['lexpr']
            if isinstance(lexpr, list):
                self.lexpr = []
                for arg in lexpr:
                    self.lexpr.append(sql_expr_analyze(arg, env))
            else:
                self.lexpr = sql_expr_analyze(lexpr, env)
        else:
            self.lexpr = None
        if 'rexpr' in args:
            rexpr = args['rexpr']
            if isinstance(rexpr, list):
                self.rexpr = []
                for arg in rexpr:
                    self.rexpr.append(sql_expr_analyze(arg, env))
            else:
                self.rexpr = sql_expr_analyze(rexpr, env)
        else:
            self.rexpr = None

    @property
    def concerned_aliases(self):
        res = set()
        if self.lexpr:
            if isinstance(self.lexpr, list):
                for expr in self.lexpr:
                    res |= expr.concerned_aliases
            else:
                res |= self.lexpr.concerned_aliases
        if self.rexpr:
            if isinstance(self.rexpr, list):
                for expr in self.rexpr:
                    res |= expr.concerned_aliases
            else:
                res |= self.rexpr.concerned_aliases
        return res

    @property
    def concerned_columns(self):
        res = set()
        if self.lexpr:
            if isinstance(self.lexpr, list):
                for expr in self.lexpr:
                    res |= expr.concerned_columns
            else:
                res |= self.lexpr.concerned_columns
        if self.rexpr:
            if isinstance(self.rexpr, list):
                for expr in self.rexpr:
                    res |= expr.concerned_columns
            else:
                res |= self.rexpr.concerned_columns
        return res

    def to_str(self, oracle=False):
        if oracle:
            if isinstance(self.lexpr, list):
                lexpr = f'({", ".join(map(lambda x: x.oracle(), self.lexpr))})'
            else:
                lexpr = self.lexpr.oracle()
            if self.kind in (self.BETWEEN, self.NOT_BETWEEN):
                rexpr = f'{self.rexpr[0].oracle()} AND {self.rexpr[1].oracle()}'
            else:
                if isinstance(self.rexpr, list):
                    rexpr = f'({", ".join(map(lambda x: x.oracle(), self.rexpr))})'
                else:
                    rexpr = self.rexpr.oracle()
        else:
            if isinstance(self.lexpr, list):
                lexpr = f'({", ".join(map(str, self.lexpr))})'
            else:
                lexpr = str(self.lexpr)
            if self.kind in (self.BETWEEN, self.NOT_BETWEEN):
                rexpr = f'{self.rexpr[0]} AND {self.rexpr[1]}'
            else:
                if isinstance(self.rexpr, list):
                    rexpr = f'({", ".join(map(str, self.rexpr))})'
                else:
                    rexpr = str(self.rexpr)

        if self.kind == self.LIKE:
            # like
            if self.name == '!~~':
                return f'{lexpr} NOT LIKE {rexpr}'
            return f'{lexpr} LIKE {rexpr}'
        if self.kind == self.ILIKE:
            # ilike
            if self.name == '!~~*':
                return f'{lexpr} NOT ILIKE {rexpr}'
            return f'{lexpr} ILIKE {rexpr}'
        if self.kind == self.IN:
            if self.name == '<>':
                return f'{lexpr} NOT IN {rexpr}'
            return f'{lexpr} IN {rexpr}'
        if self.kind == self.BETWEEN:
            return f'{lexpr} BETWEEN {rexpr}'
        if self.kind == self.NOT_BETWEEN:
            return f'{lexpr} NOT BETWEEN {rexpr}'
        assert self.kind == self.ARITHMETIC, f'Unknown operator {self.name}'
        return f'{lexpr} {self.name} {rexpr}'

    def swap(self, inplace=False):
        supported = {
            '<': '>',
            '<=': '>=',
            '=': '=',
            '<>': '<>',
            '>': '<',
            '>=': '<=',
        }
        assert self.kind == self.ARITHMETIC and self.name in supported, f'Unsupported operator type: {self.kind} {self.name}'
        if inplace:
            res = self
        else:
            res = self.clone()
        res.lexpr, res.rexpr = res.rexpr, res.lexpr
        res.name = supported[res.name]
        return res

    def __str__(self):
        return self.to_str(False)

    def oracle(self):
        return self.to_str(True)

class SubLink(Element):
    def default_setup(self, env=None):
        self.type = None
        self.test_expr = None
        self.op = None
        self.subselect = None

    def setup(self, args : dict, env=None):
        self.type = args['subLinkType']
        if 'testexpr' in args:
            # TODO: is this a bug? env is None?
            self.test_expr = sql_expr_analyze(args['testexpr'], env)
            self.op = args['operName'][0]['String']['str']
        else:
            self.test_expr = None
            self.op = None
        self.subselect = SelectStatement(args['subselect'], env)

    def is_column(self):
        return False

    @property
    def concerned_aliases(self):
        res = self.test_expr.concerned_aliases if self.test_expr else set()
        res |= self.subselect.concerned_aliases
        return res

    @property
    def concerned_columns(self):
        res = self.test_expr.concerned_columns if self.test_expr else set()
        res |= self.subselect.concerned_columns
        return res

    def to_str(self, oracle=False):
        if self.type == 0:
            sublink = 'exists'
        elif self.type == 1:
            sublink = 'all'
        elif self.type == 2:
            sublink = 'any'
        elif self.type == 4:
            sublink = ''
        else:
            raise Exception(f'Unknown sublink type: {self.type}')
        if self.test_expr is not None:
            if oracle:
                test_expr = self.test_expr.oracle()
            else:
                test_expr = str(self.test_expr)
            return f'{test_expr} {self.op} {sublink}({self.subselect})'
        return f'{sublink}({self.subselect})'

    def __str__(self):
        return self.to_str(False)

    def oracle(self):
        return self.to_str(True)

class FuncCall(Element):
    def default_setup(self, env=None):
        self.name = None
        self.class_name = None
        self.star = False
        self.args = []
        self.column_ref = None
        self.over = None
        self.distinct = False

    def setup(self, args : dict, env=None):
        name = args['_FuncName']
        if name is None:
            if len(args['funcname']) == 1:
                self.name = args['funcname'][0]['String']['str']
                self.class_name = None
            else:
                self.name = args['funcname'][1]['String']['str']
                self.class_name = args['funcname'][0]['String']['str']
        else:
            self.name = name
            self.class_name = None
        self.star = 'agg_star' in args
        self.args = []
        if not self.star:
            _args = args.get('args', [])
            for arg in _args:
                self.args.append(sql_expr_analyze(arg, env))
        if len(self.args) == 1 and isinstance(self.args[0], ColumnRef):
            self.column_ref = self.args[0]
        else:
            self.column_ref = None
        if 'over' in args:
            self.over = sql_expr_analyze(args['over'], env)
        else:
            self.over = None
        self.distinct = args.get('agg_distinct', False)

    @property
    def concerned_aliases(self):
        res = self.over.concerned_aliases if self.over else set()
        for arg in self.args:
            res |= arg.concerned_aliases
        return res

    @property
    def concerned_columns(self):
        res = self.over.concerned_columns if self.over else set()
        for arg in self.args:
            res |= arg.concerned_columns
        return res

    @property
    def alias(self):
        if self.column_ref:
            return self.column_ref.alias
        return None

    @property
    def column_name(self):
        if self.column_ref:
            return self.column_ref.column_name
        return None

    def to_str(self, oracle=False):
        if self.class_name is not None:
            name = f'{self.class_name}.{self.name}'
        else:
            name = self.name
        if self.over is not None:
            over = f' over ({self.over})'
        else:
            over = ''
        if self.distinct:
            distinct = 'distinct '
        else:
            distinct = ''
        if self.star:
            return f'{name}(*){over}'
        if oracle:
            # TODO: oracle representation?
            return f'{name}({distinct}{", ".join(map(lambda x: x.oracle(), self.args))}){over}'
        return f'{name}({distinct}{", ".join(map(str, self.args))}){over}'

    def __str__(self):
        return self.to_str(False)

    def oracle(self):
        return self.to_str(True)

class TargetTable(Element):
    @classmethod
    def star_target(cls):
        res = cls()
        res.element = ColumnRef.from_name(star=True)
        res.column_ref = True
        return res

    @classmethod
    def from_column_ref(cls, column_ref : ColumnRef):
        res = cls()
        res.element = column_ref
        res.column_ref = True
        return res

    def default_setup(self, env=None):
        self.name = None
        self.element = None
        self.column_ref = False
        self.func_call = False

    def setup(self, args : dict, env=None):
        arg = args['val']
        self.name = args.get('name', None)
        self.element = sql_expr_analyze(arg, env)
        self.column_ref = isinstance(self.element, ColumnRef)
        self.func_call = isinstance(self.element, FuncCall)

    @property
    def concerned_aliases(self):
        return self.element.concerned_aliases

    @property
    def concerned_columns(self):
        return self.element.concerned_columns

    @property
    def res_name(self):
        if self.name is not None:
            return self.name
        if self.func_call:
            if self.element.class_name is None:
                return self.element.name
            return f'{self.element.class_name}.{self.element.name}'
        if self.column_ref:
            return self.element.column_name
        return '?'

    @property
    def star(self):
        if self.func_call:
            return self.element.star
        if self.column_ref:
            return self.element.star
        return False

    @property
    def value(self):
        if self.func_call:
            self.element : FuncCall
            if self.element.column_ref is not None:
                return str(self.element.column_ref)
            return None
        if self.column_ref:
            return str(self.element)
        return None

    @property
    def alias(self):
        if self.func_call:
            if self.element.column_ref is not None:
                return self.element.column_ref.alias
        if self.column_ref:
            return self.element.alias
        return None

    @property
    def column_name(self):
        if self.func_call:
            if self.element.column_ref is not None:
                return self.element.column_ref.column_name
        if self.column_ref:
            return self.element.column_name
        return None

    def to_str(self, oracle=False):
        if self.name is not None:
            if not _re.match('[A-Za-z_][A-Za-z0-9_]*', self.name):
                name = f'"{self.name}"'
            else:
                name = self.name
            as_clause = f' AS {name}'
        else:
            as_clause = ''
        if oracle:
            return self.element.oracle() + as_clause
        return str(self.element) + as_clause

    def oracle(self):
        return self.to_str(True)

    def __str__(self):
        return self.to_str(False)

class FromTable(Element):
    def default_setup(self, env=None):
        self.fullname = None
        self.aliasname = None

    def setup(self, args : dict, env=None):
        self.fullname = args["relname"]
        self.aliasname = args["alias"]["Alias"]["aliasname"] if 'alias' in args else None

    @property
    def concerned_aliases(self):
        return {self.alias}

    @property
    def concerned_columns(self):
        return set()

    @property
    def alias(self):
        if self.aliasname is not None:
            return self.aliasname
        return self.fullname

    def __str__(self):
        return self.fullname + " AS " + self.alias

    def oracle(self):
        return self.oracle_str()

    def oracle_str(self):
        return self.fullname + " " + self.alias


class Subquery(Element):
    def default_setup(self, env=None):
        self.alias = None
        self.subquery = None

    def setup(self, args : dict, env=None):
        sql = args['subquery']
        self.alias = args['alias']['Alias']['aliasname']
        column_names = args.get('colnames', None)
        if column_names:
            targets = sql['SelectStmt']['targetList']
            for target, column_name in zip(targets, column_names):
                target['ResTarget']['name'] = column_name
        self.subquery = SelectStatement(sql, env=env)

    @property
    def concerned_aliases(self):
        return self.subquery.concerned_aliases | {self.alias}

    @property
    def concerned_columns(self):
        return self.subquery.concerned_columns

    def __str__(self):
        return f'({str(self.subquery)}) {self.alias}'

class GroupingSet(Element):
    def default_setup(self, env=None):
        self.kind = None
        self.content = []

    def setup(self, args : dict, env=None):
        self.kind = args['kind']
        content = args.get('content', [])
        self.content = []
        for element in content:
            self.content.append(sql_expr_analyze(element, env))

    @property
    def concerned_aliases(self):
        res = set()
        for arg in self.content:
            res |= arg.concerned_aliases
        return res

    @property
    def concerned_columns(self):
        res = set()
        for arg in self.content:
            res |= arg.concerned_columns
        return res

    def __str__(self):
        if self.kind == 2:
            name = 'rollup'
        elif self.kind == 3:
            name = 'cube'
        elif self.kind == 4:
            name = 'grouping sets'
        else:
            name = ''
        args = ', '.join(map(str, self.content))
        return f'{name}({args})'

class GroupClause(Element):
    def default_setup(self, env=None):
        self.element = None

    def setup(self, args : dict, env=None):
        self.element = sql_expr_analyze(args, env)

    @property
    def concerned_aliases(self):
        return self.element.concerned_aliases

    @property
    def concerned_columns(self):
        return self.element.concerned_columns

    @property
    def alias(self):
        if isinstance(self.element, ColumnRef):
            return self.element.alias
        return None

    @property
    def column_name(self):
        if isinstance(self.element, ColumnRef):
            return self.element.column_name
        return None

    def __str__(self):
        return str(self.element)

    def oracle(self):
        return self.element.oracle()

class OrderClause(Element):
    def default_setup(self, env=None):
        self.direction = None
        self.element = None

    def setup(self, args : dict, env=None):
        o = args['SortBy']
        self.direction = o['sortby_dir']
        node = o['node']
        self.element = sql_expr_analyze(node, env)

    @property
    def concerned_aliases(self):
        return self.element.concerned_aliases

    @property
    def concerned_columns(self):
        return self.element.concerned_columns

    @property
    def alias(self):
        if isinstance(self.element, ColumnRef):
            return self.element.alias
        return None

    @property
    def column_name(self):
        if isinstance(self.element, ColumnRef):
            return self.element.column_name
        return None

    def __str__(self):
        if self.direction == 1:
            direction = ' asc'
        elif self.direction == 2:
            direction = ' desc'
        else:
            direction = ''
        return f'{str(self.element)}{direction}'

    def oracle(self):
        if self.direction == 1:
            direction = ' asc'
        elif self.direction == 2:
            direction = ' desc'
        else:
            direction = ''
        return f'{self.element.oracle()}{direction}'

class LimitClause(Element):
    @classmethod
    def from_value(cls, value):
        if not isinstance(value, int):
            raise ValueError(f"'{value}' is not an integer")
        res = cls()
        res.element = Const.from_value(value)
        return res

    def default_setup(self, env=None):
        self.element = None

    def setup(self, args : dict, env=None):
        self.element = sql_expr_analyze(args, env)

    @property
    def concerned_aliases(self):
        return self.element.concerned_aliases

    @property
    def concerned_columns(self):
        return self.element.concerned_columns

    def __str__(self):
        return f'LIMIT {self.element}'

    def oracle(self):
        return f'rownum <= {self.element}'

class HavingClause(Element):
    def default_setup(self, env=None):
        self.element = None

    def setup(self, args : dict, env=None):
        self.element = sql_expr_analyze(args, env)

    @property
    def concerned_aliases(self):
        return self.element.concerned_aliases

    @property
    def concerned_columns(self):
        return self.element.concerned_columns

    def __str__(self):
        return f'HAVING {self.element}'

class SelectStatement(Element):
    def default_setup(self, env=None):
        self.from_tables = []
        self.from_subqueries = []
        self.target_tables = []
        self.where_clause = None
        self.distinct = False
        self.group_clauses = []
        self.having_clause = None
        self.order_clauses = []
        self.limit_clause = None

    def setup(self, args : dict, env=None):
        self.from_tables = []
        self.from_subqueries = []
        for x in args['fromClause']:
            if 'RangeVar' in x:
                self.from_tables.append(FromTable(x['RangeVar'], env))
            else:
                self.from_subqueries.append(Subquery(x['RangeSubselect'], env))

        alias_to_table = {x.alias : x.fullname for x in self.from_tables}
        new_env = env.clone()
        new_env.alias_to_table.update(alias_to_table)

        self.target_tables = [TargetTable(x["ResTarget"], new_env) for x in args["targetList"]] \
                if "targetList" in args else [TargetTable.star_target()]

        if 'whereClause' in args:
            self.where_clause = sql_expr_analyze(args['whereClause'], new_env)
        else:
            self.where_clause = None

        self.distinct = 'distinctClause' in args

        self.group_clauses = []
        if 'groupClause' in args:
            for clause in args['groupClause']:
                self.group_clauses.append(GroupClause(clause, new_env))

        self.having_clause = HavingClause(args['havingClause']) if 'havingClause' in args else None

        self.order_clauses = []
        if 'sortClause' in args:
            for clause in args['sortClause']:
                self.order_clauses.append(OrderClause(clause, new_env))

        self.limit_clause = LimitClause(args['limitCount'],
                                        new_env) if 'limitCount' in args else None

    @property
    def concerned_aliases(self):
        from_aliases = set()
        for ft in self.from_tables:
            from_aliases |= ft.concerned_aliases

        target_aliases = set()
        for tt in self.target_tables:
            target_aliases |= tt.concerned_aliases

        where_aliases = self.where_clause.concerned_aliases if self.where_clause else set()

        others = [*self.group_clauses, *self.order_clauses]
        if self.having_clause:
            others.append(self.having_clause)
        if self.limit_clause:
            others.append(self.limit_clause)
        other_aliases = set()
        for other in others:
            other_aliases |= other.concerned_aliases

        return (target_aliases | where_aliases | other_aliases) - from_aliases

    @property
    def concerned_columns(self):
        from_aliases = set()
        for ft in self.from_tables:
            from_aliases |= ft.concerned_aliases

        target_columns = set()
        for tt in self.target_tables:
            target_columns |= tt.concerned_columns

        where_columns = self.where_clause.concerned_columns if self.where_clause else set()

        others = [*self.group_clauses, *self.order_clauses]
        if self.having_clause:
            others.append(self.having_clause)
        if self.limit_clause:
            others.append(self.limit_clause)
        other_columns = set()
        for other in others:
            other_columns |= other.concerned_columns

        all_columns = target_columns | where_columns | other_columns
        res = set()
        for col in all_columns:
            t, c = col
            if not t in from_aliases:
                res.add(col)

        return res

    def tail_clauses(self, oracle=False):
        if oracle:
            _str = lambda x: x.oracle()
        else:
            _str = str
        if self.group_clauses:
            group = f' group by {", ".join(map(_str, self.group_clauses))}'
        else:
            group = ''
        if self.having_clause is not None:
            having = f' having {_str(self.having_clause)}'
        else:
            having = ''
        if self.order_clauses:
            order = f' order by {", ".join(map(_str, self.order_clauses))}'
        else:
            order = ''
        if self.limit_clause is not None and not oracle:
            limit = f' {_str(self.limit_clause)}'
        else:
            limit = ''
        res = f'{group}{having}{order}{limit}'
        return res

    def __str__(self):
        res = [
            'SELECT ',
            'DISTINCT ' if self.distinct else '',
            ',\n'.join(str(x) for x in self.target_tables),
        ]

        from_tables = self.from_tables + self.from_subqueries
        if from_tables:
            res.extend((
                '\nFROM ',
                ',\n'.join(str(x) for x in from_tables),
            ))

        if self.where_clause:
            res.extend((
                '\nWHERE ',
                str(self.where_clause),
                '\n',
            ))

        res.append(self.tail_clauses(oracle=False))
        return ''.join(res)

    def oracle(self):
        res = [
            'SELECT ',
            #'DISTINCT ' if self.distinct else '',
            ',\n'.join(x.oracle() for x in self.target_tables),
        ]

        from_tables = self.from_tables + self.from_subqueries
        if from_tables:
            res.extend((
                '\nFROM ',
                ',\n'.join(x.oracle() for x in from_tables),
            ))

        if self.where_clause:
            res.extend((
                '\nWHERE ',
                self.where_clause.oracle(),
                '\n',
            ))

        res.append(self.tail_clauses(oracle=True))
        res = ''.join(res)

        if self.limit_clause is not None:
            res = f'SELECT * FROM ({res}) WHERE {self.limit_clause.oracle()}'

        return res

def parse_select(sql, parser_env : ParserEnvironment = None):
    """
    Returns the parsed object of the first select statement.
    """
    if parser_env is None:
        parser_env = ParserEnvironment()

    parse_result_all = _psqlparse.parse_dict(sql)

    select = None
    for i, p in enumerate(parse_result_all):
        if select is None and 'SelectStmt' in p:
            select = i
            break
    assert select is not None, 'No select statement'

    parse_result = parse_result_all[select]['SelectStmt']

    # sql_arr = list(map(lambda x: x + ';', filter(lambda x: x.strip(), sql.split(';'))))
    # pre_actions = sql_arr[:select]
    # post_actions = sql_arr[select + 1:]

    return SelectStatement(parse_result, parser_env)

class SqlBase:
    def __init__(self, element):
        self.element = element
        self.__init()

    def __str__(self):
        return str(self.element)

    def oracle(self):
        return self.element.oracle()

    def __init(self):
        self.alias_to_table = {t.alias : t for t in self.element.from_tables}
        self.aliases = set(self.alias_to_table.keys())

        if self.element.where_clause is None:
            # no conditions
            conditions = []
        elif isinstance(self.element.where_clause, BoolExpr) and \
                self.element.where_clause.op == BoolExpr.AND:
            # multiple conditions combined with AND operator
            conditions = self.element.where_clause.args
        else:
            # other expressions, might be complicated
            conditions = [self.element.where_clause]

        eqjoin_predicates = []
        neqjoin_predicates = []
        complicated_predicates = []
        filter_predicates = []
        constant_predicates = []

        for condition in conditions:
            if isinstance(condition, MathExpr):
                # Comparison between columns or complicated conditions
                lexpr_is_column = isinstance(condition.lexpr, ColumnRef)
                rexpr_is_column = isinstance(condition.rexpr, ColumnRef)
                if lexpr_is_column and rexpr_is_column:
                    # join predicate
                    assert condition.kind == MathExpr.ARITHMETIC, f'unknown condition type for join predicate: {condition.kind} {condition.name}'
                    if condition.lexpr.alias == condition.rexpr.alias:
                        # self join, considered as a filter
                        filter_predicates.append(condition)
                    elif condition.name == '=':
                        # equi-join
                        eqjoin_predicates.append(condition)
                    else:
                        neqjoin_predicates.append(condition)
                elif lexpr_is_column or rexpr_is_column:
                    if rexpr_is_column:
                        print(f'swap operands: {condition} -> {condition.swap(inplace=False)}', file=sys.stderr)
                        condition.swap(inplace=True)
                    # the rexpr might be a list, for example: a.year in (2001, 2003, 2005)
                    rexpr_columns = concerned_columns(condition.rexpr)
                    if rexpr_columns:
                        # complicated conditions (e.g.: a > (b / c + 100))
                        rexpr_aliases = set(map(lambda x: x[0], rexpr_columns))
                        if len(rexpr_aliases) == 1 and tuple(rexpr_aliases)[0] == condition.lexpr.alias:
                            # complicated condition on only one table
                            filter_predicates.append(condition)
                        elif len(rexpr_columns) == 1:
                            # technically it can be an equi-join, but it cannot be used to infer possible edges
                            neqjoin_predicates.append(condition)
                        else:
                            complicated_predicates.append(condition)
                    else:
                        # filter predicate
                        filter_predicates.append(condition)
                else:
                    # complicated or constant
                    lexpr_columns = concerned_columns(condition.lexpr)
                    rexpr_columns = concerned_columns(condition.rexpr)
                    if lexpr_columns and rexpr_columns:
                        lexpr_aliases = set(map(lambda x: x[0], lexpr_columns))
                        rexpr_aliases = set(map(lambda x: x[0], rexpr_columns))
                        if len(lexpr_aliases) == 1 and len(rexpr_aliases) == 1 and tuple(lexpr_aliases)[0] == tuple(rexpr_aliases)[0]:
                            # complicated condition on only one table
                            filter_predicates.append(condition)
                        elif len(lexpr_columns) == 1 and len(rexpr_columns) == 1:
                            neqjoin_predicates.append(condition)
                        else:
                            complicated_predicates.append(condition)
                    elif lexpr_columns or rexpr_columns:
                        if rexpr_columns:
                            print(f'swap operands: {condition} -> {condition.swap(inplace=False)}', file=sys.stderr)
                            condition.swap(inplace=True)
                            lexpr_columns, rexpr_columns = rexpr_columns, lexpr_columns
                        lexpr_aliases = set(map(lambda x: x[0], lexpr_columns))
                        if len(lexpr_aliases) == 1:
                            # filter predicate for only one table, can contain more than one column
                            filter_predicates.append(condition)
                        else:
                            # complicated predicate, including filters for different columns of one table
                            complicated_predicates.append(condition)
                    else:
                        # not related to any column
                        constant_predicates.append(condition)
            else:
                # complicated or constant
                _concerned_columns = concerned_columns(condition)
                if len(_concerned_columns) == 1:
                    filter_predicates.append(condition)
                elif len(_concerned_columns) > 1:
                    complicated_predicates.append(condition)
                else:
                    constant_predicates.append(condition)

        self.eqjoin_predicates = eqjoin_predicates
        self.neqjoin_predicates = neqjoin_predicates
        self.complicated_predicates = complicated_predicates
        self.filter_predicates = filter_predicates
        self.constant_predicates = constant_predicates

        self.__eqjoin_fix()

        self.table_eqjoin_predicates = {}
        for eqjoin in self.eqjoin_predicates:
            left_column, right_column = eqjoin.lexpr, eqjoin.rexpr

            left_table_predicates = self.table_eqjoin_predicates.get(left_column.alias, None)
            if left_table_predicates is None:
                left_table_predicates = []
                self.table_eqjoin_predicates[left_column.alias] = left_table_predicates
            left_table_predicates.append((right_column.alias, eqjoin))

            right_table_predicates = self.table_eqjoin_predicates.get(right_column.alias, None)
            if right_table_predicates is None:
                right_table_predicates = []
                self.table_eqjoin_predicates[right_column.alias] = right_table_predicates
            right_table_predicates.append((left_column.alias, eqjoin))

        self.table_neqjoin_predicates = {}
        for neqjoin in self.neqjoin_predicates:
            left_alias, right_alias = list(concerned_aliases(neqjoin.lexpr))[0], list(concerned_aliases(neqjoin.rexpr))[0]

            left_table_predicates = self.table_neqjoin_predicates.get(left_alias, None)
            if left_table_predicates is None:
                left_table_predicates = []
                self.table_neqjoin_predicates[left_alias] = left_table_predicates
            left_table_predicates.append((right_alias, neqjoin))

            right_table_predicates = self.table_neqjoin_predicates.get(right_alias, None)
            if right_table_predicates is None:
                right_table_predicates = []
                self.table_neqjoin_predicates[right_alias] = right_table_predicates
            right_table_predicates.append((left_alias, neqjoin))

        self.table_complicated_predicates = {}
        self.other_complicated_predicates = []
        for complicated in self.complicated_predicates:
            _concerned_aliases = concerned_aliases(complicated)
            if len(_concerned_aliases) == 1:
                # table complicated filter, theoretically this cannot happen
                alias = list(_concerned_aliases)[0]
                table_complicated_predicates = self.table_complicated_predicates.get(alias, None)
                if table_complicated_predicates is None:
                    table_complicated_predicates = []
                    self.table_complicated_predicates[alias] = table_complicated_predicates
                table_complicated_predicates.append(complicated)
            else:
                # cannot handle more complicated conditions
                self.other_complicated_predicates.append((_concerned_aliases, complicated))

        self.table_filter_predicates = {}
        for filter_ in self.filter_predicates:
            _concerned_columns = concerned_columns(filter_)
            _concerned_aliases = set(map(lambda x: x[0], _concerned_columns))
            assert len(_concerned_aliases) == 1, f'filter predicate with more than one table: {filter_}'
            if len(_concerned_columns) == 1:
                concerned_column = list(_concerned_columns)[0]
                alias, column = concerned_column
                table_filter_predicates = self.table_filter_predicates.get(alias, None)
                if table_filter_predicates is None:
                    table_filter_predicates = {}
                    self.table_filter_predicates[alias] = table_filter_predicates
                column_filter_predicates = table_filter_predicates.get(column, None)
                if column_filter_predicates is None:
                    column_filter_predicates = []
                    table_filter_predicates[column] = column_filter_predicates
                column_filter_predicates.append(filter_)
            else:
                # complicated filter predicate on only one table
                alias = list(_concerned_aliases)[0]
                table_complicated_predicates = self.table_complicated_predicates.get(alias, None)
                if table_complicated_predicates is None:
                    table_complicated_predicates = []
                    self.table_complicated_predicates[alias] = table_complicated_predicates
                table_complicated_predicates.append(filter_)

    def __eqjoin_fix(self):
        disjoint_set = DisjointSet()
        edge_set = set()
        for eqjoin in self.eqjoin_predicates:
            left_column, right_column = eqjoin.lexpr, eqjoin.rexpr
            assert isinstance(left_column, ColumnRef) and isinstance(right_column, ColumnRef), f'{eqjoin}'
            left_alias_column = (left_column.alias, left_column.column_name)
            right_alias_column = (right_column.alias, right_column.column_name)
            disjoint_set.set(left_alias_column, right_alias_column)
            edge_set.add((left_alias_column, right_alias_column))
            edge_set.add((right_alias_column, left_alias_column))

        joined_sets = disjoint_set.to_dict()
        for joined_set in joined_sets.values():
            joined_set = list(joined_set)
            for i in range(len(joined_set)):
                for j in range(i + 1, len(joined_set)):
                    left_alias_column, right_alias_column = joined_set[i], joined_set[j]
                    edge = (left_alias_column, right_alias_column)
                    if edge not in edge_set:
                        # hidden join condition
                        new_column_left = ColumnRef.from_name(left_alias_column[1], left_alias_column[0])
                        new_column_right = ColumnRef.from_name(right_alias_column[1], right_alias_column[0])
                        new_condition = MathExpr()
                        new_condition.kind = MathExpr.ARITHMETIC
                        new_condition.name = '='
                        new_condition.lexpr = new_column_left
                        new_condition.rexpr = new_column_right

                        self.eqjoin_predicates.append(new_condition)

class Operators(IntEnum):
    default = 0b000
    nested_loop_join = 0b110
    merge_join = 0b101
    hash_join = 0b011
    no_nested_loop_join = 0b001
    no_merge_join = 0b010
    no_hash_join = 0b100

    leading = 0b11111111

    @classmethod
    def hint_name(cls, value, oracle=False):
        if oracle:
            return {
                cls.default: None,
                cls.nested_loop_join: 'USE_NL',
                cls.merge_join: 'USE_MERGE',
                cls.hash_join: 'USE_HASH',
                cls.no_nested_loop_join: 'NO_USE_NL',
                cls.no_merge_join: 'NO_USE_MERGE',
                cls.no_hash_join: 'NO_USE_HASH',
                cls.leading: 'LEADING',
            }.get(value, None)
        return {
            cls.default: None,
            cls.nested_loop_join: 'NestLoop',
            cls.merge_join: 'MergeJoin',
            cls.hash_join: 'HashJoin',
            cls.no_nested_loop_join: 'NoNestLoop',
            cls.no_merge_join: 'NoMergeJoin',
            cls.no_hash_join: 'NoHashJoin',
            cls.leading: 'Leading',
        }.get(value, None)

    @classmethod
    def to_onehot(cls, value):
        res = [0 for i in range(7)]
        if value in (
            cls.nested_loop_join, cls.merge_join, cls.hash_join,
            cls.no_nested_loop_join, cls.no_merge_join, cls.no_hash_join,
        ):
            res[value.value] = 1
        else:
            res[0] = 1
        return tuple(res)

    @classmethod
    def to_binary(cls, value):
        if value in (
            cls.nested_loop_join, cls.merge_join, cls.hash_join,
            cls.no_nested_loop_join, cls.no_merge_join, cls.no_hash_join,
        ):
            res = [0 for i in range(3)]
            label = value | 0x111
            for i in range(3):
                res[i] = 1 - (label & 1)
                label >>= 1
        else:
            res = (1 for i in range(3))
        return tuple(res)

class BranchNode:
    def __init__(self, index, left_child, right_child, join_method=Operators.default):
        self.index = index
        self.left_child = left_child
        self.right_child = right_child
        self.join_method = join_method

    def __str__(self):
        return f'{self.__class__.__name__}(index={self.index}, left_child={self.left_child}, right_child={self.right_child}, join_method={self.join_method.name})'

    def __repr__(self):
        return self.__str__()


class PlanParser:
    def __init__(self, plan : dict):
        self._aliases = set()
        self._root_node = None
        self._branch_nodes = []
        self._parent = {}
        self._attrs = {}
        self._joined_leaf_nodes = set()
        self._join_candidates = []
        self._is_left_deep = True
        self._parse(plan)

    @property
    def is_left_deep(self):
        return self._is_left_deep

    def _get_alias_from_node_attrs(self, attrs):
        res = []
        while attrs is not None:
            if 'Alias' in attrs:
                res.append(attrs['Alias'])
            attrs = attrs.get('Plans', None)
        if res:
            return res[-1]
        return None

    def _parse(self, plan: dict, parent=None, right_tree=False, attrs=None):
        node_type = plan['Node Type']
        this_node_attrs = self._attributes(plan)
        if attrs is None:
            node_attrs = this_node_attrs
        else:
            _attrs = attrs
            while 'Plans' in _attrs:
                _attrs = _attrs['Plans']
            _attrs['Plans'] = this_node_attrs
            node_attrs = attrs

        child_plans = plan.get('Plans', None)
        if child_plans is None:
            # a leaf node (generally a scan operator)
            if not node_type.endswith('Scan'):
                print(f"Warning: unknown leaf node type '{node_type}'", file=sys.stderr)

            alias = plan.get('Alias', self._get_alias_from_node_attrs(node_attrs))
            if alias is None:
                raise RuntimeError(f"cannot get alias from plan: {node_attrs}")
            if alias in self._aliases:
                raise RuntimeError(f"alias '{alias}' appears twice in the plan")
            self._aliases.add(alias)
            self._attrs[alias] = node_attrs
            self._parent[alias] = parent
            if parent is not None:
                parent_node = self._branch_nodes[parent]
                if right_tree:
                    parent_node.right_child = alias
                else:
                    parent_node.left_child = alias
            else:
                self._root_node = alias
        elif len(child_plans) == 2:
            # a branch join node
            if node_type not in ('Nested Loop', 'Merge Join', 'Hash Join'):
                print(f"Warning: unknown branch node type '{node_type}'", file=sys.stderr)

            plans = plan['Plans']
            if right_tree:
                # this node is in the right tree
                self._is_left_deep = False
            node_index = len(self._branch_nodes)
            join_type = {
                'Nested Loop': Operators.nested_loop_join,
                'Merge Join': Operators.merge_join,
                'Hash Join': Operators.hash_join,
            }.get(node_type, Operators.default)
            new_node = BranchNode(node_index, None, None, join_type)
            self._branch_nodes.append(new_node)
            self._attrs[node_index] = node_attrs
            self._parent[node_index] = parent
            if parent is not None:
                parent_node = self._branch_nodes[parent]
                if right_tree:
                    parent_node.right_child = node_index
                else:
                    parent_node.left_child = node_index
            else:
                self._root_node = node_index
            left, right = plans
            self._parse(left, parent=node_index, right_tree=False)
            self._parse(right, parent=node_index, right_tree=True)
        elif len(child_plans) == 1:
            # pass the node attributes to the child node
            self._parse(child_plans[0], parent=parent, right_tree=right_tree, attrs=node_attrs)
        else:
            # theoretically this will not happen
            raise RuntimeError(f"Unknown node type '{node_type}': plan")
        for condition_name in ('Recheck Cond', 'Index Cond'):
            condition = plan.get(condition_name, None)
            if condition is None:
                continue
            alias = plan.get('Alias', None)
            if alias is not None:
                self._parse_condition(condition, alias)
        for condition_name in ('Hash Cond', 'Merge Cond', 'Join Filter'):
            condition = plan.get(condition_name, None)
            if condition is None:
                continue
            self._parse_condition(condition)

    def _parse_condition(self, condition, alias=None):
        if _re.search(r'\sand\s', condition, _re.IGNORECASE):
            # remove parenthesis and split
            conditions = _re.split('\sand\s', condition[1:-1], flags=_re.IGNORECASE)
            # simply split the conditions and parse them
            for _condition in conditions:
                self._parse_condition(_condition, alias=alias)
            return

        joined = [] if alias is None else [alias]
        joined.extend(_re.findall(r'[( ]([A-Za-z0-9_]+)\.', condition))
        if len(joined) != 2:
            # print(f"Warning: the number of joined tables is not 2: {', '.join(joined)}", file=sys.stderr)
            return
        left_alias, right_alias = joined
        left_joined, right_joined = left_alias in self._joined_leaf_nodes, right_alias in self._joined_leaf_nodes

        if self._joined_leaf_nodes and not left_joined and not right_joined:
            # already has 2 leaf nodes joined, but these two leaf nodes are both not joined,
            # which indicates these two leaves create another subtree, and thus the plan is not left-deep
            self._is_left_deep = False
        if not left_joined or not right_joined:
            if left_joined:
                join_candidate = (left_alias, right_alias)
            else:
                join_candidate = (right_alias, left_alias)
            self._join_candidates.append(join_candidate)
            self._joined_leaf_nodes.update(join_candidate)

    def _attributes(self, plan: dict):
        return {k: v for k, v in filter(lambda x: x[0] != 'Plans', plan.items())}

    def _nodes_join_order(self):
        nodes = []
        stack = [self._root_node]
        while stack:
            # dfs traverse
            node = stack.pop()
            if isinstance(node, int):
                branch_node = self._branch_nodes[node]
                stack.append(branch_node.right_child)
                stack.append(branch_node.left_child)
                nodes.append(branch_node)
        nodes.reverse()
        return nodes

    @property
    def join_order(self):
        _cache = getattr(self, '_join_order_cache', None)
        if _cache is not None:
            return _cache

        nodes = self._nodes_join_order()
        node_mapping_dict = {}
        res = []
        for branch_node in nodes:
            parent, left, right, join = branch_node.index, branch_node.left_child, branch_node.right_child, branch_node.join_method
            # assign a new node index for the parent node
            new_index = len(node_mapping_dict)
            node_mapping_dict[parent] = new_index
            left = node_mapping_dict.get(left, left)
            right = node_mapping_dict.get(right, right)
            res.append((left, right, join))
        res = tuple(res)
        setattr(self, '_join_order_cache', res)
        return res

    @property
    def join_order_by_aliases(self):
        _cache = getattr(self, '_join_order_by_aliases_cache', None)
        if _cache is not None:
            return _cache

        nodes = self._nodes_join_order()
        candidates = set(self._join_candidates)
        children = {k : {k} for k in self._aliases}
        res = []
        for branch_node in nodes:
            children[branch_node.index] = children[branch_node.left_child] | children[branch_node.right_child]
            for left_alias, right_alias in candidates:
                if left_alias in children[branch_node.left_child] and right_alias in children[branch_node.right_child]:
                    res.append((left_alias, right_alias))
                    candidates.remove((left_alias, right_alias))
                    break
                elif right_alias in children[branch_node.left_child] and left_alias in children[branch_node.right_child]:
                    res.append((right_alias, left_alias))
                    candidates.remove((left_alias, right_alias))
                    break
        if len(res) != len(nodes):
            # failed to reconstruct join order
            res = ()
        else:
            res = tuple(res)
        setattr(self, '_join_order_by_aliases_cache', res)
        return res

    def __str__(self):
        return str(self.join_order)

    def plan_attributes(self, plan):
        """
        Map attributes to plan nodes. 'plan' is a PlanBase object.
        """
        to_plan_indices = {k : k for k in self._aliases}
        # traverse the tree from bottom
        def traverse(node, visited):
            visited.add(node)
            this_parent = self._parent.get(node, None)
            if this_parent is None:
                return
            plan_node = to_plan_indices.get(node, None)
            if plan_node is not None:
                plan_parent = plan.parent[plan_node]
                if plan_parent is not None:
                    to_plan_indices[this_parent] = plan_parent
            traverse(this_parent, visited=visited)
        visited = set()
        for alias in self._aliases:
            traverse(alias, visited)
        res = {}
        for alias, plan_alias in to_plan_indices.items():
            res[plan_alias] = self._attrs[alias]
        return copy.deepcopy(res)
