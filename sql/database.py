import os
import sys
import asyncio
import re
import time
import pickle
from collections.abc import Iterable
from collections import OrderedDict

import psycopg as pg

from lib.database import postgres


def sql_repr(value):
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value)

class Postgres:
    def __init__(self, *, timeout=1000000, retry_limit=3, auto_save_interval=400, auto_save_path=None):
        self._dbname = None
        self._connection = None

        self.auto_save_interval = auto_save_interval
        self.retry_limit = retry_limit
        self._timeout = timeout

        self._auto_save_count = 0
        self._auto_save_cache = {}
        self._auto_save_path = auto_save_path
        self._hooks = {}

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        self._timeout = value
        if not self._no_connection:
            self.__ensure_execute(f'set statement_timeout = {value};', pre_settings=False)
            _timeout = self.get_settings('statement_timeout')

    def add_hook(self, hook_type, hook_name, callback):
        if hook_type not in self._hooks:
            hooks = OrderedDict()
            self._hooks[hook_type] = hooks
        else:
            hooks = self._hooks[hook_type]
        hooks[hook_name] = callback

    def remove_hook(self, hook_type, hook_name):
        if hook_type not in self._hooks:
            hooks = OrderedDict()
            self._hooks[hook_type] = hooks
        else:
            hooks = self._hooks[hook_type]
        if hook_name not in hooks:
            return None
        callback = hooks[hook_name]
        hooks.pop(hook_name)
        return callback

    def _execute_hooks(self, hook_type, *args, **kwargs):
        if hook_type not in self._hooks:
            return ()
        hooks = self._hooks[hook_type]
        for hook in hooks.values():
            hook(*args, **kwargs)

    @property
    def dbname(self):
        return self._dbname

    @property
    def connection(self):
        return self._connection

    def _auto_save(self):
        self._auto_save_count += 1
        if 0 < self.auto_save_interval <= self._auto_save_count:
            self._auto_save_count = 0
            self._cache_backup()

    @property
    def __auto_save_path(self):
        return f'.{self._dbname}.cache.pkl' if self._auto_save_path is None else self._auto_save_path

    def _cache_backup(self):
        if self._connection is None:
            return
        with open(self.__auto_save_path, 'wb') as f:
            pickle.dump(self._auto_save_cache, f)

    def __cache_load(self):
        if self._connection is None:
            return
        filename = self.__auto_save_path
        if not os.path.isfile(filename):
            return
        with open(filename, 'rb') as f:
            self._auto_save_cache = pickle.load(f)

    def setup(self, dbname, *args, **kwargs):
        if 'no_connection' in kwargs:
            self._no_connection = bool(kwargs['no_connection'])
            kwargs.pop('no_connection')
        else:
            self._no_connection = False
        if 'use_hint' in kwargs:
            self._use_hint = bool(kwargs['use_hint'])
            kwargs.pop('use_hint')
        else:
            self._use_hint = True

        self._dbname = dbname

        self._execute_hooks('before_setup', self)

        if not self._no_connection:
            kwargs['dbname'] = dbname
            self.__db_args = (args, kwargs)
            self.__db_settings = {}
            self.__refresh_cursor()

        self.__cache_load()

        self._execute_hooks('after_setup', self)

    def clear_cache(self):
        self._auto_save_cache.clear()

    def _get_settings_preprocess(self, key, value):
        if key == 'statement_timeout':
            if isinstance(value, str):
                if value.endswith('ms'):
                    return int(value[:-2])
                if value.endswith('s'):
                    return 1000 * int(value[:-1])
                if value.endswith('min'):
                    return 60000 * int(value[:-3])
                return int(value)
            return value
        return value

    def get_settings(self, key):
        assert self._connection is not None
        key = str(key).strip()
        try:
            self.__cur.execute(f"show {key};")
        except Exception as e:
            self.__refresh_cursor(pre_settings=True)
            self.__cur.execute(f"show {key};")
        res = self.__cur.fetchall()[0][0]
        return self._get_settings_preprocess(key, res)

    def set_settings(self, key, value):
        self.__db_settings[key] = value
        if self._connection is not None:
            try:
                self.__cur.execute(f"set {key} = %s;", (value, ))
            except Exception as e:
                self.__refresh_cursor(pre_settings=True)

    def __pre_check(self):
        if self._use_hint:
            assert ('pg_hint_plan' in self.get_settings('session_preload_libraries') +
                    self.get_settings('shared_preload_libraries')), \
                "pg_hint_plan is not loaded"

    def __refresh_cursor(self, pre_settings=True):
        self._connection = postgres.connect(*self.__db_args[0], **self.__db_args[1])
        self.__cur = self._connection.cursor()
        self.__pre_check()
        if pre_settings:
            self.__pre_settings()

    def __ensure_execute(self, commands : Iterable, *args, pre_settings=True, retry_limit=None, **kwargs):
        if isinstance(commands, str):
            commands = (commands, )
        if retry_limit is None:
            retry_limit = self.retry_limit
        for i in range(retry_limit):
            try:
                for command in commands:
                    self.__cur.execute(command, *args, **kwargs)
                break
            except pg.errors.QueryCanceled as e:
                self.__refresh_cursor(pre_settings=pre_settings)
        else:
            raise Exception(f'retry limit ({retry_limit}) exceeded')

    def __pre_settings(self):
        command_list = [
            "set from_collapse_limit = 1;",
            "set join_collapse_limit = 1;",
            f"set geqo_threshold = {1024 if self._use_hint else self.get_settings('geqo_threshold')};",
            f"set statement_timeout = {self.timeout};",
            *(f"set {k} = {sql_repr(v)};" for k, v in self.__db_settings.items())
        ]
        self.__ensure_execute(command_list, pre_settings=False)
        return command_list

    def execute(self, sql, cache=False, retry_limit=None):
        assert self._connection is not None
        if sql in self._auto_save_cache:
            return self._auto_save_cache[sql]
        self.__ensure_execute(sql, retry_limit=retry_limit)
        res = tuple(self.__cur.fetchall())
        if cache:
            self._auto_save_cache[sql] = res
        return res

    def column_boundary(self, table, column):
        table_name = table.split(' ')[-1]
        max_ = self.execute(f"select max({table_name}.{column}) from {table};", cache=True)[0][0]
        min_ = self.execute(f"select min({table_name}.{column}) from {table};", cache=True)[0][0]
        res = (max_, min_)
        return res

    def table_size(self, table):
        total_rows = self.execute(f'select count(*) from {table};', cache=True)[0][0]
        return total_rows

    def selectivity(self, table, where, explain=True, detail=False):
        total_rows = self.table_size(table)
        if explain:
            select_rows = self.execute(f'explain select * from {table} where {where};', cache=True)[0][0]
            select_rows = int(re.search(r'rows=([0-9]+)', select_rows).group(1))
        else:
            select_rows = self.__cur.execute(f'select count(*) from {table} where {where};', cache=True)[0][0]
        res = select_rows / total_rows
        res = (res, select_rows, total_rows)
        if detail:
            return res
        return res[0]

    def cost(self, sql, cache=True):
        res = self.execute(f'EXPLAIN {sql}', cache=cache)[0][0]
        cost = float(res.split("cost=")[1].split("..")[1].split(" ")[0])
        self._connection.commit()
        return cost

    def plan_time(self, sql):
        assert self._connection is not None
        now = time.time()
        self.cost(sql, cache=False)
        res = time.time() - now
        return res

    def plan(self, sql, geqo=False):
        explain_sql = "EXPLAIN (COSTS, FORMAT JSON) " + sql
        geqo_threshold = self.get_settings('geqo_threshold')
        if geqo:
            self.set_settings('geqo_threshold', 2)
        try:
            res = self.execute(explain_sql, cache=False)
        except Exception as e:
            raise e
        finally:
            self.set_settings('geqo_threshold', geqo_threshold)
        return res

    @staticmethod
    async def _async_retry_on_exception(coroutine, retry_limit=None, *args, **kwargs):
        count = 0
        while True:
            try:
                res = await coroutine(*args, **kwargs)
                break
            except asyncio.CancelledError:
                raise
            except Exception:
                if retry_limit is not None:
                    count += 1
                    if count >= retry_limit:
                        raise
        return res

    async def async_plan(self, sql, geqo=False, retry_limit=None):
        async def _async_plan(self, sql, geqo):
            async with await pg.AsyncConnection.connect(*self.__db_args[0], **self.__db_args[1]) as conn:
                async with conn.cursor() as cur:
                    await cur.execute("set from_collapse_limit = 1;")
                    await cur.execute("set join_collapse_limit = 1;")
                    await cur.execute(f'set geqo_threshold = {2 if geqo else 1024};')
                    await cur.execute(f"set statement_timeout = {self.timeout};")
                    await cur.execute("EXPLAIN (COSTS, FORMAT JSON) " + sql)
                    res = await cur.fetchall()
            return res
        res = self._async_retry_on_exception(_async_plan, retry_limit, self, sql, geqo)
        return res

    def plan_latency(self, sql, cache=True):
        res = self.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql, cache=cache, retry_limit=1)
        return res

    def table_rows(self, table_name, filter=None, schema_name=None, time_limit=None):
        assert self._connection is not None
        if time_limit is None:
            return list(
                postgres.filter_iter_table(self._connection, table_name, schema_name=schema_name, filter=filter))

        statement_timeout = self.get_settings('statement_timeout')
        self.set_settings('statement_timeout', time_limit)
        res = None
        try:
            res = list(postgres.filter_iter_table(self._connection, table_name, schema_name=schema_name, filter=filter))
        except:
            pass
        finally:
            self.set_settings('statement_timeout', statement_timeout)
        return res

    def list_foreign_keys(self, multi_columns=False):
        sql = '''select conname as fk_name, pg_class.relname as t_name, 
pg_catalog.pg_get_constraintdef(pg_constraint.oid, true) as fk_def 
from pg_constraint, pg_class 
where pg_class.oid = pg_constraint.conrelid 
and contype = 'f' 
and connamespace = 'public'::regnamespace'''
        res = self.execute(sql)
        re_fk_def = re.compile(r'^foreign key\s+\(([A-Za-z0-9_]+(?:\s*,\s*[A-Za-z0-9_]+)*)\)\s+references\s+([A-Za-z0-9_]+)\(([A-Za-z0-9_]+(?:\s*,\s*[A-Za-z0-9_]+)*)\)', re.IGNORECASE)
        fks = []
        for fk_name, table_name, fk_def in res:
            m = re.match(re_fk_def, fk_def)
            if m:
                column_name, fk_table_name, fk_column_name = m.groups()
                column_name = tuple(map(lambda x: x.strip(), column_name.split(',')))
                fk_column_name = tuple(map(lambda x: x.strip(), fk_column_name.split(',')))
                if not multi_columns:
                    if len(column_name) > 1 or len(fk_column_name) > 1:
                        continue
                    assert len(column_name) == 1 and len(fk_column_name) == 1
                    fks.append((fk_name, (table_name, column_name[0]), (fk_table_name, fk_column_name[0])))
                else:
                    fks.append((fk_name, (table_name, column_name), (fk_table_name, fk_column_name)))
        return fks

    def list_primary_keys(self):
        sql = """SELECT tc.table_name, c.column_name, c.data_type 
FROM information_schema.table_constraints tc 
JOIN information_schema.constraint_column_usage AS ccu USING (constraint_schema, constraint_name) 
JOIN information_schema.columns AS c ON c.table_schema = tc.constraint_schema 
AND tc.table_name = c.table_name AND ccu.column_name = c.column_name WHERE constraint_type = 'PRIMARY KEY'"""
        res = self.execute(sql)
        pks = []
        for table_name, column_name, _ in res:
            pks.append((table_name, column_name))
        return pks

    def has_table(self, table_name, schema_name='public'):
        res = self.execute(f'select count(*) from pg_tables where schemaname = \'{schema_name}\' and tablename = \'{table_name}\'')[0][0]
        return res != 0

    def table_columns(self, table_name, schema_name=None):
        assert self._connection is not None
        if schema_name is None:
            schema_name = 'public'
        return list(map(lambda x: x[1], postgres.table_structure(self._connection, table_name, schema_name=schema_name)))

    def __latency(self, sql, cache=True, return_plan=False):
        res = self.plan_latency(sql, cache=cache)
        plan = res[0][0][0]['Plan']
        cost = plan['Actual Total Time']
        if return_plan:
            return cost, plan
        return cost

    def latency(self, sql, origin=None, return_plan=False, cache=True, throw=False):
        assert self._connection is not None
        latency_cache_key = ('latency', sql)
        if cache and latency_cache_key in self._auto_save_cache:
            if return_plan:
                return self._auto_save_cache[latency_cache_key]
            return self._auto_save_cache[latency_cache_key][0]

        timeout_limit = self.get_settings('statement_timeout')

        if origin is None:
            cost_ratio = None
        else:
            cost = self.cost(sql)
            cost_origin = self.cost(origin)
            cost_ratio = cost / cost_origin

        latency = None
        plan = None
        try:
            latency, plan = self.__latency(sql, cache=cache, return_plan=True)
        except Exception as e:
            self._connection.commit()
            if throw:
                raise e
        finally:
            if latency is None:
                if cost_ratio is None:
                    latency = timeout_limit
                else:
                    latency = min(cost_ratio * self.latency(origin, cache=cache), timeout_limit)
            self._auto_save_cache[latency_cache_key] = (latency, plan)
            self._auto_save()
        if return_plan:
            return latency, plan
        return latency


def postgres_type(type_name):
    if type_name in (
        'bigint', 'int8',
        'bigserial', 'serial8',
        'integer', 'int', 'int4',
        'smallint', 'int2',
        'smallserial', 'serial2',
        'serial', 'serial4',
    ):
        return 1
    if type_name in (
        'double precision', 'float8',
        'numeric', 'decimal',
        'real', 'float4',
    ):
        return 2
    return 0 # non-numeric types


class DataTable:
    def __init__(self, struct, name, schema_name ='public'):
        self.columns = {}
        self.column_indexes = {}
        self.column_types = {}
        self.name = name
        self.schema_name = schema_name
        self.size = len(struct)
        self.row_count = 0
        self.primary_keys = []
        self.foreign_keys = []
        self.foreign_keys_rev = []
        for index, (i, name, typ, *_) in enumerate(struct):
            self.columns[index] = name
            self.column_indexes[name] = index
            self.column_types[name] = typ

    @classmethod
    def from_state_dict(cls, state_dict):
        obj = cls((), None)
        obj.columns = state_dict['columns']
        obj.column_indexes = state_dict['column_indexes']
        obj.name = state_dict['name']
        obj.schema_name = state_dict['schema_name']
        obj.size = state_dict['size']
        obj.column_types = state_dict['column_types']
        obj.row_count = state_dict['row_count']
        obj.primary_keys = state_dict['primary_keys']
        obj.foreign_keys = state_dict['foreign_keys']
        obj.foreign_keys_rev = state_dict['foreign_keys_rev']
        return obj

    def state_dict(self):
        return {
            'columns': self.columns,
            'column_indexes': self.column_indexes,
            'name': self.name,
            'schema_name': self.schema_name,
            'size': self.size,
            'column_types': self.column_types,
            'row_count': self.row_count,
            'primary_keys': self.primary_keys,
            'foreign_keys': self.foreign_keys,
            'foreign_keys_rev': self.foreign_keys_rev,
        }

    def __len__(self):
        return self.size

def _convert_anylist(data : str, is_numeric=False):
    data = data[1:-1]
    in_quote = False
    skip = False
    has_quote = False
    buf = []
    res = []
    for index, c in enumerate(data):
        if skip:
            skip = False
            buf.append(c)
            continue
        if c == '\\':
            skip = True
        elif c == '"':
            in_quote = not in_quote
            has_quote = True
        elif not in_quote and c == ',':
            _res = ''.join(buf)
            if _res == 'null' and not has_quote:
                res.append(None)
            else:
                res.append(_res)
            buf = []
            has_quote = False
        else:
            buf.append(c)
    _res = ''.join(buf)
    if _res == 'null' and not has_quote:
        res.append(None)
    else:
        res.append(_res)
    if is_numeric:
        res = [float(x) if x is not None else None for x in res]
    return res

class PgStats:
    def __init__(self, data, is_numeric):
        self.schema, self.table, self.column, self.inherited, self.null_frac, \
        self.avg_width, self.n_distinct, self.most_common_vals, self.most_common_freqs, \
        self.histogram_bounds, self.correlation, self.most_common_elems, self.most_common_elem_freqs, \
        self.elem_count_histogram = data
        self.is_numeric = is_numeric

        for anylist_field in ('most_common_vals', 'histogram_bounds', 'most_common_elems', 'elem_count_histogram'):
            value = getattr(self, anylist_field)
            if value is not None and isinstance(value, str):
                setattr(self, anylist_field, _convert_anylist(value, self.is_numeric))

    def state_dict(self):
        return {k : getattr(self, k) for k in (
            'schema', 'table', 'column', 'inherited', 'null_frac', 'avg_width',
            'n_distinct', 'most_common_vals', 'most_common_freqs', 'histogram_bounds',
            'correlation', 'most_common_elems', 'most_common_elem_freqs', 'elem_count_histogram',
            'is_numeric',
        )}

    @classmethod
    def from_state_dict(cls, state_dict):
        data = [state_dict[k] for k in (
            'schema', 'table', 'column', 'inherited', 'null_frac', 'avg_width',
            'n_distinct', 'most_common_vals', 'most_common_freqs', 'histogram_bounds',
            'correlation', 'most_common_elems', 'most_common_elem_freqs', 'elem_count_histogram',
        )]
        return cls(data, state_dict['is_numeric'])

    @classmethod
    def _like_to_re(cls, value):
        buf = []
        is_escaped = False
        for c in value:
            if is_escaped:
                if not c in '\\%_':
                    buf.append('\\\\')
                buf.append(c)
                is_escaped = False
                continue
            if c == '\\':
                is_escaped = True
            elif c == '%':
                buf.append('.*')
            elif c == '_':
                buf.append('.')
            elif c in '.[]*+()?^${}|':
                buf.append('\\' + c)
            else:
                buf.append(c)
        regex = ''.join(buf)
        return regex

    def _like_hist_encoding(self, value, hist_lower, hist_upper, case_insensitive=False):
        like_re = self._like_to_re(value)
        re_flag = re.IGNORECASE if case_insensitive else 0
        matched = re.fullmatch(like_re, hist_lower, re_flag)
        if matched:
            return True
        matched = re.fullmatch(like_re, hist_upper, re_flag)
        if matched:
            return True
        return False

    def hist_encoding(self, value, like=False, size=100):
        hist = self.histogram_bounds
        if hist is None:
            hist = self.most_common_vals
            if hist is None:
                return [0 for i in range(size)]
            hist = [(i, i) for i in hist]
        else:
            hist = [(hist[i], hist[i + 1]) for i in range(len(hist) - 1)]
        origin = []
        if not self.is_numeric:
            # string
            for index, (hist_lower, hist_upper) in enumerate(hist):
                if like:
                    origin.append(self._like_hist_encoding(value, hist_lower, hist_upper))
                else:
                    origin.append(value == hist_lower or hist_lower < value < hist_upper)
        else:
            for index, (hist_lower, hist_upper) in enumerate(hist):
                origin.append(value == hist_lower or hist_lower < value < hist_upper)
        if len(origin) < size:
            origin.extend((False for i in range(size - len(origin))))
        else:
            origin = origin[:size]
        return origin


class Schema:
    def __init__(self, database : Postgres = None):
        if database is not None:
            db = database._connection
            tables = postgres.tables(database._connection)

            tables = list(filter(lambda x: x[0] == 'public', tables))

            self.tables = []
            self.name_to_indexes = {}
            self.table_names = []
            self.name_to_table = {}
            self.size = len(tables)

            self.table_structures = {}

            self.total_columns = 0
            self.max_columns = 0

            self.columns = []
            self.column_indexes = {}
            self.column_db_indexes = {}

            for i, (sname, tname) in enumerate(tables):
                table_obj = postgres.table_structure(db, tname, sname)
                self.max_columns = max(self.max_columns, len(table_obj))

                for j, name, typ, *_ in table_obj:
                    c = (sname, tname, name)
                    self.column_indexes[c] = len(self.columns)
                    self.columns.append(c)
                    self.column_db_indexes[c] = []

                table_obj = DataTable(table_obj, tname, sname)
                self.tables.append(table_obj)
                self.table_names.append(tname)
                self.name_to_indexes[tname] = i
                self.name_to_table[tname] = table_obj
                self.total_columns += len(table_obj)

                row_count_query = f'select count(*) from {sname + "." if sname else ""}{tname};'
                row_count = database.execute(row_count_query)[0][0]
                table_obj.row_count = row_count

                index_query = f'select indexname, indexdef from pg_indexes ' \
                              f'where schemaname = \'{"public" if sname is None else sname}\' and tablename = \'{tname}\';'
                res = database.execute(index_query)
                for index_name, index_def in res:
                    match = re.search(r'\(([A-Za-z0-9_]+(?:, *[A-Za-z0-9_]+)*)\)', index_def)
                    if match:
                        columns = match.group(0)
                        columns = re.findall(r'[A-Za-z0-9_]+', columns)
                        for c in columns:
                            c = (sname, tname, c)
                            assert c in self.column_db_indexes
                            self.column_db_indexes[c].append((index_name, index_def))

            self.table_columns = {t.name : list(t.columns.values()) for t in self.tables}

            self.foreign_keys = {}
            self.foreign_keys_from = {}
            for fk_id, (table_name, column_name), (fk_table_name, fk_column_name) in database.list_foreign_keys():
                fk_list = self.foreign_keys.setdefault(table_name, [])
                fk_list.append((column_name, fk_table_name, fk_column_name))
                fk_rev_list = self.foreign_keys_from.setdefault(fk_table_name, [])
                fk_rev_list.append((fk_column_name, table_name, column_name))

            for fk_id, (table_name, column_name), (fk_table_name, fk_column_name) in database.list_foreign_keys(multi_columns=True):
                fk_table = self.name_to_table[table_name]
                pk_table = self.name_to_table[fk_table_name]
                fk_table.foreign_keys.append((column_name, fk_table_name, fk_column_name))
                pk_table.foreign_keys_rev.append((fk_column_name, table_name, column_name))

            self.primary_keys = {}
            for table_name, column_name in database.list_primary_keys():
                if not table_name in self.name_to_table:
                    continue
                pk_list = self.primary_keys.setdefault(table_name, [])
                pk_list.append(column_name)
                pk_table = self.name_to_table[table_name]
                pk_table.primary_keys.append(column_name)

            self.pg_stats = {}
            pg_stats_query = f'select * from pg_stats where schemaname = \'public\';'
            pg_stats_data = database.execute(pg_stats_query)
            for res in pg_stats_data:
                schema, table, column = res[:3]
                table_obj = self.name_to_table[table]
                column_type = table_obj.column_types[column]
                is_numeric = postgres_type(column_type) != 0
                self.pg_stats[(table, column)] = PgStats(res, is_numeric)

    def load_state_dict(self, state_dict):
        tables = state_dict['tables']
        self.tables = []
        for t in tables:
            self.tables.append(DataTable.from_state_dict(t))
        for name in (
            'name_to_indexes', 'table_names', 'name_to_table', 'size', 'table_structures',
            'total_columns', 'max_columns', 'columns', 'column_indexes', 'table_columns',
            'foreign_keys', 'foreign_keys_from', 'primary_keys',
        ):
            setattr(self, name, state_dict[name])

    def state_dict(self):
        res = {}
        for name in (
            'name_to_indexes', 'table_names', 'name_to_table', 'size', 'table_structures',
            'total_columns', 'max_columns', 'columns', 'column_indexes', 'table_columns',
            'foreign_keys', 'foreign_keys_from', 'primary_keys',
        ):
            res[name] = getattr(self, name)
        res['tables'] = [i.state_dict() for i in self.tables]
        return res

    def column_index(self, table_name, column_name, schema_name='public'):
        return self.column_indexes[schema_name, table_name, column_name]

    def __len__(self):
        return self.size

    def table(self, name):
        return self.name_to_indexes[name], self.name_to_table[name]
