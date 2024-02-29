class SizedWrapper:
    def __init__(self, iter, size):
        self.size = size
        self.iter = iter

    def __getattr__(self, item, default=NotImplemented):
        if default is NotImplemented:
            return getattr(self.iter, item)
        return getattr(self.iter, item, default)

    def __len__(self):
        return self.size

    def __iter__(self):
        return self.iter

    def __next__(self):
        return next(self.iter)

def fetcher(cursor):
    res = cursor.fetchone()
    while res is not None:
        yield res
        res = cursor.fetchone()
    cursor.close()

def unpack(it):
    for i in it:
        for j in i:
            yield j

def cursor_to_data(cursor, filter_type=None):
    if filter_type is not None:
        return map(lambda x: tuple(filter(lambda y: isinstance(y, filter_type), x)), fetcher(cursor))
    return fetcher(cursor)

def cursor_iter(database, select, filter_type=None):
    cursor = database.cursor()
    cursor.execute(select)
    data = cursor_to_data(cursor, filter_type=filter_type)
    return SizedWrapper(data, cursor.rowcount)
