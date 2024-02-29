from collections.abc import Set

class DisjointSet(Set):
    def __init__(self):
        self.__set = {}

    def __contains__(self, item):
        return item in self.__set

    def __len__(self):
        return len(self.__set)

    def __iter__(self):
        return self.__set.keys()

    def clear(self):
        self.__set.clear()

    def roots(self):
        return set(self.__set.values())

    def copy(self):
        res = self.__class__()
        res.__set = self.__set.copy()
        return res

    def __check_contains(self, value):
        if value not in self:
            raise ValueError(f'\'{value}\' is not in disjoint set')

    def root(self, value, default=None):
        if value not in self:
            return default
        temp = value
        root = self.__set[temp]
        while temp != root:
            temp = root
            root = self.__set[root]
        parent = self.__set[value]
        while value != parent:
            self.__set[value] = root
            value = parent
            parent = self.__set[parent]
        return parent

    def add(self, value, dst=None):
        if dst is None:
            dst = value
            self.__set[value] = value
        else:
            self.__check_contains(dst)
            dst = self.root(dst)
            self.__set[value] = dst
        return dst

    def set(self, src, dst):
        if src not in self:
            self.add(src)
        if dst not in self:
            self.add(dst)
        root_src = self.root(src)
        root_dst = self.root(dst)
        if root_src != root_dst:
            self.__set[root_src] = root_dst
        return root_dst

    def set_root(self, value, check=False):
        if check:
            self.__check_contains(value)
        root = self.root(value)
        if root is None:
            self.add(value)
        elif value != root:
            self.__set[root] = value
            self.__set[value] = value

    def is_in_same_set(self, src, dst):
        if src not in self or dst not in self:
            return False
        return self.root(src) == self.root(dst)

    def to_dict(self):
        flattened = {k : self.root(k) for k in self.__set.keys()}
        res = {v : set() for v in set(flattened.values())}
        for k, v in flattened.items():
            res[v].add(k)
        return res
