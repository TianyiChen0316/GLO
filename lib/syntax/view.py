from types import MappingProxyType
from collections.abc import Mapping


class view:
    @classmethod
    def getter_view(cls, getter):
        return cls(getitem=getter, doc=getter.__doc__)

    @classmethod
    def setter_view(cls, setter):
        return cls(setitem=setter, doc=setter.__doc__)

    @classmethod
    def deleter_view(cls, deleter):
        return cls(delitem=deleter, doc=deleter.__doc__)

    def __init__(
            self,
            call=None,
            getitem=None,
            setitem=None,
            delitem=None,
            getters=None,
            setters=None,
            deleters=None,
            methods=None,
            doc=None,
    ):
        self._call = call
        self._getitem = getitem
        self._setitem = setitem
        self._delitem = delitem
        if getters is None:
            getters = {}
        self._getters = getters
        if setters is None:
            setters = {}
        self._setters = setters
        if deleters is None:
            deleters = {}
        self._deleters = deleters
        if methods is None:
            methods = {}
        self._methods = methods
        if doc is None and call is not None:
            doc = call.__doc__
        self.__doc__ = doc
        self._view = None

    def __get__(self, instance, owner=None):
        if self._view is None:
            class ViewMeta(type):
                def __new__(cls, name, bases, dic, *args, **kwargs):
                    for method_name, method in self._methods.items():
                        if method_name in dic:
                            continue
                        def wrapper(self, *args, **kwargs):
                            return method(self._instance, *args, **kwargs)
                        wrapper.__name__ = method.__name__
                        wrapper.__qualname__ = method.__qualname__
                        dic[method_name] = wrapper
                    return super().__new__(cls, name, bases, dic, **kwargs)

            class View(metaclass=ViewMeta):
                __fields__ = ['_parent', '_instance']

                def __init__(self, parent, instance):
                    self._parent = parent
                    self._instance = instance

                def __call__(self, *args, **kwargs):
                    if not callable(self._parent._call):
                        raise AttributeError("uncallable view")
                    return self._parent._call(self._instance, *args, **kwargs)

                def __getitem__(self, item):
                    if not callable(self._parent._getitem):
                        raise AttributeError("unreadable view")
                    return self._parent._getitem(self._instance, item)

                def __setitem__(self, key, value):
                    if not callable(self._parent._setitem):
                        raise AttributeError("unwritable view")
                    return self._parent._setitem(self._instance, key, value)

                def __delitem__(self, key):
                    if not callable(self._parent._delitem):
                        raise AttributeError("undeletable view")
                    return self._parent._delitem(self._instance, key)

                def __getattr__(self, item):
                    if item not in self._parent._getters:
                        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")
                    return self._parent._getters[item](self._instance)

                def __setattr__(self, key, value):
                    if key in self.__fields__:
                        return super().__setattr__(key, value)
                    if key not in self._parent._setters:
                        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
                    return self._parent._setters[key](self._instance, value)

                def __delattr__(self, item):
                    if item not in self._parent._deleters:
                        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")
                    return self._parent._deleters[item](self._instance)

            View.__qualname__ = f'{self.__class__.__qualname__}.{View.__name__}'
            self._view = View(self, instance)
        else:
            self._view._instance = instance
        return self._view

    def getitem(self, getter):
        return self.__class__(self._call, getter, self._setitem, self._delitem, self._getters, self._setters,
                              self._deleters, self._methods, self.__doc__)

    def setitem(self, setter):
        return self.__class__(self._call, self._getitem, setter, self._delitem, self._getters, self._setters,
                              self._deleters, self._methods, self.__doc__)

    def delitem(self, deleter):
        return self.__class__(self._call, self._getitem, self._setitem, deleter, self._getters, self._setters,
                              self._deleters, self._methods, self.__doc__)

    def getter(self, field: str):
        def getter(func):
            new_getters = self._getters.copy()
            new_getters[field] = func
            return self.__class__(self._call, self._getitem, self._setitem, self._delitem, new_getters, self._setters,
                                  self._deleters, self._methods, self.__doc__)

        return getter

    def setter(self, field: str):
        def setter(func):
            new_setters = self._setters.copy()
            new_setters[field] = func
            return self.__class__(self._call, self._getitem, self._setitem, self._delitem, self._getters, new_setters,
                                  self._deleters, self._methods, self.__doc__)

        return setter

    def deleter(self, field: str):
        def deleter(func):
            new_deleters = self._deleters.copy()
            new_deleters[field] = func
            return self.__class__(self._call, self._getitem, self._setitem, self._delitem, self._getters, self._setters,
                                  new_deleters, self._methods, self.__doc__)

        return deleter

    def str(self, str):
        new_methods = self._methods.copy()
        new_methods['__str__'] = str
        return self.__class__(self._call, self._getitem, self._setitem, self._delitem, self._getters, self._setters,
                              self._deleters, new_methods, self.__doc__)

    def repr(self, repr):
        new_methods = self._methods.copy()
        new_methods['__repr__'] = repr
        return self.__class__(self._call, self._getitem, self._setitem, self._delitem, self._getters, self._setters,
                              self._deleters, new_methods, self.__doc__)

    def method(self, name: str):
        def method(func):
            new_methods = self._methods.copy()
            new_methods[name] = func
            return self.__class__(self._call, self._getitem, self._setitem, self._delitem, self._getters, self._setters,
                                  self._deleters, new_methods, self.__doc__)
        return method


class dict_view:
    def __init__(self, mapping):
        if not isinstance(mapping, Mapping):
            raise TypeError(f"'{mapping.__class__.__name__}' is not a mapping")
        self.__mapping = MappingProxyType(mapping)

    @property
    def mapping(self):
        return self.__mapping

    def __len__(self):
        return len(self.__mapping)

    def __iter__(self):
        raise NotImplementedError

    def __repr__(self):
        return '{}([{}])'.format(self.__class__.__name__, ', '.join(map(repr, self)))
    __str__ = __repr__


class dict_keys(dict_view):
    def __iter__(self):
        for k in self.mapping.keys():
            yield k


class dict_values(dict_view):
    def __iter__(self):
        for k in self.mapping.values():
            yield k


class dict_items(dict_view):
    def __iter__(self):
        for k in self.mapping.items():
            yield k


__all__ = ['view', 'dict_keys', 'dict_values', 'dict_items']
