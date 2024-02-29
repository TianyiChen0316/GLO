from abc import ABCMeta as _ABCMeta, abstractmethod

def interfacemethod(funcobj):
    """
    Unlike abstract methods, interface methods do not need implementation,
     and __interfacemethod__ will not be inherited from the parent.
    """
    funcobj.__isinterfacemethod__ = True
    return funcobj

class InterfaceMeta(_ABCMeta):
    def __new__(__mcls, __name, __bases, __namespace, **__kwargs):
        __interfacemethods__ = set()
        for k, v in __namespace.items():
            if getattr(v, '__isinterfacemethod__', None):
                __interfacemethods__.add(k)
        __namespace['__interfacemethods__'] = frozenset(__interfacemethods__)
        if '__subclasshook__' not in __namespace:
            @classmethod
            def __subclasshook__(cls, subclass):
                if not cls.__abstractmethods__ and not cls.__interfacemethods__:
                    # not an interface
                    return NotImplemented
                for abstract_method in cls.__abstractmethods__:
                    if not any(abstract_method in getattr(p, '__dict__', ()) for p in subclass.__mro__):
                        return NotImplemented
                for interface_method in cls.__interfacemethods__:
                    if not any(interface_method in getattr(p, '__dict__', ()) for p in subclass.__mro__):
                        return NotImplemented
                return True
            __namespace['__subclasshook__'] = __subclasshook__
        return super().__new__(__mcls, __name, __bases, __namespace, **__kwargs)

class Interface(metaclass=InterfaceMeta):
    __slots__ = ()
