def _get_target(self, placeholder_name, init_hook):
    target = getattr(self, placeholder_name)
    if target is None:
        if init_hook:
            init_hook_func = getattr(self, init_hook)
            if init_hook_func is not None:
                if not callable(init_hook_func):
                    raise TypeError(f"'{init_hook_func.__class__.__name__}' object is not callable")
                init_hook_func()
                target = getattr(self, placeholder_name)
    if target is None:
        raise ValueError(f'The placeholder is not yet initialized')
    return target

class PlaceholderMetaclass(type):
    @classmethod
    def _attr_name(cls, attr_name, cls_name):
        if attr_name.startswith('__'):
            # private attribute
            if cls_name.startswith('_'):
                _pos = cls_name.rfind('_')
                _cname = cls_name[_pos:]
                if _cname == '_':
                    _cname = ''
            else:
                _cname = f'_{cls_name}'
            attr_name = f'{_cname}{attr_name}'
        return attr_name

    def __new__(cls, name, bases, dic, *args, type=None, placeholder_name=None, init_hook=None, **kwargs):
        if type is None:
            raise SyntaxError(f'placeholder metaclass requires a keyword argument \'type\'')
        if placeholder_name is None:
            raise SyntaxError(f'placeholder metaclass requires a keyword argument \'placeholder_name\'')
        if not isinstance(placeholder_name, str):
            raise TypeError(f'placeholder attribute name must be str')
        if init_hook is not None:
            if not isinstance(placeholder_name, str):
                raise TypeError(f'initialization method name must be str')

        placeholder_name = cls._attr_name(placeholder_name, name)
        if init_hook is not None:
            init_hook = cls._attr_name(init_hook, name)

        qualname = dic.get('__qualname__', None)
        if qualname is None:
            qualname = ''
        else:
            qualname = f'{qualname}.'

        attrs = dir(type)
        for attr in attrs:
            if (attr == placeholder_name or attr in dic or (init_hook and attr == init_hook)
                or attr in (
                '__class__', '__init__', '__init_subclass__', '__new__', '__slots__',
                '__dict__', '__doc__', '__module__', '__weakref__', '__prepare__',
                '__getattribute__', '__setattribute__', '__setattr__', '__repr__',
                '__dir__', 'mro',
                '__torch_function__',
            )):
                continue

            def wrapper_factory(placeholder_name, attr, init_hook, qualname):
                def wrapper_call(self, *args, **kwargs):
                    return getattr(_get_target(self, placeholder_name, init_hook), attr)(*args, **kwargs)
                wrapper_call.__name__ = attr
                wrapper_call.__qualname__ = qualname + attr

                def wrapper_get(self):
                    return getattr(_get_target(self, placeholder_name, init_hook), attr)
                wrapper_get.__name__ = attr
                wrapper_get.__qualname__ = qualname + attr

                def wrapper_set(self, value):
                    return setattr(_get_target(self, placeholder_name, init_hook), attr, value)
                wrapper_set.__name__ = attr
                wrapper_set.__qualname__ = qualname + attr

                wrapper_property = property(wrapper_get, wrapper_set)
                return wrapper_call, wrapper_property

            target_function = getattr(type, attr)
            wrapper_call, wrapper_property = wrapper_factory(placeholder_name, attr, init_hook, qualname)
            if callable(target_function):
                # method
                dic[attr] = wrapper_call
            else:
                # property
                dic[attr] = wrapper_property

        dic[placeholder_name] = None
        if '__torch_function__' not in dic:
            def factory(placeholder_name, init_hook):
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    new_args = []
                    for arg in args:
                        if isinstance(arg, cls):
                            new_args.append(_get_target(arg, placeholder_name, init_hook))
                        else:
                            new_args.append(arg)
                    kwargs = {} if kwargs is None else kwargs
                    new_kwargs = {}
                    for key, value in kwargs.items():
                        if isinstance(value, cls):
                            new_kwargs[key] = _get_target(value, placeholder_name, init_hook)
                        else:
                            new_kwargs[key] = value
                    return func(*new_args, **new_kwargs)
                return __torch_function__

            dic['__torch_function__'] = factory(placeholder_name, init_hook)

        res = super().__new__(cls, name, bases, dic, **kwargs)
        return res
