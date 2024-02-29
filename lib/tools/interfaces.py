from lib.syntax import Interface, interfacemethod

class StateDictInterface(Interface):
    @interfacemethod
    def state_dict(self) -> dict:
        res = {}
        for dict_field, name in getattr(self, '__statedictchildren__', {}).items():
            value = getattr(self, name, None)
            if not isinstance(value, StateDictInterface):
                res[dict_field] = value
            else:
                res[dict_field] = value.state_dict()
        return res

    @interfacemethod
    def load_state_dict(self, dic : dict):
        state_dict_fields = getattr(self, '__statedictchildren__', {})
        for dict_field, dict_value in dic.items():
            if dict_field not in state_dict_fields:
                continue
            name = state_dict_fields[dict_field]
            value = getattr(self, name, None)
            if isinstance(value, StateDictInterface):
                value.load_state_dict(dict_value)
            else:
                setattr(self, name, dict_value)
        return self

    def register(self, name, value=None, dict_field=None):
        """
        To register the state dict fields. The name must matches the
        """
        state_dict_fields = getattr(self, '__statedictchildren__', None)
        if state_dict_fields is None:
            state_dict_fields = {}
            setattr(self, '__statedictchildren__', state_dict_fields)
        if dict_field is None:
            dict_field = name
        state_dict_fields[dict_field] = name
        if value is not None:
            setattr(self, name, value)


__all__ = ['StateDictInterface']
