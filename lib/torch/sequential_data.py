import torch
import numpy as np
from collections.abc import Iterable as _Iterable

from lib.syntax.placeholder import PlaceholderMetaclass
from lib.syntax import view


def _heap_comparison(values):
    total_elements = len(values)
    heap = [*(None for i in range(total_elements - 1)), *values]
    for index in range(total_elements - 2, -1, -1):
        left_child = (index << 1) + 1
        right_child = (index << 1) + 2
        if heap[left_child] == heap[right_child]:
            heap[index] = heap[left_child]
        else:
            return False
    return True

def _slice_size(_slice : slice, length):
    start, stop, step = _slice.start, _slice.stop, _slice.step
    assert step != 0, 'slice step cannot be zero'
    if start is None:
        start = 0
    if stop is None:
        stop = length
    if step is None:
        step = 1

    if start < 0:
        start = start + length
    if stop < 0:
        stop = stop + length
    start = max(min(start, length), 0)
    stop = max(min(stop, length), 0)
    if step < 0:
        _range = start - stop
        _step = -step
    else:
        _range = stop - start
        _step = step
    if _range <= 0:
        return 0
    return np.ceil(_range / _step)

def broadcast(*shapes):
    largest_ndim = max(map(len, shapes))
    res = []
    for _shapes in zip(*((*shape[::-1], *(1 for i in range(largest_ndim - len(shape)))) for shape in shapes)):
        _shapes = list(filter(lambda x: x != 1, _shapes))
        if len(_shapes) == 0:
            res.append(1)
        else:
            _res = _shapes[0]
            if not _heap_comparison(_shapes):
                return None
            res.append(_res)
    return res[::-1]


class _SequenceFeature:
    def __init__(self, shape, device=None):
        self._shape = tuple(shape)
        self._items = {}
        self._device = device

    @property
    def device(self):
        return self._device

    def to(self, device, inplace=False):
        if inplace:
            res = self
        else:
            res = self.clone()
        res._device = device
        for key, value in res._items.items():
            if not isinstance(value, (_TensorPlaceholder, _TensorFixedPlaceholder)):
                res._items[key] = value.to(device)
        return res

    @classmethod
    def _clone_item(cls, value, new_parent, detach=False, deep=True):
        if isinstance(value, _TensorPlaceholder):
            return _TensorPlaceholder(new_parent, value._target_field, value._item, value._target_shape)
        elif isinstance(value, _TensorFixedPlaceholder):
            return _TensorFixedPlaceholder(new_parent, value._target_field, value._item, value._target_shape, value._dtype)
        if deep:
            return value.clone().detach() if detach else value.clone()
        return value.detach() if detach else value

    def clone(self, detach=False, deep=True):
        res = self.__class__(self._shape, self._device)
        for key, value in self._items.items():
            res._items[key] = self._clone_item(value, res, detach, deep)
        return res

    def update(self, sequence):
        if not isinstance(sequence, self.__class__):
            raise TypeError(f"'{sequence.__class__.__name__}' object is not a sequence")
        if sequence.device != self.device:
            sequence = sequence.to(self.device)
        for key, value in sequence._items.items():
            self._items[key] = self._clone_item(value, self, detach=False)
        return self

    def _getitem(self, item, create_placeholder=True):
        """
        Lazily updates the stored tensor and returns it.
        """
        value = self._items.get(item, None)
        if value is not None:
            if isinstance(value, (_TensorPlaceholder, _TensorFixedPlaceholder)):
                return value
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"'{value}' is not a tensor: {item}")
            shape_size = len(self._shape)
            if not value.ndim >= shape_size:
                raise ValueError(
                    f"shape mismatch between [{', '.join(map(str, value.shape))}] and target size [{', '.join(map(str, self._shape))}]")
            ori_shape = value.shape[:shape_size]
            if ori_shape != self._shape:
                smaller_shape = tuple(min(x, y) for x, y in zip(ori_shape, self._shape))
                padding_tensor = torch.zeros(*self._shape, *value.shape[shape_size:], dtype=value.dtype,
                                             device=value.device)
                smaller_shape_index = tuple(slice(0, s) for s in smaller_shape)
                padding_tensor[smaller_shape_index] = value[smaller_shape_index]
                value = padding_tensor
                self._items[item] = value
            return value
        if create_placeholder:
            placeholder = _TensorPlaceholder(self, '_items', item, self._shape)
            self._items[item] = placeholder
            return placeholder
        return None

    def _setitem(self, key, value):
        if isinstance(value, (int, float)):
            value = value + torch.zeros(*self._shape, dtype=value.__class__, device=self.device)
        elif isinstance(value, np.ndarray):
            value = torch.tensor(value, device=self.device)
        elif not isinstance(value, torch.Tensor):
            if isinstance(value, (_TensorPlaceholder, _TensorFixedPlaceholder)):
                if value._tensor is not None:
                    value = value._tensor
                else:
                    raise TypeError(f"'{value.__class__}' object is not a tensor")
            else:
                raise TypeError(f"'{value.__class__}' object is not a tensor")
        if value.ndim < len(self._shape):
            raise ValueError(f"shape mismatch between [{', '.join(map(str, value.shape))}] and target size [{', '.join(map(str, self._shape))}]")
        self._items[key] = value.to(self.device)

    def register(self, item, *shape, dtype=None, init=False):
        """
        To register an item with specified shape.
        """
        if init:
            return self._setitem(item, torch.zeros(*(0 for i in range(len(self._shape))), *shape, dtype=dtype, device=self.device))
        placeholder = _TensorFixedPlaceholder(self, '_items', item, (*self._shape, *shape), dtype)
        self._items[item] = placeholder
        return placeholder

    def _index_refine(self, index):
        if not isinstance(index, (int, slice, torch.IntTensor, torch.LongTensor)):
            if isinstance(index, _Iterable):
                indices = tuple(index)
            else:
                # ellipses and bool tensors are not allowed
                raise TypeError(f"invalid index type: '{index.__class__.__name__}'")
            if len(indices) > len(self._shape):
                raise ValueError(f'index dimension must be no greater than {len(self._shape)}')
        else:
            indices = (index, *(slice(None) for i in range(len(self._shape) - 1)))

        shapes_before_tensor = []
        shapes_after_tensor = []
        tensor_index_shapes = []
        tensor_index_move_to_top = False
        for index, shape in zip(indices, self._shape):
            if isinstance(index, int):
                # integer indices simply remove a dimension
                continue
            elif isinstance(index, slice):
                step = 1 if index.step is None else index.step
                if step <= 0:
                    raise ValueError('step must be greater than zero')
                new_shape = _slice_size(index, shape)
                if tensor_index_shapes:
                    # slices between tensor indices will move the tensor index dimensions to top
                    shapes_after_tensor.append(new_shape)
                else:
                    shapes_before_tensor.append(new_shape)
            elif isinstance(index, (torch.IntTensor, torch.LongTensor)):
                # tensor indices remove the original dimensions and create new dimensions
                tensor_index_shapes.append(index.shape)
                if shapes_after_tensor:
                    tensor_index_move_to_top = True
        if tensor_index_shapes:
            tensor_index_shapes = broadcast(tensor_index_shapes)
            if not tensor_index_shapes:
                raise IndexError(
                    f'shape mismatch: indexing tensors could not be broadcast together with shapes {", ".join(map(lambda x: str(list(x)), tensor_index_shapes))}')
        if tensor_index_move_to_top:
            new_shapes = [*tensor_index_shapes, *shapes_before_tensor, *shapes_after_tensor]
        else:
            new_shapes = [*shapes_before_tensor, *tensor_index_shapes, *shapes_after_tensor]

        return new_shapes, indices

    def _index_select(self, index):
        new_shape, index = self._index_refine(index)
        res = self.__class__(new_shape, device=self.device)
        for item in self._items.keys():
            new_value = self._getitem(item, False)[index].clone()
            res._setitem(item, new_value)
        return res

    def _index_assign(self, key, value):
        if not isinstance(value, self.__class__):
            raise TypeError(f"'{value.__class__.__name__}' object is not a sequence")
        new_shape, index = self._index_refine(key)
        if new_shape != value._shape:
            raise ValueError(f"shape mismatch between sequence {list(new_shape)} and {list(value._shape)}")
        for item in value._items.keys():
            new_value = value._getitem(item, False)
            ori_value = self._getitem(item, True)
            ori_value[index] = new_value

    def reshape(self, *shape, lazy=False):
        # all querying processes are lazy, so features do not need updates
        self._shape = tuple(shape)
        if not lazy:
            for key, value in tuple(self._items.items()):
                if isinstance(value, _TensorPlaceholder):
                    self._items[key] = _TensorPlaceholder(self, value._target_field, value._item, self._shape)
                elif isinstance(value, _TensorFixedPlaceholder):
                    new_shape = (*self._shape, *value._target_shape[len(self._shape):])
                    self._items[key] = _TensorFixedPlaceholder(self, value._target_field, value._item, new_shape, value._dtype)
                else:
                    new_shape = (*self._shape, *value.shape[len(self._shape):])
                    new_tensor = torch.zeros(new_shape, dtype=value.dtype, device=value.device)
                    indices = (*(slice(None, min(s1, s2), None) for s1, s2 in zip(self._shape, value.shape[:len(self._shape)])), ...)
                    new_tensor[indices] = value[indices]
                    self._items[key] = new_tensor
        return self

    def __getitem__(self, item):
        if isinstance(item, (int, slice, torch.IntTensor, torch.LongTensor, tuple)):
            return self._index_select(item)
        return self._getitem(item)

    def __setitem__(self, key, value):
        if isinstance(key, (int, slice, torch.IntTensor, torch.LongTensor, tuple)):
            return self._index_assign(key, value)
        return self._setitem(key, value)

    def __delitem__(self, item):
        if item in self._items:
            del self._items[item]

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items.keys())

    def __bool__(self):
        return bool(self._items)

    def __str__(self):
        len_shape = len(self._shape)
        res = [f'shape=[{", ".join(map(str, self._shape))}]']
        for k, v in self._items.items():
            if isinstance(v, _TensorPlaceholder):
                res.append(f'{str(k)}=?')
            elif isinstance(v, _TensorFixedPlaceholder):
                res.append(f'{str(k)}=[{", ".join(map(str, v._target_shape[len_shape:]))}]')
            else:
                res.append(f'{str(k)}=[{", ".join(map(str, v.shape[len_shape:]))}]')
        return f"({', '.join(res)})"

    def __repr__(self):
        return self.__str__()


class _TensorFixedPlaceholder(metaclass=PlaceholderMetaclass, type=torch.Tensor, placeholder_name='_tensor', init_hook='__create_tensor'):
    def __init__(self, parent, target_field, item, target_shape, dtype):
        self._parent = parent
        self._item = item
        self._target_field = target_field
        self._target_shape = target_shape
        self._dtype = dtype

    def __create_tensor(self):
        if self._tensor is not None:
            return
        self._tensor = torch.zeros(self._target_shape, dtype=self._dtype, device=self._parent.device)
        target_field = getattr(self._parent, self._target_field, None)
        if target_field is None:
            target_field = {}
            setattr(self._parent, self._target_field, target_field)
        target_field[self._item] = self._tensor

class _TensorPlaceholder(metaclass=PlaceholderMetaclass, type=torch.Tensor, placeholder_name='_tensor'):
    def __init__(self, parent, target_field, item, target_shape):
        self._parent = parent
        self._item = item
        self._target_field = target_field
        self._target_shape = target_shape

    def __setitem__(self, key, value: torch.Tensor):
        if self._tensor is not None:
            self._tensor[key] = value
            return

        # Originally, the indices can be integers, slices, ellipses, integer arrays, and boolean arrays
        # - Integers: specifying only one element of one dimension
        # - Slices: specifying a group of elements of one dimension
        # - Integer arrays: specifying a group of elements and move the specified dimensions to the front;
        #                   the arrays must be broadcast into the same shape;
        #                   for example: (5, 4, 6)[(2, 3), :, (3)] results in (2, 3, 4)
        # - Boolean arrays: row-majorly specifying and flattening a set of elements,
        #                   which must matches the total size of the first dimensions
        # - Ellipses: a placeholder that specifies the rest of the dimensions,
        #             which can only appear once
        if not isinstance(key, (int, slice, torch.IntTensor, torch.LongTensor, tuple)) \
                or isinstance(key, tuple) and True in map(lambda ix: (
                # ellipses and booleans are forbidden, as the shape has to be inferred
                not isinstance(ix[1], (int, slice, torch.IntTensor, torch.LongTensor))
                # tensor indices of the feature dimensions are also forbidden
                or isinstance(ix[1], (torch.IntTensor, torch.LongTensor)) and ix[0] >= len(self._target_shape)
        ), enumerate(key)):
            raise TypeError(f"Invalid index type '{key.__class__.__name__}': {key}")
        if not isinstance(key, tuple):
            key = (key,)

        if len(key) > len(self._target_shape):
            raise IndexError(f"Cannot infer shape from indices more than {len(self._target_shape)}")

        key_size_bias = 0
        tensor_indices = []
        for k in key:
            if isinstance(k, int):
                # the dimension with an integer index does not need a dimension in the value
                key_size_bias -= 1
            elif isinstance(k, (torch.IntTensor, torch.LongTensor)):
                tensor_indices.append(k.shape)
        if tensor_indices:
            total_tensor_indices_size = broadcast(*tensor_indices)
            if not total_tensor_indices_size:
                raise IndexError(
                    f'shape mismatch: indexing tensors could not be broadcast together with shapes {", ".join(map(lambda x: str(list(x)), tensor_indices))}')
            key_size_bias -= len(tensor_indices)
            key_size_bias += len(total_tensor_indices_size)

        if isinstance(value, np.ndarray):
            value = torch.Tensor(value, device=self._parent.device)

        if isinstance(value, torch.Tensor):
            minimal_value_size = len(self._target_shape) + key_size_bias
            if value.ndim < minimal_value_size:
                # the shape of the value does not match the target size
                raise ValueError(
                    f'shape mismatch between [{", ".join(value.shape)}] and target size [{", ".join(map(str, self._target_shape))}]')
            shape = [*self._target_shape, *value.shape[minimal_value_size:]]
            new_tensor = torch.zeros(shape, dtype=value.dtype, device=value.device)
            new_tensor[key] = value
        else:
            # scalar value that requires broadcasting
            if not isinstance(value, (int, float)):
                raise TypeError(f'the value must be an integer, a float, an ndarray, or a tensor')
            new_tensor = torch.zeros(*self._target_shape, device=self._parent.device, dtype=value.__class__)
            new_tensor[key] = value

        self._tensor = new_tensor

        # Theoretically the placeholder's tensor can be different from the sequence container
        #  when the container's tensor is reassigned
        target_field = getattr(self._parent, self._target_field, None)
        if target_field is None:
            target_field = {}
            setattr(self._parent, self._target_field, target_field)
        target_field[self._item] = new_tensor


class SequenceBase:
    def __init__(self, device=None):
        if device is None:
            device = torch.device('cpu')
        self._device = device
        self._features = {}

    def update(self, sequence):
        if not isinstance(sequence, self.__class__):
            raise TypeError(f"'{sequence.__class__.__name__}' object is not a sequence")
        for sequence_name, sequence_value in sequence._features.items():
            if sequence_name not in self._features:
                self._features[sequence_name] = sequence_value.clone().to(self.device)
            else:
                self._features[sequence_name].update(sequence_value.to(self.device))
        return self

    def _feature_register(self, name, *shape):
        self._features[name] = _SequenceFeature(shape, self._device)

    def _pad(self, name, shapes):
        """
        In-place zero padding for one feature group.
        """
        if not isinstance(shapes, _Iterable):
            shapes = (shapes, )
        features : _SequenceFeature = self._features.get(name, None)
        if features is None:
            raise KeyError(f"feature '{name}' does not exist")
        shapes = tuple(shapes)
        if len(shapes) != len(features._shape):
            raise ValueError(f"shape mismatch between {shapes} and {features._shape}")
        for value in shapes:
            if not isinstance(value, int):
                raise TypeError(f"invalid shape '{value}'")
            if value < 0:
                raise ValueError("padding size must be greater than or equals to 0")

        index_slices = tuple((slice(i, None, None) for i in shapes))
        rev_index_slices = tuple((slice(None, -i if i != 0 else None, None) for i in shapes))
        for feature_name in features:
            origin = features[feature_name]
            new_value = torch.zeros_like(origin, device=origin.device, dtype=origin.dtype)
            new_value[index_slices] = origin[rev_index_slices]
            features[feature_name] = new_value
        return self

    @property
    def device(self):
        return self._device

    def to(self, device, inplace=False):
        if inplace:
            res = self
        else:
            res = self.clone()
        res._device = device
        for key, value in self._features.items():
            res._features[key] = value.to(device, inplace=inplace)
        return res

    def clone(self, detach=False, deep=True):
        # to leave the rest of the work to subclasses instead of calling __init__ with arbitrary arguments
        res = object.__new__(self.__class__)
        res._device = self._device
        res._features = {k : v.clone(detach=detach, deep=deep) for k, v in self._features.items()}
        return res

class Sequence(SequenceBase):
    def __init__(self, sequence_length, device=None):
        self._sequence_length = sequence_length
        if device is None:
            device = torch.device('cpu')
        super().__init__(device)
        self._feature_register('feature', self._sequence_length)
        self._feature_register('attention', self._sequence_length, self._sequence_length)

    def clone(self, detach=False, deep=True):
        res = super().clone(detach=detach, deep=deep)
        res._sequence_length = self._sequence_length
        return res

    @property
    def sequence_length(self):
        return self._sequence_length

    def register(self, item, *shape, dtype=None):
        return self._features['feature'].register(item, *shape, dtype=dtype)

    def resize(self, size, lazy=False):
        self._sequence_length = size
        self._features['feature'].reshape(size, lazy=lazy)
        self._features['attention'].reshape(size, size, lazy=lazy)
        return self

    def pad(self, size, inplace=False):
        if inplace:
            res = self
        else:
            res = self.clone(deep=False)
        res._pad('feature', (size,))
        res._pad('attention', (size, size))
        return res

    def __len__(self):
        return len(self._features['feature'])

    def __iter__(self):
        return iter(self._features['feature'])

    def __getitem__(self, item):
        return self._features['feature'].__getitem__(item)

    def __setitem__(self, key, value):
        return self._features['feature'].__setitem__(key, value)

    def __delitem__(self, item):
        return self._features['feature'].__delitem__(item)

    def __bool__(self):
        return self._features['feature'] or self._features['attention']

    def __str__(self):
        return f"{self.__class__.__name__}(" \
               f"sequence_length={self._sequence_length}, " \
               f"feature={str(self._features['feature'])}, " \
               f"attention={str(self._features['attention'])})"

    def __repr__(self):
        return self.__str__()

    @view.getter_view
    def attention(self, item):
        return self._features['attention'].__getitem__(item)

    @attention.setitem
    def attention(self, key, value):
        return self._features['attention'].__setitem__(key, value)

    @attention.delitem
    def attention(self, item):
        return self._features['attention'].__delitem__(item)

    @attention.str
    def attention(self):
        return str(self._features['attention'])

    @attention.repr
    def attention(self):
        return repr(self._features['attention'])

    @attention.method('__len__')
    def attention(self):
        return len(self._features['attention'])

    @attention.method('__iter__')
    def attention(self):
        return iter(self._features['attention'])

    @classmethod
    def concat(cls, sequences):
        if not isinstance(sequences, _Iterable):
            raise TypeError(f"'{sequences.__class__.__name__}' object is not iterable")
        sequences = tuple(sequences)
        if len(sequences) == 0:
            raise ValueError("requires at least one sequence")

        all_keys, all_attention_keys = set(), set()
        total_sequence_length = 0
        device = None
        for seq in sequences:
            if not isinstance(seq, cls):
                raise TypeError(f"'{seq.__class__.__name__}' object is not a sequence")
            if device is None:
                device = seq.device
            else:
                if device != seq.device:
                    raise RuntimeError(f"expected all sequences to be on the same device, but found at least two devices, {device} and {seq.device}")
            total_sequence_length += seq.sequence_length
            all_keys.update(seq)
            all_attention_keys.update(seq.attention)

        new_features = {k : [] for k in all_keys}
        new_features_shape = {k : None for k in all_keys}
        new_attention_features = {k : [] for k in all_attention_keys}
        new_attention_features_shape = {k : None for k in all_attention_keys}

        for seq in sequences:
            exist_keys = all_keys.intersection(seq)
            not_exist_keys = all_keys - exist_keys
            for key in exist_keys:
                value = seq[key]
                assert value.shape[0] == seq.sequence_length
                if new_features_shape[key] is None:
                    new_features_shape[key] = value.shape[1:]
                else:
                    if new_features_shape[key] != value.shape[1:]:
                        raise ValueError(f"shape mismatch between sequence key '{key}': "
                                         f"[{', '.join(map(str, new_features_shape[key]))}] and [{', '.join(map(str, value.shape[1:]))}]")
                new_features[key].append(value)
            for key in not_exist_keys:
                # add a placeholder
                new_features[key].append(seq.sequence_length)

            exist_keys = all_attention_keys.intersection(seq.attention)
            not_exist_keys = all_attention_keys - exist_keys
            for key in exist_keys:
                value = seq.attention[key]
                assert value.shape[0] == value.shape[1] and value.shape[0] == seq.sequence_length
                if new_attention_features_shape[key] is None:
                    new_attention_features_shape[key] = value.shape[2:]
                else:
                    if new_attention_features_shape[key] != value.shape[2:]:
                        raise ValueError(f"shape mismatch between sequence attention key '{key}': "
                                         f"[{', '.join(map(str, new_attention_features_shape[key]))}] and [{', '.join(map(str, value.shape[2:]))}]")
                new_attention_features[key].append(seq.attention[key])
            for key in not_exist_keys:
                # add a placeholder
                new_attention_features[key].append(seq.sequence_length)

        new_seq = cls(total_sequence_length, device)
        for key, _features in new_features.items():
            feature_shape = new_features_shape[key]
            assert feature_shape is not None, f'feature shape of key \'{key}\' is None'
            _features = list(map(
                lambda x: torch.zeros(x, *feature_shape, device=new_seq.device) if isinstance(x, int) else x,
                _features,
            ))
            new_seq[key] = torch.cat(_features, dim=0)
        for key, _features in new_attention_features.items():
            feature_shape = new_attention_features_shape[key]
            assert feature_shape is not None, f'feature shape of attention key \'{key}\' is None'
            new_matrix = torch.zeros(total_sequence_length, total_sequence_length, *feature_shape, device=new_seq.device)
            _total_size = 0
            for _feature in _features:
                if isinstance(_feature, int):
                    _feature_size = _feature
                else:
                    _feature_size = _feature.shape[0]
                    new_matrix[_total_size : _total_size + _feature_size, _total_size : _total_size + _feature_size] = _feature
                _total_size += _feature_size
            new_seq.attention[key] = new_matrix
        return new_seq


class BatchedSequence(SequenceBase):
    """
    Read-only copy of the batched sequences.
    """
    def __init__(self, sequences):
        sequence_length, device, sequences = self._check(sequences)
        super().__init__(device)
        self._sequence_length = sequence_length
        self._batch_size = len(sequences)
        self._feature_register('feature', self._batch_size, self._sequence_length)
        self._feature_register('attention', self._batch_size, self._sequence_length, self._sequence_length)
        self._init_sequences(sequences)

    def _check(self, sequences):
        if not isinstance(sequences, _Iterable):
            raise TypeError(f"'{sequences.__class__.__name__}' object is not iterable")
        sequences = list(sequences)
        sequence_length = None
        device = None
        for sequence in sequences:
            if not isinstance(sequence, Sequence):
                raise TypeError(f"'{sequence.__class__.__name__}' object is not a sequence")
            if sequence_length is None:
                sequence_length = sequence.sequence_length
                device = sequence.device
            else:
                if sequence_length != sequence.sequence_length:
                    raise ValueError(f"sequence length mismatch between sequence [{sequence_length}] and [{sequence.sequence_length}]")
                if device != sequence.device:
                    raise RuntimeError(f"expected all sequences to be on the same device, but found at least two devices, {device} and {sequence.device}")
        return sequence_length, device, sequences

    def _init_sequences(self, sequences):
        shapes_feature = {}
        shapes_attention = {}
        for sequence in sequences:
            for feature_key in sequence:
                value = sequence[feature_key]
                if feature_key not in shapes_feature:
                    shapes_feature[feature_key] = value.shape
                else:
                    if shapes_feature[feature_key] != value.shape:
                        raise ValueError(f"shape mismatch between tensors [{', '.join(map(str, shapes_feature[feature_key]))}] and [{', '.join(map(str, value.shape))}]")
            for attention_key in sequence.attention:
                value = sequence.attention[attention_key]
                if attention_key not in shapes_attention:
                    shapes_attention[attention_key] = value.shape
                else:
                    if shapes_attention[attention_key] != value.shape:
                        raise ValueError(f"shape mismatch between tensors [{', '.join(map(str, shapes_attention[attention_key]))}] and [{', '.join(map(str, value.shape))}]")

        keys_feature, keys_attention = set(shapes_feature.keys()), set(shapes_attention.keys())
        batches_feature = {k : [] for k in keys_feature}
        batches_attention = {k : [] for k in keys_attention}
        for sequence in sequences:
            other_keys_feature, other_keys_attention = keys_feature.difference(sequence), keys_attention.difference(sequence.attention)
            for feature_key in sequence:
                batches_feature[feature_key].append(sequence[feature_key])
            for feature_key in other_keys_feature:
                batches_feature[feature_key].append(torch.zeros(*shapes_feature[feature_key], device=self.device))
            for attention_key in sequence.attention:
                batches_attention[attention_key].append(sequence.attention[attention_key])
            for attention_key in other_keys_attention:
                batches_attention[attention_key].append(torch.zeros(*shapes_attention[attention_key], device=self.device))
        for k, v in batches_feature.items():
            self._features['feature'][k] = torch.stack(v, dim=0)
        for k, v in batches_attention.items():
            self._features['attention'][k] = torch.stack(v, dim=0)

    def __len__(self):
        return len(self._features['feature'])

    def __iter__(self):
        return iter(self._features['feature'])

    def __getitem__(self, item):
        return self._features['feature'].__getitem__(item)

    def __bool__(self):
        return self._features['feature'] or self._features['attention']

    def __str__(self):
        return f"{self.__class__.__name__}(" \
               f"sequence_length={self._sequence_length}, " \
               f"feature={str(self._features['feature'])}, " \
               f"attention={str(self._features['attention'])})"

    def __repr__(self):
        return self.__str__()

    @view.getter_view
    def attention(self, item):
        return self._features['attention'].__getitem__(item)

    @attention.str
    def attention(self):
        return str(self._features['attention'])

    @attention.repr
    def attention(self):
        return repr(self._features['attention'])

    @attention.method('__len__')
    def attention(self):
        return len(self._features['attention'])

    @attention.method('__iter__')
    def attention(self):
        return iter(self._features['attention'])

