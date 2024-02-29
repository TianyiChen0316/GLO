import typing
from collections.abc import Iterable

import torch

from lib.torch.sequential_data import Sequence
from .base.lstm import MultiInputLSTM


class TreeLSTM(torch.nn.Module):
    def __init__(self, feature_size, input_size):
        super().__init__()
        self.preprocess = torch.nn.Sequential(
            torch.nn.Linear(input_size, feature_size, bias=True),
            torch.nn.LayerNorm(feature_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(feature_size, feature_size),
        )
        self.lstm = MultiInputLSTM(
            feature_size,
            feature_size,
            input_branches=2,
            output_branches=1,
        )
        self.tail = torch.nn.Sequential(
            torch.nn.Linear(feature_size, feature_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(feature_size, 1),
        )

    def forward(self, seq : typing.Union[Sequence, typing.Iterable[Sequence]]):
        if not isinstance(seq, Sequence):
            if isinstance(seq, Iterable):
                seq = Sequence.concat(seq)
            else:
                raise TypeError(f"'{seq.__class__.__name__}' object is not a sequence")

        branches = [(seq['left_hidden'], seq['left_cell']), (seq['right_hidden'], seq['right_cell'])]
        input = seq['extra_input']
        normalized_input = self.preprocess(input)
        res_hidden, res_cell = self.lstm(branches, normalized_input)
        seq['node_hidden'], seq['node_cell'] = res_hidden, res_cell
        return seq

    def predict(self, embeddings):
        if not isinstance(embeddings, torch.Tensor):
            if isinstance(embeddings, Iterable):
                embeddings = torch.stack(tuple(embeddings), dim=0)
        return self.tail(embeddings).squeeze(-1)
