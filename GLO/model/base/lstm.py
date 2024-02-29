import torch


class MultiInputLSTM(torch.nn.Module):
    def __init__(self, hidden_size, in_feature_size=None, input_branches=2, output_branches=1):
        super().__init__()
        self.out_feature_size = hidden_size
        self.input_branches = input_branches
        self.output_branches = output_branches

        if in_feature_size is None:
            in_feature_size = 0
        self.fc = torch.nn.Linear(hidden_size * input_branches + in_feature_size, (input_branches + 3) * hidden_size * output_branches)

    def forward(self, branches, input=None):
        hs, cs = zip(*branches)
        if input is not None:
            fc_input = torch.cat([*hs, input], dim=-1)
        else:
            fc_input = torch.cat(hs, dim=-1)

        lstm_in = self.fc(fc_input)
        a, i, o, *fs = lstm_in.chunk(self.input_branches + 3, -1)
        c = a.tanh() * i.sigmoid()
        for f, _c in zip(fs, cs):
            _c = _c.repeat(*(1 for i in range(_c.ndim - 1)), self.output_branches)
            c = c + f.sigmoid() * _c
        h = o.sigmoid() * c.tanh()
        return h, c
