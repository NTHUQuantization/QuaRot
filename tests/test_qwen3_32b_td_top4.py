import torch

from e2e.qwen3_32b_td_top4_train_windowed import trainable_top_layers


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(4, 4, bias=False) for _ in range(28)]
        )


class _Draft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Model()
        for parameter in self.parameters():
            parameter.requires_grad_(False)


def test_trainable_top_layers_only_unfreezes_last_four():
    draft = _Draft()
    selected = trainable_top_layers(draft, 4)

    assert len(selected) == 4
    assert [name.split(".")[2] for name, _ in selected] == ["24", "25", "26", "27"]
    assert not any(
        parameter.requires_grad
        for layer in draft.model.layers[:24]
        for parameter in layer.parameters()
    )
    assert all(
        parameter.requires_grad
        for layer in draft.model.layers[24:]
        for parameter in layer.parameters()
    )
