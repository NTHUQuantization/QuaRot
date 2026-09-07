from types import SimpleNamespace

import pytest
import torch

from e2e.benchmark_accuracy import full_perplexity


class ToyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size=11):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 4)
        self.vocab_size = vocab_size

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        self.last_use_cache = use_cache
        self.last_past_key_values = past_key_values
        target = (input_ids + 1).remainder(self.vocab_size)
        logits = torch.zeros(
            *input_ids.shape, self.vocab_size, device=input_ids.device)
        logits.scatter_(-1, target.unsqueeze(-1), 2.0)
        cache_length = (past_key_values or 0) + input_ids.shape[1]
        return SimpleNamespace(logits=logits, past_key_values=cache_length)


def test_full_perplexity_uses_complete_non_overlapping_blocks():
    model = ToyCausalLM()
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    result = full_perplexity(
        model, input_ids, context_length=4, batch_size=2)
    assert model.last_use_cache is False
    blocks = input_ids[:, :8].reshape(2, 4)
    logits = model(blocks).logits[:, :-1]
    labels = blocks[:, 1:]
    losses = torch.nn.functional.cross_entropy(
        logits.permute(0, 2, 1), labels, reduction="none")
    expected_mean_nll = losses.float().mean(dim=1).mean()
    expected_nll = losses.double().sum().item()
    assert result["block_count"] == 2
    assert result["scored_tokens"] == 6
    assert result["used_input_tokens"] == 8
    assert result["truncated_tail_tokens"] == 2
    assert result["total_nll"] == pytest.approx(expected_nll)
    assert result["mean_nll"] == pytest.approx(expected_mean_nll.item())
    assert result["perplexity"] == pytest.approx(
        torch.exp(expected_mean_nll).item())
    assert result["execution"] == "full_sequence_no_cache"
    assert result["kv_precision"] == "float16"


def test_full_perplexity_rejects_dataset_shorter_than_context():
    with pytest.raises(ValueError, match="fewer than one"):
        full_perplexity(
            ToyCausalLM(), torch.tensor([[1, 2, 3]]),
            context_length=4)
