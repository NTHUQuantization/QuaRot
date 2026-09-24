import torch

from e2e.qwen3_32b_td_projection_train import exact_chunked_loss_and_hidden_grad


def test_exact_chunked_loss_and_gradient_matches_full_autograd():
    torch.manual_seed(7)
    n, hs, ht, vocab = 5, 4, 6, 11
    student = torch.randn(n, hs, requires_grad=True)
    teacher = torch.randn(n, ht)
    sw, tw = torch.randn(vocab, hs), torch.randn(vocab, ht)
    labels = torch.tensor([1, 4, -100, 8, 2])
    weights = torch.tensor([1.0, 0.8, 0.0, 0.3, 0.1])
    sl, tl = student @ sw.T, teacher @ tw.T
    valid = labels != -100; denom = (weights * valid).sum()
    ce = torch.nn.functional.cross_entropy(sl, labels, reduction="none", ignore_index=-100)
    kd = torch.nn.functional.kl_div(torch.log_softmax(sl, -1), torch.softmax(tl, -1), reduction="none").sum(-1)
    full = (0.1 * (ce * weights).sum() + (kd * weights).sum()) / denom
    full.backward(); expected = student.grad.clone()
    got, got_ce, got_kd, grad = exact_chunked_loss_and_hidden_grad(
        student.detach(), teacher, labels, weights, sw, tw, chunk_size=3)
    assert torch.allclose(got, full.detach(), atol=2e-6)
    assert torch.allclose(grad, expected, atol=2e-6)
