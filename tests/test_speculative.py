from types import SimpleNamespace

import pytest
import torch

from e2e.speculative import (
    AdaptiveK,
    FusedPardRuntime,
    GenerationResult,
    Pard2Spec,
    SelectedHiddenCollector,
    TD_PROXY_QWEN3_14B_ON_QWEN3_32B,
    TargetFeatEmbedWarp,
    _ExactSmallChunkRows,
    _RowIndependentRMSNorm,
    _RowIndependentRMSNormQuant,
    _basis_from_rotation_metadata,
    _load_td_projection_state,
    _normalized_hadamard_cpu,
    _preflight_pard2,
    fold_td_projection_weight,
    greedy_accept,
    load_td_target_basis,
)
from quarot.transformers.kv_cache import (
    CacheTransaction,
    MultiLayerPagedKVCache4Bit,
)
from e2e.pard2 import parser as generation_parser
from e2e.benchmark_pard2 import parser as benchmark_parser
from e2e.pard2_calibrate import fit_affine
from e2e.qualify_pard2 import validate_formal_payload


def _draft_config(target_dim=16384):
    return SimpleNamespace(
        model_type="qwen3", hidden_size=1024, intermediate_size=3072,
        num_hidden_layers=28, num_attention_heads=16,
        num_key_value_heads=8, vocab_size=151936, pard_token=151670,
        pard2_target_dim=target_dim,
        pard2_target_layers=[-1, -8, -16, -24],
        pard2_scale=0.02, pard2_proj_bias=False)


def _target_config_8b():
    return SimpleNamespace(
        model_type="qwen3", hidden_size=4096, intermediate_size=12288,
        num_hidden_layers=36, num_attention_heads=32,
        num_key_value_heads=8, head_dim=128, vocab_size=151936)


def _target_config_32b(*, with_basis=False):
    revision = "9216db5781bf21249d130ec9da846c4624c16137"
    values = dict(
        model_type="qwen3_quarot", hidden_size=5120,
        intermediate_size=25600, num_hidden_layers=64,
        num_attention_heads=64, num_key_value_heads=8, head_dim=128,
        vocab_size=151936,
        tokenizer_name_or_path=(
            f"/cache/models--Qwen--Qwen3-32B/snapshots/{revision}"))
    if with_basis:
        values.update(
            quarot_rotation_format="hadk_v1",
            quarot_rotation_width=5120,
            quarot_rotation_remainder=40,
            quarot_rotation_inner=128,
            quarot_rotation_seed=0,
            quarot_rotation_device="cuda",
            quarot_rotation_dtype="float32",
            quarot_rotation_signs=[1, -1] * 2560,
            quarot_final_norm_weight=[1.0] * 5120)
    return SimpleNamespace(**values)


def test_pard2_pinned_config_contract():
    spec = Pard2Spec()
    target = _target_config_8b()
    draft = _draft_config()
    spec.validate(target, draft)
    draft.pard_token = 1
    with pytest.raises(ValueError, match="pard_token"):
        spec.validate(target, draft)


def test_pard2_ti_accepts_qwen3_32b_without_loading_warp(monkeypatch):
    def fail_load(*_args, **_kwargs):
        raise AssertionError("TI must not load warp_model.bin")

    monkeypatch.setattr(torch, "load", fail_load)
    spec, projection, basis = _preflight_pard2(
        "pard2-ti", _target_config_32b(), _draft_config(),
        draft_snapshot="/does/not/exist")
    assert spec.target_dim == 16384
    assert projection is None
    assert basis is None


def test_pard2_td_32b_requires_explicit_proxy_profile():
    with pytest.raises(ValueError, match="hidden_size must be 4096"):
        Pard2Spec().validate(
            _target_config_32b(), _draft_config(),
            mode="pard2-td")


def test_explicit_32b_profile_reports_target_identity_and_keeps_shared_ti_draft():
    spec = Pard2Spec.for_benchmark_profile("qwen3_32b", mode="pard2-ti")
    assert spec.target_model_id == "Qwen/Qwen3-32B"
    assert spec.target_revision == "9216db5781bf21249d130ec9da846c4624c16137"
    assert spec.model_id == "amd/PARD2-Qwen3-8B"
    spec.validate(_target_config_32b(), _draft_config(), mode="pard2-ti")
    with pytest.raises(ValueError, match="projection width"):
        spec.validate(_target_config_32b(), _draft_config(), mode="pard2-td")


def test_pard2_14b_on_32b_proxy_validates_pins_shapes_and_basis(
        tmp_path, monkeypatch):
    spec = Pard2Spec.for_proxy_profile(
        TD_PROXY_QWEN3_14B_ON_QWEN3_32B)
    snapshot = tmp_path / "snapshots" / spec.revision
    snapshot.mkdir(parents=True)
    (snapshot / "warp_model.bin").touch()
    weight = torch.empty((1024, 20480), device="meta")
    monkeypatch.setattr(
        torch, "load", lambda *_args, **_kwargs: {
            "target_proj.weight": weight})

    actual, projection, basis = _preflight_pard2(
        "pard2-td", _target_config_32b(with_basis=True),
        _draft_config(target_dim=20480), draft_snapshot=snapshot,
        td_proxy_profile=TD_PROXY_QWEN3_14B_ON_QWEN3_32B)

    assert actual == spec
    assert actual.target_alignment == "cross_target_proxy"
    assert projection["weight"].shape == (1024, 20480)
    assert basis[0].shape == basis[1].shape == (5120,)


def test_pard2_proxy_rejects_unpinned_draft_snapshot():
    spec = Pard2Spec.for_proxy_profile(
        TD_PROXY_QWEN3_14B_ON_QWEN3_32B)
    with pytest.raises(ValueError, match="pinned PARD2-Qwen3-14B"):
        spec.validate(
            _target_config_32b(with_basis=True),
            _draft_config(target_dim=20480), mode="pard2-td",
            draft_snapshot="/cache/snapshots/not-the-pinned-revision")


def test_pard2_warp_shape_is_checked_before_model_load(tmp_path, monkeypatch):
    spec = Pard2Spec.for_proxy_profile(
        TD_PROXY_QWEN3_14B_ON_QWEN3_32B)
    (tmp_path / "warp_model.bin").touch()
    monkeypatch.setattr(
        torch, "load", lambda *_args, **_kwargs: {
            "target_proj.weight": torch.empty((1024, 16384))})
    with pytest.raises(ValueError, match=r"shape \(1024, 20480\)"):
        _load_td_projection_state(
            tmp_path, spec, _draft_config(target_dim=20480))


def test_rotation_metadata_is_self_contained_and_validated():
    config = _target_config_32b(with_basis=True)
    signs, norm = _basis_from_rotation_metadata(config)
    assert signs.shape == norm.shape == (5120,)
    target = SimpleNamespace(config=config)
    loaded_signs, loaded_norm = load_td_target_basis(target, None)
    torch.testing.assert_close(loaded_signs, signs)
    torch.testing.assert_close(loaded_norm, norm)

    config.quarot_rotation_remainder = 20
    with pytest.raises(ValueError, match="factorization differs"):
        _basis_from_rotation_metadata(config)


def test_partial_rotation_metadata_fails_closed():
    config = _target_config_32b()
    config.quarot_rotation_format = "hadk_v1"
    with pytest.raises(ValueError, match="incomplete QuaRot TD basis"):
        _basis_from_rotation_metadata(config)


def test_ar_and_speculative_modes_enforce_draft_ownership():
    target = SimpleNamespace()
    draft = SimpleNamespace()
    with pytest.raises(ValueError, match="AR mode requires draft=None"):
        FusedPardRuntime(
            mode="ar", target=target, draft=draft, tokenizer=None,
            compile_mode="eager")
    with pytest.raises(ValueError, match="requires a PARD2 drafter"):
        FusedPardRuntime(
            mode="pard2-ti", target=target, draft=None, tokenizer=None,
            compile_mode="eager")


def test_compiled_drafter_allows_all_fixed_proposal_shapes(monkeypatch):
    compiled = {}

    def fake_compile(fn, **kwargs):
        compiled.update(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 8)
    target = SimpleNamespace(config=SimpleNamespace(eos_token_id=0))
    draft = SimpleNamespace(forward=lambda *args, **kwargs: None)
    FusedPardRuntime(
        mode="pard2-ti", target=target, draft=draft, tokenizer=None,
        compile_mode="max-autotune",
    )
    assert torch._dynamo.config.recompile_limit == Pard2Spec().draft_k + 1
    assert compiled == {
        "mode": "max-autotune", "fullgraph": True, "dynamic": False,
    }


class _ToyNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_dim = 4
        self.eps = 1e-6

    def forward(self, value):
        return value


class _ToyDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = _ToyNorm()
        self.post_attention_layernorm = _ToyNorm()
        self.self_attn = torch.nn.Module()
        self.mlp = torch.nn.Module()
        from quarot.nn import Quantizer
        self.self_attn.quantizer = Quantizer(0.9)
        self.mlp.quantizer = Quantizer(1.0)


class _ToyRuntimeTarget(torch.nn.Module):
    def __init__(self, layers=1, quantized=False):
        super().__init__()
        self.config = SimpleNamespace(eos_token_id=0)
        if quantized:
            self.cache_dtype = "int4"
        self.model = torch.nn.Module()
        self.model.norm = _ToyNorm()
        self.model.layers = torch.nn.ModuleList(
            [_ToyDecoderLayer() for _ in range(layers)])
        self.lm_head = torch.nn.Linear(4, 7, bias=False)


def _toy_runtime(**kwargs):
    return FusedPardRuntime(
        mode="ar", target=_ToyRuntimeTarget(), draft=None, tokenizer=None,
        compile_mode="eager", **kwargs)


def test_exact_row_norm_and_rowwise_lm_head_have_independent_defaults():
    runtime = _toy_runtime()
    assert runtime.exact_row_norm is True
    assert runtime.rowwise_lm_head is False
    assert isinstance(runtime.target.model.norm, _RowIndependentRMSNorm)
    assert not isinstance(runtime.target.lm_head, _ExactSmallChunkRows)


def test_rowwise_lm_head_can_be_enabled_without_exact_row_norm():
    runtime = _toy_runtime(exact_row_norm=False, rowwise_lm_head=True)
    assert not isinstance(runtime.target.model.norm, _RowIndependentRMSNorm)
    assert isinstance(runtime.target.lm_head, _ExactSmallChunkRows)


@pytest.mark.parametrize(("legacy", "wrapped"), [(True, True), (False, False)])
def test_legacy_exact_small_chunk_alias_controls_both_paths(legacy, wrapped):
    runtime = _toy_runtime(exact_small_chunk=legacy)
    assert isinstance(runtime.target.model.norm, _RowIndependentRMSNorm) is wrapped
    assert isinstance(runtime.target.lm_head, _ExactSmallChunkRows) is wrapped


def test_batched_and_rowwise_lm_head_preserve_top1():
    torch.manual_seed(17)
    head = torch.nn.Linear(4, 7, bias=False)
    values = torch.randn(1, 16, 4)
    batched = head(values)
    rowwise = _ExactSmallChunkRows(head, max_rows=16)(values)
    torch.testing.assert_close(rowwise, batched)
    assert torch.equal(rowwise.argmax(dim=-1), batched.argmax(dim=-1))


def test_fused_norm_quant_guards_and_default():
    runtime = _toy_runtime()
    assert runtime.fused_norm_quant is False
    with pytest.raises(ValueError, match="exact_row_norm"):
        _toy_runtime(exact_row_norm=False, fused_norm_quant=True)
    with pytest.raises(ValueError, match="quantized target"):
        _toy_runtime(fused_norm_quant=True)


def test_fused_norm_quant_wraps_72_layer_norms_but_not_final_norm():
    target = _ToyRuntimeTarget(layers=36, quantized=True)
    runtime = FusedPardRuntime(
        mode="ar", target=target, draft=None, tokenizer=None,
        compile_mode="eager", fused_norm_quant=True)
    assert isinstance(runtime.target.model.norm, _RowIndependentRMSNorm)
    assert not isinstance(
        runtime.target.model.norm, _RowIndependentRMSNormQuant)
    layer_norms = [norm for layer in runtime.target.model.layers for norm in
                   (layer.input_layernorm, layer.post_attention_layernorm)]
    assert len(layer_norms) == 72
    assert all(isinstance(norm, _RowIndependentRMSNormQuant)
               for norm in layer_norms)
    assert all(layer.input_layernorm.input_clip_ratio == 0.9
               and layer.post_attention_layernorm.input_clip_ratio == 1.0
               for layer in runtime.target.model.layers)


def test_packed_quantizer_passthrough_and_logical_shape():
    import quarot
    storage = torch.zeros((2, 3, 4), dtype=torch.int8)
    scales = torch.ones((6, 1), dtype=torch.float16)
    packed = quarot.PackedQuantizedTensor(
        storage, scales, logical_shape=(2, 3, 8))
    assert quarot.nn.Quantizer()(packed) is packed
    assert packed.logical_shape == (2, 3, 8)
    assert packed.size() == storage.size()


def test_runtime_close_removes_td_hooks_idempotently():
    target = _ToyRuntimeTarget(layers=36)
    draft = SimpleNamespace(forward=lambda *_args, **_kwargs: None)
    runtime = FusedPardRuntime(
        mode="pard2-td", target=target, draft=draft, tokenizer=None,
        compile_mode="eager")
    tapped = [target.model.layers[index] for index in runtime.collector.indices]
    assert all(module._forward_hooks for module in tapped)
    runtime.close()
    runtime.close()
    assert all(not module._forward_hooks for module in tapped)


@pytest.mark.parametrize("make_parser", [generation_parser, benchmark_parser])
def test_cli_defaults_select_exact_norm_and_batched_lm_head(make_parser):
    argv = ["--mode", "ar"]
    if make_parser is benchmark_parser:
        argv += ["--dataset", "humaneval", "--output", "unused.json"]
    args = make_parser().parse_args(argv)
    assert args.exact_row_norm is True
    assert args.rowwise_lm_head is False
    assert args.td_cache_basis is True
    assert args.td_lazy_features is False
    assert args.td_unique_projection is False
    assert args.fused_norm_quant is False
    assert args.exact_small_chunk is None
    if make_parser is benchmark_parser:
        assert args.precompile_draft_shapes is False


@pytest.mark.parametrize(("predictions", "accepted"), [
    ([9, 2, 3, 4], 0),
    ([1, 2, 9, 4], 2),
    ([1, 2, 3, 4], 3),
])
def test_greedy_accept_zero_partial_all(predictions, accepted):
    candidates = torch.tensor([[1, 2, 3]])
    assert greedy_accept(candidates, torch.tensor([predictions])) == accepted


def test_transaction_commit_and_rollback_leave_stale_storage_logical():
    cache = SimpleNamespace(length=127, _transaction=None)
    transaction = CacheTransaction(cache)
    cache.length = 143
    transaction.proposed_length = 143
    transaction.commit(6)
    assert cache.length == 133
    assert cache._transaction is None
    rollback = CacheTransaction(cache)
    cache.length = 149
    rollback.proposed_length = 149
    rollback.rollback()
    assert cache.length == 133


def test_transaction_rejects_invalid_commit():
    cache = SimpleNamespace(length=10, _transaction=None)
    transaction = CacheTransaction(cache)
    transaction.proposed_length = 15
    with pytest.raises(ValueError):
        transaction.commit(6)
    transaction.rollback()


def _cpu_paged_cache(*, max_length=256, native_gqa=True, fused_k1=True):
    return MultiLayerPagedKVCache4Bit(
        batch_size=1, page_size=128, max_seq_len=max_length,
        device=torch.device("cpu"), n_layers=1, num_heads=4,
        num_kv_heads=2, native_gqa=native_gqa,
        fused_decode_append=True, fused_k1=fused_k1,
        head_dim=64, disable_quant=False, hadamard_dtype=torch.float16)


def test_persistent_metadata_tracks_transaction_commit_and_rollback(
        monkeypatch):
    monkeypatch.delenv("QUAROT_PERSISTENT_KV_METADATA", raising=False)
    cache = _cpu_paged_cache()
    cache.length = 127
    transaction = cache.begin()
    cache.length = 143
    transaction.proposed_length = 143
    provisional = cache.get_cache_specs_for_flash_infer(None)
    assert provisional["last_page_offset"].tolist() == [15]
    assert provisional["kv_indptr"].tolist() == [0, 2]
    transaction.commit(6)
    committed = cache.get_cache_specs_for_flash_infer(None)
    assert cache.length == 133
    assert committed["last_page_offset"].tolist() == [5]
    assert committed["kv_indptr"].tolist() == [0, 2]

    rollback = cache.begin()
    cache.length = 149
    rollback.proposed_length = 149
    rollback.rollback()
    restored = cache.get_cache_specs_for_flash_infer(None)
    assert cache.length == 133
    assert restored["last_page_offset"].tolist() == [5]


def test_k1_is_disabled_for_transactions_masks_and_expanded_oracle():
    cache = _cpu_paged_cache()
    cache._needs_init[0] = False
    assert cache.can_fuse_k1(0, None)
    transaction = cache.begin()
    assert not cache.can_fuse_k1(0, None)
    transaction.rollback()
    assert not cache.can_fuse_k1(0, torch.ones(1, 1))

    expanded = _cpu_paged_cache(native_gqa=False)
    expanded._needs_init[0] = False
    assert not expanded.can_fuse_k1(0, None)
    disabled = _cpu_paged_cache(fused_k1=False)
    disabled._needs_init[0] = False
    assert not disabled.can_fuse_k1(0, None)


@pytest.mark.parametrize("static_metadata", ["0", "1"])
def test_graph_metadata_and_transactions_are_mutually_exclusive(monkeypatch, static_metadata):
    monkeypatch.delenv("QUAROT_PERSISTENT_KV_METADATA", raising=False)
    monkeypatch.setenv("QUAROT_STATIC_KV_METADATA", static_metadata)
    cache = _cpu_paged_cache(max_length=128)
    cache.length = 5
    cache.enable_cuda_graph_decode()
    assert cache.get_cache_specs_for_flash_infer(
        None)["last_page_offset"].tolist() == [6]
    cache.advance_cuda_graph_decode()
    assert cache.get_cache_specs_for_flash_infer(
        None)["last_page_offset"].tolist() == [7]
    with pytest.raises(RuntimeError, match="transactions"):
        cache.begin()


class _ToyTarget(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [torch.nn.Linear(4, 4, bias=False) for _ in range(32)])


def test_selected_hidden_tap_order_and_shape():
    target = _ToyTarget()
    collector = SelectedHiddenCollector(target, (-1, -8, -16, -24))
    x = torch.ones(1, 2, 4)
    for index, layer in enumerate(target.model.layers):
        with torch.no_grad():
            layer.weight.copy_(torch.eye(4) * (index + 1))
        layer(x)
    features = collector.features()
    assert collector.indices == (31, 24, 16, 8)
    assert features.shape == (1, 2, 16)
    sliced = collector.features(slice(0, 1))
    torch.testing.assert_close(sliced, features[:, :1])
    assert torch.equal(features[..., :4], x * 32)
    assert torch.equal(features[..., 4:8], x * 25)
    collector.close()

def test_selected_hidden_basis_is_cached_once():
    target = _ToyTarget()
    collector = SelectedHiddenCollector(
        target, (-1, -8, -16, -24),
        torch.ones(4, dtype=torch.float64),
        torch.ones(4, dtype=torch.float64),
        cache_basis=True)
    assert collector.basis_cached
    assert collector.rotation_signs.dtype == torch.float32
    signs_ptr = collector.rotation_signs.data_ptr()
    norm_ptr = collector.final_norm_weight.data_ptr()
    collector.cache_basis(torch.device("cpu"), torch.float32)
    assert collector.rotation_signs.data_ptr() == signs_ptr
    assert collector.final_norm_weight.data_ptr() == norm_ptr
    collector.close()


class _RecordingCollector:
    def __init__(self):
        self.rows = []

    def features(self, rows):
        self.rows.append(rows.stop)
        return torch.arange(rows.stop, dtype=torch.float32).view(1, rows.stop, 1)


class _ImmediateTimer:
    def __init__(self):
        self.names = []

    def record(self, name, fn):
        self.names.append(name)
        return fn()


def test_lazy_td_features_use_only_emitted_rows():
    runtime = object.__new__(FusedPardRuntime)
    runtime.collector = _RecordingCollector()
    timer = _ImmediateTimer()
    prefill = torch.tensor([[[10.0], [20.0]]])

    first_zero = runtime._accepted_td_features(prefill, 1, True, timer)
    assert torch.equal(first_zero, prefill[:, -1:])
    assert runtime.collector.rows == []

    first_partial = runtime._accepted_td_features(prefill, 4, True, timer)
    assert runtime.collector.rows == [3]
    assert first_partial.shape[1] == 4
    later_partial = runtime._accepted_td_features(prefill, 4, False, timer)
    assert runtime.collector.rows == [3, 4]
    assert later_partial.shape[1] == 4
    assert timer.names == ["td_feature_restore", "td_feature_restore"]


def test_unique_projection_matches_legacy_repeated_features():
    class ToyDraft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=3)
            self.embedding = torch.nn.Embedding(11, 3)

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, inputs_embeds=None, **_kwargs):
            return inputs_embeds

    torch.manual_seed(4)
    calibration = {
        "raw_scale": torch.linspace(0.8, 1.2, 4),
        "raw_bias": torch.linspace(-0.1, 0.1, 4),
        "projected_scale": torch.tensor([0.9, 1.0, 1.1]),
        "projected_bias": torch.tensor([-0.2, 0.0, 0.2]),
    }
    wrapper = TargetFeatEmbedWarp(
        ToyDraft(), target_dim=4, scale=0.02, proj_bias=True,
        calibration=calibration)
    runtime = object.__new__(FusedPardRuntime)
    runtime.draft = wrapper
    unique = torch.randn(1, 3, 4)
    mask_rows = 5
    repeated = torch.cat((
        unique,
        unique[:, -1:].expand(-1, mask_rows, -1),
    ), dim=1)
    input_ids = torch.arange(repeated.shape[1]).view(1, -1)

    legacy = wrapper(input_ids=input_ids, target_feat=repeated)
    projected = runtime._project_td_features(unique, mask_rows)
    optimized = wrapper(
        input_ids=input_ids, projected_target_feat=projected)

    assert projected.shape == (1, repeated.shape[1], 3)
    torch.testing.assert_close(optimized, legacy, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("width", [16, 5120])
def test_fused_rotation_basis_round_trip(width):
    source = torch.randn(8, width)
    signs = torch.tensor(
        ([1, -1] * (width // 2)), dtype=torch.float32)
    rotated = _normalized_hadamard_cpu(source * signs)
    restored = _normalized_hadamard_cpu(rotated, transpose=True) * signs
    torch.testing.assert_close(restored, source, atol=2e-5, rtol=2e-5)


def test_generalized_td_projection_fold_matches_explicit_restoration():
    torch.manual_seed(9)
    hidden_size = 5120
    source = [torch.randn(2, hidden_size) for _ in range(4)]
    signs = torch.tensor([1, -1] * (hidden_size // 2), dtype=torch.float32)
    gamma = torch.linspace(0.75, 1.25, hidden_size)
    rotated = [
        _normalized_hadamard_cpu(value * signs) for value in source]
    weight = torch.randn(3, 4 * hidden_size)
    folded = fold_td_projection_weight(
        weight, signs, gamma, hidden_size)

    expected_features = torch.cat(
        [source[0] * gamma, *source[1:]], dim=-1)
    actual_features = torch.cat(rotated, dim=-1)
    expected = torch.nn.functional.linear(expected_features, weight)
    actual = torch.nn.functional.linear(actual_features, folded)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-4)


def test_adaptive_k_and_metrics():
    selector = AdaptiveK()
    assert selector.choose() == 15
    for _ in range(20):
        selector.update(0, 15)
    assert selector.choose() == 8
    result = GenerationResult([1, 2, 3], 1, 2, 3, 1, 1, 15, 2, 1,
                              [3], [3], {"draft": 1.0}, 123)
    metrics = result.metrics()
    assert metrics["mean_accept_length"] == 3
    assert metrics["end_to_end_tokens_per_s"] == 1000
    assert metrics["conditional_acceptance"] == pytest.approx(2 / 3)
    assert metrics["conditional_acceptance_by_position"][:3] == [1.0, 1.0, 0.0]


def test_per_channel_affine_calibration_recovers_mapping():
    source = torch.randn(3, 5, 7)
    expected_scale = torch.linspace(0.5, 1.5, 7)
    expected_bias = torch.linspace(-0.2, 0.2, 7)
    scale, bias = fit_affine(source, source * expected_scale + expected_bias)
    torch.testing.assert_close(scale.float(), expected_scale, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(bias.float(), expected_bias, atol=1e-3, rtol=1e-3)


def test_qualification_rejects_smoke_contract():
    payload = {
        "contract": {
            "mode": "ar", "dataset": "math_500", "qualified": False,
            "prompt_count": 1, "generated_tokens": 32, "batch_size": 1,
            "warmups": 1, "sweeps": 1, "greedy": True,
        },
        "runs": [{}],
    }
    with pytest.raises(ValueError, match="not a formal benchmark"):
        validate_formal_payload(payload, "ar", "math_500")
