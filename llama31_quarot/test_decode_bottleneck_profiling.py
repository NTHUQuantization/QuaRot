import torch

from attention_fusion import make_uniform_paged_kv_metadata
from llama31_quarot.hf_quarot_model import FusionConfig, make_active_metadata
from llama31_quarot.summarize_decode_bottlenecks import classify_kernel, interval_union_ns


def test_fusion_presets():
    assert FusionConfig.from_name("unfused_INT4") == FusionConfig()
    assert FusionConfig.from_name("k1_k2_fused") == FusionConfig(k1=True, k2=True)
    assert FusionConfig.from_name("full_fused_hadacore256").backend == "hadacore256"


def test_active_metadata_uses_capacity_page_ids():
    allocation = make_uniform_paged_kv_metadata(2, 300, 128, "cuda")
    active = make_active_metadata(allocation, 129, "cuda")
    assert active.pages_per_batch == 2
    assert active.total_pages == 6
    assert active.indptr.cpu().tolist() == [0, 2, 4]
    assert active.indices.cpu().tolist() == [0, 1, 3, 4]
    assert active.last_page_offset.cpu().tolist() == [1, 1]


def test_interval_union_avoids_double_counting():
    assert interval_union_ns([(0, 10), (5, 20), (30, 35)]) == 25


def test_kernel_categories():
    assert classify_kernel("BatchDecodeWithPagedKVGQAKernel<__half, flashinfer::quant::__precision__s4>") == "K2_INT4"
    assert classify_kernel("BatchDecodeWithPagedKVGQAKernel<__half, __half>") == "K2_FP16"
    assert classify_kernel("output_had_quant_kernel") == "K3"
    assert classify_kernel("Cijk_Alik_Bljk") == "rocBLAS_GEMM"


if __name__ == "__main__":
    test_fusion_presets()
    test_active_metadata_uses_capacity_page_ids()
    test_interval_union_avoids_double_counting()
    test_kernel_categories()
    torch.cuda.synchronize()
    print("decode bottleneck profiling tests: PASS")
