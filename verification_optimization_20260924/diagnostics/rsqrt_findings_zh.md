# CPU-only：HIP rsqrt overload 與既有 PyTorch 語意

日期：2026-09-24。此診斷只執行 HIP 編譯，**沒有啟動 GPU kernel，也不是效能量測**。

## 重現命令

在既有 `qwen3_32b_quarot_clean` 容器執行：

```bash
cd /workspace_root/fused_v1
/opt/rocm/bin/hipcc --offload-arch=gfx1201 --cuda-device-only -O1 -S -emit-llvm \
  verification_optimization_20260924/diagnostics/verification_rsqrt_overload_probe.hip \
  -o /tmp/verification_rsqrt_overload_probe_reproduced.ll
```

原始診斷使用相同命令參數，輸入與輸出路徑分別是容器內 `/tmp/verification_rsqrt_overload_probe.hip`、`/tmp/verification_rsqrt_overload_probe.ll`。隨附 LLVM IR 保留原始編譯的 source filename 和 compiler version，故重新編譯時路徑、module ID 等非算術 metadata 可不同。

工具鏈：HIP 7.2.26015-fc0010cf6a；AMD clang 22.0.0git（roc-7.2.0 26014，LLVM revision `7b800a19466229b8479a78de19143dc33c3ab9b5`）。

## 結果與解讀

probe 把同一個 FP32 input 分別傳給 `::rsqrt(x)` 與 `::rsqrtf(x)`，產生兩個 FP32 outputs。隨附 LLVM IR 顯示：

| 表達式 | 此 ROCm 工具鏈實際產生的算術 |
|---|---|
| `::rsqrt(float)` | `fpext float → double`；`llvm.amdgcn.rsq.f64` 與 FP64 FMA refinement；`fptrunc double → float` |
| `::rsqrtf(float)` | `llvm.amdgcn.rsq.f32`，另含 subnormal 處理 |

本機 clang HIP math header `/opt/rocm/lib/llvm/lib/clang/22/include/__clang_hip_math.h` 定義 `float rsqrtf(float)` 與 `double rsqrt(double)`，沒有 global float `rsqrt` overload。因此 global `::rsqrt(float)` 會發生隱含 double promotion。`c10::cuda::compat::rsqrt(float)` 是另一個 namespace 的函式，不能當成 global overload。

PyTorch 2.9.1 的 [`UnaryOpsKernel.cu`](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/native/cuda/UnaryOpsKernel.cu) 中，FP32 `rsqrt_wrapper` 呼叫的是 global `::rsqrt(v)`。因此 chunk Q/K norm 若要維持既有 PyTorch 執行語意，不能逕以 `rsqrtf` 替代；可以明確使用 FP64 rsqrt 後轉回 FP32。

這項修正是**保留既有 PyTorch overload 與 rounding 語意**，沒有放寬 packed bytes、FP16 scales、logits 或 tokens 的驗證門檻，也沒有變更 checkpoint/權重。實際 kernel 是否已通過嚴格 parity，仍以 GPU 測試記錄為準。

ATen mean 的本機 source review 另確認：width=128 的 FP32 mean 採 input vector4；同 lane 依序合併四值後，以 offsets 1、2、4、8、16 的 shuffle-down 合併。這與修正後 chunk kernel 的 reduction 次序相符。相關本機 header 是 `ATen/native/cuda/Reduce.cuh` 與 `ATen/native/SharedReduceOps.h`。
