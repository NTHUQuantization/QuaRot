# QuaRot INT4 Coverage 架構稽核

日期：2026-07-21

## 結論摘要

目前 `fused_current` 不是完整的 W4A4KV4 QuaRot，而是：

> FP16 HuggingFace weights/projections + INT4 KV cache + K3/FFN activation 的短暫 INT4 round-trip。

真正跨 token 保持 INT4 的只有 K/V paged cache。K3 與 FFN 雖輸出 packed INT4，下一行就 dequant 回 FP16，再送進原始 FP16 `o_proj` 或 `down_proj`。所有 Q/K/V/O、gate/up/down、LM-head weights 與 projection GEMM 仍為 FP16。QuaRot 論文的完整目標則是 weights、matrix-multiplication activations 與 KV cache 皆可 4-bit，讓矩陣乘法使用 4-bit operands。

## 現況資料流

```text
token id
  -> FP16 embedding
  -> FP16 residual / RMSNorm
  -> FP16 Q/K/V weights x FP16 activation
  -> Q FP16
  -> K/V FP16 -> K1 Hadamard + quant -> INT4 KV cache
  -> K2: Q FP16 x KV INT4, kernel 內 dequant/FP accumulation -> attention out FP16
  -> K3 Hadamard + quant -> INT4 packed
  -> 立即 dequant -> FP16 o_in
  -> FP16 O weight x FP16 o_in
  -> FP16 residual / RMSNorm
  -> FP16 gate/up weights x FP16 activation
  -> gate/up FP16 -> SiLU * up -> Hadamard + quant -> INT4 packed
  -> 立即 dequant -> FP16 ffn_in
  -> FP16 down weight x FP16 ffn_in
  -> FP16 residual
  -> FP16 final norm / LM head -> FP16 logits
```

## 逐區塊稽核

| 區塊 | 目前 storage/input | 目前運算 | 完整 QuaRot / 可行目標 | 判定 |
| --- | --- | --- | --- | --- |
| Embedding weight | FP16 | FP16 lookup | 可 INT4 儲存、取出單列後 dequant | 可量化，decode bandwidth 收益小 |
| Residual hidden state | FP16 | FP16 add | FP16/BF16 | 應保留 FP16 |
| RMSNorm weights/output | FP16 | FP16/FP32 reduction | FP16/BF16，輸出在進 GEMM 前動態 INT4 quant | Norm 本身不應硬做 INT4 |
| Q/K/V weights | FP16 | rocBLAS FP16 GEMM | rotated INT4 weights + INT4 activation GEMM | **目前缺少，應優先補** |
| Q activation | FP16 | K2 接收 `__half` | 完整 W4A4 attention 可量化 Q；務實版本可先保留 FP16 Q | 可進一步 INT4，目前仍 FP16 |
| K/V projection output | FP16 temporary | K1 讀 FP16 | 可由 W4A4 projection accumulator 經 RoPE/Hadamard 後直接 pack | FP16 temporary 合理，但前後 GEMM 尚未 INT4 |
| KV cache payload | packed INT4 | K2 直接讀 INT4 | INT4 | **已完成** |
| KV scale metadata | FP16 `half2` | kernel 內 scale/zero | FP16/FP32 metadata | 應保留 FP16，不算漏量化 |
| K2 attention | Q FP16、KV INT4 | KV 在 kernel 內 dequant，attention FP accumulation | 可發展 Q4 × KV4；softmax/accumulator仍高精度 | 目前是 mixed FP16×INT4，不是純 INT4 attention |
| K3 attention output | 先 FP16，後 packed INT4 | 隨即 dequant FP16 | packed INT4 直接餵 rotated INT4 `o_proj` | **INT4 邊界目前被浪費** |
| O projection weight | FP16、未折入對應 rotation | rocBLAS FP16 GEMM | rotated INT4 weight，直接消費 K3 packed INT4 | **最高優先缺口之一** |
| gate/up projection weights | FP16 | rocBLAS FP16 GEMM | rotated INT4 weights + INT4 normed activation | **目前缺少** |
| gate/up outputs | FP16 | SiLU 與乘法 | nonlinear/乘法可在 FP16 accumulator 執行 | 暫存 FP16 合理 |
| FFN fused output | packed INT4 | 隨即 dequant FP16 | packed INT4 直接餵 rotated INT4 `down_proj` | **INT4 邊界目前被浪費** |
| down projection weight | FP16、未折入對應 rotation | rocBLAS FP16 GEMM | rotated INT4 weight，直接消費 FFN packed INT4 | **最高優先缺口之一** |
| LM-head weight | FP16 | rocBLAS FP16 GEMM | INT4 weight + quantized hidden input；必要時保留高精度輸出 | 可量化，需品質驗證 |
| Logits / softmax / sampling | FP16/FP32 | 高精度 reduction | FP16/FP32 | 應保留高精度 |

## 程式碼證據

1. 模型由 HuggingFace 直接以 `dtype=float16` 載入，沒有 quantization config 或 weight conversion：`llama31_quarot/common.py:80-88`。
2. Q/K/V 直接呼叫原始 HF linear modules：`llama31_quarot/hf_quarot_model.py:113-118`。
3. K3 產生 `packed, scales` 後，在 `o_proj` 前立即呼叫 `dequant_grouped`：`llama31_quarot/hf_quarot_model.py:395-414`。
4. FFN 產生 `ffn_packed, ffn_scales` 後，在 `down_proj` 前立即 dequant：`llama31_quarot/hf_quarot_model.py:422-450`。
5. K2 的 cache dtype 是 signed INT4，但 Q 與 output 明確是 `__half`：`flashinfer.hip:60-71`。
6. K1 明確要求 FP16 K/V input，cache payload 才是 `uint8` packed INT4：`attention_fusion/attention_fusion.hip:438-463`。
7. FFN fused kernel 明確要求 gate/up 為 FP16，輸出才是 packed `uint8`：`ffn_fusion/ffn_fusion.hip:230-255`。

## Projection Weight 流量缺口

Llama-3.1 8B 的 32 層 projections 加 LM head，在目前實作中約有 75.05 億個 FP16 elements：

| Weight group | FP16 storage/最低每-token stream | INT4 payload | 理論減少 |
| --- | ---: | ---: | ---: |
| Q projection，32 層 | 1.074 GB | 0.268 GB | 0.805 GB |
| K projection，32 層 | 0.268 GB | 0.067 GB | 0.201 GB |
| V projection，32 層 | 0.268 GB | 0.067 GB | 0.201 GB |
| O projection，32 層 | 1.074 GB | 0.268 GB | 0.805 GB |
| gate projection，32 層 | 3.758 GB | 0.940 GB | 2.819 GB |
| up projection，32 層 | 3.758 GB | 0.940 GB | 2.819 GB |
| down projection，32 層 | 3.758 GB | 0.940 GB | 2.819 GB |
| LM head | 1.051 GB | 0.263 GB | 0.788 GB |
| **合計** | **15.009 GB** | **3.752 GB** | **11.257 GB** |

INT4 還需要 scale/zero-point metadata，實際流量會略高於 3.752 GB；但主要結論不變：現在 roofline 中最大的 FP16 weight stream 尚未被 QuaRot 化。

## 數學接線缺口

目前不能只把 FP16 `o_proj/down_proj` 換成任意 INT4 weight kernel，還必須先完成正確的 model transformation：

1. 把全域 residual rotation 離線折入相鄰 weights。
2. 把 K3 的 Hadamard basis 對應折入 `o_proj` weight，使 `INT4(Hx)` 被正確的 rotated weight 消費。
3. 把 FFN Hadamard basis 對應折入 `down_proj` weight，使 `INT4(H·SiLU(gate)·up)` 被正確消費。
4. K cache 若做 head-wise Hadamard，Q 必須使用相容 rotation，確保 QK inner product 的語意不變。
5. V rotation與 attention output/K3 rotation必須互相抵消或被 O weight吸收。
6. 對 rotated weights 做正式 INT4 quantization/calibration，再建立能讀 packed activation + packed weight 的 HIP GEMM。

目前 K3/FFN 將 Hadamard 後 activation dequant，再餵未轉換的原始 HF weight；這不是完整 QuaRot 的 rotation folding。這也解釋了既有報告中 FP16 與 current QuaRot logits 誤差仍偏大的限制。

## 建議優先順序

### P0：先完成正確的 rotated model conversion

- 建立 Llama-3.1 8B weight rotation/quantization exporter。
- 固定 K1/K2/K3 與 FFN 使用的 rotation convention、group layout、scale layout。
- 先用 PyTorch dequant reference 驗證 transformed FP16 model 與原模型近似等價，再加入 W4A4 誤差。

### P1：讓 FFN INT4 真正延續到 down projection

- 新增 `INT4 activation × INT4 rotated down weight -> FP16 accumulator/output` kernel。
- 直接消費既有 FFN `packed + scale`，移除 `ffn_dequant` tensor與 FP16 down-proj weight stream。
- 這是最大單一 weight group之一，32 層 FP16 down weights 約 3.758 GB/token。

### P2：讓 K3 INT4 真正延續到 O projection

- 新增 `INT4 K3 activation × INT4 rotated O weight -> FP16 output`。
- 移除 `k3_dequant`，保留 residual add 為 FP16。

### P3：量化 normed activation 與 QKV/gate/up weights

- RMSNorm output 保持 FP16 計算，但在 GEMM 邊界動態量化。
- Q/K/V、gate/up 使用 W4A4 kernel；SiLU與 elementwise product仍使用 FP16 accumulator。

### P4：LM head、embedding與 Q4 attention

- LM head weight-only INT4 或 W4A4 可再減少約 1.051 GB FP16 stream，但須單獨驗證 logits/top-k。
- Embedding可用 INT4 storage + row dequant，主要改善模型容量而非 decode latency。
- Q4 × KV4 attention需另做 kernel與品質分析；目前 FP16 Q + INT4 KV 已經保住最重要的 cache容量收益，可晚於 projection W4A4。

## 哪些 FP16 應保留

以下不是缺陷，不建議為了「全 INT4」而強行量化：

- RMSNorm 的 reduction與 normalization。
- RoPE 的 sin/cos與旋轉計算。
- SiLU、gate乘 up等 nonlinear/elementwise 計算。
- Attention score accumulator、online softmax與 value reduction accumulator。
- Linear/GEMM accumulator與輸出，通常至少 FP16，部分 reduction用 FP32。
- Residual stream與 residual add。
- Quantization scales/zero points。
- Final logits與 sampling reduction。

「W4A4」應理解為 GEMM 的 weight/input operands 為 4-bit，而不是 decoder 中每個 intermediate、accumulator與 nonlinear operation 都必須以 4-bit 儲存或計算。

## 來源

- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456)
- [NeurIPS 2024 QuaRot paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/b5b939436789f76f08b9d0da5e81af7c-Paper-Conference.pdf)
