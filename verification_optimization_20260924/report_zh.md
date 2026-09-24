# fused_v1 PARD-2 verification 非 GEMM 最佳化實測報告

日期：2026-09-24。GPU：AMD Radeon AI PRO R9700（gfx1201）；PyTorch 2.9.1+git5bc97ba／HIP 7.2。模型為既有 Qwen3-8B legacy W4A4KV4 checkpoint：36 layers、hidden 4096、32 Q heads／8 KV heads、H128。

此處的 36 layers 指 **36 個 Transformer decoder blocks，每個 block 有一個 FFN，因此本模型有 36 個 FFN**；32 是每層的 Q attention heads 數，並非 FFN 層數。依據本次 [Qwen3-8B checkpoint config](../../qwen3_8b_fused_v1_rtn_w4a4kv4/config.json) 的 `num_hidden_layers=36`、`num_attention_heads=32`。專案內另一個 [Llama-3.1-8B checkpoint](../../QuaRot/artifacts/llama31-8b-quarot-rtnclip090-w4a4kv4/config.json) 則為 32 個 decoder blocks；本報告的效能數據不是該模型的結果。上述 checkpoint 連結為本機路徑，不包含在 Git；[主機與容器路徑表](../e2e/FUSED_ONLY_AND_PARD2_ZH.md) 列出本次配置。

全部啟用時，context=128、M=16 的 target verification median 為 **19.041 ms**，最終 baseline 為 **82.617 ms**，實測 **4.34×**、延遲降低 **77.0%**。GPU kernel 數 **6,564 → 554**；CPU launch API 次數 **6,564 → 2**。

## 主要數據：各項獨立與累積 A/B

Target 欄為 B=1、固定 128-token prefix、M=16 的合成 tokens，使用 exact-row RMSNorm，未載入 drafter／TD collector；未開 profiler 的六次 HIP event median。TI／TD 為實際 PARD-2 生成的 steady tokens/s：同一 HumanEval prompt、32 generated tokens、一次 warmup、兩次 sweep、eager drafter、ignore-eos；不是 target-only token 換算值。

|設定|GPU kernels|CPU launch APIs|Verify ms|TI tokens/s|TD tokens/s|
|---|---:|---:|---:|---:|---:|
|初始 baseline|6564|6564|80.909|22.358|30.359|
|最終 baseline|6564|6564|82.617|22.849|30.418|
|① H128|1956|1956|34.855|38.897|52.217|
|② norm/quant|6132|6132|78.793|23.393|32.433|
|③ metadata|6566|6566|78.842|24.277|32.326|
|③ metadata＋graph|6566|2|57.604|29.261|38.809|
|④ Q/KV chunk|984|984|28.194|43.217|57.302|
|①＋②|1524|1524|30.987|42.273|56.377|
|①＋②＋metadata|1526|1526|24.600|46.973|62.388|
|①＋②＋③|1526|2|22.689|48.227|62.936|
|①＋②＋③＋④|554|2|19.041|51.976|68.391|

③ graph 的獨立效果應比較 metadata＋graph 與 metadata；④ 的累積效果應比較最後兩列。Chunk 已包含 Q Hadamard，因此各項加速比與 launch 減量不能直接相加。

本次採序列 A/B，未做隨機交錯或信賴區間估計；只有數個百分點的 差異可能包含時序與 clock 波動，不能據此宣稱穩定收益。

以下是 M16 eager trace 的部分 kernel 類別，直接核對非 GEMM 減量；各欄只列選定類別，並非總數。

|Kernel 類別|Baseline|① H128|② norm/quant|④ chunk|
|---|---:|---:|---:|---:|
|INT4 GEMM|144|144|144|144|
|舊 Q Hadacore|2304|0|2304|0|
|FP16 fill|2304|0|2304|0|
|Batched H128|0|36|0|0|
|Fused row norm/quant|0|0|72|0|
|Chunk Q|0|0|0|36|
|Chunk KV|0|0|0|36|

原始 M16 eager trace 另有 2,412 次 memcpy：2,304 次 D2D 與 108 次 H2D。H128 單項移除前者，共用 metadata 移除後者；H128＋norm＋metadata 的 eager trace 為 0 次 memcpy。這些是 trace 事件數，不與 kernel launch 數混算，也不把 graph 的漏記事件當成零。

## Fusion 之後，graph 還有多少額外收益？

以下兩組配對只切換 verification graph；皆為 B=1、context=128、M=16，沿用各 stage 的六次 HIP event median。未加 graph 的版本也已啟用共用 metadata，避免把 metadata 的效果混算成 graph 收益。

|已啟用的最佳化|未加 graph（ms）|加 graph（ms）|減少時間（ms）|延遲下降|
|---|---:|---:|---:|---:|
|Metadata|78.842|57.604|21.238|26.9%|
|H128＋norm fusion＋metadata|24.600|22.689|1.911|7.8%|

**Fusion 越完整，graph 的額外收益通常就越小**：fusion 已減少 kernel 數量與中間讀寫，也減少 CPU 逐次派發工作；graph 再消除剩下的派發成本，可節省的空間因而可能縮小。本次兩組配對的絕對與相對收益都呈現此趨勢。這是本次數據支持的解釋，不是必然單調的定律；fusion 也會改變 GPU 執行時間，實際收益仍取決於剩餘 CPU 開銷、kernel 時間與工作量。

第一組 GPU kernels 在 graph 開／關時都是 6,566 個，第二組都是 1,526 個；CPU launch API 呼叫則分別由 6,566／1,526 降至 2。Graph 不會把這些 kernels 合成一個 kernel。最終 554-kernel 的全融合配置尚無 graph 開／關獨立配對，因此不能用其 19.041 ms 推算該配置的 graph 單項收益。上述比較也保留序列 A/B、小樣本、無信賴區間的限制。

## 實際 PARD-2 verify 與完整生成

此表比較 PARD-2 最佳化前後，不是有／無 PARD-2。另行規劃的 [目前 fused AR 對 PARD-2 phase 比較](../pard2_speedup_20260924/README.md) 因 GPU 忙碌等待逾時，尚無量測結果。

|模式|Baseline verify ms/step|全開 verify ms/step|Baseline steady / E2E tokens/s|全開 steady / E2E tokens/s|Steady 倍率|
|---|---:|---:|---:|---:|---:|
|pard2-ti|81.254|18.995|22.849 / 20.442|51.976 / 42.726|2.27×|
|pard2-td|81.432|19.146|30.418 / 26.031|68.391 / 53.365|2.25×|

E2E 包含這次生成的 prefill 與 decode；此處 graph 已在 warmup 建立，不代表首次使用的冷啟動。acceptance、每輪 proposal／accept length、拒絕次數與輸出 token 對照見 [完整實測表](measurements_zh.md) 及 [summary.json](summary.json)。Steady TPS 排除整個首輪的時間與 emitted tokens；verify ms/step 是每次生成的 verify event 總和除以步數，再取 兩次生成的 median，包含 M15 與 M16，並非逐 step latency median。

## M=1／15／16 與初始化成本

|M|Baseline GPU kernels|全開 GPU kernels|Baseline ms|全開 ms|
|---:|---:|---:|---:|---:|
|1|2136|482|32.968|17.955|
|15|6276|554|80.401|18.975|
|16|6564|554|82.617|19.041|

M=15 是首輪獨立 graph，M=16 是後續 verification graph；M=1 沿用 eager。模型邊界測試使用 contexts=127／128／129，實際跨越 page。

|模式|M15 graph 建立 ms|M16 graph 建立 ms|
|---|---:|---:|
|pard2-ti|84.213|53.694|
|pard2-td|75.705|53.701|

建立時間包含兩次 target warmup、capture、instantiate 與 HIP node 列舉。量測 sweep 沒有新增 capture；cache／graph 跨 generate 呼叫重用。

## Kernel 設計與其他修改

1. **Batched H128**：每 workgroup 保留最多八個有效 rows，其他 WMMA rows 以零 fragment 處理；M=16 的 512 Q rows 用 64 workgroups、一次 launch。直接寫完整輸出，移除 padding tensor、fill 與 cat；保留原 H16／H8 的 FP16 中間 rounding，沒有直接刪除八 rows 限制。
2. **RMSNorm＋scale＋INT4 packing**：每 row 一個 256-thread workgroup，FP16 normalized row 留在 LDS；維持原 reduction tree、FP16 /7、checkpoint clipping、tiny clamp、FP16 divide、nearest-even 與 signed nibbles。clipping 從 attention／MLP 的實際 quantizer 取得。
3. **固定 KV metadata＋graph**：預配置 M=1／15／16 的 append 與 causal metadata，跨層共用；GPU 從 base length 更新 positions、page indices 及每個 query 的有效長度。拒絕後只回退 logical length，下一輪重寫 metadata／provisional slots。固定 IDs、positions、KV 與 graph pool 位址，分開捕捉 M15／M16。TD 保留每張 graph 的 hidden tap references，replay 後恢復 collector，feature 還原與 acceptance 流程不變。
4. **兩個 chunk kernels**：Q 為 norm→RoPE→安全 H128；KV 為 K norm→RoPE→FP32 Hadamard→原非對稱 KV4 量化→page append，並量化／寫入 V。支援 projection 的 strided QKV views，省去中間 materialization 與 contiguous copies。保留 PyTorch 在此環境的 rsqrt FP64 overload、FP32 rounding 邊界及分開的 FP16 RoPE products／add，禁止 half FMA 省略中間 rounding。

其他修改包含 GEMM／quant／FlashInfer 改用 PyTorch current HIP stream，使 graph 完整捕捉；graph 路徑移除重複 target arange；close 時清除 TD tensor references；超大輸入加入 INT32 索引容量檢查，grouped checkpoint 的 M1 保留既有 K1 dispatch。INT4 GEMM tile 設計、權重、checkpoint 及 greedy verification 規則均未改動。詳細設計：[H128／chunk](preprocess_design_zh.md)、[norm／residual](norm_design_zh.md)、[metadata／graph](metadata_graph_design_zh.md)。

**Residual 評估**：primitive 使用 B=1、hidden=4096、clipping=0.9 （模型 checkpoint 為 1.0）。M16 的 add＋已融合 norm 為 2 launches／0.013390 ms，全融合為 1 launch／0.008678 ms；packed、scale 與保留的 FP16 residual 均 bitwise 相等。此 prototype 尚未接入 decoder；局部的 post-attention add＋norm 可以保留 residual 輸出，跨層融合則另需維持 TD hidden-tap 邊界。本次不把 primitive 結果計為 verify／TPS 收益。

## 驗證、量測範圍與重現

最終整合測試 **229 passed**，見 [final_tests.log](final_tests.log)。涵蓋 H128 row 隔離、M=1／15／16、FP16 rounding／clipping／極值、B=2 strided kernel 與非順序 page mapping、跨 page、causal、全拒絕／部分接受後覆寫、graph replay 與既有 PARD benchmark contract。完整模型與 PARD-2 效能／正確性範圍為 **B=1、此 8B checkpoint**；B=2 kernel 測試不等同其他 batch 的完整 runtime qualification。

各 target stage 的九個案例均對照原 baseline packed KV、scale 與 argmax，另對照逐 token 執行與 future-token causal；所有實際 TI／TD 輸出均與 AR oracle 完全一致。本次 90 個同 M 的 optimization／baseline logits 比較也全部 bitwise 相等（max absolute error=0）；此觀察限於受測案例。沒有降低既有測試門檻。這是單 prompt smoke 性能診斷，未重跑正式三資料集品質 qualification，也不外推至其他模型、context 或硬體。

各完整模型 A/B subprocess 前保存連續三次 GPU 0%／無 KFD process 的 idle preflight；所有 GPU 工作序列執行，沒有終止其他使用者的工作。eager launch 數來自 GPU trace。ROCtracer 7.2 會漏記 graph child events，因此 graph 使用 HIP kernel-node 列舉＋graph 外 kernel API 計數，CPU launch APIs 包括 hipGraphLaunch、但不包含 memcpy API；不使用不完整 trace 的 duration sum 宣稱 kernel 成本。

原 [tile audit](../m16_tile_audit_20260924/report_zh.md) 使用 plain target，M16 為 7076 launches／90.368 ms；本 A/B 使用 PARD 所需的 exact-row RMSNorm，因此不能把兩組 baseline 混算加速比。初始與最終 baseline 皆保留，主要倍率採最終 baseline。所有 stage 保存 command、flags、source／extension hashes、原始 samples、trace、logits／KV 比對資料。最終全開與 baseline 在容量／M1 fallback 保護加入並重新 build／test 後 重測；較早的獨立／累加矩陣使用相同 8B 算術路徑，差異來源檔保存於 [來源快照](diagnostics/pre_hardening_sources/README.md)，重測前的兩組結果 保存在 `archive_pre_hardening/`。

重現完整 A/B 與 build／test 指令見 [README](README.md)。以下啟動受測配置：

```bash
cd /workspace_root/fused_v1
bash verification_optimization_20260924/run_optimized_pard2.sh --mode pard2-ti
```

可用 `QUAROT_BATCHED_H128=0`、`QUAROT_STATIC_KV_METADATA=0`、`QUAROT_VERIFICATION_GRAPH=0`、`QUAROT_CHUNK_PREPROCESS=0`、`--no-fused-norm-quant` 各自切回 fallback。H128／chunk dispatch 對未支援的 dtype／head dim／chunk shape／mask 保留原路徑；fused norm 是 CUDA/HIP FP16 的 opt-in，其他 dtype 應關閉該選項。prefill 仍使用既有 attention 路徑。

## 方法的設計動機與既有技術來源

**本專題的判斷過程**：先由 [tile audit](../m16_tile_audit_20260924/report_zh.md) 確認 M1／M16 的 INT4 GEMM tile 數相同，再從 trace 找到大量短 kernels、Hadamard padding／copy，以及各層重複的 metadata H2D。因各層的 KV 內容 不同、但 token 位置與有效長度一致，將位置表改成跨層共用、GPU 每輪更新。接著利用 verification 反覆出現的固定 M15／M16 形狀，將固定 buffer 與 graph replay 配合。這是依本專題瓶頸套用既有方法，不是新發明的 graph 演算法。

**既有技術脈絡與引用**：PyTorch 的 *Accelerating PyTorch with CUDA Graphs* （Nguyen 等，2021）說明以一次 graph launch 提交多個 GPU 操作、降低 CPU 派發成本，並要求重播時維持 tensor 位址、在相同輸入 buffer 填入新資料。本次的固定 IDs／metadata 與 replay 採相同原理；下列來源是在補充報告時 查核的技術依據，不代表本次實作直接移植該文程式。[PyTorch 官方文章](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)。

AMD 的 *HIP graphs* 官方文件說明以 nodes 表示操作、edges 表示依賴，透過 stream capture、instantiate、launch 重複執行固定工作。這是本專題 AMD GPU 上對應的執行機制；雖使用 `torch.cuda.CUDAGraph` API 名稱，實際量測與 node 列舉使用 HIP runtime。[AMD HIP graphs 文件](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html)。

PyTorch 的 *Accelerating Generative AI with PyTorch II: GPT, Fast* （2023）也討論以預配置 static KV cache 處理逐 token 增長造成的動態性，以利降低 CPU overhead，並將 prefill 與 decode 分開處理。本專題原本已有 paged KV storage；此次新增的是共用固定 metadata、GPU 有效長度更新，以及適用 PARD-2 的 M15／M16 verification graphs，並非重新發明 static KV cache。[PyTorch GPT, Fast 官方文章](https://pytorch.org/blog/accelerating-generative-ai-2/)。

**本次具體捕捉範圍**：一次 target verification forward 的 GPU 工作，包含 metadata 更新、embedding／RoPE、36 個 decoder blocks、final norm 與 LM head。模型載入、prefill、drafter、接受／拒絕、logical-length 回退與 TD feature 重建留在 graph 外。先 warmup 兩次完成 lazy 初始化，再捕捉 GPU 操作與依賴；每輪只更新固定 IDs 與 base-length buffer，再重播。每個 kernel 的每次出現 都保留為操作，並非將 36 層縮成一次計算或重用先前 logits。實作見 [verification_graph.py](../e2e/verification_graph.py) 與 [kv_cache.py](../quarot/transformers/kv_cache.py)。
