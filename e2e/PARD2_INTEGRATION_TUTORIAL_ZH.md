# 從 AR 到 Exact Parity：在 fused_v1 整合 PARD2-TI/TD 的實作教程

> 適用讀者：準備在 `fused_v1` 上研究其他推論最佳化、但尚不熟悉 speculative decoding、量化 KV cache 或 PARD2 的組員。
>
> 本文範圍到「PARD2 與 fused target 完成整合，並通過 exact greedy parity」為止。Parity 之後的效能最佳化、採用決策與負向實驗，請接續閱讀 `PARD2_OPTIMIZATION_TUTORIAL_ZH.md`。

## 1. 我們究竟要整合什麼？

一般 autoregressive decoding（AR）每次只讓 target model 產生一個 token。模型很大時，每個 token 都要完整走過所有 transformer layers，因此 GPU 很容易被大量小矩陣、kernel launch 與記憶體存取拖慢。

Speculative decoding 的想法是：先用較便宜的 drafter 一次猜多個 token，再讓 target 用一個小 chunk 一次驗證。若一次接受多個候選，就能用一次 target forward 換得多個正確 token。

本專案固定 `draft_k=15`。每輪 verifier 最多處理：

```text
pending target token + 15 draft candidates = 16 tokens
```

PARD2 有兩種 drafter 輸入方式：

- TI（token-independent）：drafter 主要根據 token 與自己的 cache 提案，不需要 target hidden features。
- TD（token-dependent）：除了 token，還利用 target 指定 layers 的 hidden features，通常 acceptance 較高，但多了 feature 擷取、basis restore 與 projection 成本。

整合的真正難點並不是「把兩個模型接起來」，而是讓以下四件事同時成立：

1. target 仍走 fused_v1 的 W4A4KV4 HIP 執行路徑；
2. 一次驗證 2–16 rows 時，結果要等同逐 token AR；
3. 接受或拒絕候選後，量化 KV cache 必須回到正確的邏輯狀態；
4. AR、TI、TD 必須使用同一份 target 與相同數值契約，才可公平比較。

## 2. 先固定研究邊界：模型與執行契約

### 2.1 為何從 Llama 3.1 改成 Qwen3-8B

最初計畫以 Llama 3.1 8B 為目標，但實際檢查後發現 fused_v1 對該 checkpoint 的完整驗證程度不足。繼續硬接 PARD2 會同時面對「模型 runtime 未成熟」與「speculative decoding 新邏輯」兩組變因，出錯時難以定位。

因此首版改用 dense Qwen3-8B：

- fused_v1 已有 Qwen3 dense runtime；
- 有官方對應的 `amd/PARD2-Qwen3-8B`；
- shape 固定為 36 layers、hidden size 4096、8 KV heads／32 query heads；
- 可以明確排除 MoE routing 等額外變因。

這個決策反映一個重要研究原則：**第一版先選擇能建立可靠 oracle 的模型，而不是選擇最想展示的模型。** 否則 correctness 問題會和模型支援問題糾纏在一起。

### 2.2 固定資源

- Target source：`Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`
- Fused target：RTN W4A4KV4 checkpoint
- PARD2 drafter：`amd/PARD2-Qwen3-8B@67a1516c8f6fc145cda99916799a0cbb3a4af135`
- AMD PARD reference：commit `6f279bf3f1680e0b5d71c562ca5b91bdeef4c038`
- PARD token：`151670`
- TD taps：`[-1, -8, -16, -24]`
- TD concatenated dimension：`4 × 4096 = 16384`
- Projection scale：`0.02`

Runtime 會在 `Pard2Spec.validate()` 檢查 model type、layer count、hidden size、vocabulary、PARD token 與 TD metadata。這些檢查不是多餘的防呆；錯誤的 vocabulary 或 tap contract 仍可能讓程式正常執行，卻產生完全沒有意義的 acceptance 結果。

### 2.3 首版不處理的項目

- sampling distribution equivalence；
- batch serving；
- MoE；
- drafter retraining；
- Qwen 以外的 PARD2 組合；
- adaptive-k、graph capture 等第二層最佳化。

把範圍限制在 batch 1、greedy decoding 的原因，是 greedy acceptance 可以用 exact token equality 定義；這讓 cache、logits 與輸出序列都能建立強 oracle。

## 3. 整體架構：一輪 speculative step 如何流動

```text
prompt
  │
  ├─ target prefill ──> pending target prediction
  │                     + quantized target KV cache
  │
  ├─ drafter proposal ─> 15 candidate tokens
  │       TI: token/cache
  │       TD: token/cache + selected target features
  │
  └─ target verifier chunk
          pending token（第二輪起）+ candidates
          │
          ├─ target predictions
          ├─ longest matching prefix
          ├─ correction/bonus token
          └─ cache transaction commit/rollback
```

在 `e2e/speculative.py` 中，`FusedPardRuntime` 統一處理 AR、TI、TD；`GenerationResult` 統一回報 TTFT、steady/E2E latency、target/draft forward 次數、accept lengths、conditional acceptance、VRAM 與輸出 token。

這種統一介面很重要：若 AR 與 PARD2 各自使用不同 runner，任何 tokenizer、EOS、cache 或 timing 差異都可能被誤認成 speculative speedup。

## 4. 實作步驟一：先建立共同 AR target

第一步不是立刻寫 speculative loop，而是確認 fused target 的單 token AR：

1. 載入同一份 W4A4KV4 checkpoint；
2. 建立相同 paged KV cache；
3. 固定 tokenizer/chat template 與 greedy top-1；
4. 記錄 output IDs，而不只保存解碼後文字；
5. 把這條路徑當作 verifier 的 correctness oracle。

文字看起來相同，不代表 token sequence 一定相同；反過來，token 差一個後後續整段都可能改變。因此正式 parity 一律比較 token IDs。

AR baseline 也必須使用新 runtime 所需的共同數值契約。不能拿普通 PyTorch AR 與經過特殊 small-chunk kernel 的 PARD2 比較，否則觀察到的 token 差異可能來自 target arithmetic，而不是 speculative algorithm。

## 5. 實作步驟二：讓 KV cache 支援「暫存、接受、撤銷」

### 5.1 為何普通 cache append 不夠

AR 每次 append 的 token 都已確定，不需要撤回。Speculative verifier 則先把整個候選 chunk 寫入 cache，之後才知道接受幾個。

假設本輪驗證 15 candidates，只接受前 5 個。Cache 必須保留：

```text
已確認的舊 prefix + accepted 5 tokens + target correction token
```

其餘 rejected slots 雖然物理資料可以留著，但邏輯長度不可包含它們；下一輪會覆寫這些 stale slots。

### 5.2 Transaction contract

`quarot/transformers/kv_cache.py` 新增 `CacheTransaction`：

- `begin()`：記住 base logical length；
- verifier append：暫時增加 proposed length；
- `commit(keep_tokens)`：只推進接受部分；
- `rollback()`：恢復原 logical length；
- capacity 與 nested transaction 都要檢查。

這裡刻意分離「物理 storage」與「邏輯 sequence length」。回滾不必清零整段 KV；只要後續 attention metadata 看不到 rejected positions，並保證未來會覆寫即可。這能省下不必要的 device write。

### 5.3 Page boundary 是必要測項

Paged cache 最常見的隱藏 bug出現在 `127 → 128 → 129` 這類邊界。測試不能只用短 context，至少要涵蓋：

- context `1/127/128/129/1024/4096`；
- chunk `1/15/16`；
- 0、部分、全部接受；
- rollback 後再 append；
- cache capacity 上限。

## 6. 實作步驟三：保留 native GQA，而不是永久展開 MHA

Qwen3-8B 有 32 query heads，但只有 8 KV heads。最簡單的 correctness 實作是把 KV heads 展開成 32 heads，讓既有 MHA decode kernel 使用；缺點是 cache、頻寬與 VRAM 都被放大。

主線做法是：

- cache 永久保存原生 8 KV heads；
- 加入 native GQA INT4/FP16 paged decode；
- 加入支援 prefill、decode、small chunk 的 fused KV append；
- expanded-MHA 只作 correctness oracle。

KV4 使用 asymmetric scale/zero contract。K 會在既定 Hadamard/rotation basis 下寫入，V 保持原 basis；append 與 decode kernel 必須對這個 convention 完全一致。量化資料、scale、zero 任一欄位錯位，都可能在短測試看似正常，長 context 才逐漸漂移。

研究上最有效的策略是保留兩條路：

```text
native GQA（候選主線）  vs.  expanded MHA（慢但容易理解的 oracle）
```

若兩者 logits/KV 不同，先修 kernel；若兩者都以相同方式偏離 AR，問題通常在更上層的 verifier arithmetic 或 cache transaction。

## 7. 實作步驟四：把 verifier chunk 當成新的 execution regime

既有 fused runtime 通常只有兩種思維：

- q_len=1：decode；
- q_len 很大：prefill。

PARD2 的 q_len=2..16 兩者都不是。若直接落入大型 prefill 路徑，kernel geometry、causal mask、KV append 與量化 reduction 都可能不合適。

因此新增 small-chunk 路徑：

- q_len `2..16` 專用 dispatch；
- M=16 W4A4 GEMM shape；
- Q/K/V 與 gate/up execution-time grouped projections；
- fused SiLU–Hadamard–quantize–down FFN；
- virtual causal metadata。

Virtual causal metadata 的目的，是讓 chunk 中第 `i` 個 token 只能看見舊 prefix 與 chunk 內 `≤i` 的位置。不能因為整段 KV 已先 append，就讓較早 token 偷看未來 candidate。

這裡的研究教訓是：**speculative verifier 不是「比較短的 prefill」，而是具有 transaction 與虛擬因果關係的獨立 regime。**

## 8. 實作步驟五：接入 TI 與 TD drafter

### 8.1 TI

TI 不需要 `warp_model.bin`，只載入官方 BF16 drafter。Prompt 任意長度先 eager prefill，decode proposal 固定在 bounded shape，並使用 `torch.compile(max-autotune)`。

Compile recompile budget 必須覆蓋 PARD2 可能出現的 proposal lengths；否則看似偶發的 latency spike，可能只是 Dynamo 反覆重新編譯。

### 8.2 TD hidden features

TD 只收集四個指定 taps，不要求 target 回傳完整 hidden-state tuple。`SelectedHiddenCollector` 保存：

```text
[-1, -8, -16, -24]
```

Qwen3-8B 對應 final layer 與三個中間 layers。四個 4096-wide tensors 串成 16384 維，再交給官方 `target_proj`。

QuaRot target activations 位於 rotated basis。Parity baseline 先用清楚、可檢查的做法恢復：

1. 對每個 tap 做 inverse normalized Hadamard；
2. 乘 rotation signs；
3. final tap 再做原始 RMSNorm 與 gamma；
4. concat 後投影到 drafter hidden size。

Target features 要左移一格，第一個位置補零。原因是 position `t` 的 drafter 輸入應使用 target 在先前位置已產生的資訊，不能直接偷看同位置的 target output。

### 8.3 為何先保留直觀路徑

Inverse Hadamard、concat 與 hooks 可能有成本，但 parity 階段優先保留可解釋的 feature pipeline。等 correctness 穩定後，再個別評估 basis folding、lazy rows 或 fusion；這正是第二篇教程的主題。

## 9. 實作步驟六：接受規則要和官方 greedy PARD2 一致

設 drafter candidates 為 `d[0:k]`，target predictions 至少包含 `k+1` 個位置。接受長度是最長 matching prefix：

```python
accepted = first index i where d[i] != target[i]
accepted = k if all candidates match
```

然後輸出：

- accepted candidate prefix；
- target 在第一個 mismatch 的 correction token；
- 若全部接受，則輸出 target bonus token。

第一輪與後續輪的 token alignment 不完全相同：第一輪 verifier 沒有額外 pending input；後續輪需要把前次 correction/pending token 放在 candidates 前。這也是 features、logits 與 cache position 容易 off-by-one 的地方。

接受規則測試至少要顯式覆蓋：

- 0 accepted；
- partial accepted；
- all accepted；
- EOS 出現在 candidate、correction 或 bonus；
- max token 截斷；
- ignore-EOS 固定長度 microbenchmark。

## 10. 最關鍵的 parity 事故：為何 MATH prompt 14 在很後面才偏離

### 10.1 症狀

早期整合可以通過短 smoke，但 MATH prompt 14 約在 generated token 171 出現 TI/TD 同步偏離 AR。這類問題最危險，因為前面一百多個 token 都正確，很容易誤判為資料或模型隨機性。

Greedy decoding 沒有隨機性；只要 token 不同，就一定有數值或狀態差異。

### 10.2 如何逐層縮小範圍

我們依序做了：

1. native GQA 換成 expanded-MHA；偏離仍存在，排除 native GQA 特有錯誤；
2. 關閉 fused KV append；偏離仍存在，排除 unified append；
3. 用 fresh sequential cache 重算；parity 恢復，指向 chunk execution；
4. 比較每層 hidden output、KV data/scales 與 logits；
5. 定位到 full-accept chunk 後 layer 8/9 的 MLP/KV 開始出現微小差異。

這個順序值得保留給後續研究：先用大模組 oracle 二分，再進入逐層 tensor comparison。直接盯最終 token 幾乎無法找出根因。

### 10.3 先釐清 M=1 與 M=16 代表什麼

令 hidden tensor 的 shape 為 `[M, D]`：`M` 是這次一起處理的 token rows 數量，`D` 是每個 token 的 hidden columns 數量。一般 AR decode 每次只處理下一個 token，因此通常是 `M=1`；PARD2 verifier 要一次驗證 pending token 與最多 15 個 candidates，因此最多是 `M=16`。

RMSNorm 對每個 row `i` 各自計算：

```text
rms_i = sqrt((x_i0² + x_i1² + ... + x_i(D-1)²) / D + eps)
y_ij  = x_ij * gamma_j / rms_i
```

從數學定義看，不同 rows 完全獨立。`M=16` 的其他 15 個 rows **不會被加進**第 `i` row 的平方和。問題是 tensor shape 從 `M=1` 變成 `M=16` 後，GPU kernel 可能改變「如何讓 threads 和 warps 合作算完同一個 row」。因此我們需要區分：

- **數學上的 row independence**：每個 row 只使用自己的 values；
- **execution 上的 row invariance**：同一個 row 放在不同 `M` 中，仍使用相同 reduction tree，得到 bitwise 相同的結果。

原本的實作滿足前者，卻不保證後者。

### 10.4 為何 M 會改變同一個 row 的 reduction tree

GPU 不會讓單一 thread 從第一個 hidden column 一路加到最後一個。它通常讓多個 threads 各算一部分平方和，再透過 warp shuffle、shared memory 或多層 tree reduction 合併 partial sums。

`M=1` 時只有一個 row 可供平行化，kernel 可能讓較多 warps 合作處理它；`M=16` 時有 16 個 rows 可同時執行，kernel 可能改成每個 row 使用較少 warps，以取得較好的 occupancy。隨著 block size、每個 row 的 warp 數、vector load width 或 column traversal 改變，partial sums 的分組也會跟著改變。

用一個 row 只有 8 個平方值的簡化例子說明。令這 8 個 values 為：

```text
a, b, c, d, e, f, g, h
```

`M=1` kernel 若把唯一的 row 分給較多 threads，可能形成：

```text
thread 0: a + e
thread 1: b + f
thread 2: c + g
thread 3: d + h

final: ((a + e) + (b + f)) + ((c + g) + (d + h))
```

`M=16` kernel 為了同時處理更多 rows，可能採用另一種 column grouping：

```text
thread 0: a + b
thread 1: c + d
thread 2: e + f
thread 3: g + h

final: ((a + b) + (c + d)) + ((e + f) + (g + h))
```

這只是幫助理解的 reduction tree 範例，不代表 PyTorch kernel 必然逐字採用這兩種 mapping。重點是：兩次計算讀取的是同一個 row、同一組 8 個 values，沒有混入其他 rows；只有加總的括號與順序不同。

在實數算術中兩棵 tree 相等，但浮點加法每一步都可能 rounding，因此不具結合律：

```text
(a + b) + c  !=  a + (b + c)
```

也就是說，`M` 並未改變 RMSNorm 公式，卻可能藉由 kernel dispatch 與平行 reduction geometry，改變同一個 row 的最後幾個 bits。

### 10.5 為何微小差異會在 W4A4KV4 中變成 token divergence

若後續都是 BF16/FP16 dense operations，RMSNorm 最後幾個 bits 的差異經常不會改變 top-1。但 fused_v1 接著會把 activation 量化成 INT4。把量化簡化成：

```text
q = round((x - zero) / scale)
```

假設同一個 normalized value 恰好靠近量化邊界：

```text
M=1  result: 3.49998 * scale  -> quantized value 3
M=16 result: 3.50002 * scale  -> quantized value 4
```

RMSNorm 的連續值只差一點，INT4 表示卻差了一個完整 level。這個差異接著經過 W4A4 projection、attention 和 MLP，還可能被寫進 KV4 cache，讓未來 token 持續讀到不同狀態。於是差異不一定在第一個 verifier step 立刻改變 top-1；它可能累積許多 decode steps，直到兩個 logits 的排序交換。這正是 MATH prompt 14 前面約 170 個 generated tokens 都相同、之後才偏離的原因。

因此，不能用「目前 top-1 還相同」判斷 chunk verifier 已具備 exact parity。必須往前比較 layer outputs、quantized activation、KV packed data/scales 與 cache commit 後的下一個 token。

### 10.6 為何不直接逐 row 呼叫 M=1 RMSNorm

最直接的 correctness oracle，是把 `M=16` 拆成 16 次 `M=1`：每個 row 都走 AR 已知的 execution shape，確實能消除跨 `M` 差異。但它也把一次 kernel launch 變成最多 16 次，增加 launch、dispatch 與同步成本，抵銷 speculative decoding 一次驗證多個 tokens 的主要優勢。

把 rows 分成 M4 或 M8 也不是根本解法。只要 kernel geometry 仍可能隨 shape 改變，就沒有建立 `M=1/4/8/16` 共用的 bitwise contract。這些做法適合作為 diagnostic oracle，不適合作為預設 runtime。

### 10.7 修復：row-independent HIP RMSNorm

最終新增 `rms_norm_rows`：每一個 row 使用相同的 workgroup mapping，固定每個 thread 負責的 hidden columns，以及 partial sums 的 reduction tree。`M` 只決定啟動多少個相同的 row programs，不再改變單一 row 內部的算術順序。如此 M=16 的第 `i` row 與單獨 M=1 執行具有相同加總順序。

所有 target input layernorm、post-attention layernorm 與 final norm 都換成這個共同 contract。結果是：

- M=16 與 16 次 M1 bit-exact；
- chunk verifier logits/top-1 對 sequential AR；
- commit 後下一 token 對 fresh recomputation；
- AR、TI、TD 正式 prompts exact greedy parity。

需要注意：這個新 contract 與普通 PyTorch RMSNorm 數學等價，但未承諾每一 bit 都等於舊 PyTorch AR。正式比較的 reference 因此改成「同樣使用 row-independent target 的 AR」。這不是降低 parity 標準，而是先明確定義跨 shape 穩定的 target arithmetic。

## 11. 測試策略：從局部 bit parity 到完整 generation

### 11.1 單元與 contract

- pinned model/revision/vocabulary/PARD token；
- tap order、shape、feature shift；
- acceptance 0/partial/all；
- transaction commit/rollback/capacity；
- EOS 與 fixed-length ignore-EOS；
- CLI defaults 與 benchmark contract。

### 11.2 GPU kernel correctness

- native GQA vs expanded-MHA；
- q_len 1/15/16；
- context page boundaries；
- KV packed bytes、scale/zero；
- M=16 verifier vs sequential M1；
- 每個 acceptance length `0..15` commit 後的下一 token。

### 11.3 End-to-end parity

正式資料為：

- HumanEval 80；
- GSM8K 80；
- MATH-500 20；
- 每 prompt 256 generated tokens；
- batch 1、greedy、8 warmups、3 measured sweeps。

每個 mode/dataset 使用獨立程序，GPU 既有占用超過 1 GiB 就拒絕執行，不終止其他工作。

## 12. Parity 成功時的正式結果

下表是 parity qualification 當時的結果。它證明核心整合成立；後來採用的 batched LM-head 與 cached TD basis 屬於後續最佳化，不能回填到這張表假裝是同一輪 formal run。

| Dataset | TI paired speedup | TD paired speedup | TI mean accept | TD mean accept | Parity |
|---|---:|---:|---:|---:|---|
| HumanEval | 1.667× | 1.921× | 5.968 | 6.696 | exact |
| GSM8K | 2.361× | 2.802× | 5.450 | 6.373 | exact |
| MATH-500 | 1.586× | 1.743× | 5.536 | 6.517 | exact |

三資料集的 speedup bootstrap 95% CI 下界均大於 1.0，run-level CV 均低於 5%，且 VRAM 保留超過 10%。因此第一階段 hard gate 通過。

TD 相對官方 Qwen3 PARD2 文獻速度仍只有 HumanEval 28.5%、GSM8K 43.5%；這表示「演算法已正確且快於 AR」不等於「系統效率已接近論文」。兩個結論要分開陳述。

## 13. 從這個案例學到的研究邏輯

### 13.1 Correctness oracle 要保留到最後

Expanded-MHA、sequential M1、fresh recomputation 與 rowwise LM-head 都很慢，但它們能回答「錯在哪一層」。Oracle 不應因為主線變快就刪掉；應放在 flag 或測試中。

### 13.2 不要把微小數值差異當成可忽略

在純 FP16 model 中，小誤差可能不改 token；在 W4A4KV4 中，它可能跨過量化門檻並被 cache 長期保存。研究量化 speculative decoding 時，必須同時比較 logits、hidden、packed KV 與 scales。

### 13.3 先讓 target contract 穩定，再優化 drafter

若 target chunk 本身不等價，acceptance、drafter agreement 與 speedup 都沒有解釋價值。先完成 verifier/cache parity，再研究 feature calibration、drafter quantization 或 adaptive-k。

### 13.4 Smoke、diagnostic、formal 是三種不同證據

- Diagnostic：回答哪個 tensor、哪個 layer、哪個 kernel 有差異；
- Smoke：快速判斷方案是否值得繼續；
- Formal：固定資料、warmups、sweeps、CI/CV 後才可做採用結論。

不能用一個 prompt 的正結果取代 formal，也不能因 aggregate TPS 看起來高，就忽略 paired samples 或 compile 污染。

## 14. 程式地圖與基本重現

主要檔案：

- `e2e/speculative.py`：PARD2 spec、TI/TD runtime、acceptance、selected features；
- `e2e/pard2.py`：單次 generation CLI；
- `e2e/benchmark_pard2.py`：資料 hash、GPU preflight、正式 runner；
- `e2e/qualify_pard2.py`：parity、bootstrap CI、CV 與 gate；
- `quarot/transformers/kv_cache.py`：native GQA KV4 與 transaction；
- `quarot/kernels/fused_hip.hip`：row-independent norm 與 fused kernels；
- `tests/test_speculative.py`、`tests/test_pard2_gpu.py`、`tests/test_fused_hip.py`：各層 oracle。

單次執行範例：

```bash
python e2e/pard2.py \
  --mode pard2-td \
  --max-new-tokens 256 \
  --ignore-eos
```

Smoke benchmark 範例：

```bash
python e2e/benchmark_pard2.py \
  --mode pard2-td \
  --dataset math_500 \
  --offset 14 \
  --limit 3 \
  --generated-tokens 128 \
  --warmups 2 \
  --sweeps 3 \
  --ignore-eos \
  --output /tmp/pard2_td_smoke.json
```

使用 `--limit` 或 `--offset` 的輸出會標記為 `qualified: false`；這是刻意防止 smoke 被誤引用為正式結果。

## 15. 小結：完成 parity 後，系統才真正可研究

這次整合的核心成果不是一個 speculative loop，而是一套可被驗證的共同 execution contract：native GQA KV4、transactional cache、virtual causal verifier、TI/TD alignment，以及跨 M=1/M=16 穩定的 row-independent RMSNorm。

到這個節點，我們才有資格問「哪個部分值得優化」。下一篇會看到，多數直覺上省 FLOPs 的方案並沒有改善 E2E；真正被採用的，是能通過 parity、paired latency、CV 與維護成本共同審查的 batched LM-head 與 cached TD basis。

## 16. Post-merge qualification：把整合成果放回最新 fused_v1

前面的第 12 節記錄「PARD2 首次通過 parity」時的 execution contract。後來合併 `origin/fused_v1@cc4ffca`，target 加入 grouped-scale GEMM、fused projections、persistent metadata 等路徑；因此不能把舊 AR 或舊 acceptance 直接當成新 runtime 的 oracle。正確做法是讓 post-merge AR、TI、TD 共用同一個 target，再重新跑完整 qualification。

### 16.1 為何正式量測前要預編譯 proposal shapes

實際 drafter input 的 M 會落在 15–30。直接以 cudagraph 預編譯全部 shape，會讓 private pools 與 `StaticCache` buffers 累積，在 M=25/26 接近 30.8 GiB 並 OOM。最後採用 `max-autotune-no-cudagraphs`：每個 M 建立短命 cache、eager prefill 一 token，再於 `torch.inference_mode()` 中 materialize compiled drafter；每個 shape 完成後立即釋放 cache。如此 measured sweeps 不再支付 input-dependent compile 成本，也不會為了暖機犧牲正式量測所需的 VRAM headroom。

### 16.2 三資料集正式結果

以下仍是 batch 1、greedy、正常 EOS、256-token 上限、8 warmups、3 sweeps；每個 mode/dataset 使用獨立程序。Speedup 是逐 `(sweep, prompt)` 配對後的 median，CI 為 10,000-sample bootstrap。

| Dataset | Mode | Steady tok/s | Paired / post-merge AR | 95% CI | Mean accept | Peak VRAM | Parity |
|---|---|---:|---:|---:|---:|---:|---|
| HumanEval | TI | 49.783 | 1.607× | [1.553, 1.702] | 5.656 | 8.62 GiB | exact |
| HumanEval | TD | 57.438 | 1.854× | [1.830, 1.901] | 6.431 | 8.70 GiB | exact |
| GSM8K | TI | 49.785 | 1.602× | [1.557, 1.669] | 5.548 | 8.20 GiB | exact |
| GSM8K | TD | 59.542 | 1.910× | [1.811, 1.982] | 6.355 | 8.25 GiB | exact |
| MATH-500 | TI | 52.688 | 1.789× | [1.565, 2.006] | 5.787 | 8.53 GiB | exact |
| MATH-500 | TD | 60.209 | 2.015× | [1.840, 2.222] | 6.489 | 8.60 GiB | exact |

Post-merge AR steady TPS 為 HumanEval 30.967、GSM8K 31.036、MATH-500 30.005。六組 CI 下界皆大於 1.0，steady run-level CV 皆低於 5%，VRAM headroom 為 72.7–74.2%，因此 `hard_gate=true`。MATH TD 另做 cache-warm rerun：steady 與第一次只差 −0.085%，E2E CV 由 8.31% 降至 1.83%，證明第一次 E2E 波動來自首載，不是 runtime 不穩定。

### 16.3 回歸測試與研究結論

最終 `pytest -q tests` 為 **239 passed**。TI/TD 對 post-merge AR 分別完成 HumanEval/GSM8K 各 240/240、MATH-500 各 60/60 exact token parity；clean HIP extension 亦保留 26/26 pybind exports。

這輪最重要的教學結論是：合併 target 最佳化後，必須重新定義共同 oracle，而不是要求新 target 重現舊 rounding。舊數值路徑可作 diagnostic，但若它不再對 current AR parity，就不能拿來決定 speculative acceptance。完整衝突與 ablation 證據見 `FUSED_V1_MERGE_CONFLICTS_ZH.md`。
