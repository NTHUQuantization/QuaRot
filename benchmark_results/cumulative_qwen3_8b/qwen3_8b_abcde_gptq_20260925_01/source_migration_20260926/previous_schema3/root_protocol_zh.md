# Qwen3-8B 累加消融：測速與呈現規範

日期：2026-09-25；版本 3（A=HIP、B=Hadacore、C=GEMM、D=Fusion、E=PARD2）。用途：交接給實作 benchmark、執行實驗與整理結果的 agent。
本文件記錄本次討論的方法；寫入本文件不代表已完成程式改寫、資格驗證或正式測速。


**版本 3 補充（本次執行生效）**

使用者已選定「保留各分支計算實作」。只施加載入／cache API 相容層，不改 HIP kernel、精度或 rounding。跨版差異照實保留，不作正式測量失敗條件；同版跨 repeats／phase 必須一致。

- 代號歷史分兩階段：最初四組的 C（Fusion）→現行 D、D（PARD2）→現行 E；其後五組曾將 GEMM 標為 B、Hadacore 標為 C，本次修正為 B=Hadacore、C=GEMM。A 呈現名稱改為 HIP；D 固定名為 Fusion。
- 原始四版本文件封存於 `cumulative_ablation_abcde_20260925/protocol_original_abcd.md`；既有 raw record 保留原件，另提供明確改標的 D／E 結果與 mapping manifest。
- 延續既有 96 題正式、12 題不重疊 pilot、完整 input IDs、sweep 順序與生成設定。
- 新增 P10／P90：主表以每題 3 次 median 的 32 題分布計算，使用線性插值；另列全部 96 測次的 P10／P90，明列口徑，不把 repeats 當成獨立題目。
- 新增 peak memory：phase 開始前同步並 reset PyTorch peak counters（decode 在 prefix 準備完成後 reset），計時後讀取 peak allocated／reserved 與 baseline、increment；包含模型常駐記憶體，不冒稱整卡 VRAM。
- 舊 D／E 測量沒有 peak memory，須補測才能填此欄；不能從延遲或 GPU snapshot 推算。
- 任何分支與共同 checkpoint／數值契約不相容，先保留診斷、修正相容性或明確記錄差異；不得混用重新量化權重冒充純 runtime 消融。
- 全部測量驗證後，於各分支以相同 `benchmark_results/cumulative_qwen3_8b/<run_id>/` 格式存放該組結果與重現資訊，使用正常 fast-forward push，不 force push。

**1. 實驗目標與五版本定義**

比較同一份 Qwen3-8B INT4 checkpoint，在相同 prompt 與生成設定下，各指定分支的延遲與吞吐量。分支數值實作保留，生成軌跡與輸出長度可能不同。
消融敘事與所有報表固定依 A→B→C→D→E 呈現。來源稽核確認 B 的 GEMM／Linear4bit 與 A 相同，C 的 Hadamard kernel 與 B 相同；因此先 Hadacore、再較新的 GEMM，最後 Fusion 與 PARD2。

| 版本 | 英文名稱 | 固定來源 | 本步重點 |
| --- | --- | --- | --- |
| A | HIP | `NTHUQuantization/QuaRot_Version`：`main` | HIP QuaRot INT4 基準；呈現名稱由 main 改為 HIP |
| B | Hadacore | 同 repository：`Hadacore` | 保留 HIP GEMM，加入 HadaCore |
| C | GEMM | 同 repository：`GEMM` | 較新的實作；保留 HadaCore，加入 GEMM 優化 |
| D | Fusion | `NTHUQuantization/QuaRot`：`fused_pard2`，AR mode | 融合 runtime |
| E | PARD2 | 同一 fused runtime，PARD2-TI mode | 投機解碼 |

- A→B 對應 Hadacore，B→C 對應 GEMM，C→D 對應 Fusion，D→E 對應 PARD2。
- `main` 保留為真實 source branch 名稱，不把 branch 改名誤寫成來源 Git 操作；圖表與版本名稱使用 HIP。
- 各版若生成內容或長度不同，配對耗時比不能解讀成等輸出的純 runtime 加速。D→E 須獨立驗證等輸出。
- 本次 B／C 對調僅校正標示與呈現順序，不更換已測分支、權重、數值或 prompt。
- 原始執行 log 保持時間順序並校正代號，不偽造執行先後。GPTQ pilot 的實際順序改標後是 D→A→C→B→E；正式剩餘測量依 B→C→E 接續 D、A。
- 更早四組版本的 C/D（Fusion/PARD2）與本次五組 B/C 對調是兩次不同改標，原始證據已明確封存並附 mapping。
- mean accept 另有獨立實驗；原始接受統計仍保留供追查。

**2. 已決定的原則、執行預設與尚待固定的設定**

確定的原則：五個指定版本，是否累加以來源稽核為準；同 prompt 配對；真實 AR／PARD2 生成；prefill／decode／E2E 同步 wall time；主表包含兩種 tokens/s；以 E2E 配對 speedup 為主要加速指標。

以下沿用討論中的首輪建議，作為本文件執行預設；若調整，需在正式測速前更新 manifest，所有版本共用新設定：

| 項目 | 首輪預設 |
| --- | --- |
| Target | Qwen3-8B，同一份 INT4 量化模型的數值設定 |
| 資料規模 | 3 個資料集，各 32 個不同 prompts，共 96 題 |
| 建議資料集 | HumanEval、GSM8K、MATH-500；這是程式／數學子集，不能宣稱已涵蓋一般對話 |
| Batch | 1；五版本依序使用同一張 GPU |
| 解碼 | Greedy，不啟用 sampling；Qwen3 thinking 關閉 |
| 輸入長度 | 真實 prompt 含 chat template 的實際 token 數，不固定為 2048 |
| Pilot 最大新增 tokens | 128；僅供初步正確性檢查與估時 |
| 正式最大新增 tokens | 256；五版本一致 |
| 停止方式 | 主實驗保留自然 EOS；共同 EOS IDs、stop 規則與長度上限 |
| 每題正式重複 | 3 次，分成 3 個 sweeps |
| 初始暖機 | 每版本、每個 phase 至少 3 次代表性 workload；遇到尚未暖機的編譯／graph shape 再補暖機 |
| 隨機種子 | 抽樣、順序、bootstrap 各記錄種子；未有專案設定時使用 0 |
| Pilot | 3 個資料集各 4 題，共 12 題，先驗證五版本與量測成本 |

正式執行前，agent 必須從現有專案／資源確認並記錄以下值；本次對話未指定其具體值：

1. A、B、C、D、E 的 source checkout／commit、載入的 HIP extension、各項有效優化開關。不能只按目錄名稱推定版本。
2. 共用 target checkpoint 的來源、量化／rotation／clip／KV 設定。若各 runtime 需要不同 packing，需證明是相同量化值的格式轉換；不得混入重新量化差異。
3. E 的 PARD2 模式：選定 TI 或 TD 作主表 E，固定 draft checkpoint、draft 精度、draft_k、adaptive-k 與 AR fallback 策略。不得將 TI／TD 混成一列或逐題擇快；另一模式可另列補充實驗。
4. 資料集來源 revision／本地檔案 hash、split、chat template、system prompt、EOS／stop IDs。
5. 共用非消融設定：attention backend、target dtype、cache page size／capacity 政策、compile／graph 政策等。若某項變動是本步優化的一部分，明列在該步差異清單，不得隱藏。

上述欄位未確認時可繼續資料準備與 harness 整合；不得把來源不明的結果標示為已完成五版本正式消融。

**3. Prompt 與輸出工作量**

1. 從固定版本的 held-out 資料集，以固定種子抽取 32 個不同 sample IDs；在查看速度或接受率之前凍結清單。保存抽樣演算法、來源筆數與 sample IDs，不能只保存 seed。
2. 統一 prompt 格式後只 tokenize 一次，保存完整 input_ids、input token 數、sample ID、prompt/template hash。五版載入完全相同的 IDs。
3. 保留完整題意。不同題目的輸入長度可以不同，不需補到 2048，也不重複無意義 token。過長輸入的排除／截斷政策須預先固定，不能刪掉某版本慢或不易接受的題目。
4. 不同資料集的絕對 ms 不需要對齊；同一題在五版之間配對即可。可另外按輸入 token 長度分組，但不可每版各自分出不同題目。
5. A、B、C、D 每一步都用模型預測的 token 繼續生成；E 必須實際執行 draft、verify、accept/reject 與 cache 更新。禁止固定餵入 token 100 或 teacher forcing 代替主測速。
6. Pilot 使用 `max_new_tokens=128`，正式實驗使用 `max_new_tokens=256`；兩者均保留自然 EOS。這是新增輸出 token 上限，並非保證輸出數，也不是輸入長度。原組員 benchmark 的 2048 是固定輸入長度、128 是固定輸出長度；本主實驗不沿用其合成輸入或忽略 EOS 的設定。主測速不得默默啟用 ignore_eos，pilot 與正式結果不得混合彙整。
7. 記錄真正提交的 output_ids、停止原因、是否碰到上限。定義 N=len(output_ids)，包含生成且已提交的 EOS／stop tokens，不包含 prompt、被拒絕的 draft tokens 或最後一輪超過停止點的候選 tokens。
8. E 在一輪中跨過 EOS 或當次輸出上限（pilot 128／正式 256）時，只提交截至共同停止點的 tokens；該輪已付出的 draft／verification 時間仍全部計入，不事後扣除。
9. Pilot 供正確性／成本檢查。若拿 pilot 調 draft_k 等超參數，使用與正式清單不重疊的開發題目；正式題目不按觀察到的接受率選擇。

如需固定輸出量的補充實驗，可另用固定 N 與 ignore_eos=True，但必須獨立命名、五版共同執行、與主表分開。自然結束後強制續寫可能產生重複尾段並扭曲接受率；不能用此結果取代自然停止的主消融。

**4. 先驗證工作量與輸出，再解讀速度**

- 計時外逐題比較 A／B／C／D 的 greedy output IDs、N、EOS／停止位置與停止原因；另比較 D 的 target AR 與 E 的 PARD2。只與同模式自己的舊 runtime 一致不足以證明 D→E 一致。Pilot 的 128-token 一致性不能代替正式 256-token 範圍的驗證。
- EOS 提早結束會縮短耗時，但同一題五版本若在相同位置停止，仍是公平配對。不同題目可以在不同位置停止；若只有某一版提早停止，不可把減少輸出量造成的時間差全部當成運算加速。
- 短輸出會提高 prefill 在 E2E 的占比，也可能降低 PARD2 的 E2E 加速幅度；這是短請求的實際效果，不應為提高加速比而忽略 EOS。
- 同一版本不同 repeats，以及 decode-only／E2E 兩條路徑，應得到相同輸出。
- 如有差異，保留該題結果與首次差異位置，釐清數值 rounding、cache、EOS、驗證規則或實作差異。不得刪掉不一致的題目來提高主表速度。
- 若長度或生成軌跡差異尚未釐清，結果可留作診斷，但不可直接解讀成相同輸出工作量下的純執行加速。先修正或明確說明變更，再重測受影響結果。
- 固定輸出長度也不能證明 speculative decoding 的正確性；E 仍需獨立 AR oracle 驗證。
- 正式 256-token cap 仍可能截斷答案；主結果標示為「256-token 輸出預算下的生成效能」，不把碰到 cap 當成成功完成任務，另報 cap_reached_fraction。

**5. 計時邊界與實作契約**

採 `time.perf_counter()` 加 phase 起訖 GPU synchronization 的 wall time；ROCm 使用 PyTorch 的 `torch.cuda.synchronize()` API。
包含 host 控制、kernel launch、runtime 所需 CPU/GPU 資料傳遞與解碼控制成本。量測額外加入的同步只放在 phase 邊界；生成演算法本身必要的 CPU/GPU 同步保留並計時，不額外逐 token 同步量測或開啟 profiler。

| Phase | 起訖與包含項目 |
| --- | --- |
| Prefill | 從開始準備本 request 的 cache／prefix 起，到完整 prefix 狀態就緒、尚未選出第一個輸出 token 為止。A–D 包含 target prefix；E 包含必要 target／draft prefix 與 feature 準備。 |
| Decode | 每次量測前，在計時外用該版本重新準備乾淨 prefix。計時從 prefix 就緒起，到共同停止點的 output token IDs 完成為止。包含第一個 token 的選擇，以及 E 的第一輪 draft／verification、接受判斷與 cache 操作。 |
| E2E | 從本 request 的 cache／prefix 準備開始，到共同停止點的 output token IDs 完成，直接量完整路徑。不得在 prefill／decode 分界插入額外量測同步。 |

- 模型載入、權重轉換、tokenization、input tensor 準備與搬入 GPU、編譯、graph capture、暖機、輸出 correctness 比較、文字 detokenization、寫檔與統計均放在計時外；五版一致。
- E2E 是上述本地模型生成的 E2E，不包含網路、服務排隊或 tokenizer，表註寫明。
- E2E 直接量測，不能以 prefill median + decode median 代替。這是三種分開量測的 workload，統計數值不必可相加。
- Decode 準備 prefix 雖不計入 decode_ms，仍會消耗實際實驗時間；記錄整批 wall time 估算成本。
- 每次 request 使用乾淨的邏輯 KV cache；可保留 runtime 正常的常駐 buffer，但不能把上一題的 prefix／draft 狀態帶入下一題。禁止只改 cache.length 就假設所有狀態重設完成。
- Target prompt prefill 的 LM head 工作量要一致（目前可用 logits_to_keep=1）。PARD2 verification 需要多位置 logits，不可把此參數全域固定為 1。
- E 的額外 prefill、feature 與 verification 等必要成本都必須在上述 phase 中計入。若 runtime 將部分初始化延後到 decode，就算在 decode，不得移到兩者以外。
- 本規範的 prefill 不等於 TTFT。不得以「第一個 accepted chunk 完成時間」代替 prefill，也不得套用排除第一輪 tokens 的 steady-decode 指標。

**6. 執行順序、暖機與重複**

1. 建立並凍結 dataset manifest、五版本 manifest、共用生成／計時契約。
2. 用 pilot 完成跨版本輸出檢查、三 phase 邊界檢查與實際整批耗時估計。每個版本的模型載入、暖機、正式量測成本分開記錄。
3. 按版本載入模型，再跑多題；避免每題重載 checkpoint。用不同 process／隔離載入路徑確認當前 import 與 extension 屬於該版本。
4. 每版本每 phase 先跑至少 3 次代表性暖機，補足正式測試需要的 compile／graph shape。不能讓首次編譯或 graph capture 落入正式樣本。
5. 正式每題每 phase 跑 3 次，保存原始次序。每個 sweep 使用預先產生的題目順序，五版本在同 sweep 使用同一順序。
6. **不是每題再暖機 3 次。** 若重用現有 benchmark_one，初始／必要 shape 暖機完成後，各題正式呼叫需使用 warmup=0，避免舊預設自動放大實驗時間。
7. 預先安排版本／sweep 順序以分散溫度與時脈漂移；可按 sweep 輪換版本，例如 ABCD、BCDA、CDAB。若採單次載入跑完一個版本，記錄此取捨與 GPU 狀態。
8. 使用相同 GPU 與功耗／時脈政策，量測時避免 GPU 競爭。若發生競爭、編譯、OOM 或計時異常，保存原因與原始樣本，按預先設定的有效性規則重測；不因為某筆很慢就刪除。
9. 若 pilot 顯示量測波動大，先排查原因，再決定增加所有受比較版本的 repeats；變更寫入 manifest。不要只多跑某版直到出現較好 median。

3 次重複是首輪成本取捨；不足以估計可靠的每題 P90。不要把 repeats 當成新的獨立 prompts。

**7. 指標與統計公式**

基本測試單位是 batch=1 的一筆 request；配對／統計抽樣單位是不同 prompt。
設資料集為 d、題目為 i、版本為 v、正式重複為 r、phase 為 p。
T[v,i,r,p] 單位為 ms；N[v,i,r,p] 為該生成測次實際提交的輸出 token 數。Prefill 沒有輸出 N。

每題各 phase 先取重複測量 median：

    t[v,i,p] = median_r T[v,i,r,p]

主表每個資料集的 latency：

    latency[v,d,p] = median_{i in d} t[v,i,p]

每次生成測量分別算吞吐量，再依「先 repeats、再 prompts」取 median：

    decode_tps[v,i,r] = 1000 * N[v,i,r,decode] / T[v,i,r,decode]
    e2e_tps[v,i,r]    = 1000 * N[v,i,r,e2e]    / T[v,i,r,e2e]
    decode_tps[v,d]   = median_{i in d} median_r decode_tps[v,i,r]
    e2e_tps[v,d]      = median_{i in d} median_r e2e_tps[v,i,r]

- E2E tokens/s 分子仍只計輸出 tokens，不加 input tokens。N=0 或非正時間視為異常，記錄原因，不產生無限大／無意義吞吐量。
- 上述是 median request throughput。若另報整批 throughput=sum(N)/sum(time)，要另命名，不能混用。
- tokens/s 是輔助指標；context 長度與任務仍影響成本，不能宣稱換成 tokens/s 就消除了所有 workload 差異。

以 E2E 為主，先按題計算配對加速，分母不使用跨題 median：

    cumulative[v,i] = t[A,i,e2e] / t[v,i,e2e]
    incremental[B,i] = t[A,i,e2e] / t[B,i,e2e]
    incremental[C,i] = t[B,i,e2e] / t[C,i,e2e]
    incremental[D,i] = t[C,i,e2e] / t[D,i,e2e]
    incremental[E,i] = t[D,i,e2e] / t[E,i,e2e]
    speedup[v,d] = exp(mean_{i in d}(log(cumulative[v,i])))

逐步加速同樣對 incremental 的 log 取平均再 exp。大於 1 是加速，小於 1 是減速，兩者照實呈現。
如果另算 decode／prefill speedup，替換成相應 phase，欄名寫明，不能混成 E2E speedup。

如需跨資料集整體 speedup，先取得每個資料集的幾何平均，再資料集等權：

    overall_speedup[v] = exp(mean_d(log(speedup[v,d])))

主表不把不同資料集的原始 ms 直接加總成通用延遲。不同資料集樣本數不等時，仍保持預先指定的資料集等權。
若有缺失／失敗題目，不可讓各版用不同樣本集產生可比較的正式總結；明列 coverage 與未解決問題，不默默取成功交集。

建議對 E2E speedup 報 95% bootstrap CI，按 prompt ID 配對重抽樣（例如 2000 次，seed=0），同一次重抽樣同時保留五版。跨資料集 CI 在各資料集內重抽樣後再等權彙整。
重複執行同一題的 3 個 repeats 不是 3 個獨立樣本；96 題各跑 3 次仍是 96 個不同工作負載。

**8. 報表格式**

每個資料集分別列五版；表註包含題數、batch、輸出上限、EOS 規則、thinking、重複次數、E 模式與 draft_k。
時間與兩種 tokens/s 都依第 7 節取 median；speedup 依配對幾何平均計算：

| 版本 | Prefill (ms/request) | Decode (ms/request) | E2E (ms/request) | Decode (tokens/s) | E2E (tokens/s) | E2E 相對 A 加速 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A：HIP | 待測 | 待測 | 待測 | 待測 | 待測 | 1.00× |
| B：Hadacore | 待測 | 待測 | 待測 | 待測 | 待測 | 待測 |
| C：GEMM | 待測 | 待測 | 待測 | 待測 | 待測 | 待測 |
| D：Fusion | 待測 | 待測 | 待測 | 待測 | 待測 | 待測 |
| E：PARD2（TI） | 待測 | 待測 | 待測 | 待測 | 待測 | 待測 |

另列 A→B、B→C、C→D、D→E 的 E2E 逐步 speedup 與建議 CI。
附每資料集的 input/output token 長度分布、cap_reached_fraction、輸出及 EOS／停止位置一致性與有效題數。若另外分短／長輸出組，使用共同參考版本 A 的輸出長度與預先固定的分組邊界，五版本在每組使用相同題目，不可各版按自己的長度分組。
若使用圖，按資料集分面／分組，或畫相對 A 的 E2E speedup；必須保留 D→E 的增量資訊，不把不同 workload 的 ms 排成同一條累加改善鏈。
本實驗不需重複呈現 mean accept 主表；原始接受記錄仍保留，以便與獨立實驗的資料／生成設定對照。

**9. 最小產物與可重現記錄**

輸出到獨立結果目錄，不覆寫歷史結果。建議結構如下；這是產物規格，檔案尚未由本文件建立：

    ablation_qwen3_8b_<run_id>/
      protocol_manifest.json
      prompts.jsonl
      variant_manifest.json
      qualification.jsonl
      runs.jsonl
      per_prompt.csv
      summary_by_dataset.csv
      speedups.csv
      report_zh.md

- protocol_manifest：本文件 hash、資料／抽樣／順序設定、全部共用參數、計時邊界、暖機策略、開始結束時間、hardware／software 環境。
- prompts：dataset revision/split、sample ID、完整 token IDs、input 長度與 hash；可由此重跑完全相同工作負載。
- variant_manifest：完整累加設定、source commit／dirty diff 或 source hash、實際 import 路徑、extension hash、權重／config 識別、有效開關、E 的全部 draft 參數。不得只記錄請求參數而不核對實際生效值。
- qualification：跨版、跨 repeats、跨 phase 的 output IDs／停止原因一致性；失敗的首次差異位置與處理紀錄。
- runs：保留原始測量順序；至少包含 run_id、variant、dataset、sample_id、input hash、sweep/repeat、phase、elapsed_ms、input/output token 數、output IDs/hash、停止原因、cap flag、有效性／失敗原因、GPU 競爭狀態。
- E 的 raw stats：verifier rounds、proposed draft tokens、accepted draft tokens、actual emitted tokens 與 fallback 使用情況；區分「接受的 draft 數」與「含 correction/bonus 的每輪輸出數」，不要用這些中間數代替 N。
- 所有報表必須由 raw records 重建。記錄程序整體 wall time 與各步耗時，供下一輪估時；不要用 median request latency 冒充整批執行成本。

**10. 現有程式可沿用處與必要調整**

以下是 2026-09-25 的唯讀檢查結果；執行 agent 應確認實際來源未改變。這些入口不是已經符合本規範的一鍵正式命令。

| 檔案 | 可沿用內容／限制 |
| --- | --- |
| [組員 benchmark 快照](QuaRot_Version_benchmark/upstream/e2e/benchmark_real_llama_runtime.py) | 可參考 wall timer 與 phase 介面。現有 decode 反覆餵固定 token，必須改為真正生成。快照本身不是完整可 import 的 runtime。 |
| [快照檢查報告](QuaRot_Version_benchmark/REVIEW_ZH.md) | 說明來源、共同格式與計時陷阱；其中 GEMM／HadaCore 分支尚未擷取檢查，不能由此直接認定 B 已可用。 |
| [目前共用 harness](fused_v1/e2e/benchmark_real_llama_runtime.py) | 可沿用同步 wall time／decode prepare。預設 mode 與 warmup/repeats 不等於本規範；須顯式配置。 |
| [Phase adapter](fused_v1/e2e/benchmark_pard2_phases.py) | 已支援 --input-ids、真實 AR／PARD2 與完整 prefix 的 phase split；但目前強制 ignore_eos=True、固定輸出長度、單 prompt shape，須加入自然 EOS、可變 N 與多 prompt 調度。 |
| [資料與 PARD2 runner](fused_v1/e2e/benchmark_pard2.py) | 可參考 tokenizer、provenance、input hash。既有 decode 指標是 steady-decode，不能直接當成本規範的完整 decode；--limit 目前會標成 smoke/unqualified，不應藉刪旗標冒充通過舊完整協定。為本抽樣消融建立獨立且明確的 qualification 定義。 |
| [既有完整 split 資料目錄](fused_v1/qwen3_full_eval_20260925/data) | 可檢查 HumanEval／GSM8K／MATH-500 的完整 split、本地 hash 與來源，供抽樣。不得因路徑名稱就假設內容／revision 正確。 |

額外注意：

- 舊 `benchmark_pard2.py` 的 pinned benchmark 快照是 HumanEval 80、GSM8K 80、MATH-500 20 題。MATH-500 不足以直接抽 32 個不同 prompts；使用完整 split 或另一個已固定 revision 的完整來源，不可重複那 20 題湊數。
- 目前 batch=1 phase adapter 的 reference 是同模式 runtime.generate；E 仍需要額外和 D 的獨立 AR 輸出比較。
- 目前 DecodeWorkload／E2EWorkload 在被計時的呼叫內執行 adapter.validate，包括輸出比較與 checks 記錄。整合時將這些正確性檢查與記錄移到計時外，才符合第 5 節；正常生成所需的 token 選擇與資料傳遞仍計時。
- 既有 ttft_ms／steady_decode_ms 以第一個 emitted round 切分，AR 是一個 token、PARD2 可能是一個 chunk；與本規範 phase 不相容。
- 舊 percentile helper 有整數索引時的已知問題。若沿用 percentile／CI 工具，先修正並驗證；不可把 3 次 repeats 的 P10/P90 當作穩健不確定性估計。
- 完成整合後，正式跑之前要確認三個 phase 的邊界、E 成本沒有漏帳、N／EOS 截斷正確、所有版載入正確 extension，且主表 speedup 使用完全相同題目配對。

**11. 完成條件**

五個指定版本來源可追溯；96 題清單固定；正式最大新增 256 tokens、自然停止與共同生成規則生效；三次正式量測完成；五版本輸出、EOS／停止位置與計時契約驗證完成；raw records 可重建每資料集 latency、兩種 tokens/s、累計及逐步 speedup。所有失敗／缺失／未釐清差異如實保留，不把部分完成描述成完整消融。

先交付 128-token pilot 的正確性與整批 wall time，再估算 256-token 正式測試時間。外推時拆開固定的 prefill／載入／暖機成本與隨實際輸出量變動的 decode 成本；EOS 與接受率會影響生成長度及速度，不能把總時間直接乘二。可先用少量題目做 256-token 試跑校準估時。此前討論的「約半天至兩天」只是歷史合成測速的粗略排程預算，不是本自然語言協定的執行承諾。
