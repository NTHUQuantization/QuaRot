# Qwen3 thinking 品質：v2-12h

本規格取代本次執行中的 v1 全題集／三 seeds／32K 輸出計畫。v1 原始檔案與輸出保留，但不混入 v2 統計。此輪只測 **Qwen3-8B**，不擴到 14B／32B。

## 所有組一致的固定設定

|項目|設定|
|---|---|
|Task|HumanEval 64／164；GSM8K test 64／1319；MATH-500 64／500|
|選題|對完整 v1 題集，以 SHA256(`protocol_id\|subset\|task\|id`) 最小的 64 個 ID 選取，再保留原始題序；完整 ID 在 selection.json；與模型輸出無關|
|樣本數|單一 replicate seed=20260926；每題一次；**192 次正式生成／variant**|
|Thinking|enable_thinking=true、add_generation_prompt=true，固定零樣本 messages 與 input token hashes|
|Sampling|temperature=.6、top_k=20、top_p=.95、min_p=0，單一輸出；無 repetition/presence/frequency 額外懲罰|
|輸出預算|**4096 new tokens，thinking+final 合計**；自然 EOS [151645,151643]；不在 think end 停止，不強制關閉 thinking|
|Batch／併發|1；無跨題 prefix cache；同一 GPU 的各 variant 串行|
|Pilot|每 task 封裝第一題，共 3 題／variant；正式仍重跑完整 192 題|
|評分|只評 </think> 後 final；HumanEval 原測試 execution pass@1；GSM8K boxed numeric EM；MATH-500 boxed conservative normalized EM|
|報告|每 task correct/64、accuracy/pass@1、95% Wilson CI、截斷／未完成 thinking／無 final／擷取失敗比率；**不報三 seed 平均或 SD**|

只有 v2 的 `common.sample_seed` 能決定每題 seed。它使用 v2 protocol ID，因此 **不得直接挪用 v1 已生成樣本**。scoring.py/sandbox.py 的計分規則與 v1 相同；抽樣／seed／輸出預算改變，必須另開結果目錄。

## 12 小時約束

本次共同起點：**2026-09-26 23:59:03 Asia/Taipei**。

- GPU 生成停止派工並停止自己的未完成生成：**2026-09-27 10:59:03 Asia/Taipei**。
- 評分、整理與封裝硬截止：**2026-09-27 11:59:03 Asia/Taipei**。
- 這是這輪所有 variants 的共同期限，不是每個 agent 重新獲得 12 小時。較晚收到指令的 agent 使用剩餘時間；已過期則不得啟動新測試。
- 每組最多 786432 正式 output tokens。以截圖提供的約 53 tok/s 計，純生成滿額約 4.12 小時／組；BF16+NVFP4 在同一 GPU 串行約 8.24 小時。這是估算，不包含初始化、prefill、評分及硬體波動。pilot 必須更新 ETA，時間不足立即報告風險，不能私自改題数或 cap。
- 優先完成這個固定方案，不再安排舊品質／舊診斷／14B/32B 或其他補測。停止各自原 v1 service／worker，保留完整紀錄與中斷資訊，不刪除歷史輸出。
- 到期限仍缺樣本或評分時，寫 `budget_incomplete`，列出缺少的 ID 與原因；不可把 infrastructure timeout 算成普通答錯、不可宣稱已完成、不可另選已跑完的題目拼成 64 題。

## 防止 BF16 與 NVFP4 搶同一張 RTX GPU

兩個 agent 都必須在**主機端**取得排他鎖 `/tmp/qwen3-thinking-v2-<GPU-UUID>.lock`，由保持存活的 supervisor 持有到 GPU worker 完全退出。若容器 /tmp 不共用，仍由 host supervisor 持鎖，不是在各自容器建立兩把不同鎖。GPU 有其他工作則等待，不終止非本次工作。等待也計入共同 deadline。記錄 UUID，不能用 GPU index 猜測是否同卡。

CPU 評分可與下一個 variant 的 GPU 生成並行，前提是資源足夠且評分程序不建立 GPU context。BF16與NVFP4各自不得為取得鎖而停止對方新 v2 工作。

## Preflight 與 pilot

1. 驗證 SHA256SUMS、固定 model/tokenizer revision、全部 192 題 input token hashes、effective sampling settings、context 容量、原實作仍生效。
2. **在正式 GPU 生成前就確認 HumanEval sandbox**。Docker 可用時執行固定 digest image 的 pass/fail/timeout 控制。可使用 rootless Podman 執行同一 image／相同限制，但須保存適配程式與命令；不要更改 Python版本、測試、10秒限制或 scorer。禁止在未隔離的主機直接執行模型程式。
3. 若 sandbox 權限／安裝確實無法解決，立即回報，繼續可獨立完成的數學生成與評分，保留 HumanEval 輸出並標待評分；不能等全部生成完才揭露缺少 sandbox，也不能冒稱三 task 完成。
4. 每 task 第一題以正式設定作 pilot。pilot 檢查工程行為與吞吐量；4K 預算下截斷、未完成 thinking 或答案錯誤是品質結果，**不單獨構成停止條件**。identity mismatch、NaN、OOM、錯誤精度或不正確停止行為仍必須處理。
5. Pilot 通過後直接正式跑，禁止為追求高分改 prompt、重新抽 seed、挑最好答案或悄悄擴充 token budget。

## 成績有效性與 paper 表述

固定名稱：**Qwen3-8B, thinking, 4K output budget, 64-example/task fixed subset, single-sample evaluation**。它是時間／生成預算受限的品質對照，不是完整 HumanEval/GSM8K/MATH-500 官方成績，也不是完整長推理能力評估。

無 think end、重複 think end、無 final、無合法答案均算錯並保留在 64 題分母；已有 final 的 cap 樣本照固定 scorer 評分並標截斷。不同 GPU/後端相同 seed 不保證相同文字，分數差也不能全部歸因於量化。64 題的 CI 較寬，不據此宣稱極小退化／統計等價。

不設定答案正確率／截斷率門檻來選擇是否公布結果。若截斷很高，將其與準確率一起報告，說明 4K 限制。

## 交付

`results/qwen3-thinking-quality-v2-12h/8b/<variant>/<hardware-id>/<run-id>/`

保存 manifest、v2 protocol/selection/input/scorer hashes、模型/實作版本、完整有效設定、deadline、GPU鎖資訊；逐題 input/output IDs、原始文字、final、derived seed、終止原因、elapsed/peak memory（附帶資訊）、分數與錯誤；輸出 per_example.csv、summary.csv、quality.json、progress.json、RESULTS_ZH.md、檔案 hashes 與不含權重的結果包。

完成條件：三 task 各 64 個唯一正式樣本、全部評分完成；每題與固定 ID/seed/input hash 相符。所有組只使用同一 v2 統計比較，差距以百分點表示。完整題集結果、v1 三 seeds、不同 token cap 的資料另列。

來源：v1 immutable commit `a1fb51e4c9eb3be0096ef1581bd1400c1731858d`。本版固定 prompts/scorers/token hashes 全在套件，不需使用者傳檔案。
