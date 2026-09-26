# Qwen3 thinking 品質共同規格 v1

這是新一輪 **thinking + sampling** 品質評估。與既有 non-thinking + greedy 品質結果分開存放；它不替換現行測速或品質 queue，也不是官方 leaderboard 的完全複現。

## 直接取得（不必手動搬移 ZIP）

Repository：`https://github.com/NTHUQuantization/QuaRot.git`；branch：`benchmark/qwen3-thinking-quality-handoff`；套件目錄：`benchmarks/qwen3_thinking_quality_v1`。請使用獨立 clone，並記錄／遵守交接訊息提供的 commit。執行指令見 `AGENT_PROMPT_ZH.md`。此套件不含模型權重、不含既有實驗結果。

## 研究問題與範圍

- RTX 6000：原始 Qwen3 BF16、AR 品質基準。
- R9700：同來源模型的 GPTQ W4A4KV4、Fusion AR；PARD2 待 sampling 正確性與長 cache 驗證後加入。
- 目前 `fused_v1/e2e/speculative.py` 的 AR／PARD2 使用 argmax；既有 max_cache_len=8192。不可以直接把舊程式標為符合本規格。
- 主實驗先完成 **8B**。14B／32B 使用本套規格的固定 revision；在主機資源許可且排程授權涵蓋時依序擴充。不可因 OOM 改精度、縮短輸出、量化 BF16 baseline 或偷截 prompt。
- 跨 GPU 的正確率可作對照，但差異包含後端與 sampling 波動。若只有跨平台 baseline，不將小幅差距全部歸因於量化。RTX 6000 的時間／記憶體只能作該平台附帶資訊，不能計算 R9700 的 kernel 加速比。

## 固定設定

|項目|配置|
|---|---|
|模型|`protocol.json` 的 Qwen/Qwen3-8B、14B、32B 原始 instruct 模型及 immutable revision；不是 Base、2507 或量化發布版|
|BF16|模型權重 BF16、KV cache BF16；attention／softmax 的原生 FP32 累加可保留並記錄|
|資料|HumanEval test 164；GSM8K main/test 1319；MATH-500 test 500；沿用既有鎖定題集|
|prompt|零樣本；只讀 `prompts/*.jsonl` 的 messages；system 固定 `You are a helpful assistant.`|
|thinking|`enable_thinking=True`、`add_generation_prompt=True`；禁止 fallback template|
|sampling|do_sample=true、temperature=0.6、top_k=20、top_p=0.95、min_p=0；依 temperature→top_k→top_p 順序處理 logits|
|其他生成控制|num_beams=1、num_return_sequences=1、repetition_penalty=1.0、無 presence/frequency penalty、無 grammar／forced answer／工具／self-consistency／retry-on-wrong|
|預算|每题 max_new_tokens=32768，**thinking 與 final 合計**；min_new_tokens=0；自然 EOS `[151645,151643]`，pad=151643|
|context|保留模型原始 config/rope；不啟用 YaRN；input+32768 必須 <= 原始 max_position_embeddings；PARD2 額外保留 lookahead|
|batch|1；每題獨立 RNG；不使用跨題 prefix cache|
|replicates|固定 seeds `[20260926,20260927,20260928]`，每題每 seed 一次，總共 **5949 次生成／模型**|
|每題 seed|由 `common.sample_seed(task,id,replicate_seed)` 產生；每題重設 Python、NumPy、torch CPU／CUDA RNG，使續跑／順序不影響 seed|
|生成引擎|BF16 以 HF Transformers 4.57.6、tokenizers 0.22.2、PyTorch 2.9.1 CUDA build、SDPA、eval/inference mode 為 v1 參考；禁止默換 vLLM／SGLang|
|環境|CUDA build／driver 依主機相容版本固定並記錄，Python 3.10；若版本不可用，回報偏差，勿混為同一 run|

PyTorch CUDA/ROCm 即使 seed 相同也不保證相同輸出；這是多次 sampling 的品質比較，不是逐 token parity 測試。

## Prompt（必須使用封裝檔案，以下僅說明）

- HumanEval：`Complete the code I provided.\n\n` + 原始程式 prompt + `\n\nReturn the complete Python implementation, including required imports, in a single fenced python code block in your final answer.`
- GSM8K／MATH-500：原題 + `\n\nPlease reason step by step, and put your final answer within \\boxed{}.`（實際 prompt 中是一個反斜線。）
- 不向模型提供 answer、solution、canonical_solution、test；這些只存在 `ground_truth/`。
- 用 pinned tokenizer 直接 `apply_chat_template(..., tokenize=True)`，不要對已套 template 的字串再加入 BOS。逐題輸入 token hash 必須等於 `input_contracts/`。

## 評分規格

1. 保存完整 generated token IDs 與 `skip_special_tokens=False` 的原始文字。依 **token ID 151668 (`</think>`)** 劃分：只評分該 token 之後的 final。這批固定 template 以 `<|im_start|>assistant\n` 結尾，不額外手動插入 `<think>`；解析邊界不依賴 opening tag 是否由模型生成。
2. 沒有 closing token、重複 closing token、final 空白：標記無有效最終答案、算錯；不得從 thinking 中抓最後一個數字或程式。不因答錯／截斷重抽。
3. HumanEval：final 中擷取 Python candidate，執行原始 test 的 pass@1，不是 HumanEval+。採封裝 `scoring.py` 的 code_candidate；不修程式、不依測試回饋重試。使用固定 Python sandbox image、10 秒 candidate timeout、128 MiB、1 CPU、32 PIDs、無網路；容器/主機錯誤算 infrastructure error，該 task 未完成，不能當答錯。
4. GSM8K：只擷取 final 的最後一個完整 `\\boxed{}`，做精確數值比較；支援逗號、Decimal、分數及簡單 LaTeX 分數。沒有有效 boxed numeric answer 算錯；不回退至全文最後一個數字。
5. MATH-500：只擷取 final 的最後一個完整 `\\boxed{}`，用同一 `scoring.py` 的 conservative normalized exact match。這不是完整符號等價評分，可能低估等價答案；禁止僅某一組換用更寬鬆 grader。可日後對所有組同時新增版本化評分。
6. 觸及 token cap 的樣本仍留在原分母；若已有 final，照固定 scorer 評分並另標 cap；未完成 thinking 算錯。EOS 恰在最後一 token 與 cap_without_eos 分開記錄。
7. 每個 task 分別報每 seed 的 correct/N、accuracy、95% Wilson CI；總結報 **3 seeds accuracy 的平均與 sample SD**（ddof=1），可另報按題目 cluster bootstrap 95% CI（10000 次，seed=20260926，同題三 seed 一起重抽）。不做 best-of-3／pass@3，也不把三個 task 平均成一個總分。
8. 同時報 cap_without_eos、missing/multiple think end、empty final、extraction failure、exception/OOM 各自數量與比率。分母是完整題數，不排除失敗題來提高分數。

## 執行與驗證

- 先用 `python3 test_contract.py`、`common.verify_bundle()` 驗證封裝。下載 pinned 模型（不含在此封裝）後，以 `model_hashes.json` 比對所有列出的檔案，再做全部輸入 token hash 與 context 容量 preflight。
- 每 task 的固定前 4 題、第一個 replicate seed 作 pilot；使用正式 32768 上限及完全相同設定，pilot 輸出單獨保存。pilot 只測工程正確性，不能挑選高分設定。正式仍重新執行完整題集，pilot 不計入總分。
- 技術錯誤、OOM、版本／hash 改變：停止該 run 並保存錯誤。pilot 有 cap 或未完成 thinking 時先回報，不擅自縮小題集、改 budget 或重抽。pilot 答錯但流程有效，不阻止正式測試。
- 正式次序：replicate seed 外圈，HumanEval → GSM8K → MATH-500，題序使用封裝原順序。seed 1 全部完成即可提供 provisional 結果；3 seeds 都完整才標正式 complete。
- 正式截斷／final 擷取失敗率若 >1%，照原規格繼續保留全量結果，但結論加註限制；不能因分數不好就丟棄 run。任何新預算／grader 都另開 protocol version，且所有比較組一起變更。
- 單卡 BF16 優先；若模型不能在該 GPU 容納，可使用多卡 BF16（記錄 device map，僅品質比較），但不得搶占他人工作或自動改為 CPU/disk offload。資源不足標 blocked，保留已完成的 8B。
- 不終止原有 R9700 queue；本文件不會啟動新的 GPU 測量。

## 交付格式

`results/<protocol-id>/<size>/bf16-ar/<hardware-id>/<run-id>/`

- `manifest.json`：protocol/file hashes、模型與 tokenizer revision/hash、完整有效 GenerationConfig、torch/transformers/tokenizers/CUDA/driver/Python、GPU 完整名稱/UUID/VRAM、attention backend、device map、source hash、開始時間。
- `pilot/`、`formal/seed_<seed>/<task>/runs.jsonl`：每題立即 flush/fsync。鍵為 model/variant/task/id/replicate_seed；保留 input_ids、input hash、output_ids、原文、final、derived seed、input/output/thinking/final token counts、EOS/cap/think completion、elapsed、peak memory（附帶資訊）、status/error。
- `quality.json`／`per_example.csv`／`summary.csv`：scorer hash、逐題判定、各 seed/task 與跨 seed 統計。HumanEval sandbox image digest 與 candidate hashes 必須保留。
- `progress.json`／`errors.jsonl`／`commands.jsonl`／`environment.txt`／`SHA256SUMS.json`／`RESULTS_ZH.md`；結果壓縮包不含模型權重。
- resume 僅在 protocol/model/source/environment hash 相同時跳過已完整且唯一的紀錄；損壞尾行先另存，禁止重複 key。基礎設施失敗修復後可用原 sample seed 重試，但留下 audit，不挑選答案。

來源：https://huggingface.co/Qwen/Qwen3-8B#best-practices 、https://docs.pytorch.org/docs/stable/notes/randomness.html 。此封裝內的資料、token hashes、scorer 與 JSON protocol 才是跨主機比較的固定依據。
