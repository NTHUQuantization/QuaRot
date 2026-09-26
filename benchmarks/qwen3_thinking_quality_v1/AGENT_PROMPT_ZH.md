你在 RTX 6000 主機上執行 Qwen3 thinking 的 BF16 品質基準。請完成實作、preflight、pilot、正式生成、評分與結果封裝，不要只提供計畫。

請直接從 GitHub 取得套件，不需要使用者搬檔案：repository `https://github.com/NTHUQuantization/QuaRot.git`，branch `benchmark/qwen3-thinking-quality-handoff`，目錄 `benchmarks/qwen3_thinking_quality_v1`（不含權重）。使用獨立 clone，若使用者提供 commit 則 checkout 該 commit；記錄實際 HEAD。將含 `protocol.json` 的目錄記為 BENCH。也可使用同內容的 ZIP。完整方法在 `README_ZH.md`，JSON、資料與共用 Python helper 都以 `SHA256SUMS.json` 鎖定。這是新 thinking run，禁止覆寫任何舊 non-thinking／測速結果。

本次必做 **Qwen3-8B BF16 AR，三個 task、三個 seeds**。14B／32B 的 revision/hash 也已附上，先報主機能否容納；本次不要自行擴大到這兩個模型。

### 1. 固定測試合約

- 模型 `Qwen/Qwen3-8B`，revision `b968826d9c46dd6066d109eabc6255188de91218`；使用原始 BF16 權重及 BF16 KV cache，不可改成 FP16、FP8、INT8、GPTQ 或其他 checkpoint。
- HumanEval test 164、GSM8K main/test 1319、MATH-500 test 500。只讀封裝 `prompts/*.jsonl` 的 messages 生成，不從 `ground_truth/` 讀任何文字作 prompt。zero-shot；不重寫 prompt、不換 dataset download、不抽樣取代完整題集。
- `enable_thinking=True`、`add_generation_prompt=True`，pinned tokenizer 的 `apply_chat_template(..., tokenize=True)`。逐題比對 `input_contracts/8b_<task>.jsonl`；不得手動增減特殊 tokens，也不得在 mismatch 時重建 expected hashes。
- `do_sample=True, temperature=0.6, top_k=20, top_p=0.95, min_p=0.0, num_beams=1, num_return_sequences=1, repetition_penalty=1.0, no_repeat_ngram_size=0`。
- `max_new_tokens=32768`，包含 thinking 與 final；`min_new_tokens=0`；EOS `[151645,151643]`，pad=151643；不設其他 stop string，不在 `</think>` 停止，不強制插入結束 thinking 或答案。
- 每題 3 次獨立採樣，replicate seeds `[20260926,20260927,20260928]`，每題 seed 使用 `common.sample_seed(task,id,replicate_seed)`，並在每次生成前重設 Python／NumPy／torch CPU、CUDA RNG。batch=1；禁止 best-of-N、majority vote、答錯重試。每模型正式共 5949 次生成。
- BF16 runner：Python 3.10、PyTorch 2.9.1（適配本機 driver 的 CUDA build）、transformers 4.57.6、tokenizers 0.22.2，HF AutoModelForCausalLM + SDPA、eval + inference_mode。建立獨立環境，不破壞既有環境；若指定版本不相容，記錄阻礙，不自動改版本繼續正式結果。
- 單卡優先；BF16 品質可以多卡，但須明列 device map、每卡 KV/weight dtype。不得自動 CPU/disk offload。不要搶占其他工作、終止其他程序、租用付費資源或更改 GPU 系統設定。

### 2. 先完成 preflight

1. 檢查 RTX 6000 的完整世代／名稱、UUID、顯存、driver、CUDA、原生 BF16 支援與空閒狀態；不能只依「6000」名稱假設能力。若不支援 BF16 或容量不足，回報實際限制，不把 FP16 標成 BF16。
2. 執行 `python3 test_contract.py` 及 `common.verify_bundle()`。這個封裝提供固定資料／評分工具，**未提供已在該 GPU 驗證的生成 runner**；請在 BENCH 外的獨立工作目錄實作 `run_bf16.py`、`score_results.py`，不要修改鎖定檔案。
3. 從 HF 下載固定 revision 或核對已有 snapshot，逐檔比對 `model_hashes.json` 中 8B 的全部列出檔案（含權重），禁用 trust_remote_code。保存實際 hashes。
4. `common.verify_tokenized(tokenizer,'8b',task)` 驗證全部題目的 input token hash；所有題目必須滿足 input_tokens+32768 <= 原始 config.max_position_embeddings。不截斷 prompt、不修改 RoPE、不開 YaRN。
5. generation config 用 `GenerationConfig(**protocol['generation'])` 明確建立並保存完整有效設定，不讓 model.generation_config 的隱藏設定覆寫合約。溫度／top-k／top-p 的順序沿用該版 Transformers，GSM8K／HumanEval 等三 task 完全相同。
6. 固定 runner/scorer source hashes、pip freeze、GPU環境與參數；實作的自我檢查至少包含斷點續跑不重複 key、缺少 think end 不得評分 reasoning、seed 依題目固定、OOM 不當答錯、score 不做 best-of-3。

### 3. Pilot 後自動進入正式測試

- 每 task 封裝原順序前 4 題，使用 replicate seed=20260926，完整 32768 token budget；放 `pilot/`，不混進正式。
- 試跑成功、輸入一致、無 OOM／程序錯誤、thinking 正常結束並有 final 後，直接啟動正式，不再要求一般確認。答錯不算工程失敗，不改 prompt 追求 pilot 高分。
- pilot 如有 cap／未結束 thinking／擷取失敗，保留結果並回報，先檢查 runner／tokenizer／parser；不得擅自換 seed、調 sampling、縮短或延長 budget。技術修復若改到凍結合約，另建版本並回報。
- 正式按 replicate seed 外圈，HumanEval → GSM8K → MATH-500，題序依封裝原順序。每題都保存並 fsync；可安全續跑；第一次完整 seed 結果標 provisional，三次全數完成才標 complete。
- 記錄 `output_ids` 必須只包含新生成 tokens（排除 prompt），保存 `skip_special_tokens=False` 原文。使用 `common.final_ids(output_ids)`；只把最後答案 token IDs 解碼後交給 scorer。缺失／重複 `</think>` 或空 final 算錯，不从 reasoning 抓答案。不要輸出／執行 reasoning 內的 code。
- token cap 的樣本仍在分母中；已完成 thinking 且有 final 的截斷樣本照固定 scorer 評分並另標 cap。沒有 final 算錯，不重抽。infra errors/OOM 使 task incomplete，修復後可同 seed 重試並保留 audit。

### 4. 使用封裝評分器

- GSM8K／MATH-500：使用 `scoring.score_math_final(task, final_text, expected)`；GSM8K exact numeric match，MATH-500 conservative normalized EM。不得改成 LLM judge 或只給某一組更寬鬆的 symbolic grader。
- HumanEval：用 `scoring.code_candidate(truth['prompt'], final_text, truth['entry_point'])`，再附加 `truth['test']` 與 `check(entry_point)`。只評原始 HumanEval，不是 HumanEval+。候選檔名 `s<seed_index>_p<prompt_index:03d>.py`；每題存 candidate hash。
- 不在主機直接執行模型生成的程式。取得 `protocol.json` 中固定 digest 的 Python Docker image，使用封裝 `sandbox.py --candidates-dir ... --output ... --workers 4`。先測正確程式、assertion failure、無窮迴圈三種控制，確認 pass/fail/10 秒 timeout。
- Sandbox 基礎設施錯誤必須報錯，不算答錯或成功；保存 image digest、candidate hash、status。無 final 的 HumanEval 記錯，不建立可執行 candidate。
- 每 task 每 seed 報 correct/N、accuracy、95% Wilson CI；跨三 seeds 報平均 accuracy 與 sample SD（ddof=1）。不算 pass@3，不挑最好一次，也不把三 task 合成一個平均分。
- 若實作 aggregate CI，使用按題目 cluster bootstrap：同一題三 seed 一起重抽，10000 次、seed=20260926；不可把 3N 答案當獨立題目套 Wilson。比較其他組時報百分點差並保留跨平台限制。

### 5. 保存與回報

- 結果目錄：`results/qwen3-thinking-quality-v1/8b/bf16-ar/<hardware-id>/<run-id>/`，新 run 不覆寫舊資料。
- `manifest.json`：bundle／protocol／model／tokenizer／runner／scorer hashes，完整有效 GenerationConfig，軟硬體版本、GPU、dtype、attention、device map、開始／結束時間。
- 每題 JSONL：task、id、replicate_seed、derived_seed、input_ids/input_hash、output_ids、raw_text、final_text、thinking_complete、input/output/thinking/final token counts、EOS/cap、elapsed、peak memory、status/error；分數保存在獨立逐題檔案。
- `quality.json`、`per_example.csv`、`summary.csv`、`progress.json`、`errors.jsonl`、`commands.jsonl`、`environment.txt`、`RESULTS_ZH.md`、檔案 SHA256 清單與壓縮包，不含模型權重。
- 先回報硬體與 preflight／pilot 結果及 ETA；正式期間定期更新完成題數、當前 task/seed 與錯誤。結束後回報三 task 各 seed、平均±SD、截斷與無 final 比率，提供結果包位置。
- 此測試是品質對照；RTX 6000 的 elapsed／peak memory 只作附帶記錄，不和 R9700 的數據直接計算 kernel 加速比。
