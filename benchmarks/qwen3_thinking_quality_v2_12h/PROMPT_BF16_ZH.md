請立即把本機 Qwen3-8B BF16 thinking 品質測試切換到 **v2-12h**，完成測試、評分和封裝，不要只提供計畫。

從 `https://github.com/NTHUQuantization/QuaRot.git` 的 `benchmark/qwen3-thinking-quality-handoff` 分支取得 `benchmarks/qwen3_thinking_quality_v2_12h`。若使用者指定 commit，必須 checkout 該 commit。先讀 README_ZH.md、protocol.json 和本文件。

1. 停止**自己**先前的 v1 5949 次生成 service／worker，確保沒有自動重啟；保留舊紀錄和中斷資訊。不要終止別人的工作，不覆寫 v1。另開 v2 結果目錄。
2. v2 固定 HumanEval/GSM8K/MATH-500 **各64題**，使用套件原樣 prompts 和 input token hashes；**1 seed=20260926、每題最多4096 new tokens（thinking+final）**。每題 derived seed 必須用此版 common.sample_seed，不能使用 v1 seed 函數，也不能混入舊樣本。正式共192次生成。
3. BF16 模型沿用 `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`。權重與 KV cache 必須是 BF16。HF Transformers/PyTorch 參考版本沿用 v1 已配置的固定環境；核對 model_hashes.json。不得換精度、checkpoint 或 vLLM 來加速本 BF16 variant。
4. thinking=true，temperature=.6、top_k=20、top_p=.95、min_p=0、batch=1、num_return_sequences=1、num_beams=1、repetition_penalty=1；無 presence/frequency penalty、無額外 stop、無工具/grammar。max_new_tokens=4096、min_new_tokens=0、自然EOS [151645,151643]，不在 </think> 停止。不截 prompt、不強制結束 thinking、不開 YaRN。
5. 在**host**持有 `/tmp/qwen3-thinking-v2-<GPU-UUID>.lock` 後才能啟動 GPU worker；NVFP4 agent 也會用同一把鎖。同卡必須串行，不能各自在容器私有 /tmp 鎖定。GPU忙時等待，不殺掉對方新 v2 工作。把 lock owner/UUID 記在 manifest。
6. 共同 GPU 生成截止 **2026-09-27 10:59:03 Asia/Taipei**，評分/封裝硬截止 **11:59:03**。等待也算時間；不要從自己開始時另計12小時。supervisor 必須在截止時停止自己的生成，保留 partial，未完整則標 budget_incomplete。不要擅自減題或改 token cap。
7. 先處理截圖中已指出的 HumanEval Docker/sandbox問題：核對固定digest image，執行 pass/fail/timeout控制；可用同image與相同限制的rootless Podman適配器，保存程式/命令。不在主機直接執行模型程式。若確實無法解決，立即回報限制，繼續數學部分並將HumanEval標待評分，不能冒稱全數完成。
8. 各task第一題，共3題pilot；用4096正式設定，核對工程行為與ETA。cap/未完成thinking/答錯是4K品質結果，不因此重抽或停掉正式；OOM/NaN/輸入不一致等仍需處理。通過後直接跑完整192題，pilot另存不計分。
9. 使用固定 common.final_ids，只評 </think> 後的final。原始output token IDs和skip_special_tokens=False文字完整保存。無final/未結束thinking算錯；截斷仍在64題分母。GSM8K/MATH用 scoring.score_math_final，HumanEval用共同code_candidate、原tests和sandbox，不換grader。
10. 各task報 correct/64、accuracy/pass@1、95% Wilson CI、截斷/無final/擷取失敗數。這是**固定64題子集、單seed、4K預算thinking**，不是完整benchmark；不報三seed平均/SD，不選best-of，不和v1結果混合。
11. 逐題落盤，支援相同source/model/protocol的續跑；輸出manifest、per_example.csv、summary.csv、quality.json、RESULTS_ZH.md與不含權重的結果包。只與符合相同v2 hashes和規格的其他variant比较。

先回報停止舊service、GPU鎖、sandbox與pilot狀態，再持續執行並定期更新完成題數、實測ETA與截止風險；若估計超時，及早回報，不隱瞞或自行改規格。
