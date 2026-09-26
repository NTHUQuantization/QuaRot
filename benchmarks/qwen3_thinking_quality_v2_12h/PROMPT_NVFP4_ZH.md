請立即把本機 **Qwen3-8B ModelOpt NVFP4 W4A4＋vLLM RHT-INT4 KV cache** 品質測試切換到 **v2-12h**，完成測試、評分和封裝，不要只提供計畫。

從 `https://github.com/NTHUQuantization/QuaRot.git` 的 `benchmark/qwen3-thinking-quality-handoff` 分支取得 `benchmarks/qwen3_thinking_quality_v2_12h`。若使用者指定 commit，必須 checkout 該 commit。先讀 README_ZH.md、protocol.json 和本文件。

1. 停止**自己**舊v1品質service/worker，保留資料並禁止自動重啟；不覆寫舊結果、不停止別人的工作。另開v2目錄。
2. 正式固定三task各64題、單一replicate seed=20260926、每題最多4096 new tokens（thinking+final），共192次。固定selection/prompts/input token hashes；用此版common.sample_seed產生每題seed，明確傳入該request的SamplingParams.seed。不得重用v1輸出或v1 seed函數。
3. **沿用測速時同一個NVFP4 checkpoint、ModelOpt版本、vLLM fork/commit、RHT-INT4 KV kernel、scales、rotation及相關flags**。保存checkpoint/source/extension hashes並證明實際生效。不要換成HF推論、重做量化、改FP8/BF16 KV或一般vLLM。protocol的bf16_reference環境要求只適用BF16組，不要求NVFP4改成該環境。
4. 追查來源模型是否為Qwen/Qwen3-8B revision b968826d9c46dd6066d109eabc6255188de91218。NVFP4權重另存自身hash，不能拿BF16 weight hash比對NVFP4。若來源無法確認，明列限制，不宣稱同來源的純量化退化。
5. 固定tokenizer/template，thinking=true、add_generation_prompt=true；直接送核對過的prompt token IDs避免二次template。SamplingParams：n=1、temperature=.6、top_k=20、top_p=.95、min_p=0、repetition_penalty=1、presence_penalty=0、frequency_penalty=0、max_tokens=4096、min_tokens=0、ignore_eos=False、stop_token_ids=[151645,151643]。不設額外stop string/grammar/logit bias/thinking budget，不在</think>停止；不使用speculative、best-of或beam。保存實際有效參數，不能被server預設覆寫。
6. batch/concurrency=1，無跨題prefix caching；保留原始RoPE，不開YaRN、不截prompt。確保context能容納input+4096。必要的max_model_len變更另記錄，其他計算實作維持測速版本。
7. 在**host**取得 `/tmp/qwen3-thinking-v2-<GPU-UUID>.lock`，持有到worker完全退出。BF16 agent用同一把鎖，同GPU必須串行，不在各自容器私有/tmp使用不同鎖，不停止對方新v2工作。等待也算共同預算。
8. 所有agent的GPU生成截止 **2026-09-27 10:59:03 Asia/Taipei**，評分/封裝硬截止 **11:59:03**；不是從自己啟動再算12小時。supervisor到截止停止自己未完成生成、保存partial、標budget_incomplete。禁止私自減題、改cap或把缺失題當答錯來宣稱完成。
9. 正式前確認HumanEval固定digest Python sandbox，驗證pass/fail/timeout；可用同image/相同限制的rootless Podman適配，保留程式/命令。禁止主機直接執行生成程式。權限真的無法解決時立即報告，數學部分繼續、HumanEval標待評分。
10. 各task第一題作pilot，共3題，正式4096設定。核對engine確實使用NVFP4和RHT-INT4、輸入hash、sampling、EOS與ETA。cap/無final/答錯是budgeted品質結果，不改seed重抽；工程檢查通過後直接正式192題。
11. 保存完整generated token IDs、原文、finish_reason/stop_reason。若vLLM省略終止EOS，保存實際metadata，不偽造token。用common.final_ids，只評</think>後final。沒有合法final算錯、截斷保留分母；沿用共同scoring.py與HumanEval tests/sandbox，不換grader。
12. 交付每task correct/64、accuracy/pass@1、Wilson95% CI、截斷/未完成thinking/無final/擷取失敗；完整逐題檔、manifest、summary.csv、quality.json、RESULTS_ZH.md及不含權重結果包。不報三seed均值/SD，不稱完整benchmark。

若同機BF16的v2結果已存在，先核對protocol/selection/input/scorer hashes、預算和seed，再產生逐task比較CSV，差距用百分點。只把它解讀為量化＋後端的整套實作品質差異，相同seed不要求相同文字。不混入v1、greedy或不同budget結果。

先回報舊service停止、原測速版本識別、GPU鎖、sandbox及pilot結果，然後持續執行並定期更新ETA；預估超时要立即報告，不能默改規格。
