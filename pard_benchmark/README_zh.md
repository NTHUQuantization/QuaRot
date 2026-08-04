# PARD / PARD2 Llama 3.1 Decode Benchmark

這是一套獨立的 Transformers+ speculative decoding 評估，不會修改現有 QuaRot decode wrapper。
PARD verifier 流程改寫自 AMD-AGI/PARD commit
`6f279bf3f1680e0b5d71c562ca5b91bdeef4c038`，上游採 MIT License。

## 1. HF token

AMD PARD/PARD2 與官方測試用的 Unsloth target 都是公開模型；目前專案使用的
`meta-llama/Llama-3.1-8B` 是 gated model。若本機 cache 不完整：

```bash
cp .hf_env.example .hf_env
# 編輯 .hf_env，只填在等號後方：HF_TOKEN=hf_xxx
```

`.hf_env` 已被 Git 忽略。runner 透過 Docker env-file 傳入，程式不會把 token
寫入命令列、log、JSON 或 CSV。既有 `.hf_env` 不需要重建，也不會被 setup 覆寫。
新下載統一放在 `.hf_cache/pard`；setup 只以 symlink 重用既有 Base snapshot，
不修改舊 cache 或 lock。

## 2. 執行

GPU preflight 預設要求執行前的 VRAM 使用量不超過 1 GiB。若有其他工作，runner
會直接停止且不會終止外部程序。

```bash
# 只下載／讀取 config，先驗證模型家族、vocab、PARD token 與 PARD2 TD shape
./.venv-pard/bin/python -m pard_benchmark.compat_check

# 先做 eager smoke：Base 與 Instruct、context 128、16 tokens
./run_pard_benchmark.sh --phase smoke

# k=4/8/12/15 tuning
./run_pard_benchmark.sh --phase tune

# 使用 tuning 最佳 k 跑 context 128/1024/4096
./run_pard_benchmark.sh --phase formal

# 依序執行全部階段
./run_pard_benchmark.sh --phase all
```

若模型均已下載，可加上 `--local-files-only`。只有在明確接受數據受其他程序干擾時，
才使用 `--allow-busy-gpu`；該結果不適合作為正式速度或記憶體結論。

單一案例也可在容器／隔離 venv 內執行：

```bash
./.venv-pard/bin/python -m pard_benchmark.benchmark \
  --mode pard --target base --draft-k 12 \
  --context-len 128 --max-new-tokens 128 \
  --compile-mode eager --out-dir pard_decode_results
```

## 3. 模式與公平性

- `ar`：相同 target 的 target-only greedy baseline。
- `pard`：`amd/PARD-Llama-3.2-1B` target-independent parallel draft。
- `pard2-ti`：PARD2 checkpoint，不注入 target hidden states。
- `pard2-td`：注入四層 target hidden states；僅允許官方
  `unsloth/Llama-3.1-8B-Instruct` target。

Base grid 是 AR/PARD/PARD2-TI；Instruct grid 另加 PARD2-TD。所有正式結果使用
BF16、batch 1、greedy decode，並預設忽略 EOS 以固定生成完整 N tokens；如需一般
文字生成語意可加 `--stop-on-eos`。speedup 只對同 target/context/compile mode 的 AR。

## 4. 輸出

`pard_decode_results/` 會保留每個隔離程序的 JSON/log，並生成：

- `environment.json`：模型 revision、ROCm/PyTorch/Transformers/GPU。
- `latency_repeats.csv`、`latency_summary.csv`：TTFT、steady decode、E2E、接受率。
- `memory.csv`：target/draft/cache/decode 各階段的 process 與 GPU-wide 記憶體。
- `generation_check.csv`：與 AR 的 exact greedy token parity。
- `report_zh.md`：speedup、VRAM headroom 與採用判定。

token parity 失敗的 speculative 結果只供除錯，不可宣稱為 lossless speedup。
