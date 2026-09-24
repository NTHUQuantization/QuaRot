# Fused PARD2：Qwen3-8B、14B、32B

本分支為 AMD ROCm / HIP 的 dense Qwen3 W4A4KV4 runtime。
共同契約是 batch 1、greedy、關閉 thinking、正常 EOS、`draft_k=15`、native GQA KV4。
profile 用於驗證架構與 checkpoint provenance；不會下載模型或自動切換本機路徑。

## 支援矩陣

| Target | `--benchmark-profile` | 正式模式 | TI drafter | TD drafter |
|---|---|---|---|---|
| Qwen3-8B | `qwen3_8b` | AR / TI / TD | PARD2-Qwen3-8B | PARD2-Qwen3-8B |
| Qwen3-14B | `qwen3_14b` | AR / TI / TD | PARD2-Qwen3-8B | PARD2-Qwen3-14B |
| Qwen3-32B | `qwen3_32b` | AR / TI | PARD2-Qwen3-8B | 僅 experimental proxy |

32B 的 `--td-proxy-profile qwen3-14b-on-qwen3-32b` 必須明確指定，
只供實驗，不能當作 canonical TD 或加速保證。32B 的正式 target 使用 GPTQ；RTN 保留作比較。
8B / 14B 使用 RTN。所有模式的 target 仍是相同 W4A4KV4 runtime。

## 環境與編譯

在已安裝 ROCm PyTorch 的 Python 環境中，依根目錄 README 建置 hadacore 與 QuaRot。
專案預設編譯 `gfx1201`；其他架構可設 `QUAROT_HIP_ARCHS`，但不代表已通過相同驗證。
既有 32B 紀錄環境為 PyTorch `2.9.1+git5bc97ba`、HIP `7.2.26015-fc0010cf6a`、
Transformers `4.57.6`，與根目錄歷史 requirements 的版本不同；重現時請記錄實際環境。

```bash
python -m pip install -e third-party/hadacore --no-build-isolation
python -m pip install -e . --no-build-isolation
python -m pytest tests -q
```

修改 HIP 原始碼後須重建 extension。若測試找不到 `selected_bpre_kernel_name`，
先確認載入的 `quarot._HIP` 是否為本次編譯產物。

## 固定模型版本

| 資源 | Revision |
|---|---|
| `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| `Qwen/Qwen3-14B` | `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| `Qwen/Qwen3-32B` | `9216db5781bf21249d130ec9da846c4624c16137` |
| `amd/PARD2-Qwen3-8B` | `67a1516c8f6fc145cda99916799a0cbb3a4af135` |
| `amd/PARD2-Qwen3-14B` | `679eff0b65ffaf5abd2dadd21a17909562935798` |

用 `hf download MODEL_ID --revision REVISION` 取得表中的資源，保留
`models--…/snapshots/REVISION` 路徑結構以供 provenance 驗證。
後續指令中的 `SOURCE`、`DRAFT` 都指向這種完整 snapshot 目錄。

## 建立 target checkpoint

以下變數請換成本機路徑。14B / 32B 的嚴格 preflight 會核對架構、固定來源、
格式 v2、`grouped_h256_v1`、activation clip 0.9 與自包含 HadK metadata。
因此請從固定 snapshot 轉換，不要手動拼裝或省略 metadata。

```bash
SOURCE=/models/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18
TARGET=/models/qwen3_14b_w4a4kv4
python e2e/checkpoint_utils/quantize_checkpoint.py \
  --model "$SOURCE" --output "$TARGET" \
  --quant-method rtn --w-clip --seed 0 --w-groupsize -1 \
  --rotation-device cuda --rotation-dtype float32
```

8B 換成對應的 source / target。32B 換成 32B snapshot，並將量化參數改為：

```bash
python e2e/checkpoint_utils/quantize_checkpoint.py \
  --model "$SOURCE" --output "$TARGET" \
  --quant-method gptq --dataset wikitext2 --nsamples 128 --seqlen 2048 \
  --w-clip --seed 0 --w-groupsize -1 --percdamp 0.01 \
  --rotation-device cuda --rotation-dtype float32
```

streaming converter 逐層量化及寫入 shard，保留 Qwen3 q/k norm 的來源 dtype，
並輸出 rotation signs、final norm 與 provenance。權重不納入 Git。

## 生成與 smoke test

從 repo 根目錄執行。以下為 14B TI 範例；`SOURCE` 與 `TARGET` 沿用上節：

```bash
PROFILE=qwen3_14b
DRAFT=/models/hub/models--amd--PARD2-Qwen3-8B/snapshots/67a1516c8f6fc145cda99916799a0cbb3a4af135
export QUAROT_FUSED_K1=0
python e2e/pard2.py \
  --benchmark-profile "$PROFILE" --mode pard2-ti \
  --target "$TARGET" --tokenizer "$SOURCE" --draft "$DRAFT" \
  --compile-mode eager --max-cache-len 2048 --max-new-tokens 32 \
  --prompt 'Explain speculative decoding in one paragraph.'
```

8B / 32B 切換 `PROFILE`、`SOURCE`、`TARGET`，TI 仍使用同一 8B drafter。
AR 使用 `--mode ar`，不載入 draft。14B TD 改用表中的 14B draft，
並指定 `--mode pard2-td --td-basis-fold`；8B TD 使用 8B draft。
32B 保持 `QUAROT_FUSED_K1=0`，其既有正式路徑使用
`QUAROT_QWEN3_32B_GROUPED_NWAVES=4`、`QUAROT_QWEN3_32B_MULTI_NWAVES=2`。

## Benchmark 與 qualification

資料集來自固定 PARD source；benchmark 會核對三個 JSONL 的 SHA256。
取得資料後以 `--data-root` 指定，無須沿用原機器的兄弟 repo 路徑：

```bash
git clone https://github.com/AMD-AGI/PARD.git third-party/PARD
git -C third-party/PARD checkout 6f279bf3f1680e0b5d71c562ca5b91bdeef4c038
DATA_ROOT="$PWD/third-party/PARD/datas/bmk"
mkdir -p results
python e2e/benchmark_pard2.py \
  --benchmark-profile "$PROFILE" --mode pard2-ti --dataset humaneval \
  --target "$TARGET" --tokenizer "$SOURCE" --draft "$DRAFT" \
  --data-root "$DATA_ROOT" --compile-mode eager \
  --max-cache-len 2048 --generated-tokens 32 --warmups 1 --sweeps 1 --limit 1 \
  --output results/pard2-ti_humaneval_smoke.json
```

smoke 含 `--limit`，不具正式資格。完整測試移除 `--limit`，使用
`--max-cache-len 8192 --generated-tokens 256 --warmups 8 --sweeps 3`，
並分別以獨立程序執行三資料集（HumanEval / GSM8K / MATH-500）及支援的各模式。
結果命名為 `ar_humaneval.json`、`pard2-ti_humaneval.json` 等
`MODE_DATASET.json`，存入同一結果目錄。
8B / 14B 的 compiled draft 可使用 `--compile-mode max-autotune-no-cudagraphs
--precompile-draft-shapes`；32B 歷史正式測試採 eager。

```bash
python e2e/qualify_pard2.py \
  --benchmark-profile "$PROFILE" --result-dir results \
  --output results/qualification.json
```

qualifier 檢查 protocol、AR token parity、配對速度、穩定性與 VRAM。
benchmark 的 protocol 標記不等於已通過所有 gates。外部 VRAM 監測工具為
`attach_amd_smi_monitor.py`；完整重現應保存 AMD SMI CSV、runtime provenance
與 HIP binary hash。歷史 `run_qwen3_*` shell scripts 含實驗機路徑與 binary pins，
使用前須調整，不能當成跨機器預設值。

## 歷史結果與實驗工具

[Source index](PARD2_SOURCE_INDEX.md) 列出程式與報告。
8B 舊整合報告中的「parity 未完成」描述是早期狀態；後續資格紀錄見
[post-merge results](../pard2_post_merge_results/README.md)。
32B TD 校正與訓練工具保留供後續研究，與正式 AR/TI qualification 分開。
本次整理不代表重新執行全部模型的完整品質與效能評測。
