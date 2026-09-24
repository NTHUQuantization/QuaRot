# Qwen3-32B × PARD2 TD：S3–S5 執行狀態

> 此檔為 `smoke_train_r1` 當時的歷史狀態，已被 `QWEN3_32B_PARD2_TD_S3_S5_RESULTS_ZH.md` 取代；請勿把下列 STOP 視為目前結論。

日期：2026-08-28（Asia/Taipei）
狀態：S3 preflight 觸發 95% VRAM hard gate；依 frozen parent gate 與使用者原始約束停止。S4、S5 未啟動。

## 已完成的準備

- 下載並稽核 AMD 官方 PARD repository，commit `6f279bf3f1680e0b5d71c562ca5b91bdeef4c038`。
- 固定官方 Qwen3 PARD-2 loss/config：K=16、CAT、CE alpha 0.1、KD alpha 1.0、temperature 1.0、target feature dropout 0.1、COD 0.7/min 0.1。
- 因 32B target 與 drafter 不能同卡 resident，改為數學等價的離線 teacher shards；不保存 full teacher logits，而保存 legacy selected features、final hidden 與 gold-token probabilities。訓練時由 frozen 32B LM head 做 exact full-vocabulary chunked KD，沒有 top-k 近似。
- analytic chunked CE+KD gradient 已對照 full-vocabulary PyTorch autograd；連同既有測試共 5/5 passed。
- 建立 1M-token train split：2,279 samples、1,000,461 tokens，general/code/math 各約 333.5K tokens；formal overlap=0，4 筆與 S0 重疊者已拒絕。
- 建立 response-bearing training validation：12 samples、5,577 tokens，三 strata 各 4；對 formal/S0/train overlap 全為 0。

## Teacher-cache smoke

- Requested range 0–512 cumulative tokens；因維持完整 sample boundary，實際 1,011 tokens、2 samples。
- Throughput 147.90 tok/s。
- External peak VRAM 28,253 MiB / 86.60%。
- Peak hotspot/memory 71°C / 46°C。
- Shard 41,419,973 bytes，SHA256 `49f1cf8efd3d8a4dc6f662fa801e24c47a3bf7028259810b5a0fb6d542916472`。
- 結論：32B full-logits/CAT teacher extraction 通過資源 gate。

## Projection trainer smoke

第一次 `smoke_train` 因 wrapper 遺漏 batch dimension，在 rotary position embedding 前 shape mismatch；峰值僅 4,877 MiB，保留失敗 artifact 後以 `smoke_train_r1` 修正。

`smoke_train_r1` 使用第一筆 499-token code sample：

- K=16 + COD 後 expanded positions：1,964。
- Attention mask：約 3,857,296 elements / 14.71 MiB；它本身不是主要 VRAM 來源。
- 在 training 前的 validation forward，external VRAM 達 31,105 MiB / 95.34%，超過 95% hard gate。
- hotspot/memory 僅 41°C / 40°C，排除熱問題。
- gate signal 後 ROCm queue 回報 hardware exception；程序無法由 SIGINT/TERM 清理，最後以 SIGKILL 終止。GPU 已回到約 57 MiB。
- 尚未完成任何 optimizer step，沒有產生 S3 checkpoint。

## 根因與建議

峰值來自 frozen 0.6B draft 為了把 gradient 傳回 projection input，仍需保留 K=16 expanded sequence 的完整 forward activations。實作漏用了 AMD 官方 Qwen3 config 已指定的 `gradient_checkpointing: True`；同時 validation 不需要 backward，卻仍保留了 projection/draft graph。

建議先批准 `smoke_train_r2`，只做以下數值/記憶體修正：

1. 啟用官方 gradient checkpointing。
2. Validation 將 projected input detach 並以 no-grad forward，不保存 backward graph。
3. K=16、COD、loss、dropout、legacy taps、affine calibration、learning-rate schedule、exact vocabulary KD 全部不變。

這不修改 QuaRot 或 PARD2 inference/training 架構，也不改單一實驗變因。若 r2 仍觸發 gate，下一個 fallback 才是把 training context 切成最多 128 tokens；該方法會改變 context distribution，必須使用另一 experiment ID，不應靜默套用。

## Gate 決策

- S3：`STOP / awaiting guidance`。
- S4：未啟動；parent S3 未通過。
- S5：未啟動；parent S4 未通過。

主要 artifacts：

```text
qwen3_32b_results/td_calibration/S3_S5_data/train_manifest.json
qwen3_32b_results/td_calibration/S3_S5_data/training_validation_manifest.json
qwen3_32b_results/td_calibration/S3_projection_micro/smoke_teacher/teacher_manifest.json
qwen3_32b_results/td_calibration/S3_projection_micro/smoke_train/monitor.log
qwen3_32b_results/td_calibration/S3_projection_micro/smoke_train_r1/monitor.log
qwen3_32b_results/td_calibration/S3_projection_micro/smoke_train_r1/monitor_amd_smi.csv
qwen3_32b_results/td_calibration/S3_projection_micro/smoke_train_r1/monitor_safety_gate.log
qwen3_32b_results/td_calibration/S3_projection_micro/smoke_train_r1/blocker_analysis.json
```
