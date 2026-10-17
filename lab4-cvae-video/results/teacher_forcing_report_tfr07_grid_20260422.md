# Teacher Forcing 實驗報告（tfr_sde × tfr_d_step）

日期：2026-04-22

## 1. 實驗設定
- 固定模式：per-epoch（scheduled_sampling=false）
- 固定 Teacher Forcing Ratio：tfr=0.7
- 掃描參數：
  - tfr_sde：10, 15, 20, 25, 30
  - tfr_d_step：0.05, 0.10
- 目標：比較不同 decay 起始點與 decay 步長對 best_PSNR 的影響

## 2. 資料來源與統計範圍
- 統一結果檔：[results/teacher_forcing_best_psnr.csv](results/teacher_forcing_best_psnr.csv)
- 主要新跑批次：[logs/tf_sde_dstep_tfr0p7_20260421](logs/tf_sde_dstep_tfr0p7_20260421)
- 為了完整 10 組網格，本報告同時納入前次已存在且條件相同的 2 組：
  - tf_epoch_tfr0p7_sde20_d0p05（run_tag: tf_sweep_20260420_114549）
  - tf_epoch_tfr0p7_sde25_d0p10（run_tag: tf_sweep_20260420_114549）

## 3. 結果總表（完整 10 組）

| tfr_sde | tfr_d_step | exp_name | best_PSNR | status | run_tag |
|---:|---:|---|---:|---|---|
| 10 | 0.05 | tf_epoch_tfr0p7_sde10_d0p05 | 37.561539 | ok | tf_sde_dstep_tfr0p7_20260421 |
| 10 | 0.10 | tf_epoch_tfr0p7_sde10_d0p10 | NA | failed(1) | tf_sde_dstep_tfr0p7_20260421 |
| 15 | 0.05 | tf_epoch_tfr0p7_sde15_d0p05 | 36.536583 | ok | tf_sde_dstep_tfr0p7_20260421 |
| 15 | 0.10 | tf_epoch_tfr0p7_sde15_d0p10 | NA | failed(1) | tf_sde_dstep_tfr0p7_20260421 |
| 20 | 0.05 | tf_epoch_tfr0p7_sde20_d0p05 | 36.934925 | ok | tf_sweep_20260420_114549 |
| 20 | 0.10 | tf_epoch_tfr0p7_sde20_d0p10 | 36.158928 | ok | tf_sde_dstep_tfr0p7_20260421 |
| 25 | 0.05 | tf_epoch_tfr0p7_sde25_d0p05 | 31.862030 | ok | tf_sde_dstep_tfr0p7_20260421 |
| 25 | 0.10 | tf_epoch_tfr0p7_sde25_d0p10 | 37.164116 | ok | tf_sweep_20260420_114549 |
| 30 | 0.05 | tf_epoch_tfr0p7_sde30_d0p05 | 35.519661 | ok | tf_sde_dstep_tfr0p7_20260421 |
| 30 | 0.10 | tf_epoch_tfr0p7_sde30_d0p10 | 36.290157 | ok | tf_sde_dstep_tfr0p7_20260421 |

## 4. 排名（僅成功組）
1. tf_epoch_tfr0p7_sde10_d0p05：37.561539
2. tf_epoch_tfr0p7_sde25_d0p10：37.164116
3. tf_epoch_tfr0p7_sde20_d0p05：36.934925
4. tf_epoch_tfr0p7_sde15_d0p05：36.536583
5. tf_epoch_tfr0p7_sde30_d0p10：36.290157
6. tf_epoch_tfr0p7_sde20_d0p10：36.158928
7. tf_epoch_tfr0p7_sde30_d0p05：35.519661
8. tf_epoch_tfr0p7_sde25_d0p05：31.862030

## 5. 統計分析
- 完整網格：10 組
- 成功：8 組
- 失敗：2 組
- 成功組平均 best_PSNR：36.003492

### 5.1 依 tfr_d_step 分組
- d_step=0.05：平均 35.682948（5 組），最高 37.561539，最低 31.862030
- d_step=0.10：平均 36.537734（3 組），最高 37.164116，最低 36.158928

觀察：
- 在成功樣本中，d_step=0.10 的平均值較高。
- 但 d_step=0.10 在 sde=10、15 出現 2 次訓練失敗，顯示穩定性風險較高。

### 5.2 依 tfr_sde 分組
- sde=10：37.561539（僅 1 組成功；另一組失敗）
- sde=15：36.536583（僅 1 組成功；另一組失敗）
- sde=20：平均 36.546926（2 組皆成功）
- sde=25：平均 34.513073（2 組皆成功，但組內差距大）
- sde=30：平均 35.904909（2 組皆成功）

觀察：
- sde=20 與 sde=30 的穩定性相對較佳（兩個 d_step 都成功）。
- sde=25 對 d_step 高度敏感：0.10 很好、0.05 明顯掉分。

## 6. 失敗案例分析
- tf_epoch_tfr0p7_sde10_d0p10
  - log：[logs/tf_sde_dstep_tfr0p7_20260421/tf_epoch_tfr0p7_sde10_d0p10.log](logs/tf_sde_dstep_tfr0p7_20260421/tf_epoch_tfr0p7_sde10_d0p10.log)
  - 錯誤：non-finite loss at Epoch 8（mse=nan, beta=0.228571）
- tf_epoch_tfr0p7_sde15_d0p10
  - log：[logs/tf_sde_dstep_tfr0p7_20260421/tf_epoch_tfr0p7_sde15_d0p10.log](logs/tf_sde_dstep_tfr0p7_20260421/tf_epoch_tfr0p7_sde15_d0p10.log)
  - 錯誤：non-finite loss at Epoch 7（mse=nan, beta=0.200000）

## 7. 結論與建議
1. 若優先追求最高 PSNR，最佳組合為 sde=10, d_step=0.05（37.561539）。
2. 若優先追求穩定性，建議從 sde=20 或 sde=30 開始，兩者在 0.05 與 0.10 都能成功收斂。
3. d_step=0.10 可能帶來較高平均 PSNR，但在較早 decay（sde=10,15）有明顯 NaN 風險。
4. 下一輪建議：
   - 穩定路線：以 sde=20, d_step=0.05 或 sde=30, d_step=0.10 為主。
   - 挑戰路線：在 sde=10, d_step=0.05 周邊微調（例如 d_step=0.04, 0.06）並增加重複次數驗證穩定性。
