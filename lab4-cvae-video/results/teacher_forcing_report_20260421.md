# Teacher Forcing 實驗報告

日期：2026-04-21

## 1. 實驗目的
比較不同 Teacher Forcing 設定對影片預測品質（best_PSNR）的影響，並以統一紀錄檔進行橫向分析。

## 2. 資料來源
- 統一結果檔：results/teacher_forcing_best_psnr.csv
- 主要 run tag：tf_sweep_20260420_114549
- 記錄筆數：8 組
- 成功組數：7 組
- 失敗組數：1 組

備註：
目前另有手動執行的 tf_epoch_tfr0p7_sde10_d0p10、tf_epoch_tfr0p7_sde15_d0p10（可見 ckpt 目錄），但這兩組尚未寫入統一 CSV，因此本報告不納入統計。

## 3. 各組結果總表

| exp_name | tfr | tfr_sde | tfr_d_step | scheduled_sampling | best_PSNR | status |
|---|---:|---:|---:|---|---:|---|
| tf_epoch_tfr1p0_sde20_d0p05 | 1.0 | 20 | 0.05 | false | 36.579910 | ok |
| tf_epoch_tfr1p0_sde25_d0p10 | 1.0 | 25 | 0.10 | false | NA | failed(1) |
| tf_epoch_tfr1p0_sde30_d0p05 | 1.0 | 30 | 0.05 | false | 32.501438 | ok |
| tf_epoch_tfr0p7_sde20_d0p05 | 0.7 | 20 | 0.05 | false | 36.934925 | ok |
| tf_epoch_tfr0p7_sde25_d0p10 | 0.7 | 25 | 0.10 | false | 37.164116 | ok |
| tf_frame_tfr1p0_sde20_d0p05 | 1.0 | 20 | 0.05 | true | 31.862175 | ok |
| tf_frame_tfr1p0_sde25_d0p10 | 1.0 | 25 | 0.10 | true | 31.425781 | ok |
| tf_frame_tfr0p7_sde30_d0p05 | 0.7 | 30 | 0.05 | true | 30.841696 | ok |

## 4. 排名（僅成功組）
1. tf_epoch_tfr0p7_sde25_d0p10: 37.164116
2. tf_epoch_tfr0p7_sde20_d0p05: 36.934925
3. tf_epoch_tfr1p0_sde20_d0p05: 36.579910
4. tf_epoch_tfr1p0_sde30_d0p05: 32.501438
5. tf_frame_tfr1p0_sde20_d0p05: 31.862175
6. tf_frame_tfr1p0_sde25_d0p10: 31.425781
7. tf_frame_tfr0p7_sde30_d0p05: 30.841696

## 5. 統計觀察
- 成功組整體平均 best_PSNR：33.901434
- 以 Teacher Forcing 施作方式分組：
  - per-epoch（scheduled_sampling=false）平均 35.795097（4 組）
  - per-frame（scheduled_sampling=true）平均 31.376551（3 組）
- 以初始 tfr 分組：
  - tfr=0.7 平均 34.980246（3 組）
  - tfr=1.0 平均 33.092326（4 組）

解讀：
- 本次設定下，per-epoch 明顯優於 per-frame。
- tfr=0.7 在本輪資料中表現優於 tfr=1.0。
- 目前最佳設定為 tf_epoch_tfr0p7_sde25_d0p10。

## 6. 失敗案例
- 失敗實驗：tf_epoch_tfr1p0_sde25_d0p10
- 狀態：failed(1)
- 失敗位置：Epoch 17
- 錯誤摘要：non-finite loss（mse=nan, kl=0.012247..., beta=0.485714）
- 參考 log：logs/tf_sweep_20260420_114549/tf_epoch_tfr1p0_sde25_d0p10.log

## 7. 結論
本次 Teacher Forcing sweep 結果顯示：
1. 建議優先採用 per-epoch 訓練策略。
2. 在目前實驗範圍內，tfr=0.7 具較佳穩定度與 PSNR 表現。
3. 推薦下一輪以 tf_epoch_tfr0p7_sde25_d0p10 為基準，再延伸微調 tfr_sde 與 tfr_d_step。
