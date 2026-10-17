# KL Annealing 實驗報告

日期：2026-04-23

## 1. 實驗目的
在固定 Teacher Forcing 設定下，比較不同 KL annealing strategy 與 learning rate 的表現與穩定性，評估最佳設定。

## 2. 實驗設定
- 主要資料來源：results/kl_annealing_best_psnr.csv
- 主分析 run tag：kl_sweep_20260422_173414
- 固定條件：
  - tfr=0.7
  - tfr_sde=20
  - tfr_d_step=0.05
  - scheduled_sampling=false（per-epoch）
- 掃描維度：
  - lr: 5e-4, 8e-4
  - kl_anneal_type: mono, cyc
  - mono ratio: 0.5, 0.75
  - cyc cycle: 2, 5, 7, 10
  - cyc ratio: 0.5, 0.75

## 3. 執行結果摘要（主 run tag）
- 總組數：15
- 成功組數：13
- 失敗組數：2
- 成功率：86.67%
- 成功組平均 best_PSNR：35.652268
- 最佳組合：kl_mono_lr8em4_r0p5（best_PSNR=37.070621）

## 4. 各組結果總表（主 run tag）

| exp_name | lr | kl_anneal_type | cycle | ratio | best_PSNR | status |
|---|---:|---|---:|---:|---:|---|
| kl_cyc_lr5em4_c2_r0p5 | 5e-4 | cyc | 2 | 0.5 | 35.87903594970703 | ok |
| kl_cyc_lr5em4_c2_r0p75 | 5e-4 | cyc | 2 | 0.75 | 36.22402572631836 | ok |
| kl_cyc_lr5em4_c5_r0p5 | 5e-4 | cyc | 5 | 0.5 | 34.90787124633789 | ok |
| kl_cyc_lr5em4_c5_r0p75 | 5e-4 | cyc | 5 | 0.75 | 35.23333740234375 | ok |
| kl_cyc_lr5em4_c7_r0p5 | 5e-4 | cyc | 7 | 0.5 | 34.36869430541992 | ok |
| kl_cyc_lr5em4_c7_r0p75 | 5e-4 | cyc | 7 | 0.75 | 33.22911834716797 | ok |
| kl_cyc_lr5em4_c10_r0p5 | 5e-4 | cyc | 10 | 0.5 | 36.07678985595703 | ok |
| kl_cyc_lr5em4_c10_r0p75 | 5e-4 | cyc | 10 | 0.75 | 35.77939224243164 | ok |
| kl_mono_lr8em4_r0p5 | 8e-4 | mono | 1 | 0.5 | 37.070621490478516 | ok |
| kl_mono_lr8em4_r0p75 | 8e-4 | mono | 1 | 0.75 | 36.97004699707031 | ok |
| kl_cyc_lr8em4_c2_r0p5 | 8e-4 | cyc | 2 | 0.5 | 36.021663665771484 | ok |
| kl_cyc_lr8em4_c2_r0p75 | 8e-4 | cyc | 2 | 0.75 | 36.149688720703125 | ok |
| kl_cyc_lr8em4_c5_r0p5 | 8e-4 | cyc | 5 | 0.5 | NA | failed(1) |
| kl_cyc_lr8em4_c5_r0p75 | 8e-4 | cyc | 5 | 0.75 | 35.56919860839844 | ok |
| kl_cyc_lr8em4_c7_r0p5 | 8e-4 | cyc | 7 | 0.5 | NA | failed(1) |

## 5. 排名（成功組 Top 5）
1. kl_mono_lr8em4_r0p5: 37.070621
2. kl_mono_lr8em4_r0p75: 36.970047
3. kl_cyc_lr5em4_c2_r0p75: 36.224026
4. kl_cyc_lr8em4_c2_r0p75: 36.149689
5. kl_cyc_lr5em4_c10_r0p5: 36.076790

## 6. 分組統計與觀察

### 6.1 依 learning rate
- lr=5e-4：total=8, ok=8, success_rate=100.00%, avg_psnr_success=35.212283
- lr=8e-4：total=7, ok=5, success_rate=71.43%, avg_psnr_success=36.356244

解讀：
- 8e-4 平均 PSNR 較高，但穩定性明顯較差。
- 5e-4 雖平均分數稍低，但穩定性最佳（本輪 100% 成功）。

### 6.2 依 KL strategy
- mono：count=2, avg_psnr=37.020334
- cyc：count=11, avg_psnr=35.403529

解讀：
- 在本輪範圍下，mono 明顯優於 cyc（但 mono 組數較少，需注意樣本數）。

### 6.3 cyc 內部（成功組）
- by cycle:
  - cycle=2: avg_psnr=36.068604
  - cycle=10: avg_psnr=35.928091
  - cycle=5: avg_psnr=35.236802
  - cycle=7: avg_psnr=33.798906
- by ratio:
  - ratio=0.5: avg_psnr=35.450811
  - ratio=0.75: avg_psnr=35.364127

解讀：
- cycle=2、10 較佳；cycle=7 最差。
- ratio 0.5 與 0.75 差異小，0.5 略優。

## 7. 失敗案例分析
失敗組：
1. kl_cyc_lr8em4_c5_r0p5
2. kl_cyc_lr8em4_c7_r0p5

錯誤訊息（log）：
- Warning: non-finite loss at Epoch 14. mse=nan, kl=0.103022..., beta=0.010000
  - log: logs/kl_sweep_20260422_173414/kl_cyc_lr8em4_c5_r0p5.log
- Warning: non-finite loss at Epoch 10. mse=nan, kl=0.279757..., beta=0.010000
  - log: logs/kl_sweep_20260422_173414/kl_cyc_lr8em4_c7_r0p5.log

解讀：
- 失敗皆出現在 lr=8e-4 且 cyc 的特定組合，符合「較大 LR 容易在 KL 權重轉換階段不穩定」的風險。

## 8. 歷史紀錄備註
同一 CSV 內存在較早 run tag（例如 1e-4 相關）的失敗紀錄，共 27 筆資料；其中部分舊 log 檔已不存在，無法完整回溯錯誤。
本報告主分析以「log 可追溯且你本輪完成」的 run tag `kl_sweep_20260422_173414` 為主。

## 9. 結論與建議
1. 若優先追求最高 PSNR：可先用 `kl_mono_lr8em4_r0p5` 當基準（本輪最佳 37.070621）。
2. 若優先追求穩定收斂：建議優先使用 lr=5e-4（本輪成功率 100%）。
3. 若持續使用 cyc：建議優先測 cycle=2 或 10，並避開在高 LR 下的高風險組合（例如 cycle=5/7, ratio=0.5）。
4. 下一輪可做小範圍 fine-tune：
   - mono: lr 在 6e-4 到 8e-4 間微調
   - cyc: 以 cycle=2/10 搭配較保守 LR（5e-4 到 7e-4）
