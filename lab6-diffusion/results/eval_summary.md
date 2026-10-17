| Run | CFG | Sampler | Steps | Beta schedule | test.json acc | new_test.json acc | Time |
|---|---|---|---|---|---|---|---|
| eval_train_beta_linear | 3.0 | ddpm | 50 | linear | 0.7917 | 0.8214 | 7.30s |
| eval_train_beta_scaled_linear | 3.0 | ddim | 50 | scaled_linear | 0.7639 | 0.8095 | 7.69s |
| eval_train_beta_squaredcos_cap_v2 | 3.0 | ddim | 50 | squaredcos_cap_v2 | 0.7917 | 0.8333 | 7.38s |
| eval_train_concat | 3.0 | ddpm | 1000 | scaled_linear | 0.4306 | 0.5595 | 108.50s |
| eval_train_cross_attn | 3.0 | ddpm | 1000 | scaled_linear | 0.8611 | 0.9048 | 119.34s |
| train_cross_attn | 3.0 | ddpm | 1000 | scaled_linear | 0.8611 | 0.9048 | 119.01s |
| train_cross_attn_best_cfg1.0_s1000 | 1.0 | ddpm | 1000 | scaled_linear | 0.7222 | 0.8690 | 119.03s |
| train_cross_attn_best_cfg1.0_s50 | 1.0 | ddim | 50 | scaled_linear | 0.6806 | 0.8333 | 7.30s |
| train_cross_attn_best_cfg2.0_s1000 | 2.0 | ddpm | 1000 | scaled_linear | 0.7500 | 0.8929 | 119.91s |
| train_cross_attn_best_cfg2.0_s50 | 2.0 | ddim | 50 | scaled_linear | 0.7778 | 0.8929 | 7.22s |
| train_cross_attn_best_cfg3.0_ddim_s10 | 3.0 | ddim | 10 | scaled_linear | 0.7778 | 0.8452 | 2.48s |
| train_cross_attn_best_cfg3.0_ddim_s100 | 3.0 | ddim | 100 | scaled_linear | 0.7222 | 0.8452 | 12.99s |
| train_cross_attn_best_cfg3.0_ddim_s25 | 3.0 | ddim | 25 | scaled_linear | 0.7778 | 0.8571 | 4.41s |
| train_cross_attn_best_cfg3.0_ddim_s50 | 3.0 | ddim | 50 | scaled_linear | 0.7778 | 0.8452 | 7.28s |
| train_cross_attn_best_cfg3.0_ddpm_s10 | 3.0 | ddpm | 10 | scaled_linear | 0.8611 | 0.8690 | 3.12s |
| train_cross_attn_best_cfg3.0_ddpm_s1000 | 3.0 | ddpm | 1000 | scaled_linear | 0.8611 | 0.9048 | 119.37s |
| train_cross_attn_best_cfg3.0_s100 | 3.0 | ddpm | 100 | scaled_linear | 0.8333 | 0.9167 | 12.96s |
| train_cross_attn_best_cfg3.0_s1000 | 3.0 | ddpm | 1000 | scaled_linear | 0.8611 | 0.9048 | 119.67s |
| train_cross_attn_best_cfg3.0_s25 | 3.0 | ddpm | 25 | scaled_linear | 0.8750 | 0.9405 | 5.40s |
| train_cross_attn_best_cfg3.0_s50 | 3.0 | ddpm | 50 | scaled_linear | 0.8194 | 0.8810 | 7.28s |
| train_cross_attn_best_cfg5.0_s1000 | 5.0 | ddpm | 1000 | scaled_linear | 0.9167 | 0.9286 | 119.29s |
| train_cross_attn_best_cfg5.0_s50 | 5.0 | ddim | 50 | scaled_linear | 0.6806 | 0.6429 | 7.39s |
| train_cross_attn_best_cfg8.0_s1000 | 8.0 | ddpm | 1000 | scaled_linear | 0.8611 | 0.9048 | 119.41s |
| train_cross_attn_best_cfg8.0_s50 | 8.0 | ddim | 50 | scaled_linear | 0.3333 | 0.3929 | 7.32s |
