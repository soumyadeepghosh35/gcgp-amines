## Model 1: zero-mean GP, RQ kernel: results

> These numbers come from a FAST run (small subsets, fewer restarts and folds). Rerun without GCGP_FAST_RUN before using them.

- Tb (96 train / 24 test): test R² = 0.903, RMSE = 21.51 K, MAE = 16.01 K; 91.7% of test molecules inside the 95% interval.
- Tm (96 train / 24 test): test R² = 0.421, RMSE = 58.38 K, MAE = 44.01 K; 91.7% of test molecules inside the 95% interval.
- Pc (60 train / 15 test): test R² = 0.739, RMSE = 4.37 bar, MAE = 2.57 bar; 93.3% of test molecules inside the 95% interval.
- Vc (55 train / 14 test): test R² = 0.984, RMSE = 12.66 cm³/mol, MAE = 9.37 cm³/mol; 100.0% of test molecules inside the 95% interval.
- Tc (56 train / 14 test): test R² = 0.777, RMSE = 39.75 K, MAE = 32.96 K; 71.4% of test molecules inside the 95% interval.
- ΔHvap (56 train / 14 test): test R² = 0.710, RMSE = 2.33 kJ/mol, MAE = 1.92 kJ/mol; 92.9% of test molecules inside the 95% interval.
- log P (96 train / 24 test): test R² = 0.971, RMSE = 0.287, MAE = 0.222; 100.0% of test molecules inside the 95% interval.
