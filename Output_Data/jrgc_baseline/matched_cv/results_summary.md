## Matched baselines: results

All models are scored on the curated molecules of each property in the same 10 folds as notebook 04.

- **Tb** (886 molecules): raw GC estimate R² = 0.768, RMSE = 32.85 K; linear recalibration (out-of-fold) R² = 0.831, RMSE = 28.04 K; GC-GP R² = 0.848, RMSE = 26.57 K. GC-GP changes RMSE by -19.1 % against the raw GC estimate (significant) and by -5.27 % against the linear recalibration (significant).
- **Tm** (1839 molecules): raw GC estimate R² = -0.329, RMSE = 102.30 K; linear recalibration (out-of-fold) R² = 0.571, RMSE = 58.11 K; GC-GP R² = 0.616, RMSE = 55.00 K. GC-GP changes RMSE by -46.2 % against the raw GC estimate (significant) and by -5.36 % against the linear recalibration (significant).
- **Pc** (75 molecules): raw GC estimate R² = 0.817, RMSE = 5.43 bar; linear recalibration (out-of-fold) R² = 0.808, RMSE = 5.57 bar; GC-GP R² = 0.809, RMSE = 5.55 bar. GC-GP changes RMSE by +2.3 % against the raw GC estimate (not significant) and by -0.33 % against the linear recalibration (not significant).
- **Vc** (69 molecules): raw GC estimate R² = 0.983, RMSE = 20.59 cm³/mol; linear recalibration (out-of-fold) R² = 0.987, RMSE = 18.12 cm³/mol; GC-GP R² = 0.987, RMSE = 18.12 cm³/mol. GC-GP changes RMSE by -12.0 % against the raw GC estimate (not significant) and by -0.04 % against the linear recalibration (not significant).
- **Tc** (70 molecules): raw GC estimate R² = 0.881, RMSE = 30.36 K; linear recalibration (out-of-fold) R² = 0.866, RMSE = 32.18 K; GC-GP R² = 0.866, RMSE = 32.18 K. GC-GP changes RMSE by +6.0 % against the raw GC estimate (significant) and by -0.00 % against the linear recalibration (not significant).
- **ΔHvap** (70 molecules): raw GC estimate R² = 0.883, RMSE = 2.40 kJ/mol; linear recalibration (out-of-fold) R² = 0.910, RMSE = 2.11 kJ/mol; GC-GP R² = 0.922, RMSE = 1.96 kJ/mol. GC-GP changes RMSE by -18.3 % against the raw GC estimate (significant) and by -7.04 % against the linear recalibration (not significant).
- **log P** (2978 molecules): raw GC estimate R² = 0.855, RMSE = 0.750; linear recalibration (out-of-fold) R² = 0.866, RMSE = 0.723; GC-GP R² = 0.883, RMSE = 0.675. GC-GP changes RMSE by -10.0 % against the raw GC estimate (significant) and by -6.67 % against the linear recalibration (significant).

GC-GP is significantly more accurate than the raw GC estimate for Tb, Tm, ΔHvap, log P and significantly less accurate for Tc. Against the linear recalibration it is significantly more accurate for Tb, Tm, log P; for the other properties the kernel adds no significant accuracy beyond the refitted line.

A difference is called significant when the bootstrap 95% interval of the RMSE difference excludes zero and the Wilcoxon signed-rank p value is below 0.05.