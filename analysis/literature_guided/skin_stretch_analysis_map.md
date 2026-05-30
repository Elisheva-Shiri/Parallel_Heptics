# Literature-guided skin-stretch analysis map

This folder records how recent Prof. Ilana Nisky skin-stretch / cutaneous-haptics papers were translated into repository analyses.

## Sources reviewed (last 10 years: 2016-05-16 to 2026-05-16)

- Farajian, Leib, Kossowsky, Zaidenberg, Mussa-Ivaldi, Nisky, and Vaadia (2020), *Stretching the skin immediately enhances perceived stiffness and gradually enhances the predictive control of grip force*, eLife. The paper used stiffness-discrimination psychometric curves, PSE/JND, repeated-probe motor-control analysis, peak grip-force/load-force ratio, and grip-force/load-force regression.
- Farajian, Leib, and Nisky (2020 PHRI), *The Effect of Vision on the Augmentation of Perceived Stiffness by Adding Artificial Skin Stretch to Kinesthetic Force*. The paper used a forced-choice paradigm across feedback conditions and analyzed PSE/JND with repeated-measures ANOVA plus Holm-Bonferroni post-hoc tests.
- Farajian, Leib, Kossowsky, and Nisky (2021), *Visual Feedback Weakens the Augmentation of Perceived Stiffness by Artificial Skin Stretch*, IEEE Transactions on Haptics. The paper kept the forced-choice/PSE/JND framing and explicitly tested visual-feedback condition effects.
- Farajian, Leib, Kossowsky, and Nisky (2023), *Direction-Specific Effects of Artificial Skin-Stretch on Stiffness Perception and Grip Force Control*, IEEE Transactions on Haptics. The abstract reports positive/negative stretch-direction effects on perceived stiffness and grip force, and a mechanoreceptor preferred-direction model.

## Implemented mapping

| Literature method | Repository category | Implementation |
| --- | --- | --- |
| PSE/JND bias and sensitivity; Weber-style sensitivity normalization | Psychophysics | `twoafc_psychophysics.add_fit_delta_columns` now emits `pse_delta_from_standard`, `abs_pse_delta_from_standard`, `jnd_over_standard`, and `weber_fraction`; fit group/scope comparisons include them. |
| Repeated probing motor-control changes | Kinematics | `kinematics_analysis.compute_tracking_kinematics` now derives jerk (`jx_px_s3`, `jy_px_s3`, `jerk_px_s3`) plus per-segment `mean_jerk_px_s3`, `max_jerk_px_s3`, and dimensionless `normalized_jerk_cost` as smoothness/correction metrics. |
| Time course of probing contacts and active exploration | Probing | `probing_analysis.detect_probe_events` now reports `first_probe_latency_s`, center/side/exploration dwell times, and dwell fractions; probing summaries and group/scope comparisons include the relevant metrics. |
| Feedback/direction-condition comparisons | Existing group/scope framework | The prior group/all/subgroup/participant comparison expansion remains applicable to these new metrics whenever condition columns are present. |

## Not implemented yet

- Direct grip-force/load-force regression and peak grip-force/load-force ratios require raw grip/load force signals that are not present in the current tracking-only inputs.
- Mechanoreceptor preferred-direction model fitting from the 2023 direction-specific paper requires explicit signed skin-stretch direction/force cues beyond the current stiffness/finger/tracking tables.
