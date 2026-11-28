# Part 1 findings — Expected annual loss and social vulnerability

**Q1.** Total expected annual loss differs across SOVI ratings (n = 3,221, epsilon-squared = 0.048, p = 3.00e-32), with median losses of $4,152,816 in Very Low and $8,356,645 in Very High.

**Line 1 — Loss type.** Building loss follows total loss, population-equivalence dollars peak in Relatively High, and agricultural loss has a stronger rank association across ratings after division by agricultural value (epsilon-squared rises from 0.017 to 0.064).

**Line 2 — Hazard × loss type.** Hurricane population equivalence has the strongest raw association among specific loss types (rho = +0.393), while tsunami building loss has the strongest exposure-normalized association among specific loss types in a much smaller cohort (n = 108, epsilon-squared = 0.209, rho = -0.456).

### Where does SOVI matter most?

**Largest raw association.** Hurricane population-equivalence loss in the headline all-county grid (n = 3,221, epsilon-squared = 0.148, rho = +0.393, BH-adjusted p = 2.92e-100) rises with vulnerability. From Very Low to Very High, median annual losses were $0, $69, $249, $879, $7,500. The association stays positive after matching loss to exposure (n = 2,307, epsilon-squared = 0.148, rho = +0.389).

**Largest associations after controls.** Hurricane population equivalence rose 6.80% per 10 SOVI points (state-clustered 95% CI 1.22% to 12.70%, n = 2,294). Earthquake population equivalence rose 3.39% per 10 SOVI points (state-clustered 95% CI 0.50% to 6.36%, n = 3,211). These are the two positive associations whose state-clustered intervals exclude zero among the selected targets. The independent role of SOVI after controls is small. These are adjusted associations in observational, modeled data.

**Occurrence and magnitude.** Stage 3 estimates loss magnitude among counties with positive loss. The Stage 2 headline grid captures both whether counties have any loss and the size of their losses. For cold-wave agriculture, 2,219 of 3,221 counties have positive loss. Its negative all-county pattern and near-zero fully adjusted positive-loss slope answer different questions.

### Slope decomposition and practical size

Shares use the fitted log10-dollar slopes per SOVI point. They describe how the slope changes across models. Sign reversals are shown without shares.

|Target                            |n     |Model 1 b |Model 2 b |Model 3 b |State share |Exposure and size share |Remaining share |Note                     |
|:---------------------------------|:-----|:---------|:---------|:---------|:-----------|:-----------------------|:---------------|:------------------------|
|County total                      |3,213 |0.00454   |0.00196   |0.00054   |56.7%       |31.5%                   |11.8%           |                         |
|Hurricane population equivalence  |2,294 |0.01927   |0.00658   |0.00286   |65.8%       |19.3%                   |14.8%           |                         |
|Cold-wave agriculture             |2,211 |-0.00414  |0.00042   |0.00039   |—           |—                       |—               |Coefficient changes sign |
|Earthquake population equivalence |3,211 |0.01322   |0.00390   |0.00145   |70.5%       |18.5%                   |11.0%           |                         |
|Winter-weather building           |3,086 |-0.00775  |-0.00005  |0.00048   |—           |—                       |—               |Coefficient changes sign |

**Fully adjusted percent change across a 50-point SOVI gap.** Lower and upper are state-clustered 95% limits.

|Target                            |n     |Estimate |Lower   |Upper  |
|:---------------------------------|:-----|:--------|:-------|:------|
|County total                      |3,213 |6.36%    |-2.09%  |15.54% |
|Hurricane population equivalence  |2,294 |38.97%   |6.24%   |81.79% |
|Cold-wave agriculture             |2,211 |4.60%    |-27.43% |50.76% |
|Earthquake population equivalence |3,211 |18.15%   |2.55%   |36.13% |
|Winter-weather building           |3,086 |5.63%    |-16.50% |33.62% |

### Raw and normalized losses

Normalization strengthens the SOVI pattern for total and building losses. For county total, epsilon-squared increases from 0.048 to 0.085. Dividing by building value gives more weight to a given dollar loss in counties with smaller building inventories. Median building value in Very High is $4.93 billion, below the middle ratings' $6.75 billion, $5.87 billion, $6.11 billion in rating order. This denominator difference helps explain why the high end separates more clearly. The normalized pattern is J-shaped, with similar levels across the lower ratings and a rise across the top two. Raw total loss does not clearly separate Very High from Relatively High.

### Methods

Checked log-loss distributions with Q–Q plots, Shapiro–Wilk and Fligner–Killeen. Compared ranks with Kruskal–Wallis, epsilon-squared, Holm-adjusted Wilcoxon and signed Cliff's delta, and checked robustness with Welch ANOVA and Holm-adjusted Welch pairs. Used Spearman for trend, a county bootstrap for the score correlation, and LOESS and nested natural spline models for shape. Adjusted hazard Kruskal–Wallis p-values with separate Benjamini–Hochberg families for headline, applicable-only and normalized grids (62, 62, 62 tests, respectively). The headline grid places structural inapplicability at zero for the all-county comparison. The applicable-only grid retains recorded zeros and omits structural missingness. Derived agricultural losses below one cent count as zero, with a one-dollar sensitivity check. Neither cutoff changes the exported RDS. Sparse-hazard analyses separate occurrence and positive magnitude. Regressions use log10 of positive raw loss, conventional and state-clustered SEs. Forest performance uses state-grouped cross-validation. DALEX importance and profiles describe the full-data fits.

### Limitations and additional data

- FEMA EAL is modeled. Observed event losses would help validate the estimates.
- County-level summaries hide variation across neighborhoods and census tracts.
- The supplied data lack individual SOVI components such as income, disability, education, and language.
- 8 Connecticut counties lack matching Census covariates and are absent from models requiring population.
- County agricultural value approximates agricultural exposure because hazard-specific agricultural exposure is unavailable for the selected cold-wave target and other non-drought agricultural cells.
- Hazard cells overlap. Total combines building, population-equivalence, and agricultural loss where those components apply.
- Counties can be dependent within states. Clustered regression intervals and state-grouped prediction address this in Stage 3. The earlier county-based tests and bootstrap use counties as units.
- Puerto Rico (78 rated county equivalents in the input) is excluded from the forests because the four Census regions do not cover it.
- Sparse hazards such as tsunami, avalanche, and volcanic activity are excluded from Stage 3 by the 300-positive-county minimum. Coastal flooding does not rank among the selected hazards.

### Why it matters

Counties rated Very High in social vulnerability had a median expected annual loss of $8,356,645, compared with $4,152,816 in Very Low counties. That difference matters for planning because the same dollar loss can represent a larger share of what a county has exposed. Dividing by building value made the gap clearer in the highest-vulnerability counties. Hurricane population-equivalence loss, FEMA's dollar measure of impacts on people, showed the strongest raw relationship with vulnerability. After comparing positive-loss counties while accounting for state, exposure, frequency, area, and population, the hurricane and earthquake relationships remained positive but were much smaller. Adding the vulnerability score made little difference to how accurately the forests predicted loss in held-out states. For planning, these results support reading vulnerability alongside hazard exposure and county size, and asking separately where loss occurs and how large it is where it occurs. The estimates describe annual modeled losses.

### Output inventory

Paths are relative to the outputs folder. The preparation files come from the cleaning export.

|file                                                                |rubric                                                                            |
|:-------------------------------------------------------------------|:---------------------------------------------------------------------------------|
|analysis.html                                                       |Q1, Line 1, Line 2, where does SOVI matter most, data description, we wish we had |
|figures/forest_sovi_partial_dependence_ice_and_adjusted_slope.png   |Where does SOVI matter most                                                       |
|figures/hazard_loss_exposure_normalized_epsilon_squared_heatmap.png |Line 2, where does SOVI matter most                                               |
|figures/hazard_loss_raw_epsilon_squared_heatmap.png                 |Line 2, where does SOVI matter most                                               |
|figures/loss_type_raw_and_normalized_loess.png                      |Line 1                                                                            |
|figures/q1_log_total_loss_qq_by_rating.png                          |Q1                                                                                |
|figures/q1_total_loss_and_building_value_ratio_boxplots.png         |Q1                                                                                |
|figures/q1_total_loss_and_ratio_by_sovi_score_loess.png             |Q1                                                                                |
|figures/sovi_adjusted_regression_residual_diagnostics.png           |Where does SOVI matter most                                                       |
|figures/sovi_nested_regression_coefficient_intervals.png            |Where does SOVI matter most                                                       |
|findings.md                                                         |Q1, Line 1, Line 2, where does SOVI matter most, data description, we wish we had |
|part1_clean.csv                                                     |Data description                                                                  |
|part1_clean.rds                                                     |Data description                                                                  |
|part1_data_dictionary.csv                                           |Data description                                                                  |
|tables/building_value_by_sovi_rating.csv                            |Q1, data description                                                              |
|tables/forest_dalex_full_fit_permutation_importance.csv             |Where does SOVI matter most                                                       |
|tables/forest_dalex_sovi_ice_100_counties.csv                       |Where does SOVI matter most                                                       |
|tables/forest_dalex_sovi_partial_dependence_and_adjusted_slope.csv  |Where does SOVI matter most                                                       |
|tables/hazard_agricultural_cent_vs_dollar_sensitivity.csv           |Line 2, we wish we had                                                            |
|tables/hazard_agricultural_loss_cutoff_counts.csv                   |Line 2, we wish we had                                                            |
|tables/hazard_loss_applicability_sensitivity.csv                    |Line 2, where does SOVI matter most                                               |
|tables/hazard_loss_exposure_rank_changes.csv                        |Line 2, where does SOVI matter most                                               |
|tables/hazard_loss_grid_all_counties_headline.csv                   |Line 2, where does SOVI matter most                                               |
|tables/hazard_loss_grid_applicable_counties.csv                     |Line 2, where does SOVI matter most                                               |
|tables/hazard_loss_grid_exposure_normalized.csv                     |Line 2, where does SOVI matter most                                               |
|tables/hazard_loss_top_five_excluding_informational_totals.csv      |Line 2, where does SOVI matter most                                               |
|tables/hazard_loss_top_five_raw_and_normalized.csv                  |Line 2, where does SOVI matter most                                               |
|tables/loss_type_a_ratio_wilcoxon_holm_matrix.csv                   |Line 1                                                                            |
|tables/loss_type_a_ratio_wilcoxon_pairs.csv                         |Line 1                                                                            |
|tables/loss_type_b_ratio_wilcoxon_holm_matrix.csv                   |Line 1                                                                            |
|tables/loss_type_b_ratio_wilcoxon_pairs.csv                         |Line 1                                                                            |
|tables/loss_type_eal_vala_wilcoxon_holm_matrix.csv                  |Line 1                                                                            |
|tables/loss_type_eal_vala_wilcoxon_pairs.csv                        |Line 1                                                                            |
|tables/loss_type_eal_valb_wilcoxon_holm_matrix.csv                  |Line 1                                                                            |
|tables/loss_type_eal_valb_wilcoxon_pairs.csv                        |Line 1                                                                            |
|tables/loss_type_eal_valpe_wilcoxon_holm_matrix.csv                 |Line 1                                                                            |
|tables/loss_type_eal_valpe_wilcoxon_pairs.csv                       |Line 1                                                                            |
|tables/loss_type_group_medians.csv                                  |Line 1                                                                            |
|tables/loss_type_pe_per_capita_wilcoxon_holm_matrix.csv             |Line 1                                                                            |
|tables/loss_type_pe_per_capita_wilcoxon_pairs.csv                   |Line 1                                                                            |
|tables/loss_type_tests_and_effect_sizes.csv                         |Line 1                                                                            |
|tables/model_target_selection.csv                                   |Where does SOVI matter most                                                       |
|tables/q1_eal_valt_wilcoxon_holm_matrix.csv                         |Q1                                                                                |
|tables/q1_eal_valt_wilcoxon_pairs.csv                               |Q1                                                                                |
|tables/q1_fligner_log_total_loss.csv                                |Q1                                                                                |
|tables/q1_group_summary.csv                                         |Q1                                                                                |
|tables/q1_kruskal_wallis.csv                                        |Q1                                                                                |
|tables/q1_score_linear_vs_spline.csv                                |Q1                                                                                |
|tables/q1_shapiro_by_rating.csv                                     |Q1                                                                                |
|tables/q1_spearman_trend.csv                                        |Q1                                                                                |
|tables/q1_t_ratio_wilcoxon_holm_matrix.csv                          |Q1                                                                                |
|tables/q1_t_ratio_wilcoxon_pairs.csv                                |Q1                                                                                |
|tables/q1_welch_anova.csv                                           |Q1                                                                                |
|tables/q1_welch_pairs.csv                                           |Q1                                                                                |
|tables/sovi_agricultural_regressions_one_dollar_sensitivity.csv     |Where does SOVI matter most                                                       |
|tables/sovi_incremental_prediction_and_profile_summary.csv          |Where does SOVI matter most                                                       |
|tables/sovi_nested_regressions_conventional_and_clustered.csv       |Where does SOVI matter most                                                       |
|tables/sovi_slope_decomposition_and_50_point_effects.csv            |Where does SOVI matter most                                                       |
|tables/sparse_hazard_occurrence_and_positive_magnitude.csv          |Line 2, we wish we had                                                            |
|tables/state_grouped_model_comparison_by_fold.csv                   |Where does SOVI matter most                                                       |
|tables/state_grouped_model_comparison_pooled.csv                    |Where does SOVI matter most                                                       |
