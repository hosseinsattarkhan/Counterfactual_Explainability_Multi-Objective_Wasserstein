# Operational validation: referral-policy selection on WDBC

## Data and decision problem
The study uses the Wisconsin Diagnostic Breast Cancer (WDBC) data, 569 historical observations (212 malignant, 357 benign) with 30 measured features, as bundled in scikit-learn. Five stratified out-of-fold logistic-regression evaluations produce observed error and referral rates. Each policy is a referral threshold between 0.10 and 0.80. Three objectives are minimized: the false-negative rate among malignant cases, the false-positive rate among benign cases, and the overall referral rate. Each fold is one scenario, and the empirical scenario probabilities equal the test-fold proportions. The five-fold AUC is a model-discrimination diagnostic, not a measure of clinical utility.

The preference center is w = (0.65, 0.20, 0.15), with perturbation vertices (+0.10, −0.05, −0.05), (−0.10, +0.05, +0.05), and (0, 0, 0). These values are illustrative and were not elicited from clinicians. The scenario ground distance is max(0.02, |prevalence_i − prevalence_j|) off the diagonal, which is a modeling choice and not a calibrated medical distance. The Wasserstein radii 0, 0.0001, 0.001, 0.003, and 0.01 are expressed in these ground-distance units.

## Computational protocol
For each of the 8 policies and 5 Wasserstein radii, the exact uniform optimality gap over the finite-support Wasserstein ball is computed through a transport LP. If the policy is valid at the preference center, the critical preference-stability radius is computed with the linear-fractional LP solver (`exact_radius_v9.py`), and a point just below the returned radius is checked independently with the vertex-and-transport separation oracle. A bisection result is recorded for comparison.

To rerun: `cd Implementation && python scripts/operational_validation_v12.py`

## Results
- 40 policy–radius combinations were evaluated.
- Number of uniformly valid policies by Wasserstein radius: 1 at 0, 1 at 0.0001, 1 at 0.001, 0 at 0.003, and 0 at 0.01.
- The only valid policy at the first three radii has referral threshold 0.30.
- Critical stability radii are computed for these three valid cases (0.896, 0.857, and 0.489); all three below-radius independent audits pass.
- For the other 37 combinations the preference center itself fails uniform validity, so a stability radius centered at the chosen preferences is not defined.
- The mean five-fold ROC AUC is approximately 0.9950; this characterizes classifier discrimination and does not establish operational benefit.

## Interpretation
That no policy is uniformly valid at radii of 0.003 and above is an informative negative result: requiring one decision to outperform every alternative under every distribution in the ball can be too demanding even with costs derived from real observations. The infeasible combinations are reported rather than filtered out.

## Limitations
The underlying data are public and real, but the referral thresholds, preference weights, and Wasserstein metric are constructed for this experiment. The five folds are not independent deployment environments. The study includes no external validation, clinical calibration, prospective data, or clinician preference elicitation, and it supports no claim of operational readiness or of a validated medical decision rule.

## Data provenance
- UCI WDBC: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- scikit-learn loader: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
