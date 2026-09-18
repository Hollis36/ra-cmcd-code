# W1 weight fit on the training split

Boxes 1506: matched 1418, unmatched 82, ambiguous 6 (excluded). Simplex step 0.01, 5151 allocations.

- fixed 0.45/0.35/0.20: AUROC 0.9912, separation 0.422
- best allocation [0.0, 1.0, 0.0]: AUROC 0.9922, separation 0.439, Spearman vs fixed 0.9993
- AUROC range over the whole simplex [0.9826, 0.9922]; 35.8% of allocations within 0.001 of the maximum, 90.4% within 0.005; worst [0.0, 0.0, 1.0] at 0.9826
- confidence only (no structural term): AUROC 0.9893
- unconstrained logistic regression (5-fold CV): AUROC 0.9905; structural allocation from its coefficients [0.39, 0.42, 0.19]
- bootstrap arg-max (B=1000, step 0.05): mean [0.02, 0.98, 0.0], 95% interval of w_loc [-0.00, 0.00]
- minimum Spearman rho against the fixed score over the simplex: 0.9912
